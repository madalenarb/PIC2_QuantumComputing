#!/usr/bin/env python3
import math
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from time import perf_counter
from collections import Counter

def build_qft_no_meas(n_bits: int) -> QuantumCircuit:
    """Build an n-qubit QFT circuit that saves the final statevector."""
    qc = QuantumCircuit(n_bits)
    # Hadamard + controlled-phase
    for k in range(n_bits):
        qc.h(k)
        for j in range(k+1, n_bits):
            qc.cp(math.pi / 2**(j-k), j, k)
    # Swap reversal
    for i in range(n_bits // 2):
        qc.swap(i, n_bits - 1 - i)
    qc.reverse_bits()
    # *** Save the final statevector ***
    qc.save_statevector()
    return qc

def compute_l2_error(counts: dict, n_bits: int, shots: int) -> float:
    """Compute sampling-style L2 error vs uniform."""
    N = 2 ** n_bits
    ideal = np.full(N, 1 / N)
    sampled = np.zeros(N)
    for bitstr, freq in counts.items():
        idx = int(bitstr, 2)
        sampled[idx] = freq / shots
    return np.linalg.norm(sampled - ideal, 2)

def detect_backend():
    """Choose GPU AerSimulator if possible, else CPU."""
    try:
        sim = AerSimulator(method='statevector', device='GPU')
        _ = sim.run(transpile(QuantumCircuit(1), sim), shots=1).result()
        print("→ Using GPU-accelerated AerSimulator\n")
        return sim, "GPU"
    except Exception:
        sim = AerSimulator(method='statevector')
        print("→ GPU not available; using CPU AerSimulator\n")
        return sim, "CPU"

if __name__ == "__main__":
    SHOTS = 2**18
    backend, backend_label = detect_backend()

    print(f"{' q':>3} | {'time(s)':>8} | {'shots/s':>10} | {'L2 err':>9} | backend")
    print("-" * 50)

    for n in range(3, 29):
        # build & transpile
        qc  = build_qft_no_meas(n)
        tqc = transpile(qc, backend)

        # warm-up (32-shot sampling)
        _ = backend.run(tqc, shots=32).result()

        # timed 1-shot statevector evolution
        t0 = perf_counter()
        result = backend.run(tqc, shots=1).result()
        elapsed = perf_counter() - t0

        # extract statevector (now present)
        sv = result.get_statevector()
        probs = np.abs(sv) ** 2

        # sample SHOTS times in Python
        sample_idxs = np.random.choice(len(probs), size=SHOTS, p=probs)
        bitstrs = [format(idx, f'0{n}b') for idx in sample_idxs]
        counts = Counter(bitstrs)

        # compute L2 error
        l2_err = compute_l2_error(counts, n, SHOTS)
        throughput = SHOTS / elapsed

        print(f"{n:3d} | {elapsed:8.3f} | {throughput:10.1f} | {l2_err:9.3e} | {backend_label}")
