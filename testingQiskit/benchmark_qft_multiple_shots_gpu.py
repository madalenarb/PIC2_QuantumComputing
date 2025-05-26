import os
import math
import time
import importlib.util

import numpy as np
import pandas as pd

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

def qft_circuit(n_qubits: int) -> QuantumCircuit:
    """Build an n-qubit QFT with measurements on all qubits."""
    qc = QuantumCircuit(n_qubits, n_qubits)
    for j in range(n_qubits):
        qc.h(j)
        for k in range(j+1, n_qubits):
            qc.cp(math.pi / (2 ** (k - j)), k, j)
    qc.reverse_bits()
    # add measurements so we can get counts
    qc.measure(range(n_qubits), range(n_qubits))
    return qc

def compute_l2_error(counts, n_qubits, shots):
    """Compute the L2 norm between the observed and uniform distributions."""
    N = 2 ** n_qubits
    ideal = np.full(N, 1 / N)
    sampled = np.zeros(N)
    for bitstr, freq in counts.items():
        # bitstr is MSB→LSB, reverse it so index matches
        idx = int(bitstr[::-1], 2)
        sampled[idx] = freq / shots
    return np.linalg.norm(sampled - ideal, ord=2)

def make_backend():
    """Try GPU AerSimulator, fall back to CPU."""
    try:
        sim = AerSimulator(method='statevector', device='GPU')
        # quick check run
        _ = sim.run(transpile(QuantumCircuit(1,1), sim), shots=1).result()
        print("Using GPU AerSimulator\n")
        return sim, "GPU"
    except Exception:
        sim = AerSimulator(method='statevector')
        print("GPU not available, using CPU AerSimulator\n")
        return sim, "CPU"

if __name__ == "__main__":
    shot_counts = [2**i for i in range(10, 20)]  # 2^10…2^19
    max_qubits  = 27

    backend, backend_type = make_backend()

    records = []
    for shots in shot_counts:
        print(f"\n--- Benchmarking {shots} shots ---")
        for n in range(3, max_qubits+1):
            qc = qft_circuit(n)
            tqc = transpile(qc, backend)

            # warm-up
            _ = backend.run(tqc, shots=32).result()

            # timed run
            t0 = time.perf_counter()
            result = backend.run(tqc, shots=shots).result()
            elapsed = time.perf_counter() - t0

            # extract counts and compute L2 error
            counts = result.get_counts()
            l2_err = compute_l2_error(counts, n, shots)
            throughput = shots / elapsed

            print(f"{n:>2} qubits | {shots:>7} shots | {elapsed:6.3f}s | {throughput:8.1f} shots/s | L2 error={l2_err:.6f} | {backend_type}")

            records.append({
                "n_bits":    n,
                "shots":     shots,
                "sim_time_s": elapsed,
                "throughput": throughput,
                "l2_error":  l2_err,
                "backend":   backend_type
            })

    # save all results to CSV
    os.makedirs("Results", exist_ok=True)
    df = pd.DataFrame(records)
    out_path = os.path.join("Results", "benchmark_qft_gpu_multipleshots.csv")
    df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")
