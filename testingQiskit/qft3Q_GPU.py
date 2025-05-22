#!/usr/bin/env python3
"""
3-qubit Quantum-Fourier-Transform on Qiskit Aer with GPU support
"""

from math import pi
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

def build_qft3() -> QuantumCircuit:
    qc = QuantumCircuit(3, 3)
    qc.h(0)
    qc.cp(pi/2, 0, 1)
    qc.h(1)
    qc.cp(pi/2, 1, 2)
    qc.h(2)
    qc.swap(0, 2)
    qc.measure(range(3), range(3))
    return qc

if __name__ == "__main__":
    qc = build_qft3()
    print("Circuit diagram for 3-qubit QFT:")
    print(qc.draw("text"))

    SHOTS = 512 * 4

    # ← Here’s the only change: ask AerSimulator to run on the GPU
    backend = AerSimulator(
        method='statevector',
        device='GPU',
        batched_shots_gpu=True
    )

    # Transpile for the GPU‐capable backend
    qobj   = transpile(qc, backend)
    result = backend.run(qobj, shots=SHOTS).result()
    counts = result.get_counts(qobj)

    print(f"\nMeasurement distribution ({SHOTS} shots):")
    for bits, freq in sorted(counts.items()):
        print(f"  State |{bits[::-1]}⟩ : {freq}")

    print("\nProbabilities:")
    for bits, freq in sorted(counts.items()):
        print(f"  State |{bits[::-1]}⟩ : {freq/SHOTS:.3f}")

    # Compute L2 error against the ideal uniform distribution
    ideal   = np.full(2**3, 1/8)
    sampled = np.zeros(2**3)
    for bits, freq in counts.items():
        idx = int(bits[::-1], 2)
        sampled[idx] = freq/SHOTS
    l2err = np.linalg.norm(sampled - ideal, 2)

    print(f"\nL2 error from uniform: {l2err:.3e}")
