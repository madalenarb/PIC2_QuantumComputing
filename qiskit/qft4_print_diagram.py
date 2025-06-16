#!/usr/bin/env python3

import os
from math import pi
from qiskit import QuantumCircuit
from qiskit.visualization import circuit_drawer

def qft_unrolled_with_swaps(n: int) -> QuantumCircuit:
    """Explicit QFT on n qubits, including swaps"""
    qc = QuantumCircuit(n, name="QFT_unrolled")

    # QFT part (no approximation)
    for i in range(n):
        qc.h(i)
        for j in range(i + 1, n):
            angle = pi / (2 ** (j - i))
            qc.cp(angle, j, i)

    # Add final swaps to reverse qubit order
    for i in range(n // 2):
        qc.swap(i, n - 1 - i)

    return qc


def save_qft_diagram(qc, filename="qft4_unrolled_swaps.png"):
    os.makedirs("qft4-results", exist_ok=True)
    fig = qc.draw(output="mpl", fold=-1)
    fig.savefig(f"qft4-results/{filename}")


if __name__ == "__main__":
    qc = qft_unrolled_with_swaps(4)
    save_qft_diagram(qc)
    print("✅ Full QFT diagram with swaps saved as qft4-results/qft4_unrolled_swaps.png")
