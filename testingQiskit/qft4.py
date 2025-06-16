#!/usr/bin/env python3

import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend for saving files
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import QFT


def save_circuit_diagrams(initial, qft_block, full, folder):
    """Save circuit diagrams as PNG files"""
    init_name = initial.name.lower()
    init_path = os.path.join(folder, f"initial_{init_name}.png")
    qft_path = os.path.join(folder, "qft_block.png")
    full_path = os.path.join(folder, "full_circuit.png")

    initial.draw(output="mpl", fold=-1).savefig(init_path)
    print(f"Saved: {init_path}")

    qft_block.draw(output="mpl", fold=-1).savefig(qft_path)
    print(f"Saved: {qft_path}")

    full.draw(output="mpl", fold=-1).savefig(full_path)
    print(f"Saved: {full_path}")


def compute_l2_error(counts, n, shots):
    N = 2 ** n
    ideal = np.full(N, 1 / N)
    sampled = np.zeros(N)
    for bitstring, count in counts.items():
        idx = int(bitstring[::-1], 2)
        sampled[idx] = count / shots
    return np.linalg.norm(sampled - ideal, ord=2)


def main():
    n = 4
    shots = 16384

    # Choose initial state: "zero" for |0>^n, "ghz" for GHZ state
    initial_state = "ghz"  # options: "zero", "ghz"

    # Dynamically set folder based on initial_state
    folder = f"qft4-results_{initial_state}"
    os.makedirs(folder, exist_ok=True)
    print(f"Saving results to: {os.path.abspath(folder)}")

    # 1. Prepare initial state circuit
    if initial_state == "ghz":
        initial = QuantumCircuit(n, name="zero")
        initial.h(0)
        for i in range(1, n):
            initial.cx(0, i)
    else:
        initial = QuantumCircuit(n, name="ZeroState")  # |0>^n, no gates

    # 2. QFT block (with swaps)
    qft_block = QFT(num_qubits=n, do_swaps=True, name="QFT")

    # 3. Assemble full circuit with measurement
    full = QuantumCircuit(n, n)
    full.append(initial.to_instruction(), range(n))
    full.append(qft_block.to_instruction(), range(n))
    full.measure(range(n), range(n))

    # 4. Save diagrams
    save_circuit_diagrams(initial, qft_block, full, folder)

    # 5. Run simulation
    backend = AerSimulator(method="statevector")
    tqc = transpile(full, backend)
    _ = backend.run(tqc, shots=32).result()  # Warmup

    start = time.perf_counter()
    result = backend.run(tqc, shots=shots).result()
    elapsed = time.perf_counter() - start

    counts = result.get_counts()
    l2 = compute_l2_error(counts, n, shots)

    # 6. Print results
    print("\nMeasurement results:")
    for bitstring, count in sorted(counts.items()):
        print(f"{bitstring}: {count}")
    print(f"\nElapsed: {elapsed:.4f}s | L2 error: {l2:.6f}")

    # 7. Save custom histogram PNG
    hist_path = os.path.join(folder, "histogram.png")
    states = sorted(counts.keys())
    labels = [f"|{s}>" for s in states]
    values = [counts[s] for s in states]

    plt.figure(figsize=(8, 5))
    plt.bar(labels, values)
    plt.xlabel("State $|\\dots\\rangle$")
    plt.ylabel("Counts")
    plt.title(f"QFT Output Distribution ({n}-qubit, {initial_state.upper()})")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(hist_path)
    plt.close()
    print(f"Saved: {hist_path}")

    # 8. Save counts CSV
    df = pd.DataFrame({
        "bitstring": list(counts.keys()),
        "counts": list(counts.values())
    })
    csv_path = os.path.join(folder, "histogram.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
