#!/usr/bin/env python3

import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QFT
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel,
    depolarizing_error,
    amplitude_damping_error,
    phase_damping_error,
    pauli_error,
)

def compute_l2_error(counts, n, shots):
    N = 2 ** n
    ideal = np.full(N, 1 / N)
    sampled = np.zeros(N)
    for bitstring, count in counts.items():
        idx = int(bitstring[::-1], 2)
        sampled[idx] = count / shots
    return np.linalg.norm(sampled - ideal, ord=2)

def create_noise_model(noise_type):
    noise_model = NoiseModel()

    # Strong error parameters
    pam = 0.4  # high amplitude damping probability

    if noise_type == "depolarizing":
        err_1q = depolarizing_error(0.2, 1)  # strong 1-qubit noise
        err_2q = depolarizing_error(0.2, 2)  # stronger 2-qubit noise
        noise_model.add_all_qubit_quantum_error(err_1q, ["u1", "u2", "u3", "h", "x", "measure"])
        noise_model.add_all_qubit_quantum_error(err_2q, ["cx"])

    elif noise_type == "amplitude_damping":
        amp = amplitude_damping_error(0.2)
        noise_model.add_all_qubit_quantum_error(amp, ["u1", "u2", "u3", "h", "x", "measure"])
        # optional: noise_model.add_all_qubit_quantum_error(amp.tensor(amp), ["cx"])

    elif noise_type == "phase_damping":
        phase = phase_damping_error(0.2)  # strong dephasing
        noise_model.add_all_qubit_quantum_error(phase, ["u1", "u2", "u3", "h", "x", "measure"])

    elif noise_type == "bit_flip":
        p = 0.8  # or higher for visibility
        bit_flip = pauli_error([("X", p), ("I", 1 - p)])
        noise_model.add_all_qubit_quantum_error(bit_flip, ["u1", "u2", "u3", "h", "x", "measure"])


    return noise_model


def simulate_qft(n, shots, noise_type):
    folder = f"qft4-{noise_type}"
    os.makedirs(folder, exist_ok=True)

    initial = QuantumCircuit(n, name="ZeroState")
    qft_block = QFT(num_qubits=n, do_swaps=True, name="QFT")

    full = QuantumCircuit(n, n)
    full.append(initial.to_instruction(), range(n))
    full.append(qft_block.to_instruction(), range(n))
    full.measure(range(n), range(n))

    noise_model = create_noise_model(noise_type)
    backend = AerSimulator(noise_model=noise_model, method="density_matrix")

    tqc = transpile(full, backend)
    _ = backend.run(tqc, shots=32).result()

    start = time.perf_counter()
    result = backend.run(tqc, shots=shots).result()
    elapsed = time.perf_counter() - start

    counts = result.get_counts()
    l2 = compute_l2_error(counts, n, shots)

    print(f"[{noise_type}] Elapsed: {elapsed:.4f}s | L2 error: {l2:.6f}")

    # Histogram
    hist_path = os.path.join(folder, f"histogram_{noise_type}.png")
    labels = [f"|{s}>" for s in sorted(counts)]
    values = [counts[s] for s in sorted(counts)]

    plt.figure(figsize=(8, 5))
    plt.bar(labels, values)
    plt.xlabel("State $|\\dots\\rangle$")
    plt.ylabel("Counts")
    plt.title(f"QFT Output with {noise_type.replace('_', ' ').title()} Noise")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(hist_path)
    plt.close()
    print(f"Saved: {hist_path}")

    # CSV
    df = pd.DataFrame({"bitstring": list(counts.keys()), "counts": list(counts.values())})
    csv_path = os.path.join(folder, "histogram.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

if __name__ == "__main__":
    n = 4
    shots = 16384
    noise_types = ["depolarizing", "amplitude_damping", "phase_damping", "bit_flip"]
    for noise in noise_types:
        simulate_qft(n, shots, noise)
