#!/usr/bin/env python3
import math
import numpy as np
import cudaq
import time
import os
import csv
from typing import List
from collections import Counter

# ──────────────────────────────────────────────────
# 1) Kernel: GHZ preparation → QFT (no swaps)
# ──────────────────────────────────────────────────
@cudaq.kernel
def quantum_fourier_transform_ghz(n_bits: int):
    q = cudaq.qvector(n_bits)
    h(q[0])
    for i in range(1, n_bits):
        x.ctrl(q[0], q[i])
    for i in range(n_bits):
        h(q[i])
        for j in range(i + 1, n_bits):
            angle = (2 * math.pi) / (2 ** (j - i + 1))
            cr1(angle, [q[j]], q[i])
    mz(q)  # Add measurement for histogram


# ──────────────────────────────────────────────────
# 2) Ideal statevector of GHZ → QFT
# ──────────────────────────────────────────────────
def ideal_qft_ghz_state(n_bits: int) -> np.ndarray:
    N = 1 << n_bits
    psi_in = np.zeros(N, complex)
    psi_in[0] = psi_in[-1] = 1 / np.sqrt(2)
    ω = np.exp(2j * np.pi / N)
    F = np.array([[ω ** (j * k) for k in range(N)] for j in range(N)], dtype=complex) / np.sqrt(N)
    return F @ psi_in


# ──────────────────────────────────────────────────
# 3) Histogram saving function
# ──────────────────────────────────────────────────
def save_histogram(counts: dict, folder: str, filename: str = "histogram.csv"):
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, filename)
    with open(filepath, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["bitstring", "count"])
        for bitstring, count in sorted(counts.items()):
            writer.writerow([bitstring, count])


# ──────────────────────────────────────────────────
# 4) Main
# ──────────────────────────────────────────────────
def main():
    cudaq.set_target("qpp-cpu")
    shots = 16384

    for n_bits in [3, 4, 5]:
        print(f"\n=== n_bits = {n_bits} ===")
        start_time = time.time()

        if n_bits == 4:
            counts = cudaq.sample(quantum_fourier_transform_ghz, n_bits, shots_count=shots)

            # Convert keys to padded binary strings
            formatted_counts = {format(int(k, 0), f"0{n_bits}b"): v for k, v in counts.items()}

            print("\nMeasurement results:")
            for bitstring, count in sorted(formatted_counts.items()):
                print(f"{bitstring}: {count}")

            save_histogram(formatted_counts, folder="results/qft4")

        else:
            # Run ideal simulation for comparison
            psi_sim = np.array(cudaq.get_state(quantum_fourier_transform_ghz, n_bits), dtype=complex)
            psi_id = ideal_qft_ghz_state(n_bits)

            print("\nSimulated amplitudes:")
            for i, amp in enumerate(psi_sim):
                print(f"  |{i:0{n_bits}b}⟩ {amp.real:+.3f}{amp.imag:+.3f}j")

            print("\nIdeal amplitudes:")
            for i, amp in enumerate(psi_id):
                print(f"  |{i:0{n_bits}b}⟩ {amp.real:+.3f}{amp.imag:+.3f}j")

            F = abs(np.vdot(psi_id, psi_sim))**2
            rho_sim = np.outer(psi_sim, psi_sim.conj())
            rho_id = np.outer(psi_id, psi_id.conj())
            frob = np.linalg.norm(rho_sim - rho_id)

            print(f"\nFidelity       = {F:.6f}")
            print(f"Frobenius norm = {frob:.6e}")

        end_time = time.time()
        print(f"Elapsed time   = {end_time - start_time:.3f} seconds")


if __name__ == "__main__":
    main()
