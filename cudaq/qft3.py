#!/usr/bin/env python3
import math
import numpy as np
import cudaq
from typing import List


# ──────────────────────────────────────────────────
# 1) Kernel: GHZ preparation → QFT (no swaps)
# ──────────────────────────────────────────────────
@cudaq.kernel
def quantum_fourier_transform_ghz(n_bits: int):  # Add type annotation
    q = cudaq.qvector(n_bits)
    # 1) GHZ prep: (|0...0⟩ + |1...1⟩)/√2
    h(q[0])
    for i in range(1, n_bits):
        x.ctrl(q[0], q[i])
    # 2) QFT proper
    for i in range(n_bits):
        h(q[i])
        for j in range(i + 1, n_bits):
            angle = (2 * math.pi) / (2 ** (j - i + 1))
            cr1(angle, [q[j]], q[i])
    # (no final swap – this matches CUDA-Q’s little-endian indexing)


# ──────────────────────────────────────────────────
# 2) Build the ideal GHZ→QFT statevector
# ──────────────────────────────────────────────────
def ideal_qft_ghz_state(n_bits: int) -> np.ndarray:
    N = 1 << n_bits
    psi_in = np.zeros(N, complex)
    psi_in[0] = psi_in[-1] = 1 / np.sqrt(2)
    ω = np.exp(2j * np.pi / N)
    F = np.array([[ω ** (j * k) for k in range(N)] for j in range(N)], dtype=complex) / np.sqrt(N)
    return F @ psi_in


# ──────────────────────────────────────────────────
# 3) Main: run, print, diagnose
# ──────────────────────────────────────────────────
def main():

    # 1) choose statevector simulator
    cudaq.set_target("qpp-cpu")

    for n_bits in [3,4,5]:
        # 2) draw the circuit
        

        # 3) run and grab the statevector
        psi_sim = np.array(cudaq.get_state(quantum_fourier_transform_ghz, n_bits), dtype=complex)

        # 4) compute ideal
        psi_id = ideal_qft_ghz_state(n_bits)

        # 5) show amplitudes
        print("\nSimulated amplitudes:")
        for i, amp in enumerate(psi_sim):
            print(f"  |{i:03b}⟩ {amp.real:+.3f}{amp.imag:+.3f}j")

        print("\nIdeal amplitudes:")
        for i, amp in enumerate(psi_id):
            print(f"  |{i:03b}⟩ {amp.real:+.3f}{amp.imag:+.3f}j")

        # 6) fidelity and Frobenius norm
        F = abs(np.vdot(psi_id, psi_sim))**2
        rho_sim = np.outer(psi_sim, psi_sim.conj())
        rho_id = np.outer(psi_id, psi_id.conj())
        frob = np.linalg.norm(rho_sim - rho_id)

        print(f"\nFidelity       = {F:.6f}")
        print(f"Frobenius norm = {frob:.6e}")


if __name__ == "__main__":
    main()
