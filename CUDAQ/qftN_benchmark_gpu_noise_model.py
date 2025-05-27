#!/usr/bin/env python3
"""
4-qubit QFT + manual Kraus noise injection (CUDA-Q on NVIDIA GPU)
Sweep p from 0.0 to 1.0 in steps of 0.1 for each channel.
"""

import math, numpy as np, cudaq

# ─── Build 4-qubit QFT kernel ─────────────────────────
def build_qft4():
    @cudaq.kernel
    def qft4():
        q = cudaq.qvector(4)
        for k in range(4):
            h(q[k])
            for j in range(k+1,4):
                theta = 2*math.pi/2**(j-k+1)
                cr1(theta, [q[j]], q[k])
    return qft4

# ─── Ideal QFT state for 4 qubits ──────────────────────
def ideal_qft4() -> np.ndarray:
    N = 1<<4
    psi0 = np.zeros(N, complex)
    psi0[0] = 1.0
    ω = np.exp(2j*math.pi/N)
    F = np.array([[ω**(j*k) for k in range(N)] for j in range(N)])/math.sqrt(N)
    return F @ psi0

# ─── Kraus definitions ─────────────────────────────────
def depolarizing_kraus(p):
    q = p/4
    K0 = math.sqrt(1 - 3*q) * np.eye(2)
    K1 = math.sqrt(q) * np.array([[0,1],[1,0]])
    K2 = math.sqrt(q) * np.array([[0,-1j],[1j,0]])
    K3 = math.sqrt(q) * np.array([[1,0],[0,-1]])
    return [K0,K1,K2,K3]

def amp_damp_kraus(p):
    K0 = np.array([[1,0],[0,math.sqrt(1-p)]])
    K1 = np.array([[0,math.sqrt(p)],[0,0]])
    return [K0,K1]

def phase_flip_kraus(p):
    K0 = math.sqrt(1-p)*np.eye(2)
    K1 = math.sqrt(p)*np.array([[1,0],[0,-1]])
    return [K0,K1]

def bit_flip_kraus(p):
    K0 = math.sqrt(1-p)*np.eye(2)
    K1 = math.sqrt(p)*np.array([[0,1],[1,0]])
    return [K0,K1]

# ─── Apply single-qubit channel ────────────────────────
def apply_channel(rho, kraus_ops, qubit):
    n = int(math.log2(rho.shape[0]))
    I = np.eye(2)
    new = np.zeros_like(rho)
    for K in kraus_ops:
        op = 1
        for q in range(n):
            op = np.kron(op, K if q==qubit else I)
        new += op @ rho @ op.conj().T
    return new

def apply_to_all(rho, kraus_fn, p):
    out = rho.copy()
    for q in range(4):
        out = apply_channel(out, kraus_fn(p), q)
    return out

# ─── Metrics ────────────────────────────────────────────
def fidelity(rho, psi):
    return float(np.real(psi.conj() @ (rho @ psi)))

def fro_norm(rho, rho0):
    return np.linalg.norm(rho - rho0)

# ─── Main ──────────────────────────────────────────────
def main():
    cudaq.set_target("nvidia")
    kern = build_qft4()
    psi = np.array(cudaq.get_state(kern))       # pure state from GPU
    rho0 = np.outer(psi, psi.conj())            # noiseless density

    psi_ideal = ideal_qft4()
    rho_ideal = np.outer(psi_ideal, psi_ideal.conj())

    print("Channel      p     Fidelity    Frobenius Norm")
    print("──────────── ───── ─────────── ─────────────")

    for label, kraus_fn in [
        ("Depolarize", depolarizing_kraus),
        ("AmpDamp",    amp_damp_kraus),
        ("PhaseFlip",  phase_flip_kraus),
        ("BitFlip",    bit_flip_kraus),
    ]:
        for p in np.linspace(0.0, 1.0, 11):
            rho_noisy = apply_to_all(rho0, kraus_fn, p)
            F = fidelity(rho_noisy, psi_ideal)
            d = fro_norm(rho_noisy, rho_ideal)
            print(f"{label:<12} {p:4.2f}    {F:.6f}       {d:.6e}")

if __name__ == "__main__":
    main()
