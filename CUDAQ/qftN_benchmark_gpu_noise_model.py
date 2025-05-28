#!/usr/bin/env python3

import math, argparse, os, time
import numpy as np
import pandas as pd
import cudaq

# ────────────────── Quantum Kernel ──────────────────
def build_qft_kernel(n_bits: int, init_state: str):
    if init_state == "ghz":
        @cudaq.kernel
        def circ():
            q = cudaq.qvector(n_bits)
            h(q[0])
            for i in range(1, n_bits):
                x.ctrl(q[0], q[i])
            for k in range(n_bits):
                h(q[k])
                for j in range(k + 1, n_bits):
                    θ = 2 * math.pi / (2 ** (j - k + 1))
                    cr1(θ, [q[j]], q[k])
        return circ
    else:
        @cudaq.kernel
        def circ():
            q = cudaq.qvector(n_bits)
            for k in range(n_bits):
                h(q[k])
                for j in range(k + 1, n_bits):
                    θ = 2 * math.pi / (2 ** (j - k + 1))
                    cr1(θ, [q[j]], q[k])
        return circ

# ──────────────── Noise Channels ────────────────
def depolarizing_kraus(p):
    K0 = np.sqrt(1 - 3*p/4) * np.eye(2, dtype=complex)
    Kx = np.sqrt(p/4) * np.array([[0,1],[1,0]], dtype=complex)
    Ky = np.sqrt(p/4) * np.array([[0,-1j],[1j,0]], dtype=complex)
    Kz = np.sqrt(p/4) * np.array([[1,0],[0,-1]], dtype=complex)
    return [K0, Kx, Ky, Kz]

def amplitude_damping_kraus(p):
    s1 = np.sqrt(1 - p)
    K0 = np.array([[1, 0], [0, s1]], dtype=complex)
    K1 = np.array([[0, np.sqrt(p)], [0, 0]], dtype=complex)
    return [K0, K1]

def phase_flip_kraus(p):
    K0 = np.sqrt(1 - p) * np.eye(2, dtype=complex)
    K1 = np.sqrt(p) * np.array([[1, 0], [0, -1]], dtype=complex)
    return [K0, K1]

def bit_flip_kraus(p):
    K0 = np.sqrt(1 - p) * np.eye(2, dtype=complex)
    K1 = np.sqrt(p) * np.array([[0, 1], [1, 0]], dtype=complex)
    return [K0, K1]

# ──────────── Apply Channel ────────────
def apply_channel(rho, Ks, qubit):
    n = int(np.log2(rho.shape[0]))
    I = np.eye(2, dtype=complex)
    result = np.zeros_like(rho)
    for K in Ks:
        ops = [K if i == qubit else I for i in range(n)]
        K_full = ops[0]
        for op in ops[1:]:
            K_full = np.kron(K_full, op)
        result += K_full @ rho @ K_full.conj().T
    return result

# ────────────────── Main ──────────────────
def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--init", choices=["zero", "ghz"], default="ghz")
    pa.add_argument("--shots", type=int, default=4096)
    pa.add_argument("--max_bits", type=int, default=10)
    pa.add_argument("--probs", nargs="*", type=float, default=[0.01, 0.1, 0.5, 0.9, 1.0])
    args = pa.parse_args()

    cudaq.set_target("nvidia")
    shots = args.shots
    print(f"{'probability':<11} {'noise_model':<11} {'n_bits':<7} {'shots':<6} {'time_s':<7} {'L2_pop':<10} {'Fro_norm':<10} {'Fidelity'}")

    print("-" * 70)

    channels = {
        "Depol": depolarizing_kraus,
        "BitFlip": bit_flip_kraus,
        "PhaseFlip": phase_flip_kraus,
        "AmpDamp": amplitude_damping_kraus,
    }

    rows = []
    # First, compute and store ideal (noiseless) results
    for n in range(3, args.max_bits + 1):
        kernel = build_qft_kernel(n, args.init)
        t0 = time.time()
        kernel()
        psi = np.array(cudaq.get_state(kernel))
        t1 = time.time()
        rho = np.outer(psi, np.conj(psi))
        dt = t1 - t0

        N = 1 << n
        psi0 = np.zeros(N, dtype=complex)
        if args.init == "ghz":
            psi0[0] = psi0[-1] = 1 / np.sqrt(2)
        else:
            psi0[0] = 1.0
        ω = np.exp(2j * np.pi / N)
        F = np.fromiter((ω**(j*k) for j in range(N) for k in range(N)), complex).reshape(N, N) / np.sqrt(N)
        psi_ideal = F @ psi0
        rho_ideal = np.outer(psi_ideal, np.conj(psi_ideal))
        pops_ideal = np.real(np.diag(rho_ideal))

        fid = float(np.real(np.trace(rho_ideal @ rho)))
        frob = float(np.linalg.norm(rho - rho_ideal))
        l2 = float(np.linalg.norm(np.real(np.diag(rho)) - pops_ideal))
        rows.append((0.0, "none", n, shots, dt, l2, frob, fid))
        print(f"{0.0:<11.2f} {'none':<11} {n:<7} {shots:<6} {dt:<7.3f} {l2:<10.3e} {frob:<10.3e} {fid:.6f}")

    # Then, loop over (prob → noise → n_bits)
    for p in args.probs:
        for name, kraus_fn in channels.items():
            for n in range(3, args.max_bits + 1):
                kernel = build_qft_kernel(n, args.init)
                kernel()
                psi = np.array(cudaq.get_state(kernel))
                rho = np.outer(psi, np.conj(psi))

                # Apply noise
                Ks = kraus_fn(p)
                t_start = time.time()
                noisy_rho = rho.copy()
                for q in range(n):
                    noisy_rho = apply_channel(noisy_rho, Ks, q)
                dt = time.time() - t_start

                # Ideal target
                N = 1 << n
                psi0 = np.zeros(N, dtype=complex)
                if args.init == "ghz":
                    psi0[0] = psi0[-1] = 1 / np.sqrt(2)
                else:
                    psi0[0] = 1.0
                ω = np.exp(2j * np.pi / N)
                F = np.fromiter((ω**(j*k) for j in range(N) for k in range(N)), complex).reshape(N, N) / np.sqrt(N)
                psi_ideal = F @ psi0
                rho_ideal = np.outer(psi_ideal, np.conj(psi_ideal))
                pops_ideal = np.real(np.diag(rho_ideal))

                # Metrics
                fid = float(np.real(np.trace(rho_ideal @ noisy_rho)))
                frob = float(np.linalg.norm(noisy_rho - rho_ideal))
                l2 = float(np.linalg.norm(np.real(np.diag(noisy_rho)) - pops_ideal))

                rows.append((p, name, n, shots, dt, l2, frob, fid))
                print(f"{p:<11.2f} {name:<11} {n:<7} {shots:<6} {dt:<7.3f} {l2:<10.3e} {frob:<10.3e} {fid:.6f}")


    # Save results
    os.makedirs("results", exist_ok=True)
    out = f"results/cudaq_qft_noise_{args.init}_{args.shots}_nvidia.csv"
    pd.DataFrame(rows, columns=[
        "probability","noise_model","n_bits","shots",
        "time_s","L2_pop","Fro_norm","Fidelity"]).to_csv(out, index=False)
    print(f"\n✅ Results saved to {out}")


if __name__ == "__main__":
    main()
