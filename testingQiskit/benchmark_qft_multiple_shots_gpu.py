#!/usr/bin/env python3
"""
qftN_benchmark_cpu_noise_model.py — QFT Noise Benchmark (CUDA‑Q)
================================================================
Benchmarks the Quantum Fourier Transform (QFT) under a variety of
noise channels using CUDA‑Q’s **density‑matrix** simulator.

Output columns (tab‑separated)
------------------------------
```
probability   noise_model   n_bits   shots   time_s   L2_pop   Fro_norm   Fidelity
```
 * **probability** – error probability *p* fed to the channel
 * **noise_model** – “none”, “Depol”, “AmpDamp”, “Phase”, or “BitFlip”
 * **n_bits** – number of qubits in the register
 * **shots** – sample shots used for the L2‑population metric
 * **time_s** – sampling wall‑time (seconds)
 * **L2_pop** – L2 distance between sampled and uniform distributions
 * **Fro_norm** – Frobenius norm ‖ρ_sim − ρ_ideal‖
 * **Fidelity** – ‖⟨ψ_id | ψ_sim⟩‖²  (or Tr(ρ_id ρ_sim) for noisy runs)

A CSV file with the same columns is written to *results/*.
"""

from __future__ import annotations

import argparse, math, os, time
from typing import Dict, Tuple
import numpy as np, pandas as pd, cudaq

# ─────────────────────────── kernel builders ────────────────────────────

def qft_kernel(n_bits: int, init_state: str = "ghz"):
    """Textbook / mathematical QFT without the final swap layer."""
    if init_state == "ghz":
        @cudaq.kernel
        def circ():
            q = cudaq.qvector(n_bits)
            h(q[0])
            for k in range(1, n_bits):
                x.ctrl(q[0], q[k])
            for i in range(n_bits):
                h(q[i])
                for j in range(i + 1, n_bits):
                    cr1(2 * math.pi / (2 ** (j - i + 1)), [q[j]], q[i])
        return circ

    @cudaq.kernel
    def circ():
        q = cudaq.qvector(n_bits)
        for i in range(n_bits):
            h(q[i])
            for j in range(i + 1, n_bits):
                cr1(2 * math.pi / (2 ** (j - i + 1)), [q[j]], q[i])
    return circ

# ─────────────────────────── ideal state vector ─────────────────────────

def ideal_qft_state(n_bits: int, init_state: str) -> np.ndarray:
    N = 1 << n_bits
    psi = np.zeros(N, complex)
    if init_state == "ghz":
        psi[0] = psi[-1] = 1 / np.sqrt(2)
    else:
        psi[0] = 1.0
    ω = np.exp(2j * np.pi / N)
    F = np.array([[ω ** (j * k) for k in range(N)] for j in range(N)]) / np.sqrt(N)
    return F @ psi

# ─────────────────────────── L2‑population helper ───────────────────────

def sample_l2_pop(kern, shots: int, n_bits: int, noise=None) -> Tuple[float, float]:
    cudaq.sample(kern, shots_count=32, noise_model=noise)  # warm‑up
    t0 = time.perf_counter()
    counts = cudaq.sample(kern, shots_count=shots, noise_model=noise)
    t1 = time.perf_counter()
    probs = np.fromiter(counts.values(), float, len(counts)) / shots
    l2 = math.sqrt(max(np.sum(probs ** 2) - 1 / (1 << n_bits), 0.0))
    return t1 - t0, l2

# ─────────────────────────── main routine ───────────────────────────────

def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--init",  choices=["zero", "ghz"], default="ghz")
    pa.add_argument("--shots", type=int, default=4096)
    pa.add_argument("--max_bits", type=int, default=12)
    pa.add_argument("--noise", action="store_true",
                    help="include noisy sweeps in addition to noiseless run")
    pa.add_argument("--probs", nargs="*", type=float, default=[0.01, 0.1, 0.5, 0.9, 1.0])
    args = pa.parse_args()

    # print header
    print("probability\tnoise_model\tn_bits\tshots\ttime_s\tL2_pop\tFro_norm\tFidelity")

    rows = []

    # ── noiseless pass on state‑vector backend ─────────────────────────
    cudaq.set_target("qpp-cpu")
    for n in range(3, args.max_bits + 1):
        kern   = qft_kernel(n, args.init)
        psi_s  = np.array(cudaq.get_state(kern))
        psi_id = ideal_qft_state(n, args.init)
        fid    = abs(np.vdot(psi_id, psi_s)) ** 2
        frob   = np.linalg.norm(np.outer(psi_s, psi_s.conj()) - np.outer(psi_id, psi_id.conj()))
        t_s, l2 = sample_l2_pop(kern, args.shots, n)
        rows.append((0.0, "none", n, args.shots, t_s, l2, frob, fid))
        print(f"0.0\tnone        \t{n}\t{args.shots}\t{t_s:.3f}\t{l2:.3e}\t{frob:.3e}\t{fid:.6f}")

    # ── noisy sweeps (optional) ───────────────────────────────────────
    if args.noise:
        cudaq.set_target("density-matrix-cpu")
        channels = {
            "Depol"  : cudaq.DepolarizationChannel,
            "AmpDamp": cudaq.AmplitudeDampingChannel,
            "Phase"  : cudaq.PhaseFlipChannel,
            "BitFlip": cudaq.BitFlipChannel,
        }
        for p in args.probs:
            for label, Chan in channels.items():
                nm = cudaq.NoiseModel()
                chan = Chan(p)
                for q in range(args.max_bits):
                    nm.add_channel("h", [q], chan)
                for n in range(3, args.max_bits + 1):
                    kern   = qft_kernel(n, args.init)
                    t_s, l2 = sample_l2_pop(kern, args.shots, n, noise=nm)
                    cudaq.set_noise(nm)
                    rho_sim = np.array(cudaq.get_state(kern))
                    cudaq.unset_noise()
                    psi_id = ideal_qft_state(n, args.init)
                    rho_id = np.outer(psi_id, psi_id.conj())
                    fid  = float(np.real(np.trace(rho_id @ rho_sim)))
                    frob = np.linalg.norm(rho_sim - rho_id)
                    rows.append((p, label, n, args.shots, t_s, l2, frob, fid))
                    print(f"{p}\t{label:<10}\t{n}\t{args.shots}\t{t_s:.3f}\t{l2:.3e}\t{frob:.3e}\t{fid:.6f}")

    # ── write CSV ─────────────────────────────────────────────────────
    os.makedirs("results", exist_ok=True)
    out = f"results/qftN_{args.init}.csv"
    df = pd.DataFrame(rows, columns=["probability","noise_model","n_bits","shots",
                                     "time_s","L2_pop","Fro_norm","Fidelity"])
    df.to_csv(out, index=False)
    print(f"\n✅ results saved → {out}")

if __name__ == "__main__":
    main()
