#!/usr/bin/env python3
"""
QFT N-qubit benchmark (CUDA-Q, noise optional)

  • noiseless run: fidelity = 1.000 for every n ≥ 3
  • optional sweep of Depol / AmpDamp / Phase / BitFlip channels
  • saves results => results/qftN_<init>_<shots>.csv
"""

from __future__ import annotations
import argparse, math, os, time
from typing import Dict, Tuple, Type
import numpy as np, pandas as pd, cudaq

# ────────────────── build QFT kernel ──────────────────
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
                    θ = 2*math.pi / (2 ** (j - k + 1))
                    cr1(θ, [q[j]], q[k])
        return circ

    @cudaq.kernel
    def circ():
        q = cudaq.qvector(n_bits)
        for k in range(n_bits):
            h(q[k])
            for j in range(k + 1, n_bits):
                θ = 2*math.pi / (2 ** (j - k + 1))
                cr1(θ, [q[j]], q[k])
    return circ

# ────────────────── ideal statevector ─────────────────
def ideal_qft_state(n_bits: int, init_state: str) -> np.ndarray:
    N, psi = 1 << n_bits, np.zeros(1 << n_bits, complex)
    psi[0] = 1 / math.sqrt(2) if init_state == "ghz" else 1.0
    if init_state == "ghz":
        psi[-1] = psi[0]
    ω = np.exp(2j * math.pi / N)
    F = np.array([[ω**(j*k) for k in range(N)] for j in range(N)]) / math.sqrt(N)
    return F @ psi

# ───────────── helper: always return ρ matrix ─────────
def get_rho(kern, n_bits: int) -> np.ndarray:
    raw = np.array(cudaq.get_state(kern))
    if raw.size == (1 << n_bits):                 # state-vector
        return np.outer(raw, raw.conj())
    return raw.reshape(1 << n_bits, 1 << n_bits)  # already density-matrix

# ───────────── helper: L2(pop) from sampling ──────
def sample_l2_pop(kern, shots: int, n_bits: int, noise=None) -> Tuple[float,float]:
    cudaq.sample(kern, shots_count=32, noise_model=noise)
    t0 = time.perf_counter()
    counts = cudaq.sample(kern, shots_count=shots, noise_model=noise)
    dt = time.perf_counter() - t0
    probs = np.fromiter(counts.values(), float, len(counts)) / shots
    l2 = math.sqrt(max(np.sum(probs**2) - 1/(1<<n_bits), 0.0))
    return dt, l2

# ─────────────────────────── main ────────────────────────────
def main():
    pa = argparse.ArgumentParser(description="Benchmark QFT circuits with configurable initial state, shots, noise, and target backend.")

    pa.add_argument(
        "--init", 
        choices=["zero", "ghz"], 
        default="ghz",
        help="Initial state to prepare before running the QFT circuit. Choose 'zero' for |0...0⟩ or 'ghz' for GHZ state."
    )

    pa.add_argument(
        "--shots", 
        type=int, 
        default=4096,
        help="Number of measurement shots to perform per circuit run."
    )

    pa.add_argument(
        "--max_bits", 
        type=int, 
        default=10,
        help="Maximum number of qubits to simulate."
    )

    pa.add_argument(
        "--probs", 
        nargs="*", 
        type=float,
        default=[0.01, 0.1, 0.5, 0.9, 1.0],
        help="List of depolarizing probabilities to apply as noise. Default: [0.01, 0.1, 0.5, 0.9, 1.0]."
    )

    
    args = pa.parse_args()

    target = "qpp-cpu"

    rows = []
    print(f"{'probability':<11} {'noise_model':<11} {'n_bits':<7} {'shots':<6} {'time_s':<7} {'L2_pop':<10} {'Fro_norm':<10} {'Fidelity'}")


    # —— noiseless baseline on state-vector back-end ——
    cudaq.set_target(target)
    for n in range(3, args.max_bits+1):
        kern = build_qft_kernel(n, args.init)
        rho_s = get_rho(kern, n)
        psi_i = ideal_qft_state(n, args.init)
        rho_i = np.outer(psi_i, psi_i.conj())
        fid   = float(np.real(np.trace(rho_i @ rho_s)))
        frob  = np.linalg.norm(rho_s - rho_i)
        t,l2  = sample_l2_pop(kern, args.shots, n)
        rows.append((0.0,"none",n,args.shots,t,l2,frob,fid))
        print(f"0.0\tnone       \t{n}\t{args.shots}\t{t:.3f}\t{l2:.3e}\t{frob:.3e}\t{fid:.6f}")

    # —— noisy sweep (optional) ————————————————
    cudaq.set_target("density-matrix-cpu")
    chans : Dict[str,Type] = {
        "Depol": cudaq.DepolarizationChannel,
        "AmpDamp": cudaq.AmplitudeDampingChannel,
        "Phase": cudaq.PhaseFlipChannel,
        "BitFlip": cudaq.BitFlipChannel,
    }
    for p_err in args.probs:
        for label, Chan in chans.items():
            nm = cudaq.NoiseModel()
            chan = Chan(p_err)
            for q in range(args.max_bits):
                nm.add_channel("h",  [q], chan)
                nm.add_channel("x",  [q], chan)
                nm.add_channel("r1", [q], chan)
            for n in range(3, args.max_bits+1):
                kern = build_qft_kernel(n, args.init)
                t,l2 = sample_l2_pop(kern, args.shots, n, noise=nm)
                cudaq.set_noise(nm)
                rho_s = get_rho(kern, n)
                cudaq.unset_noise()
                psi_i = ideal_qft_state(n, args.init)
                rho_i = np.outer(psi_i, psi_i.conj())
                fid   = float(np.real(np.trace(rho_i @ rho_s)))
                frob  = np.linalg.norm(rho_s - rho_i)
                rows.append((p_err,label,n,args.shots,t,l2,frob,fid))
                print(f"{p_err}\t{label:<10}\t{n}\t{args.shots}\t{t:.3f}"
                        f"\t{l2:.3e}\t{frob:.3e}\t{fid:.6f}")

    # —— save CSV ————————————————————————————————
    os.makedirs("results", exist_ok=True)
    out = f"results/qftN_{args.init}_{args.shots}_cpu.csv"
    pd.DataFrame(rows, columns=[
        "probability","noise_model","n_bits","shots",
        "time_s","L2_pop","Fro_norm","Fidelity"]).to_csv(out, index=False)
    print(f"\n✅ results saved → {out}")

if __name__ == "__main__":
    main()
