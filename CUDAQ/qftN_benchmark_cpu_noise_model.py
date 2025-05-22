#!/usr/bin/env python3
import math, time, os
import numpy as np
import pandas as pd
import cudaq

# ───────────────────────────────────────────────────────────────────────
# Build an N-qubit QFT kernel
# ───────────────────────────────────────────────────────────────────────
def make_qft_kernel(n_bits):
    @cudaq.kernel
    def qft():
        q = cudaq.qvector(n_bits)
        for k in range(n_bits):
            h(q[k])
            for j in range(k+1, n_bits):
                angle = math.pi / (2 ** (j - k))
                r1.ctrl(angle, q[j], q[k])
        for i in range(n_bits//2):
            swap(q[i], q[n_bits - i - 1])
    return qft

# ───────────────────────────────────────────────────────────────────────
# Population-L2 error from sampling
# ───────────────────────────────────────────────────────────────────────
def sample_l2_pop(kern, shots, noise_model, n_bits):
    t0 = time.perf_counter()
    counts = cudaq.sample(kern, shots_count=shots, noise_model=noise_model)
    t1 = time.perf_counter()
    probs = {s: c/sum(counts.values()) for s, c in counts.items()}
    N = 2**n_bits
    l2 = math.sqrt(max(sum(p*p for p in probs.values()) - 1/N, 0.0))
    return (t1 - t0, l2)

# ───────────────────────────────────────────────────────────────────────
# Ideal QFT state (uniform superposition)
# ───────────────────────────────────────────────────────────────────────
def ideal_qft_state(n_bits):
    N = 2**n_bits
    psi = np.full(N, 1/math.sqrt(N), dtype=complex)
    rho = np.outer(psi, psi.conj())
    return rho, psi

# ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    shots         = 262144
    target        = "density-matrix-cpu"
    max_bits      = 12
    probabilities = [0.01, 0.1, 0.5, 0.9, 1.0]

    # noise channels to test
    channel_ctors = {
        "Depolarizing":      cudaq.DepolarizationChannel,
        "Amplitude Damping": cudaq.AmplitudeDampingChannel,
        "Phase Damping":     cudaq.PhaseFlipChannel,
        "Bit Flip":          cudaq.BitFlipChannel
    }

    cudaq.set_target(target)

    # precompute ideal density matrices & state-vectors
    IDEAL_RHO = {}
    IDEAL_PSI = {}
    for n in range(3, max_bits+1):
        rho, psi = ideal_qft_state(n)
        IDEAL_RHO[n], IDEAL_PSI[n] = rho, psi

    records = []

# ─── 1. Run 'none' case only once with p=0 ───────────────────────────
channels = {"none": None}
for noise_name, chan in channels.items():
    for n in range(3, max_bits+1):
        kern = make_qft_kernel(n)
        sim_time_s, l2_pop = sample_l2_pop(kern, shots, None, n)
        rho_ideal = IDEAL_RHO[n]
        psi_ideal = IDEAL_PSI[n]
        rho_noisy = np.array(cudaq.get_state(kern))
        fro_norm = np.linalg.norm(rho_noisy - rho_ideal)
        fidelity = float((psi_ideal.conj() @ rho_noisy @ psi_ideal).real)
        records.append({
            "n_bits":     n,
            "shots":      shots,
            "noise":      noise_name,
            "probability": 0.0,
            "time_sampling": sim_time_s,
            "l2_pop":     l2_pop,
            "time_density": sim_time_s,
            "fro_norm":   fro_norm,
            "fidelity":   fidelity
        })
        print(f"p=0.0 {noise_name:16s} n={n:2d} "
              f"t={sim_time_s:.3f}s  L2_pop={l2_pop:.3e}  "
              f"Fro={fro_norm:.3e}  F={fidelity:.4f}")

    # ─── 2. Now run all other noise types across all p values ────────────
    for p in probabilities:
        for name, ctor in channel_ctors.items():
            chan = ctor(p)
            nm = cudaq.NoiseModel()
            for q in range(max_bits):
                nm.add_channel('h', [q], chan)

            for n in range(3, max_bits+1):
                kern = make_qft_kernel(n)
                sim_time_s, l2_pop = sample_l2_pop(kern, shots, nm, n)
                rho_ideal = IDEAL_RHO[n]
                psi_ideal = IDEAL_PSI[n]
                cudaq.set_noise(nm)
                rho_noisy = np.array(cudaq.get_state(kern))
                cudaq.unset_noise()
                fro_norm = np.linalg.norm(rho_noisy - rho_ideal)
                fidelity = float((psi_ideal.conj() @ rho_noisy @ psi_ideal).real)
                records.append({
                    "n_bits":     n,
                    "shots":      shots,
                    "noise":      name,
                    "probability": p,
                    "time_sampling": sim_time_s,
                    "l2_pop":     l2_pop,
                    "time_density": sim_time_s,
                    "fro_norm":   fro_norm,
                    "fidelity":   fidelity
                })
                print(f"p={p:<4} {name:16s} n={n:2d} "
                    f"t={sim_time_s:.3f}s  L2_pop={l2_pop:.3e}  "
                    f"Fro={fro_norm:.3e}  F={fidelity:.4f}")

    # save as long-format CSV
    df = pd.DataFrame(records)
    os.makedirs("results", exist_ok=True)
    out_csv = f"results/qft_noise_all_probs_{shots}_shots_3.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved results to {out_csv}")
