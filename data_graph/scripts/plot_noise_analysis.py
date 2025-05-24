#!/usr/bin/env python3
"""
plot_noise_analysis.py — QFT Benchmark Analysis (noisy)
=========================================================
Generates:

  • Per‐target Fidelity & Fro Norm vs. Qubits across noise probabilities
  • Multi‐target Fidelity & Fro Norm @ fixed shots & probability
  • Side‐by‐side L2 Error & Time vs. Qubits across ALL noise types
  • For a given shot & probability, compare all targets: Fidelity, Fro Norm, L2 Error, Time
  • Prints out per‐target/shot/probability summary tables
  • Progression @ n_bits=12 and shot: metrics vs. probability for all noise types on same plot per target
  • All plots → ./graphs/noise/

Usage:

  # default: all shots & probabilities in the data per type of noise
  python plot_noise_analysis.py 

  # specify shot(s), probability(ies), and target(s)
  python plot_noise_analysis.py 16384 -p 0.01 0.05 -t qiskit-aer cudaq-cpu

"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List

# ──────────────────────────────────────────────────────────────────────────────
# Configuration & Data
# ──────────────────────────────────────────────────────────────────────────────
BASE        = Path(__file__).parent
DATA_CSV    = BASE / "data_csv" / "treated_data" / "qft_merged_noise.csv"
GRAPH_DIR   = BASE / "graphs" / "noise"
GRAPH_DIR.mkdir(exist_ok=True, parents=True)

df_noise    = pd.read_csv(DATA_CSV)
# drop 'none' noise type entirely (only applies to p=0)
df_noise    = df_noise[df_noise.noise != 'none']

# derive lists from the data
TARGETS     = sorted(df_noise["target"].unique())
SHOT_LIST   = sorted(df_noise["shots"].unique())
PROB_LIST   = sorted(p for p in df_noise["probability"].unique() if p != 1)  # exclude probability=1
NOISE_TYPES = sorted(df_noise["noise"].unique())

# color cycles
target_palette = plt.rcParams['axes.prop_cycle'].by_key()['color']
color_cycle    = plt.rcParams['axes.prop_cycle'].by_key()['color']
COLORS         = {t: target_palette[i % len(target_palette)] for i, t in enumerate(TARGETS)}
NOISE_COLORS   = {n: color_cycle[i % len(color_cycle)] for i, n in enumerate(NOISE_TYPES)}

# styles for probabilities (for fidelity/fro plots)
PROB_STYLES = {
    p: style for p, style in zip(
        PROB_LIST,
        [("o","-"), ("s","--"), ("^","-."), ("D",":"), ("*","-"), ("X","--")]
    )
}

# styles for noise types (for L2/time vs. noise)
NOISE_MARKERS    = ["o","s","^","D","*","X","P","v","<",">"]
NOISE_LINESTYLES = ["-","--","-.",":","-","--","-.",":","-","--"]
NOISE_STYLES     = {
    n: (NOISE_MARKERS[i % len(NOISE_MARKERS)], NOISE_LINESTYLES[i % len(NOISE_LINESTYLES)])
    for i, n in enumerate(NOISE_TYPES)
}

# ──────────────────────────────────────────────────────────────────────────────
# Plot Function: All Noise Types for a Target & Probability
# ──────────────────────────────────────────────────────────────────────────────
def plot_all_noise_types(df: pd.DataFrame, target: str, prob: float) -> None:
    """
    Plot all noise types for a given target and probability:
      • Top-left:    L2 Error vs. Qubits
      • Top-right:   Time Sampling vs. Qubits
      • Bottom-left: Fidelity vs. Qubits
      • Bottom-right:Fro Norm vs. Qubits
    """
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    fig.suptitle(f"QFT Benchmark Analysis ({target})\nProbability: {prob:.2f}", fontsize=16)

    ax_l2, ax_time = axs[0]
    ax_fid, ax_fro  = axs[1]

    for noise in NOISE_TYPES:
        subset = df[(df.target == target) &
                    (df.probability == prob) &
                    (df.noise == noise)]
        if subset.empty:
            continue
        marker, ls = NOISE_STYLES[noise]
        color = NOISE_COLORS[noise]
        ax_l2.plot(subset.n_bits, subset.l2_error, label=noise, color=color, marker=marker, linestyle=ls)
        ax_time.plot(subset.n_bits, subset.time_sampling, label=noise, color=color, marker=marker, linestyle=ls)
        ax_fid.plot(subset.n_bits, subset.fidelity, label=noise, color=color, marker=marker, linestyle=ls)
        ax_fro.plot(subset.n_bits, subset.fro_norm, label=noise, color=color, marker=marker, linestyle=ls)

    # Set titles, labels, grids, legends
    for ax, title, ylabel in zip([ax_l2, ax_time, ax_fid, ax_fro],
                                  ["L2 Error vs. Qubits","Time Sampling vs. Qubits",
                                   "Fidelity vs. Qubits","Fro Norm vs. Qubits"],
                                  ["L2 Error","Time Sampling","Fidelity","Fro Norm"]):
        ax.set(title=title, xlabel="Qubits", ylabel=ylabel)
        ax.grid(True)
        handles, labels = ax.get_legend_handles_labels()
        if handles: ax.legend(loc="best", title="Noise Type")

    plt.tight_layout(rect=[0,0,1,0.93])
    out = GRAPH_DIR / f"{target.replace('/','_')}_all_noises_p_{prob:.2f}.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"✔ Saved all-noise types plot: {out}")

# ──────────────────────────────────────────────────────────────────────────────
# New Plot: Metrics vs Probability @ n_bits & shot per Target
# ──────────────────────────────────────────────────────────────────────────────
def plot_metrics_vs_probability(df: pd.DataFrame, target: str,
                                 shot: int, n_bits: int = 12) -> None:
    """
    For a fixed target, shot, and n_bits, plot Fidelity, Fro Norm,
    L2 Error, and Time Sampling vs. probability, with one curve per noise type.
    """
    subset_base = df[(df.target==target)&(df.shots==shot)&(df.n_bits==n_bits)]
    if subset_base.empty:
        print(f"⚠️ No data for {target} @ shots={shot}, n_bits={n_bits}")
        return
    fig, axs = plt.subplots(2,2, figsize=(10,8), sharex=True)
    fig.suptitle(f"Metrics vs Probability for {target}, shots={shot}, n_bits={n_bits}", fontsize=14)
    axes = {"fidelity": axs[0,0], "fro_norm": axs[0,1],
            "l2_error": axs[1,0], "time_sampling": axs[1,1]}
    titles = {"fidelity":"Fidelity","fro_norm":"Fro Norm",
              "l2_error":"L2 Error","time_sampling":"Time Sampling"}

    for noise in NOISE_TYPES:
        dfn = subset_base[subset_base.noise==noise]
        if dfn.empty: continue
        marker, ls = NOISE_STYLES[noise]
        color = NOISE_COLORS[noise]
        for metric, ax in axes.items():
            ax.plot(dfn.probability, dfn[metric], label=noise,
                    marker=marker, linestyle=ls, color=color)

    for metric, ax in axes.items():
        ax.set(title=f"{titles[metric]} vs Probability", xlabel="Probability", ylabel=titles[metric])
        ax.grid(True)
        handles, labels = ax.get_legend_handles_labels()
        if handles: ax.legend(loc="best", title="Noise Type")

    plt.tight_layout(rect=[0,0,1,0.95])
    out = GRAPH_DIR / f"metrics_vs_prob_{target.replace('/','_')}_shot{shot}_n{n_bits}.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"✔ Saved metrics-vs-prob plot: {out}")

# ──────────────────────────────────────────────────────────────────────────────
# main function
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="QFT Noise Analysis Plots")
    parser.add_argument('shots', nargs='*', type=int, help='Shots; default=all')
    parser.add_argument('-p','--probability', nargs='*', type=float, help='Probabilities; default=all')
    parser.add_argument('-t','--target', nargs='*', type=str, help='Targets; default=all')
    parser.add_argument('-b','--n_bits', type=int, default=12, help='Fixed qubit count for progression plots')
    args = parser.parse_args()

    shots = args.shots or SHOT_LIST
    probs = args.probability or PROB_LIST
    targets = args.target or TARGETS
    n_bits = args.n_bits

    # Per-target per-prob plots
    for shot in shots:
        df_shot = df_noise[df_noise.shots==shot]
        for prob in probs:
            for tgt in targets:
                plot_all_noise_types(df_shot, tgt, prob)

    # Metrics vs Probability progression per target & shot @ n_bits
    for shot in shots:
        for tgt in targets:
            plot_metrics_vs_probability(df_noise, tgt, shot, n_bits)

if __name__ == "__main__":
    main()

