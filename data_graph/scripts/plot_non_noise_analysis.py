#!/usr/bin/env python3
"""
plot_non_noise_analysis.py — Combined L2‐error & runtime analysis for QFT (noiseless)
=================================================================================
Generates:
  - Side‐by‐side plots of L2 error and simulation time vs. qubit count
  - Summary table of mean L2 error and mean time per shot count
  - Saves PNGs into a "graphs" subdirectory under the script directory, named by target
"""

from __future__ import annotations
import sys
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

# Base directory (script location)
BASE = Path(__file__).parent
# Input CSV (treated no-noise data)
DATA_CSV = BASE / "data_csv" / "treated_data" / "qft_merged_no_noise.csv"
# Output directory for graphs
GRAPH_DIR = BASE / "graphs"
GRAPH_DIR.mkdir(parents=True, exist_ok=True)

# Colors per simulator target
TARGET_COLORS: dict[str, str] = {
    "cudaq-cpu":  "tab:blue",
    "cudaq-gpu":  "tab:orange",
    "qiskit-aer": "tab:green",
}

# Marker & linestyle per shot count
SHOT_STYLES: dict[int, tuple[str, str]] = {
    1024:   ("o", "-"),
    2048:   ("s", "--"),
    4096:   ("^", "-."),
    8192:   ("D", ":"),
    16384:  ("*", "-"),
    32768:  ("P", "--"),
    65536:  ("X", "-"),
    131072:("h", ":"),
    262144:("H", "--"),
    524288:("v", "-"),
    1048576:("p", ":"),
}

# Default shots to analyze
DEFAULT_SHOTS: List[int] = [4096, 16384, 65536, 131072, 262144, 524288]

# ──────────────────────────────────────────────────────────────────────────────
# Plotting functions
# ──────────────────────────────────────────────────────────────────────────────

def combined_l2_time(target: str, shots: List[int]) -> pd.DataFrame:
    """
    Plot combined L2‐error and sim_time vs. n_bits for a given target and shot list.
    Returns a DataFrame summarizing mean L2‐error and mean sim_time per shot.
    """
    # load and filter data
    df_all = pd.read_csv(DATA_CSV)
    df = df_all[df_all["target"] == target]

    # prepare figure with two subplots
    fig, (ax_l2, ax_time) = plt.subplots(1, 2, figsize=(14, 6), sharex=True)
    fig.suptitle(f"{target}: L2 Error & Sim Time vs. Qubit Count", fontsize=16)

    summary: list[dict[str, float]] = []

    for shot in shots:
        sub = df[df["shots"] == shot]
        if sub.empty:
            continue

        marker, ls = SHOT_STYLES.get(shot, ("o", "-"))
        color = TARGET_COLORS.get(target, "black")

        # plot L2 error
        ax_l2.plot(
            sub["n_bits"], sub["l2_error"],
            label=f"{shot} shots", marker=marker, linestyle=ls, color=color
        )
        # plot sim_time
        ax_time.plot(
            sub["n_bits"], sub["sim_time_s"],
            label=f"{shot} shots", marker=marker, linestyle=ls, color=color
        )

        # record summary stats
        summary.append({
            "shots":        shot,
            "mean_l2_error": sub["l2_error"].mean(),
            "mean_sim_time_s": sub["sim_time_s"].mean()
        })

    # finalize L2‐error plot
    ax_l2.set_xlabel("Number of Qubits")
    ax_l2.set_ylabel("L2 Error")
    ax_l2.set_title("L2 Error vs. Qubits")
    ax_l2.grid(True)
    ax_l2.legend(title="Shots", loc="best", ncol=1)

    # finalize sim‐time plot
    ax_time.set_xlabel("Number of Qubits")
    ax_time.set_ylabel("Simulation Time (s)")
    ax_time.set_title("Simulation Time vs. Qubits")
    ax_time.grid(True)
    ax_time.legend(title="Shots", loc="best", ncol=1)

    plt.tight_layout(rect=[0, 0, 1, 0.94])

    # save figure
    out_path = GRAPH_DIR / f"combined_l2_time_{target}.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"✔ Saved figure: {out_path}")

    # return summary DataFrame
    return pd.DataFrame(summary).set_index("shots")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # parse custom shot list or use default
    shots = [int(s) for s in sys.argv[1:]] if len(sys.argv) > 1 else DEFAULT_SHOTS

    for target in TARGET_COLORS:
        print(f"\n=== Summary for {target} ===")
        df_summary = combined_l2_time(target, shots)
        print(df_summary)

    print("\nAll plots saved successfully in:")
    print(GRAPH_DIR)
