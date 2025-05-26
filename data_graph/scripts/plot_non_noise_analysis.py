#!/usr/bin/env python3
"""
plot_qft_no_noise.py — QFT Benchmark Analysis (noiseless)
=========================================================
Adds support for multiple raw CSV inputs. By default it now reads:
  • merged CSV (treated_data/qft_merged_no_noise.csv)
  • qiskit GPU raw CSV at the user‑specified absolute path

Generates:
  • Per‑target side‑by‑side L2 error & sim‑time vs. qubit count
  • A multi‑target L2‑error & sim‑time comparison at a fixed shot count
  • A comparison of all targets across each shot count
  • Prints out per‑target and per‑shot summary tables
  • All plots go into ./graphs/

Usage examples:
  python3 plot_qft_no_noise.py                  # default per‑target
  python3 plot_qft_no_noise.py --multi 131072   # all targets @ 131072 shots
  python3 plot_qft_no_noise.py 2048 8192        # custom shots per target
"""
import argparse
from pathlib import Path
from typing import List
import pandas as pd
import matplotlib.pyplot as plt

# ────────────────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent
GRAPH_DIR = BASE / "graphs/no_noise"
GRAPH_DIR.mkdir(parents=True, exist_ok=True)

# Primary merged dataset
DATA_FILES = [
    BASE / "data_csv/treated_data/qft_merged_no_noise.csv",
    Path("/home/madalenarb/Documents/PIC/PIC2_QuantumComputing/data_graph/scripts/data_csv/qiskit_gpu_no_noise.csv"),
]

TARGETS = [
    "cudaq-cpu",
    "cudaq-gpu",
    "qiskit-aer",
    "qiskit-aer-gpu",
]

COLORS = {
    "cudaq-cpu":  "tab:blue",
    "cudaq-gpu":  "tab:orange",
    "qiskit-aer": "tab:green",
    "qiskit-aer-gpu": "tab:red",
}

SHOT_STYLES = {
    1024:    ("o", "-"),
    2048:    ("s", "--"),
    4096:    ("^", "-.") ,
    8192:    ("D", ":"),
    16384:   ("*", "-"),
    32768:   ("P", "--"),
    65536:   ("X", "-"),
    131072:  ("h", ":"),
    262144:  ("H", "--"),
    524288:  ("v", "-"),
    1048576: ("p", ":"),
}

DEFAULT_SHOTS: List[int] = [4096, 16384, 65536, 131072, 262144, 524288]

# ────────────────────────────────────────────────────────────
# Helper: load & concat datasets
# ────────────────────────────────────────────────────────────

def load_data(files: List[Path]) -> pd.DataFrame:
    frames = []
    for fp in files:
        if not fp.exists():
            print(f"⚠️  Data file not found: {fp}")
            continue
        try:
            frames.append(pd.read_csv(fp))
            print(f"✔ Loaded {fp} ({len(frames[-1])} rows)")
        except Exception as exc:
            print(f"‼️  Failed to load {fp}: {exc}")
    if not frames:
        raise FileNotFoundError("No valid data CSVs were loaded.")
    return pd.concat(frames, ignore_index=True)

# ──────────────────────────────────────────────────────────────────────────────
# Plotting Helpers
# ──────────────────────────────────────────────────────────────────────────────
def plot_pair(ax_l2, ax_time, df: pd.DataFrame, *,
              label: str, color: str, marker: str, linestyle: str):
    """Plot a single (target@shots) line on both subplots, with explicit labels."""
    ax_l2.plot(
        df.n_bits, df.l2_error,
        label=label,
        color=color,
        marker=marker,
        linestyle=linestyle
    )
    ax_time.plot(
        df.n_bits, df.sim_time_s,
        label=label,
        color=color,
        marker=marker,
        linestyle=linestyle
    )


def side_by_side(df: pd.DataFrame,
                 targets: List[str],
                 shot_list: List[int],
                 multi_target: bool=False,
                 fixed_shots: int=None):
    """
    If multi_target=True, draws ALL targets on one figure at shot=fixed_shots.
    Otherwise draws each target separately, all shots in shot_list.
    """
    if multi_target:
        assert fixed_shots is not None
        fig, (ax_l2, ax_time) = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
        fig.suptitle(f"ALL Targets @ {fixed_shots} shots", fontsize=16)

        sub = df[df.shots == fixed_shots]
        for tgt in targets:
            d2 = sub[sub.target == tgt]
            if d2.empty:
                continue
            m, ls = SHOT_STYLES[fixed_shots]
            plot_pair(
                ax_l2, ax_time, d2,
                label=tgt,
                color=COLORS[tgt],
                marker=m,
                linestyle=ls
            )

        ax_l2.set(title="L2 Error vs. Qubits", xlabel="Qubits", ylabel="L2 Error")
        ax_time.set(title="Sim Time vs. Qubits", xlabel="Qubits", ylabel="Time (s)")
        for ax in (ax_l2, ax_time):
            ax.grid(True)
            ax.legend(loc="best", title="Target")

        plt.tight_layout(rect=[0, 0, 1, 0.92])
        out = GRAPH_DIR / f"compare_all_{fixed_shots}_shots.png"
        fig.savefig(out, dpi=300)
        plt.close(fig)
        print(f"✔ Saved compare-all: {out}")

    else:
        for tgt in targets:
            sub = df[df.target == tgt]
            fig, (ax_l2, ax_time) = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
            fig.suptitle(f"{tgt} — L2 & Time", fontsize=16)

            for shot in shot_list:
                d2 = sub[sub.shots == shot]
                if d2.empty:
                    continue
                m, ls = SHOT_STYLES[shot]
                plot_pair(
                    ax_l2, ax_time, d2,
                    label=f"{shot} shots",
                    color=COLORS[tgt],
                    marker=m,
                    linestyle=ls
                )

            ax_l2.set(title="L2 Error vs. Qubits", xlabel="Qubits", ylabel="L2 Error")
            ax_time.set(title="Sim Time vs. Qubits", xlabel="Qubits", ylabel="Time (s)")
            for ax in (ax_l2, ax_time):
                ax.grid(True)
                ax.legend(loc="best", title="Shots")

            plt.tight_layout(rect=[0, 0, 1, 0.92])
            out = GRAPH_DIR / f"combined_{tgt}.png"
            fig.savefig(out, dpi=300)
            plt.close(fig)
            print(f"✔ Saved per-target: {out}")


def summarize(df: pd.DataFrame,
              targets: List[str],
              shot_list: List[int],
              multi_target: bool=False,
              fixed_shots: int=None) -> pd.DataFrame:
    """
    Returns a summary table:
      • per-target & per-shot mean L2 & mean time
      • or, for multi_target, each target @ fixed_shots
    """
    rows = []
    if multi_target:
        sub = df[df.shots == fixed_shots]
        for tgt in targets:
            d2 = sub[sub.target == tgt]
            if d2.empty:
                continue
            rows.append({
                "target": tgt,
                "shots": fixed_shots,
                "mean_l2": d2.l2_error.mean(),
                "mean_time_s": d2.sim_time_s.mean()
            })
    else:
        for tgt in targets:
            for shot in shot_list:
                d2 = df[(df.target == tgt) & (df.shots == shot)]
                if d2.empty:
                    continue
                rows.append({
                    "target": tgt,
                    "shots": shot,
                    "mean_l2": d2.l2_error.mean(),
                    "mean_time_s": d2.sim_time_s.mean()
                })

    return pd.DataFrame(rows).set_index(["target", "shots"])


def compare_targets_across_shots(df: pd.DataFrame,
                                 targets: List[str],
                                 shot_list: List[int]):
    """
    For each shot in shot_list, plot all targets together:
      • L2 Error vs. Qubits
      • Sim Time vs. Qubits
    """
    for shot in shot_list:
        sub = df[df.shots == shot]
        if sub.empty:
            print(f"⚠️ No data for {shot} shots, skipping.")
            continue

        fig, (ax_l2, ax_time) = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
        fig.suptitle(f"ALL Targets @ {shot} shots", fontsize=16)

        for tgt in targets:
            d2 = sub[sub.target == tgt]
            if d2.empty:
                continue
            m, ls = SHOT_STYLES.get(shot, ("o", "-"))
            ax_l2.plot(
                d2.n_bits, d2.l2_error,
                label=tgt,
                color=COLORS.get(tgt, "black"),
                marker=m,
                linestyle=ls
            )
            ax_time.plot(
                d2.n_bits, d2.sim_time_s,
                label=tgt,
                color=COLORS.get(tgt, "black"),
                marker=m,
                linestyle=ls
            )

        ax_l2.set(title="L2 Error vs. Qubits", xlabel="Qubits", ylabel="L2 Error")
        ax_time.set(title="Sim Time vs. Qubits", xlabel="Qubits", ylabel="Time (s)")
        for ax in (ax_l2, ax_time):
            ax.grid(True)
            ax.legend(loc="best", title="Target")

        plt.tight_layout(rect=[0, 0, 1, 0.92])
        out = GRAPH_DIR / f"compare_targets_{shot}_shots.png"
        fig.savefig(out, dpi=300)
        plt.close(fig)
        print(f"✔ Saved compare-targets @ {shot} shots: {out}")


def main():
    parser = argparse.ArgumentParser(description="QFT Benchmark Analysis (noiseless)")
    parser.add_argument("shots", nargs="*", type=int, metavar="SHOTS",
                        help="Custom shot counts (per‑target mode)")
    parser.add_argument("--multi", type=int, metavar="SHOT",
                        help="Draw ALL targets @ this shot count")
    args = parser.parse_args()

    df_all = load_data(DATA_FILES)

    # Existing analysis/plotting functions are assumed to be defined above.
    if args.multi:
        side_by_side(df_all, TARGETS, [], multi_target=True, fixed_shots=args.multi)
        tbl = summarize(df_all, TARGETS, [], multi_target=True, fixed_shots=args.multi)
        print(f"\n=== Mean L2 & Time @ {args.multi} shots ===")
        print(tbl)
    else:
        shots = args.shots or DEFAULT_SHOTS
        side_by_side(df_all, TARGETS, shots)
        tbl = summarize(df_all, TARGETS, shots)
        print("\n=== Per‑target Mean L2 & Time Summary ===")
        print(tbl)
        compare_targets_across_shots(df_all, TARGETS, shots)

    print(f"\nAll plots saved to → {GRAPH_DIR}")

if __name__ == "__main__":
    main()
