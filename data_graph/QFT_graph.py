#!/usr/bin/env python3
"""plot_qft_benchmarks.py

A small utility that ingests several CSV benchmark files and produces three
PNG figures:
  • qft_l2_vs_qubits.png        – L2‑error vs number of qubits
  • qft_time_vs_qubits.png      – Simulation time vs number of qubits
  • qft_gpu_mem_vs_qubits.png   – GPU‑memory usage vs number of qubits (CUDA‑Q GPU only)

Usage::

    pip install pandas matplotlib
    python plot_qft_benchmarks.py

Any input file that is missing is skipped with a warning so the script happily
runs even if you have only a subset of the data.
"""

from __future__ import annotations

import re
import sys
import pathlib
from typing import List, Dict, Tuple, Pattern, Optional

import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# 1)  INPUT FILES --------------------------------------------------------------
# -----------------------------------------------------------------------------
FILES: List[str] = [
    "data_graph/benchmark_qft_qiskit_1024.csv",
    "data_graph/benchmark_qft_qiskit_2048.csv",
    "data_graph/a_qft_benchmark_qpp-cpu_2048.csv",
    "data_graph/a_qft_benchmark_qpp-cpu_1024.csv",
    "data_graph/a_qft_benchmark_nvidia_2048.csv",
    "data_graph/a_qft_benchmark_nvidia_1024.csv",
]

# -----------------------------------------------------------------------------
# 2)  INGEST & ANNOTATE --------------------------------------------------------
# -----------------------------------------------------------------------------
# Regex that captures the *shot count* that always sits just before the file
# extension – e.g.   ..._2048.csv   → shots = 2048
RX_SHOTS: Pattern[str] = re.compile(r"_(\d+)\.csv$")

# Mapping from a keyword occurring in the filename → canonical backend label
BACKEND_LABELS: Dict[str, str] = {
    "qiskit": "qiskit-aer",
    "nvidia": "cudaq-gpu",
    "qpp-cpu": "cudaq-cpu",
}

# Unify column names regardless of the header spelling used by each benchmark
COLUMN_ALIASES: Dict[str, str] = {
    # qubit count
    "n_bits": "n_bits",
    "n_qubits": "n_bits",
    "qubits": "n_bits",
    # simulation time
    "sim_time_s": "sim_time_s",
    "sim_time": "sim_time_s",
    # l2 error
    "l2": "l2_norm",
    "l2_error": "l2_norm",
    "l2_norm": "l2_norm",
    # gpu memory usage (%)
    "gpu_mem_pct": "gpu_mem_pct",
    "gpu_memory_pct": "gpu_mem_pct",
    "gpu_usage_pct": "gpu_mem_pct",
}

frames: List[pd.DataFrame] = []


def canonical_backend(path: pathlib.Path) -> str:
    """Return the canonical backend label based on the filename."""
    stem_lower = path.stem.lower()
    for key, label in BACKEND_LABELS.items():
        if key in stem_lower:
            return label
    return "unknown"


def extract_shots(path: pathlib.Path) -> int:
    """Extract the shot count from the filename (e.g. *_2048.csv → 2048)."""
    m = RX_SHOTS.search(path.name)
    if not m:
        raise ValueError(f"Filename '{path.name}' does not contain a shot count " \
                         "(expected ‘_*digits*.csv’ suffix)")
    return int(m.group(1))


for file_str in FILES:
    path = pathlib.Path(file_str)
    if not path.exists():
        print(f"⚠️  Warning: file '{path}' not found – skipping.", file=sys.stderr)
        continue

    # ------------------------------------------------------------------
    try:
        backend = canonical_backend(path)
        shots = extract_shots(path)
    except ValueError as err:
        print(f"⚠️  {err} – skipping {path.name}", file=sys.stderr)
        continue

    # ------------------------------------------------------------------
    df = pd.read_csv(path)

    # Normalise column names --------------------------------------------------
    new_cols: Dict[str, str] = {}
    for col in df.columns:
        key = col.lower().strip()
        if key in COLUMN_ALIASES:
            new_cols[col] = COLUMN_ALIASES[key]
    df = df.rename(columns=new_cols)

    # Sanity check ------------------------------------------------------------
    required_cols = {"n_bits", "sim_time_s", "l2_norm"}
    if not required_cols.issubset(df.columns):
        missing = ", ".join(sorted(required_cols - set(df.columns)))
        print(f"⚠️  {path.name}: missing required columns: {missing} –  skipped.",
              file=sys.stderr)
        continue

    # Attach metadata ---------------------------------------------------------
    df["backend"] = backend
    df["shots"] = shots

    frames.append(df[[
        "n_bits",
        "sim_time_s",
        "l2_norm",
        *(["gpu_mem_pct"] if "gpu_mem_pct" in df.columns else []),
        "backend",
        "shots",
    ]])

# Concatenate all ingested data ----------------------------------------------
if not frames:
    sys.exit("❌  No data loaded – aborting.")

data = pd.concat(frames, ignore_index=True)

# -----------------------------------------------------------------------------
# 3)  STYLES -------------------------------------------------------------------
# -----------------------------------------------------------------------------
backend_colors: Dict[str, str] = {
    "qiskit-aer": "#1f77b4",  # blue
    "cudaq-cpu": "#ff7f0e",   # orange
    "cudaq-gpu": "#d62728",   # red
    "unknown": "#7f7f7f",     # grey
}

shot_styles: Dict[int, str] = {1024: "-", 2048: "--"}

# Helper ----------------------------------------------------------------------

def plot_grouped(
    ax: plt.Axes,
    df: pd.DataFrame,
    y: str,
    title: str,
    y_label: str,
    logy: bool = False,
):
    """Plot *df* grouped by (backend, shots) on *ax* using shared style dicts."""
    for (backend, shots), grp in df.groupby(["backend", "shots"], sort=False):
        grp_sorted = grp.sort_values("n_bits")
        ax.plot(
            grp_sorted["n_bits"],
            grp_sorted[y],
            label=f"{backend}  {shots} shots",
            color=backend_colors.get(backend, "k"),
            linestyle=shot_styles.get(shots, "-"),
            marker="s",
        )
    ax.set(
        xlabel="n_bits",
        ylabel=y_label,
        title=title,
    )
    if logy:
        ax.set_yscale("log")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

# -----------------------------------------------------------------------------
# 4)  MAKE PLOTS ---------------------------------------------------------------
# -----------------------------------------------------------------------------
print("✏️  Generating figures …")

# --- L2 error vs qubits ------------------------------------------------------
fig1, ax1 = plt.subplots(figsize=(6, 4))
plot_grouped(
    ax1,
    data,
    y="l2_norm",
    y_label="L2 norm",
    title="QFT  L2‑norm vs number of qubits",
    logy=True,
)
fig1.tight_layout()
fig1.savefig("qft_l2_vs_qubits.png", dpi=300)
print("  ✔ qft_l2_vs_qubits.png")

# --- Simulation time vs qubits ----------------------------------------------
fig2, ax2 = plt.subplots(figsize=(6, 4))
plot_grouped(
    ax2,
    data,
    y="sim_time_s",
    y_label="Simulation time [s]",
    title="QFT  runtime vs number of qubits",
)
fig2.tight_layout()
fig2.savefig("qft_time_vs_qubits.png", dpi=300)
print("  ✔ qft_time_vs_qubits.png")

# --- GPU memory vs qubits (CUDA‑Q GPU only) ----------------------------------
if "gpu_mem_pct" in data.columns:
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    # Filter to CUDA‑Q GPU rows only
    gpu_data = data[data["backend"] == "cudaq-gpu"]
    if not gpu_data.empty:
        plot_grouped(
            ax3,
            gpu_data,
            y="gpu_mem_pct",
            y_label="GPU memory [%]",
            title="QFT  GPU memory usage vs number of qubits",
        )
        fig3.tight_layout()
        fig3.savefig("qft_gpu_mem_vs_qubits.png", dpi=300)
        print("  ✔ qft_gpu_mem_vs_qubits.png")
    else:
        print("  ℹ️  No CUDA‑Q GPU rows – GPU‑memory plot skipped.")
else:
    print("  ℹ️  Column 'gpu_mem_pct' not present – GPU‑memory plot skipped.")

print("✅  All done.")
