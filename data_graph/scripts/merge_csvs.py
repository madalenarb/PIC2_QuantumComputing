#!/usr/bin/env python3
"""
merge_csvs.py — Consolidate QFT benchmark data for both noisy and noiseless cases across targets.
"""

import pandas as pd
import re
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
BASE_DIR   = Path("/home/madalenarb/Documents/PIC/PIC2_QuantumComputing/data_graph/scripts")
DATA_DIR   = BASE_DIR / "data_csv"
OUT_DIR    = DATA_DIR / "treated_data"
OUT_DIR.mkdir(exist_ok=True, parents=True)

OUT_NON_NOISE = OUT_DIR / "qft_merged_no_noise.csv"
OUT_NOISE     = OUT_DIR / "qft_merged_noise.csv"

# ──────────────────────────────────────────────────────────────────────────────
# Source files
# ──────────────────────────────────────────────────────────────────────────────
no_noise_files = {
    "qiskit-aer":  DATA_DIR / "qiskit_no_noise.csv",
    "cudaq-gpu":   DATA_DIR / "cudaq_nvidia_no_noise.csv",
    "cudaq-cpu":   DATA_DIR / "cudaq_qpp-cpu_no_noise.csv",
}

noise_files = {
    # note: both gpu and cpu density‐matrix variants live in cudaq_noise_{shots}.csv
    "density-matrix-cpu-16384": DATA_DIR / "cudaq_noise_16384.csv",
    "qiskit-aer-16384":         DATA_DIR / "qiskit_noise_16384.csv",
    "density-matrix-cpu-262144":DATA_DIR / "cudaq_noise_262144.csv",
    "qiskit-aer-262144":        DATA_DIR / "qiskit_noise_262144.csv",
}

# ──────────────────────────────────────────────────────────────────────────────
# NO-NOISE MERGE
# ──────────────────────────────────────────────────────────────────────────────
frames = []
for target, path in no_noise_files.items():
    df = pd.read_csv(path)
    df["target"] = target

    # Standardize col names for the three no-noise sources:
    df = df.rename(columns={
        # qiskit_no_noise.csv already has these headers:
        "n_bits":     "n_bits",
        "shots":      "shots",
        "sim_time_s": "sim_time_s",
        "l2_error":   "l2_error",
        # the cudaq files call it "l2_norm"
        "l2_norm":    "l2_error",
    })

    # pick exactly these five columns:
    frames.append(df[["target","n_bits","shots","sim_time_s","l2_error"]])

pd.concat(frames, ignore_index=True).to_csv(OUT_NON_NOISE, index=False)
print(f"✔ Merged no-noise → {OUT_NON_NOISE}")

# ──────────────────────────────────────────────────────────────────────────────
# NOISE MERGE
# ──────────────────────────────────────────────────────────────────────────────
def normalize_noise_name(s: str) -> str:
    # Insert space before each uppercase letter (except first), then strip
    return re.sub(r'(?<=[a-z])([A-Z])', r' \1', s).strip()

frames = []
for tag, path in noise_files.items():
    df = pd.read_csv(path)
    # split our tag into two pieces: target & shots
    target, shots = tag.rsplit("-",1)
    df["target"] = target
    df["shots"] = int(shots)

    # Standardize column names for ALL four noise dumps:
    df = df.rename(columns={
        "noise":         "noise",
        "probability":   "probability",
        "time_sampling": "time_sampling",
        "l2_pop":        "l2_error",
        "time_density":  "time_density",
        "fro_norm":      "fro_norm",
        "fidelity":      "fidelity",
        "n_bits":        "n_bits",
    })

    # Normalize noise labels (e.g. "PhaseDamping" → "Phase Damping")
    df["noise"] = df["noise"].astype(str).apply(normalize_noise_name)

    # pick these ten columns:
    frames.append(df[[
        "target","n_bits","shots","noise","probability",
        "time_sampling","l2_error","time_density","fro_norm","fidelity"
    ]])

pd.concat(frames, ignore_index=True).to_csv(OUT_NOISE, index=False)
print(f"✔ Merged noise → {OUT_NOISE}")
