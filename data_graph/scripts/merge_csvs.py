#!/usr/bin/env python3
"""
merge_csvs.py — Consolidate QFT benchmark data for both noisy and noiseless cases
by auto-discovering files under data_csv/noise and data_csv/non_noise.
Now includes an 'init' column for the zero/ghz style.
"""

import pandas as pd
import re
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent
DATA_DIR      = BASE_DIR / "data_csv"
NOISE_DIR     = DATA_DIR / "noise"
NON_NOISE_DIR = DATA_DIR / "non_noise"

OUT_DIR = DATA_DIR / "treated_data"
OUT_DIR.mkdir(exist_ok=True, parents=True)

OUT_NON_NOISE = OUT_DIR / "qft_merged_no_noise.csv"
OUT_NOISE     = OUT_DIR / "qft_merged_noise.csv"

# ──────────────────────────────────────────────────────────────────────────────
# Helper to prettify noise-model names
# ──────────────────────────────────────────────────────────────────────────────
def normalize_noise_name(s: str) -> str:
    return re.sub(r'(?<=[a-z])([A-Z])', r' \1', s).strip()

# ──────────────────────────────────────────────────────────────────────────────
# MERGE NO-NOISE
# ──────────────────────────────────────────────────────────────────────────────
no_noise_pattern = re.compile(
    r"^(qiskit|cudaq)_qft_multishot_no_noise_(cpu|gpu)_(zero|ghz)\.csv$",
    re.IGNORECASE
)

no_noise_frames = []
for path in NON_NOISE_DIR.glob("*.csv"):
    m = no_noise_pattern.match(path.name)
    if not m:
        print(f"Skipping non-noise file (no match): {path.name}")
        continue

    vendor, device, freq = (g.lower() for g in m.groups())
    target = f"{vendor}-{device}"

    df = pd.read_csv(path)
    # rename if needed
    df = df.rename(columns={"l2_norm": "l2_error"})

    # add metadata columns
    df["target"] = target
    df["init"]   = freq

    # select and order exactly:
    no_noise_frames.append(
        df[["target", "init", "shots", "n_bits", "sim_time_s", "l2_error"]]
    )

pd.concat(no_noise_frames, ignore_index=True) \
  .to_csv(OUT_NON_NOISE, index=False)
print(f"✔ Merged no-noise → {OUT_NON_NOISE}")

# ──────────────────────────────────────────────────────────────────────────────
# MERGE NOISE
# ──────────────────────────────────────────────────────────────────────────────
noise_pattern = re.compile(
    r"^(qiskit|cudaq)_qft_noise_(zero|ghz)_(cpu|gpu)_([0-9]+)\.csv$",
    re.IGNORECASE
)

noise_frames = []
for path in NOISE_DIR.glob("*.csv"):
    m = noise_pattern.match(path.name)
    if not m:
        print(f"Skipping noise file (no match): {path.name}")
        continue

    vendor, freq, device, shots = m.group(1,2,3,4)
    vendor, freq, device = vendor.lower(), freq.lower(), device.lower()
    shots = int(shots)
    target = f"{vendor}-{device}"

    df = pd.read_csv(path)
    # unify column names across both qiskit & cudaq exports
    df = df.rename(columns={
        "noise_model": "noise",       # cudaq
        "time_s":      "time_sampling",  # cudaq
        "l2_pop":      "l2_error",     # qiskit
        "L2_pop":      "l2_error",     # cudaq
        "Fro_norm":    "fro_norm",     # cudaq
        "fro_norm":    "fro_norm",     # qiskit
        "Fidelity":    "fidelity",     # cudaq
    })

    # normalize noise-labels
    df["noise"] = df["noise"].astype(str).apply(normalize_noise_name)

    # add metadata columns
    df["target"] = target
    df["init"]   = freq
    df["shots"]  = shots

    # select and order exactly (dropping time_density):
    noise_frames.append(
        df[[
            "target", "init", "shots", "n_bits", "noise", "probability",
            "time_sampling", "l2_error", "fro_norm", "fidelity"
        ]]
    )

pd.concat(noise_frames, ignore_index=True) \
  .to_csv(OUT_NOISE, index=False)
print(f"✔ Merged noise → {OUT_NOISE}")
