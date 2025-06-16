#!/usr/bin/env python3
"""
merge_csvs.py — Consolidate QFT benchmark data for both noisy and noiseless cases
by auto-discovering files under data_csv/noise and data_csv/non_noise.

New flag:
  --ignore-cudaq-gpu    if set, any cudaq-gpu data will be omitted entirely.
"""

import argparse
import pandas as pd
import re
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Merge QFT CSVs into unified noise / no-noise tables"
)
parser.add_argument(
    "--ignore-cudaq-gpu",
    action="store_true",
    help="If set, skip any entries whose target would be 'cudaq-gpu'"
)
args = parser.parse_args()
ignore_cudaq_gpu = args.ignore_cudaq_gpu

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent
DATA_DIR      = BASE_DIR / "data_csv"
NOISE_DIR     = DATA_DIR / "noise"
NON_NOISE_DIR = DATA_DIR / "non_noise"

OUT_DIR       = DATA_DIR / "treated_data"
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

    vendor, device, init = (g.lower() for g in m.groups())
    target = f"{vendor}-{device}"

    # honor the ignore flag
    if ignore_cudaq_gpu and target == "cudaq-gpu":
        print(f"Ignoring (flagged): {path.name} → target {target}")
        continue

    df = pd.read_csv(path)
    df = df.rename(columns={"l2_norm": "l2_error"})
    df["target"] = target
    df["init"]   = init

    no_noise_frames.append(
        df[["target","init","shots","n_bits","sim_time_s","l2_error"]]
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

    vendor, init, device, shots = m.group(1,2,3,4)
    vendor, init, device = vendor.lower(), init.lower(), device.lower()
    shots = int(shots)
    target = f"{vendor}-{device}"

    if ignore_cudaq_gpu and target == "cudaq-gpu":
        print(f"Ignoring (flagged): {path.name} → target {target}")
        continue

    df = pd.read_csv(path)
    # if you just want to remap those three before normalizing:

    df = df.rename(columns={
        "noise_model": "noise",
        "time_s":      "time_sampling",
        "l2_pop":      "l2_error",
        "L2_pop":      "l2_error",
        "Fro_norm":    "fro_norm",
        "fro_norm":    "fro_norm",
        "Fidelity":    "fidelity",
    })

    df["noise"] = (
    df["noise"]
      .astype(str)
      .replace(
         regex={
           r".*Amp.*":   "Amplitude Damping",
           r".*Depol.*": "Depolarizing",
           r".*Phase.*": "Phase Damping",
         }
      )
      .apply(normalize_noise_name)
    )

    df["target"] = target
    df["init"] = init
    df["target"]  = df["target"].astype(str).apply(normalize_noise_name)
    df["init"]    = df["init"].astype(str).apply(normalize_noise_name)
    df["shots"]   = shots

    noise_frames.append(
        df[[
            "noise","target","init","shots","n_bits","probability",
            "time_sampling","l2_error","fro_norm","fidelity"
        ]]
    )

pd.concat(noise_frames, ignore_index=True) \
  .to_csv(OUT_NOISE, index=False)
print(f"✔ Merged noise → {OUT_NOISE}")
