#!/usr/bin/env python3
"""
Consolidate QFT benchmark data for both noisy and noiseless cases across targets.
"""

import os
import pandas as pd
import pathlib

# Directories
BASE_DIR = pathlib.Path("/home/madalenarb/Documents/PIC/PIC2_QuantumComputing/data_graph/scripts")
DATA_DIR = BASE_DIR / "data_csv"
OUT_DIR = DATA_DIR / "treated_data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Output files
OUT_NON_NOISE = OUT_DIR / "qft_merged_no_noise.csv"
OUT_NOISE = OUT_DIR / "qft_merged_noise.csv"

# Source files
files = {
    "qiskit_no_noise": DATA_DIR / "benchmark_qft_qiskit_various_shots_1.csv",
    "cudaq_gpu": DATA_DIR / "qftN_nvidia_multiple_shots.csv",
    "cudaq_cpu": DATA_DIR / "qftN_qpp-cpu_multiple_shots.csv",
    "cudaq_noise": DATA_DIR / "qft_noise_all_probs_16384_shots_2.csv",
    "qiskit_noise": DATA_DIR / "qft_qiskit_noise_metrics_16384shots.csv",
}

# ----------------------------
# NO-NOISE DATA TREATMENT
# ----------------------------
frames_no_noise = []

# Qiskit no-noise
df = pd.read_csv(files["qiskit_no_noise"])
df["target"] = "qiskit-aer"
df = df.rename(columns={"n_bits": "n_bits", "shots": "shots", "sim_time_s": "sim_time_s", "l2_error": "l2_error"})
frames_no_noise.append(df[["target", "n_bits", "shots", "sim_time_s", "l2_error"]])

# CudaQ no-noise (GPU)
df = pd.read_csv(files["cudaq_gpu"])
df["target"] = "cudaq-gpu"
df = df.rename(columns={"l2_norm": "l2_error"})
frames_no_noise.append(df[["target", "n_bits", "shots", "sim_time_s", "l2_error"]])

# CudaQ no-noise (CPU)
df = pd.read_csv(files["cudaq_cpu"])
df["target"] = "cudaq-cpu"
df = df.rename(columns={"l2_norm": "l2_error"})
frames_no_noise.append(df[["target", "n_bits", "shots", "sim_time_s", "l2_error"]])

pd.concat(frames_no_noise, ignore_index=True).to_csv(OUT_NON_NOISE, index=False)

# ----------------------------
# NOISE DATA TREATMENT
# ----------------------------
frames_noise = []

# CudaQ noise
df = pd.read_csv(files["cudaq_noise"])
df["target"] = "density-matrix-cpu"
df = df.rename(columns={"time_sampling": "time_sampling", "time_density": "time_density"})
frames_noise.append(df[["target", "n_bits", "shots", "noise", "probability", "time_sampling", "l2_pop", "time_density", "fro_norm", "fidelity"]])

# Qiskit noise
df = pd.read_csv(files["qiskit_noise"])
df["target"] = "qiskit-aer"
frames_noise.append(df[["target", "n_bits", "shots", "noise", "probability", "time_sampling", "l2_pop", "time_density", "fro_norm", "fidelity"]])

pd.concat(frames_noise, ignore_index=True).to_csv(OUT_NOISE, index=False)

print(f"✔ Saved: {OUT_NON_NOISE}")
print(f"✔ Saved: {OUT_NOISE}")
