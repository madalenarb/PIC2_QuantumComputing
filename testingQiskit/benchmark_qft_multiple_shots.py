#!/usr/bin/env python3
"""
Benchmark QFT simulation times and L2 error norms for N-qubit QFT circuits
using Qiskit AerSimulator, detecting and reporting the number of available CPUs,
and exporting results to CSV.

Usage:
    python benchmark_qft.py [shots] [max_qubits]
"""
import math
import time
import sys
import os
import numpy as np
import pandas as pd
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# Build an N-qubit QFT circuit
def build_qft(n_bits):
    qc = QuantumCircuit(n_bits, n_bits)
    # QFT: Hadamard + controlled-phase
    for k in range(n_bits):
        qc.h(k)
        for j in range(1, n_bits - k):
            angle = math.pi / (2 ** j)
            qc.cp(angle, k, k + j)
    # Swap reversal
    for i in range(n_bits // 2):
        qc.swap(i, n_bits - 1 - i)
    # Measurement
    qc.measure(range(n_bits), range(n_bits))
    return qc

# Compute L2 error vs uniform
def compute_l2_error(counts, n_bits, shots):
    N = 2 ** n_bits
    ideal = np.full(N, 1 / N)
    sampled = np.zeros(N)
    for bitstr, freq in counts.items():
        idx = int(bitstr[::-1], 2)
        sampled[idx] = freq / shots
    diff = sampled - ideal
    return np.linalg.norm(diff, 2)

# Run one benchmark
def run_benchmark(n_bits, simulator, shots, use_gpu=False):
    qc = build_qft(n_bits)
    qc_t = transpile(qc, simulator)
    _ = simulator.run(qc_t, shots=32).result()  # warm‑up

    t0 = time.perf_counter()
    result = simulator.run(qc_t, shots=shots).result()
    elapsed = time.perf_counter() - t0

    l2_err = compute_l2_error(result.get_counts(), n_bits, shots)

    return elapsed, l2_err


if __name__ == "__main__":
    # CLI args
    shots = [2**i for i in range(10, 20)]
    max_qubits = 28

    # Detect CPU count
    cpu_count = os.cpu_count() or 1
    print(f"Detected logical CPU cores: {cpu_count}")

    # Initialize AerSimulator with matching thread count
    simulator = AerSimulator(max_parallel_threads=cpu_count)
    print(f"AerSimulator using up to {cpu_count} parallel threads\n")

    records = []
    for num_shots in shots:
        print(f"Running QFT with {num_shots} shots")
        for n_bits in range(3, max_qubits + 1):
            elapsed, l2_err = run_benchmark(
                n_bits, simulator, num_shots
            )
            print(f"  {n_bits} qubits: {elapsed:.3f}s, L2 error: {l2_err:.6f}")

            records.append({
                "n_bits": n_bits,
                "shots": num_shots,
                "sim_time_s": elapsed,
                "l2_error": l2_err
            })


    # Export to 2 CSVs nvidia and qpp-cpu
    df = pd.DataFrame(records)
    name_file = "benchmark_qft_qiskit_various_shots.csv"
    df.to_csv(os.path.join("Results", name_file), index=False)
    print(f"Results exported to {name_file}")

