#!/usr/bin/env python3
"""
CUDAQ QFT N-qubit benchmark
===========================

This script benchmarks the QFT kernel on different targets (CPU or GPU).
It generates a QFT kernel for N qubits, runs it on the specified target,
and measures:

  • Simulation time (in seconds)
  • L2 norm of the resulting sampled distribution vs. ideal

It automatically detects the number of logical CPU cores and available NVIDIA GPUs,
and saves all results to a CSV file for further analysis.

───────────────────────────────────────────────────────
Usage:

  python3 qftN_runall.py --target [TARGET] --init [zero | ghz]

Options:
  --target [qpp-cpu | nvidia]   Backend target to run on (default: nvidia)
  --init   [zero | ghz]         Initial state of qubits (default: zero)

Examples:
  python3 qftN_runall.py
  python3 qftN_runall.py --target qpp-cpu --init ghz
"""

import math
import time
import os
import argparse
import cudaq
import pandas as pd

def get_cpu_count():
    return os.cpu_count()

def make_qft_kernel(n_bits, init_state="zero"):
    if init_state == "ghz":
        @cudaq.kernel
        def qft():
            q = cudaq.qvector(n_bits)
            h(q[0])
            for i in range(1, n_bits):
                x.ctrl(q[0], q[i])
            # Apply QFT
            for k in range(n_bits):
                h(q[k])
                for j in range(k + 1, n_bits):
                    angle = math.pi / (2 ** (j - k))
                    r1.ctrl(angle, q[j], q[k])
            for i in range(n_bits // 2):
                swap(q[i], q[n_bits - i - 1])
        return qft

    else:
        @cudaq.kernel
        def qft():
            q = cudaq.qvector(n_bits)
            # Apply QFT directly
            for k in range(n_bits):
                h(q[k])
                for j in range(k + 1, n_bits):
                    angle = math.pi / (2 ** (j - k))
                    r1.ctrl(angle, q[j], q[k])
            for i in range(n_bits // 2):
                swap(q[i], q[n_bits - i - 1])
        return qft


def get_probabilities(counts):
    total = sum(counts.values())
    return {state: freq / total for state, freq in counts.items()}

def get_l2_norm(probs, n_bits):
    N = 2 ** n_bits
    sum_p2 = sum(p * p for p in probs.values())
    return math.sqrt(max(sum_p2 - 1.0 / N, 0.0))

def run_benchmark(n_bits, target, shots=1024, init='zero'):
    cudaq.set_target(target)
    kernel = make_qft_kernel(n_bits, init)
    _ = cudaq.sample(kernel, shots_count=32)  # Warm-up

    t_start = time.perf_counter()
    counts = cudaq.sample(kernel, shots_count=shots)
    t_end = time.perf_counter()

    probs = get_probabilities(counts)
    l2_norm = get_l2_norm(probs, n_bits)
    return (t_end - t_start, l2_norm)

def main():
    parser = argparse.ArgumentParser(description="Run QFT benchmark with CUDA-Q.")
    parser.add_argument('--target', type=str, choices=['qpp-cpu', 'nvidia'], default='nvidia',
                        help='Backend target (default: nvidia)')
    parser.add_argument('--init', type=str, choices=['zero', 'ghz'], default='zero',
                        help='Qubit initialization (default: zero)')
    parser.add_argument('--max_bits', type=int, default=28,
                        help='Maximum number of qubits to benchmark (default: 28 for nvidia, 21 for qpp-cpu)')
    args = parser.parse_args()

    target = args.target
    init_state = args.init
    requested_max_bits = args.max_bits
    target = args.target

    # Define safe limits
    MAX_BITS_LIMIT = {
        "nvidia": 28,
        "qpp-cpu": 21
    }
    GHZ_LIMIT_ON_NVIDIA = 27  # Safety bound due to GHZ entanglement overhead
    MIN_BITS = 3

    # Determine default max_bits based on target and init
    default_max = GHZ_LIMIT_ON_NVIDIA if (init_state == "ghz" and target == "nvidia") else MAX_BITS_LIMIT.get(target, 21)

    # Clamp and warn if necessary
    if requested_max_bits < MIN_BITS or requested_max_bits > default_max:
        print(f"⚠️  Requested --max_bits={requested_max_bits} is out of bounds for target='{target}' and init='{init_state}'.")
        print(f"   Using max_bits={default_max} instead.")
        max_bits = default_max
    else:
        max_bits = requested_max_bits




    print(f"Detected logical CPU cores: {get_cpu_count()}")
    print(f"Detected NVIDIA GPUs      : {cudaq.num_available_gpus()}")

    cudaq.set_target(target)
    print(f"\n🎯 Running on target: {target} with init: {init_state}")

    if target == "qpp-cpu":
        print(f"🧠 Using CPU cores: {get_cpu_count()}")
    else:
        print(f"🖥️  Using GPUs: {cudaq.num_available_gpus()}")

    shots_list = [2 ** i for i in range(12, 20)]
    records = []

    for shots in shots_list:
        print(f"\n🔁 Benchmarking {shots} shots:")
        for n_bits in range(3, max_bits + 1):
            sim_time, l2 = run_benchmark(n_bits, target, shots, init=init_state)
            print(f"  {n_bits:2d} qubits → time: {sim_time:.4f}s | L2: {l2:.3e}")
            records.append({
                "target":     target,
                "init":       init_state,
                "n_bits":     n_bits,
                "shots":      shots,
                "sim_time_s": sim_time,
                "l2_norm":    l2
            })

    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    output_path = os.path.join(results_dir, f"qftN_{target}_{init_state}_multiple_shots.csv")

    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    print(f"\n✅ Results saved to → {output_path}")

if __name__ == "__main__":
    main()
