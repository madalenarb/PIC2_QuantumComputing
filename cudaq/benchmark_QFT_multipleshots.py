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

  python3 qftN_runall.py [options]

Options:
  --target [qpp-cpu | nvidia]   Backend target to run on (default: nvidia)
  --init   [zero | ghz]         Initial state of qubits (default: zero)
  --max-bits N                  Max number of qubits (default: 28/21)
  
  Mutually exclusive (one required):
    --shots S                   Single shot count to use (default: 131072)
    --multi                     Run multiple shots sweep (2^12 … 2^19)

Examples:
  python3 qftN_runall.py
  python3 qftN_runall.py --target qpp-cpu --init ghz --shots 50000
  python3 qftN_runall.py --multi
  python3 qftN_runall.py --multi --max-bits 20
"""

import math
import time
import os
import argparse
import subprocess

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

def run_benchmark(n_bits, target, shots, init='zero'):
    cudaq.set_target(target)
    kernel = make_qft_kernel(n_bits, init)
    _ = cudaq.sample(kernel, shots_count=32)  # Warm-up

    t0 = time.perf_counter()
    counts = cudaq.sample(kernel, shots_count=shots)
    t1 = time.perf_counter()

    probs = get_probabilities(counts)
    l2 = get_l2_norm(probs, n_bits)
    return (t1 - t0, l2)

def parse_args():
    p = argparse.ArgumentParser(description="Run QFT benchmark with CUDA-Q.")
    p.add_argument('--target', choices=['qpp-cpu', 'nvidia'], default='nvidia',
                   help='Backend target (default: nvidia)')
    p.add_argument('--init', choices=['zero', 'ghz'], default='zero',
                   help='Qubit initialization (default: zero)')
    p.add_argument('--max-bits', type=int, default=None,
                   help='Maximum number of qubits to benchmark '
                        '(default depends on target/init)')
    group = p.add_mutually_exclusive_group()
    group.add_argument('--shots', type=int,
                       help='Single shot count to use (default: 131072)')
    group.add_argument('--multi', action='store_true',
                       help='Run the full multiple-shots sweep (2^12 … 2^19)')
    return p.parse_args()

def main():
    args = parse_args()

    # Determine safe defaults
    MAX_BITS = {'nvidia': 28, 'qpp-cpu': 21}
    GHZ_LIMIT = 27
    min_q = 3

    default_max = GHZ_LIMIT if (args.init == 'ghz' and args.target == 'nvidia') else MAX_BITS[args.target]
    max_bits = args.max_bits or default_max
    if not (min_q <= max_bits <= default_max):
        print(f"⚠️  Clamping --max-bits to {default_max}")
        max_bits = default_max

    # Build shot list
    if args.multi:
        shots_list = [2 ** i for i in range(12, 20)]
    else:
        single = args.shots or 2**17
        shots_list = [single]

    print(f"Detected CPU cores: {get_cpu_count()}")
    print(f"Detected GPUs     : {cudaq.num_available_gpus()}")
    print(f"Running on target : {args.target}")
    print(f"Init state        : {args.init}")
    print(f"Max qubits        : {max_bits}")
    print(f"Shots            : {shots_list}\n")

    records = []
    for shots in shots_list:
        print(f"🔁 Benchmarking {shots} shots:")
        for n in range(min_q, max_bits + 1):
            t, l2 = run_benchmark(n, args.target, shots, init=args.init)
            print(f"  {n:2d} qubits → time {t:.4f}s | L2 {l2:.3e}")
            records.append({
                "target":     args.target,
                "init":       args.init,
                "n_bits":     n,
                "shots":      shots,
                "sim_time_s": t,
                "l2_norm":    l2
            })

    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"qftN_{args.target}_{args.init}.csv")
    pd.DataFrame(records).to_csv(out_path, index=False)
    print(f"\n✅ Results saved → {out_path}")

if __name__ == "__main__":
    main()
