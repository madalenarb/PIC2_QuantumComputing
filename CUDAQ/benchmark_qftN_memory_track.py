#!/usr/bin/env python3
"""
QFT benchmark with live GPU-utilisation sampling (MiB + %).
Requires:   pip install cudaq nvidia-ml-py3 pandas openpyxl
Usage:      python qft_benchmark_gpu_mon.py [options]
"""

import math
import time
import os
import sys
import threading
import subprocess
import argparse

import pandas as pd
import cudaq
from pynvml import (
    nvmlInit, nvmlShutdown,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetUtilizationRates,
    nvmlDeviceGetMemoryInfo
)

# ---------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------
def parser():
    p = argparse.ArgumentParser(
        description="GPU-only QFT benchmark with live utilization monitoring."
    )
    p.add_argument("-s", "--shots",    type=int, default=1024,
                   help="Number of measurement shots per circuit run.")
    p.add_argument("-q", "--max-bits", type=int, default=27,
                   help="Maximum number of qubits to simulate.")
    p.add_argument("-o", "--out-dir",  type=str, default="results/GPUmonitored",
                   help="Directory to save CSV and Excel outputs.")
    return p

# ---------------------------------------------------------------------
# 1.  Detect available resources
# ---------------------------------------------------------------------
get_cpu_count = os.cpu_count

def get_gpu_count() -> int:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--list-gpus"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, check=True
        ).stdout
        return len(out.strip().splitlines())
    except Exception:
        return 0

# ---------------------------------------------------------------------
# 2.  GPU monitor
# ---------------------------------------------------------------------
class GPUUtilMonitor:
    """Sample GPU core-% and memory-% every `interval_sec`, compute actual memory used."""
    def __init__(self, idx: int = 0, interval_sec: float = 0.1):
        self.idx      = idx
        self.interval = interval_sec
        self.core_pct = []
        self.mem_pct  = []
        self._stop    = threading.Event()
        self._thr     = threading.Thread(target=self._worker, daemon=True)

    def __enter__(self):
        nvmlInit()
        self.handle    = nvmlDeviceGetHandleByIndex(self.idx)
        total          = nvmlDeviceGetMemoryInfo(self.handle).total
        self.total_mib = total / 1024**2
        self._thr.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        self._thr.join()
        nvmlShutdown()

    def _worker(self):
        while not self._stop.is_set():
            util = nvmlDeviceGetUtilizationRates(self.handle)
            self.core_pct.append(util.gpu)
            self.mem_pct.append(util.memory)
            time.sleep(self.interval)

    @property
    def avg_core_pct(self):
        return sum(self.core_pct)/len(self.core_pct) if self.core_pct else 0.0

    @property
    def avg_mem_pct(self):
        return sum(self.mem_pct)/len(self.mem_pct) if self.mem_pct else 0.0

    @property
    def avg_mem_mib(self):
        return (self.avg_mem_pct/100.0)*self.total_mib

# ---------------------------------------------------------------------
# 3.  QFT kernel + helper
# ---------------------------------------------------------------------
def make_qft_kernel(n_bits):
    @cudaq.kernel
    def qft():
        q = cudaq.qvector(n_bits)
        for k in range(n_bits):
            h(q[k])
            for j in range(k+1, n_bits):
                angle = math.pi / (2**(j-k))
                r1.ctrl(angle, q[j], q[k])
        for i in range(n_bits//2):
            swap(q[i], q[n_bits-i-1])
    return qft

def l2_norm(counts, n_bits):
    total = sum(counts.values())
    probs = [v/total for v in counts.values()]
    return math.sqrt(max(sum(p*p for p in probs) - 1/(2**n_bits), 0.0))

# ---------------------------------------------------------------------
# 4.  Benchmark wrapper
# ---------------------------------------------------------------------
def run_benchmark(n_bits, target, shots, monitor_gpu=False):
    cudaq.set_target(target)
    kernel = make_qft_kernel(n_bits)
    _ = cudaq.sample(kernel, shots_count=32)  # fresh warm-up per kernel
    if monitor_gpu:
        with GPUUtilMonitor() as mon:
            t0     = time.perf_counter()
            counts = cudaq.sample(kernel, shots_count=shots)
            t1     = time.perf_counter()
        core    = mon.avg_core_pct
        mem_pct = mon.avg_mem_pct
        mem_mib = mon.avg_mem_mib
    else:
        t0     = time.perf_counter()
        counts = cudaq.sample(kernel, shots_count=shots)
        t1     = time.perf_counter()
        core = mem_pct = mem_mib = 0.0
    return (t1-t0, l2_norm(counts, n_bits), core, mem_pct, mem_mib)

# ---------------------------------------------------------------------
# 5.  Main loop (GPU-only, zero init)
# ---------------------------------------------------------------------
def main():
    args     = parser().parse_args()
    shots    = args.shots
    max_bits = args.max_bits
    out_dir  = args.out_dir

    # Print parameters
    print("▶ Parameters:")
    print(f"    shots    = {shots}")
    print(f"    max_bits = {max_bits}")
    print(f"    out_dir  = {out_dir}\n")

    # Check GPU
    gpus = get_gpu_count()
    if gpus == 0:
        print("⚠️ No NVIDIA GPU detected. Exiting.")
        sys.exit(1)
    print(f"▶ Found {gpus} NVIDIA GPU(s)\n")

    target = "nvidia"
    rows   = []
    os.makedirs(out_dir, exist_ok=True)

    # Global warm-up for largest kernel
    print("▶ Performing initial warm-up on max_bits kernel...")
    _ = run_benchmark(max_bits, target, shots=32, monitor_gpu=False)
    print("✔ Warm-up complete\n")

    print(f"### Benchmarking zero-initialized QFT on GPU target: {target}")
    header = f"{'n_bits':>6} {'time_s':>8} {'L2':>8} {'GPU%':>6} {'Mem%':>6} {'MemMiB':>8}"
    print(header)
    print("-"*len(header))

    for n_bits in range(3, max_bits+1):
        t, l2, core, mem_pct, mem_mib = run_benchmark(
            n_bits, target, shots, monitor_gpu=True
        )
        print(f"{n_bits:6d} {t:8.4f} {l2:8.3e} {core:6.1f} {mem_pct:6.1f} {mem_mib:8.1f}")
        rows.append({
            "n_bits":       n_bits,
            "shots":        shots,
            "sim_time_s":   t,
            "l2_norm":      l2,
            "gpu_core_pct": core,
            "gpu_mem_pct":  mem_pct,
            "gpu_mem_mib":  mem_mib
        })

    # Save CSV + Excel
    df        = pd.DataFrame(rows)
    csv_path  = os.path.join(out_dir, f"qft_gpu_mon_{shots}.csv")
    xlsx_path = os.path.join(out_dir, f"qft_gpu_mon_{shots}.xlsx")
    df.to_csv(csv_path, index=False)
    df.to_excel(xlsx_path, index=False)
    print(f"\n✅ GPU-monitored results saved →\n   {csv_path}\n   {xlsx_path}")

if __name__ == "__main__":
    main()
