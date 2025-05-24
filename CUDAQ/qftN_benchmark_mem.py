#!/usr/bin/env python3
"""
QFT benchmark with live GPU‑utilisation sampling (MiB + %).
Requires:  pip install cudaq nvidia-ml-py3 pandas
Usage:     python qft_benchmark_gpu_mon.py [shots]
"""

import math, time, os, sys, threading, subprocess
import pandas as pd
import cudaq
from pynvml import (
    nvmlInit, nvmlShutdown,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetUtilizationRates,
    nvmlDeviceGetMemoryInfo
)

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
# 2.  GPU monitor (background thread)
# ---------------------------------------------------------------------
class GPUUtilMonitor:
    """Sample GPU core‑% and memory (%, MiB) every `interval_sec`."""
    def __init__(self, idx: int = 0, interval_sec: float = 0.1):
        self.idx       = idx
        self.interval  = interval_sec
        self.core_pct  = []
        self.mem_pct   = []
        self.mem_mib   = []
        self._stop     = threading.Event()
        self._thr      = threading.Thread(target=self._worker, daemon=True)

    def __enter__(self):
        nvmlInit()
        self.handle = nvmlDeviceGetHandleByIndex(self.idx)
        self._thr.start()
        return self

    def __exit__(self, *_):
        self._stop.set(); self._thr.join(); nvmlShutdown()

    def _worker(self):
        while not self._stop.is_set():
            util = nvmlDeviceGetUtilizationRates(self.handle)
            mem  = nvmlDeviceGetMemoryInfo(self.handle)
            self.core_pct.append(util.gpu)             # %
            self.mem_pct .append(util.memory)          # %
            self.mem_mib .append(mem.used / 1024**2)   # MiB
            time.sleep(self.interval)

    @property
    def avg_core_pct(self): return sum(self.core_pct)/len(self.core_pct) if self.core_pct else 0.0
    @property
    def avg_mem_pct (self): return sum(self.mem_pct) /len(self.mem_pct) if self.mem_pct  else 0.0
    @property
    def avg_mem_mib(self): return sum(self.mem_mib)/len(self.mem_mib) if self.mem_mib else 0.0

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
    return math.sqrt(max(sum(p*p for p in probs) - 1/2**n_bits, 0.0))

# ---------------------------------------------------------------------
# 4.  Benchmark wrapper
# ---------------------------------------------------------------------
def run_benchmark(n_bits, target, shots=1024, monitor_gpu=False):
    cudaq.set_target(target)
    kernel = make_qft_kernel(n_bits)

    # ── Warm-up: trigger JIT / context creation ──
    _ = cudaq.sample(kernel, shots_count=32)

    if monitor_gpu and target == "nvidia":
        with GPUUtilMonitor() as mon:
            t0 = time.perf_counter()
            counts = cudaq.sample(kernel, shots_count=shots)
            t1 = time.perf_counter()
        core, mem_pct, mem_mib = mon.avg_core_pct, mon.avg_mem_pct, mon.avg_mem_mib
    else:
        t0 = time.perf_counter()
        counts = cudaq.sample(kernel, shots_count=shots)
        t1 = time.perf_counter()
        core = mem_pct = mem_mib = 0.0

    return (t1-t0, l2_norm(counts, n_bits), core, mem_pct, mem_mib)

# ---------------------------------------------------------------------
# 5.  Main loop
# ---------------------------------------------------------------------
if __name__ == "__main__":
    shots = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
    print(f"Logical CPU cores : {get_cpu_count()}")
    print(f"NVIDIA GPUs       : {get_gpu_count()}")

    rows = []
    for target, max_bits in [("qpp-cpu", 5), ("nvidia", 28)]:
        print(f"\n### Target: {target}")
        for n_bits in range(3, max_bits+1):
            t, l2, core, mem_pct, mem_mib = run_benchmark(
                n_bits, target, shots, monitor_gpu=True
            )
            print(f"{n_bits:2d}q → {t:8.4f}s  L2={l2:8.3e}  "
                  f"GPU={core:5.1f}%  MEM={mem_pct:5.1f}%/{mem_mib:6.1f} MiB")
            rows.append(dict(
                target=target, n_bits=n_bits, shots=shots,
                sim_time_s=t, l2_norm=l2,
                gpu_core_pct=core, gpu_mem_pct=mem_pct, gpu_mem_mib=mem_mib
            ))

    # Export to CSV - 2 files one per target
    df = pd.DataFrame(rows)
    for target in ["qpp-cpu", "nvidia"]:
        df[df.target == target].drop(columns=["target"]).to_csv(
            f"a_qft_benchmark_{target}_{shots}.csv", index=False
        )
    print("Done.")
