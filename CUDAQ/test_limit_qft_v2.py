#!/usr/bin/env python3
"""
CUDA-Q raw-speed benchmark (GPU trajectory backend)

• Sweeps qubit-count from START … STOP   (inclusive)
• Runs SHOTS executions per size
• Prints wall-time, host RAM, GPU RAM and per-size speed-up
"""

import math, os, time, cudaq

# ───── optional memory helpers ───────────────────────────────────────────
try:
    import psutil
    _proc = psutil.Process(os.getpid())
    host_mb = lambda: _proc.memory_info().rss / 2**20
except ImportError:
    host_mb = lambda: float('nan')

try:
    from pynvml import (
        nvmlInit, nvmlShutdown, nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetMemoryInfo,
    )
    nvmlInit()
    _h = nvmlDeviceGetHandleByIndex(0)
    gpu_mb = lambda: nvmlDeviceGetMemoryInfo(_h).used / 2**20
except Exception:
    gpu_mb = lambda: float('nan')
# ─────────────────────────────────────────────────────────────────────────

START = int(os.getenv("START", 1))
STOP  = int(os.getenv("STOP" , 26))
SHOTS = int(os.getenv("SHOTS", 2048))

cudaq.reset_target()
cudaq.set_target("nvidia")
assert cudaq.num_available_gpus() > 0, "No NVIDIA GPU visible to CUDA-Q"

# ───── QFT kernel factory ────────────────────────────────────────────────
def qft_kernel(n_qubits: int):
    @cudaq.kernel
    def qft():
        q = cudaq.qvector(n_qubits)
        for k in range(n_qubits):
            h(q[k])
            for j in range(k + 1, n_qubits):
                ang = math.pi / (2 ** (j - k))      #  <-- no bit-shift
                r1.ctrl(ang, q[j], q[k]) 
        # bit-reverse
        for i in range(n_qubits // 2):
            swap(q[i], q[n_qubits - 1 - i])
    return qft
# ─────────────────────────────────────────────────────────────────────────

header = (
    f"{'q':>3} |{'shots':>6} |{'sec':>7} |{'shots/s':>11} |"
    f"{'host ΔMB':>9} |{'GPU ΔMB':>8} |{'⇑':>5}"
)
print(header)
print('-' * len(header))

baseline = None
for n in range(START, STOP + 1):
    ker = qft_kernel(n)
    cudaq.sample(ker, shots_count=32)        # warm-up / JIT

    h0, g0 = host_mb(), gpu_mb()
    t0 = time.perf_counter()
    cudaq.sample(ker, shots_count=SHOTS)
    dt = time.perf_counter() - t0
    h1, g1 = host_mb(), gpu_mb()

    host_delta = h1 - h0
    gpu_delta  = g1 - g0
    thr = SHOTS / dt
    if baseline is None:
        baseline = dt
    speedup = baseline / dt

    print(
        f"{n:3d} |{SHOTS:6d} |{dt:7.3f} |{thr:11.1f} |"
        f"{host_delta:9.1f} |{gpu_delta:8.1f} |{speedup:5.2f}"
    )

try:
    nvmlShutdown()
except Exception:
    pass
