#!/usr/bin/env python3
"""
Raw-speed benchmark for CUDA-Q on GPU
=====================================

• GPU trajectory backend (“nvidia”)
• No noise model
• Sweeps qubits = start … stop  (default 6 … 20)
• Runs <shots> executions per size (default 256)
• Prints shots/s so you can see where the GPU saturates
"""

import math, time, sys, cudaq

# ── CLI params:  start  stop  shots ──────────────────────────────────────
START = 1
STOP  = 30   # inclusive
SHOTS = 2048
TARGET = "nvidia"  # "qpp-cpu" for CPU backend
cudaq.set_target(TARGET)

# ── 1)  GPU backend, no noise ────────────────────────────────────────────
cudaq.reset_target()
cudaq.set_target("nvidia")                    # fastest GPU simulator
assert cudaq.num_available_gpus() > 0, "GPU not visible to CUDA-Q!"

# ── 2)  QFT kernel factory (loop-safe gates) ─────────────────────────────
def qft_kernel(n):
    @cudaq.kernel
    def qft():
        q = cudaq.qvector(n)
        # Hadamard + controlled-phase “staircase”
        for k in range(n):
            h(q[k])
            for j in range(k + 1, n):
                ang = math.pi / (2 ** (j - k))
                r1.ctrl(ang, q[j], q[k])
        # bit-reversal swaps
        for i in range(n // 2):
            swap(q[i], q[n - 1 - i])
    return qft

# ── 3)  Benchmark loop ───────────────────────────────────────────────────
print(f"{'qubits':>6}  {'shots':>6}  {'sec':>8}  {'shots/s':>10}")
for n in range(START, STOP + 1):
    kernel = qft_kernel(n)
    cudaq.sample(kernel, shots_count=32)      # warm-up compile & cache

    t0 = time.perf_counter()
    cudaq.sample(kernel, shots_count=SHOTS)
    dt = time.perf_counter() - t0

    print(f"{n:6d}  {SHOTS:6d}  {dt:8.3f}  {SHOTS/dt:10.1f}")
