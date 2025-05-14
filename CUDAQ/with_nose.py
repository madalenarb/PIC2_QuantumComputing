#!/usr/bin/env python3
"""
Noisy-QFT benchmark on the NVIDIA GPU trajectory simulator.

• Depolarizing noise p = 1 % applied to H, R1, and CX gates
• Generates an N-qubit QFT kernel (loop-safe gate calls)
• Measures average runtime and throughput (shots / second)
"""

import math, time, statistics, cudaq

# ── 1)  GPU backend + depolarizing noise ──────────────────────────────
cudaq.reset_target()
cudaq.set_target("nvidia")               # GPU trajectory simulator
assert cudaq.num_available_gpus() > 0, "CUDA-Q can't see a GPU!"

p = 0.01                                  # 1 % depolarizing
noise = cudaq.NoiseModel()
depo  = cudaq.DepolarizationChannel(p)
for g in ("h", "r1", "cx"):               # skip "swap" (it’s 3 CXs)
    noise.add_all_qubit_channel(g, depo)

cudaq.set_noise(noise)                    # <── new API name

# ── 2)  QFT kernel factory ────────────────────────────────────────────
def make_qft(n):
    @cudaq.kernel
    def qft():
        q = cudaq.qvector(n)
        for k in range(n):
            h(q[k])
            for j in range(k + 1, n):
                theta = math.pi / 2 ** (j - k)
                r1.ctrl(theta, q[k], q[j])
        for i in range(n // 2):
            swap(q[i], q[n - 1 - i])
    return qft

# ── 3)  Benchmark & histogram ─────────────────────────────────────────
nqubits = 10
shots   = 1024
kernel  = make_qft(nqubits)

cudaq.sample(kernel, shots_count=32)      # warm-up compilation

times = []
for _ in range(5):
    t0 = time.perf_counter()
    cudaq.sample(kernel, shots_count=shots)
    times.append(time.perf_counter() - t0)

avg = statistics.mean(times)
print(f"\n{nqubits}-qubit QFT with 1 % depolarizing noise:")
print(f"  mean time  = {avg:.4f} s")
print(f"  throughput = {shots/avg:.1f} shots/s")

print("\nSampled measurement distribution:")
counts = cudaq.sample(kernel, shots_count=shots)
for bits, c in sorted(counts.items()):
    print(f"  |{bits}⟩ : {c}")
