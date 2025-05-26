#!/usr/bin/env python3
"""
Sweep-benchmark: depolarizing-noisy QFT on the NVIDIA GPU simulator
-------------------------------------------------------------------
usage:
    python noisy_qft_bench.py              # default sweep 6–14 qubits, p=1 %
    python noisy_qft_bench.py 8 18 2048    # 8–18 qubits, 2048 shots
    python noisy_qft_bench.py --ps 0.5 2   # run p = 0.5 % and 2 %

• GPU trajectory backend (“nvidia”)
• Depolarizing noise applied to H, R1, CX
• Loop-safe gate calls
"""

import argparse, math, time, statistics, cudaq

# ── CLI ────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("start", nargs="?", type=int, default=6,
                    help="first qubit count (default 6)")
parser.add_argument("stop",  nargs="?", type=int, default=14,
                    help="last qubit count  (inclusive, default 14)")
parser.add_argument("shots", nargs="?", type=int, default=512,
                    help="shots per data point (default 512)")
parser.add_argument("--ps", "--p-list", nargs="+", type=float,
                    help="list of depolarizing rates in %% (e.g. 0.5 1 2)")
args = parser.parse_args()

P_LIST = [x/100 for x in (args.ps or [1.0])]   # convert % → probability

# ── GPU backend ────────────────────────────────────────────────────────
cudaq.reset_target()
cudaq.set_target("nvidia")
assert cudaq.num_available_gpus() > 0, "GPU not visible!"

# ── QFT kernel factory ────────────────────────────────────────────────
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

# ── Benchmark sweep ────────────────────────────────────────────────────
print(f"{'p(%)':>5}  {'n':>3}  {'shots':>6}  {'sec':>7}  {'shots/s':>10}")
for p in P_LIST:
    # build noise model once per p
    noise = cudaq.NoiseModel()
    depo  = cudaq.DepolarizationChannel(p)
    for g in ("h", "r1", "cx"):
        noise.add_all_qubit_channel(g, depo)
    cudaq.set_noise(noise)

    for n in range(args.start, args.stop + 1):
        kern = make_qft(n)
        # Warm-up run (32 shots) to trigger JIT compilation & context setup
        _ = cudaq.sample(kern, shots_count=32)

        t0 = time.perf_counter()
        cudaq.sample(kern, shots_count=args.shots)
        dt = time.perf_counter() - t0

        print(f"{p*100:5.1f}  {n:3d}  {args.shots:6d}  {dt:7.3f}  "
              f"{args.shots/dt:10.1f}")

