#!/usr/bin/env python3
"""
Raw-speed benchmark for Qiskit Aer
==================================

• Can run on CPU (default) or NVidia GPU if qiskit-aer-gpu is installed
• No noise model
• Sweeps qubits  = START … STOP       (default 6 … 30)
• Runs <SHOTS> executions per size    (default 2 048)
• Prints shots/s so you can eyeball saturation

Requires:  qiskit  >= 1.0,  qiskit-aer >= 0.13
"""

import math, time, sys, argparse, os

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


parser = argparse.ArgumentParser()
parser.add_argument("-s", "--start", type=int, default=6,   help="min #qubits")
parser.add_argument("-e", "--end",   type=int, default=25,  help="max #qubits (inclusive)")
parser.add_argument("-n", "--shots", type=int, default=2048,help="#shots per circuit")
parser.add_argument("--gpu", action="store_true",           help="try GPU backend")
args = parser.parse_args()

START, STOP, SHOTS = args.start, args.end, args.shots


# ── 1)  Pick Aer backend (GPU preferred if flag & available) ─────────
if args.gpu:
    try:
        backend = AerSimulator(method="statevector", device="GPU")
        DEVICE  = "GPU"
    except AerError as err:
        print("⚠️  GPU simulator unavailable – falling back to CPU\n", file=sys.stderr)
        backend = AerSimulator(method="statevector", device="CPU")
        DEVICE  = "CPU"
else:
    backend = AerSimulator(method="statevector", device="CPU")
    DEVICE  = "GPU"


# ── 2)  QFT circuit factory ──────────────────────────────────────────
def qft_circuit(n: int) -> QuantumCircuit:
    qc = QuantumCircuit(n)
    # Hadamard + controlled-phase staircase
    for k in range(n):
        qc.h(k)
        for j in range(k + 1, n):
            angle = math.pi / (2 ** (j - k))
            qc.cp(angle, k, j)
    # bit-reversal swaps
    for i in range(n // 2):
        qc.swap(i, n - 1 - i)
    qc.measure_all()
    return qc


# benchmark loop 
hdr = f"  q | shots |    sec |   shots/s | backend={DEVICE}"
print(hdr)
print("-" * len(hdr))

for n in range(START, STOP + 1):
    circ   = qft_circuit(n)
    t_circ = transpile(circ, backend, optimization_level=1)

    # warm-up / jit-compile
    backend.run(t_circ, shots=32).result()

    t0 = time.perf_counter()
    backend.run(t_circ, shots=SHOTS).result()
    dt = time.perf_counter() - t0

    print(f"{n:3d} | {SHOTS:5d} | {dt:6.3f} | {SHOTS/dt:9.1f}")




