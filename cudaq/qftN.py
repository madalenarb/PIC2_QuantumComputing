#!/usr/bin/env python3
"""
Generic N-qubit Quantum Fourier Transform in CUDA-Q
---------------------------------------------------
Usage:
    python qftN.py            # default N = 3
    python qftN.py 5          # 5-qubit QFT
"""

import math
import time
import cudaq
import sys

# Set target to qpp-cpu or nvidia from command line
if len(sys.argv) > 2 and (sys.argv[1] == "qpp-cpu" or sys.argv[1] == "nvidia"):
    TARGET = sys.argv[2]
else:
    TARGET = "qpp-cpu"

cudaq.set_target(TARGET)
target = cudaq.get_target()
numgpu = cudaq.num_available_gpus()
print("Current target  :", target.name)
print("Available GPUs  :", numgpu)

# Get N_BITS from command line
def parse_args():
    if len(sys.argv) > 1:
        return int(sys.argv[1])
    return 3

N_BITS = parse_args()
print(f"Number of qubits: {N_BITS}")

# ------------------------------------------------------------
# QFT kernel
define_kernel = """
@cudaq.kernel
def qft():
    q = cudaq.qvector(N_BITS)

    # Hadamard + controlled-phase ladder
    for k in range(N_BITS):            # MSB … LSB
        h(q[k])
        for j in range(k + 1, N_BITS):
            angle = math.pi / (2 ** (j - k))    # π / 2^{j-k}
            r1.ctrl(angle, q[j], q[k])          # control = q[k], target = q[j]

    # Bit-reversal swap network
    for i in range(N_BITS // 2):
        swap(q[i], q[N_BITS - i - 1])
"""
@cudaq.kernel  # setting the kernel so I can create the function
def qft():
    q = cudaq.qvector(N_BITS)

    # Hadamard + controlled-phase ladder
    for k in range(N_BITS):            # MSB … LSB
        h(q[k])
        for j in range(k + 1, N_BITS):
            angle = math.pi / (2 ** (j - k))    # π / 2^{j-k}
            r1.ctrl(angle, q[j], q[k])          # control = q[k], target = q[j]

    # Bit-reversal swap network
    for i in range(N_BITS // 2):
        swap(q[i], q[N_BITS - i - 1])

# ------------------------------------------------------------
# Utility functions
def get_probabilities(counts: dict) -> dict:
    """Convert raw counts into probability distribution."""
    total = sum(counts.values())
    return {state: count / total for state, count in counts.items()}


def get_L2_norm(probs: dict, ideal: float) -> float:
    """Efficient L2 norm calculation from ideal uniform distribution."""
    norm_squared = 0.0
    num_states = 2 ** int(math.log2(len(probs)))  # assumes full set

    for i in range(num_states):
        state = f"{i:0{int(math.log2(num_states))}b}"
        p = probs.get(state, 0.0)
        norm_squared += (p - ideal) ** 2

    return math.sqrt(norm_squared) / num_states

# ------------------------------------------------------------
# Run a demo: forward QFT on |0…0⟩
if __name__ == "__main__":
    # Draw the circuit
    print(f"Circuit diagram for {N_BITS}-qubit QFT:\n")
    print(cudaq.draw(qft))

    # Simulation parameters
    SHOTS = 1024

#    Warm-up run (32 shots) to trigger JIT compilation & context setup
    _ = cudaq.sample(qft, shots_count=32)

    # Time the simulation
    t_start = time.perf_counter()
    counts = cudaq.sample(qft, shots_count=SHOTS)
    t_end = time.perf_counter()

    # Report measurement distribution
    print(f"\nMeasurement distribution ({SHOTS} shots):")
    for bitstr, freq in sorted(counts.items()):
        print(f"  |{bitstr}⟩ : {freq}")

    # Compute probabilities and L2 norm
    probs = get_probabilities(counts)
    print("\nProbabilities:")
    for state, prob in sorted(probs.items()):
        print(f"  |{state}⟩ : {prob:.4f}")
    ideal = 1 / (2 ** N_BITS)
    l2_norm = get_L2_norm(probs, ideal)
    print(f"\nL2 norm from uniform: {l2_norm:.3e}")

    # Report simulation time
    sim_time = t_end - t_start
    print(f"Simulation time: {sim_time:.6f} seconds")

    # Report target info
    print(f"Target: {target.name}")
    print(f"Available GPUs: {numgpu}")
