#!/usr/bin/env python3
# qft3_cudaq.py – 3-qubit QFT on CPU or NVIDIA GPU backend using CUDA-Q

import math
import numpy as np
import cudaq

# --------------------------------------------------------------------
# Define the 3-qubit QFT kernel
# --------------------------------------------------------------------
@cudaq.kernel
def qft3():
    q = cudaq.qvector(3)
    h(q[0])
    r1.ctrl(math.pi / 2, q[1], q[0])
    r1.ctrl(math.pi / 4, q[2], q[0])
    h(q[1])
    r1.ctrl(math.pi / 2, q[2], q[1])
    h(q[2])
    swap(q[0], q[2])


def run_sampling(kernel, shots: int):
    """Run the quantum kernel and return result counts."""
    return cudaq.sample(kernel, shots_count=shots)

def get_probabilities(counts: dict) -> dict:
    """Convert raw counts into probability distribution."""
    total = sum(counts.values())
    return {state: count / total for state, count in counts.items()}

def print_distribution(probs: dict, ideal: float):
    """Print comparison with ideal distribution."""
    diffs = []
    print("\nMeasured vs Ideal Probabilities:")
    for i in range(8):
        state = f"{i:03b}"
        p = probs.get(state, 0.0)
        diff = abs(p - ideal)
        diffs.append(diff)
        print(f"  |{state}⟩: {p:.4f} (ideal: {ideal:.4f}, diff: {diff:.4f})")
    return np.linalg.norm(np.array(diffs))

# --------------------------------------------------------------------
# Main execution
# --------------------------------------------------------------------
if __name__ == "__main__":
    TARGET = "qpp-cpu"  # "nvidia"
    SHOTS = 4096

    cudaq.set_target(TARGET)  
    print("Backend target:", cudaq.get_target().name)
    print("Available GPUs:", cudaq.num_available_gpus() if TARGET == "nvidia" else "N/A")

    print("\nQuantum Circuit for 3-Qubit QFT:")
    print(cudaq.draw(qft3))

    # Run sampling
    counts = run_sampling(qft3, SHOTS)
    probs = get_probabilities(counts)

    # Print measured probabilities
    print("\nMeasured Probabilities:")
    for state in sorted(probs):
        print(f"  |{state}⟩: {probs[state]:.4f}")

    # Compare to ideal uniform distribution
    L2_error = print_distribution(probs, ideal=1/8)
    print(f"\nL2 Error from ideal distribution: {L2_error:.4e}")
