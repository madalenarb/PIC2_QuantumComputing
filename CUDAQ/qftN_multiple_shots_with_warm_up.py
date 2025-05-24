"""
CUDAQ QFT N-qubit benchmark
This script benchmarks the QFT kernel on different targets (CPU and GPU).
It generates a QFT kernel for N qubits, runs it on the specified target,
and measures the time taken and the L2 norm of the resulting state vector
against the ideal uniform distribution.
The script also detects the number of logical CPU cores and available GPUs,
and reports this information.
The results are saved to a CSV file for further analysis.

Usage:
    python qftN_multiple_shots.py [target]
    where [target] is "nvidia" (GPU) or "qpp-cpu" (CPU). Default is "nvidia".
"""
import math
import time
import os
import sys
import cudaq
import pandas as pd

def get_cpu_count():
    """Return the number of logical CPU cores available to this process."""
    return os.cpu_count()

def make_qft_kernel(n_bits):
    """Factory to generate an N-qubit QFT kernel."""
    @cudaq.kernel
    def qft():
        q = cudaq.qvector(n_bits)
        # Hadamard + controlled-phase ladder
        for k in range(n_bits):
            h(q[k])
            for j in range(k + 1, n_bits):
                angle = math.pi / (2 ** (j - k))
                r1.ctrl(angle, q[j], q[k])
        # Bit-reversal swap network
        for i in range(n_bits // 2):
            swap(q[i], q[n_bits - i - 1])
    return qft

def get_probabilities(counts):
    """Convert raw counts to probabilities."""
    total = sum(counts.values())
    return {state: freq / total for state, freq in counts.items()}

def get_l2_norm(probs, n_bits):
    """
    Compute L2 norm vs ideal uniform:
    L2 = sqrt(sum((p - 1/N)^2)) = sqrt(sum(p^2) - 1/N).
    """
    N = 2 ** n_bits
    sum_p2 = sum(p * p for p in probs.values())
    return math.sqrt(max(sum_p2 - 1.0 / N, 0.0))

def run_benchmark(n_bits, target, shots=1024):
    """
    Run a single benchmark and L2 measurement for given N and target,
    including a warm-up sample to trigger JIT/GPU setup.
    """
    cudaq.set_target(target)
    kernel = make_qft_kernel(n_bits)

    # Warm-up run (32 shots) to trigger JIT and GPU context
    cudaq.sample(kernel, shots_count=32)

    # Timed run
    t_start = time.perf_counter()
    counts = cudaq.sample(kernel, shots_count=shots)
    t_end = time.perf_counter()

    probs = get_probabilities(counts)
    l2_norm = get_l2_norm(probs, n_bits)
    return (t_end - t_start, l2_norm)

if __name__ == "__main__":
    # Detect resources
    cpu_count = get_cpu_count()
    gpu_count = cudaq.num_available_gpus()

    print(f"Detected logical CPU cores: {cpu_count}")
    print(f"Detected NVIDIA GPUs   : {gpu_count}")

    # Parse target from CLI
    target = sys.argv[1] if len(sys.argv) > 1 else "nvidia"
    # If user asked for GPU but none found, fall back to CPU
    if target == "nvidia" and gpu_count == 0:
        print("⚠️  No GPUs detected — falling back to CPU target\n")
        target = "qpp-cpu"

    max_bits = 28 if target == "nvidia" else 21

    cudaq.set_target(target)
    print(f"\nBackend target: {target}")
    if target == "qpp-cpu":
        print(f"Using CPU cores: {cpu_count}")
    else:
        print(f"Using GPUs: {gpu_count}")

    # Test shots from 2^10 to 2^19
    shots_list = [2 ** i for i in range(10, 20)]
    records = []

    for shots in shots_list:
        print(f"\nRunning benchmark with {shots} shots")
        for n_bits in range(3, max_bits + 1):
            sim_time, l2 = run_benchmark(n_bits, target, shots)
            print(f"  {n_bits:2d} qubits -> time: {sim_time:.6f}s, L2 norm: {l2:.3e}")
            records.append({
                "target":    target,
                "n_bits":    n_bits,
                "shots":     shots,
                "sim_time_s": sim_time,
                "l2_norm":   l2
            })

    # Create 2 dataframes and export to CSV
    df = pd.DataFrame([r for r in records if r["target"] == target])
    name_file = f"qftN_{target}_multiple_shots_warmup.csv"
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    df.to_csv(os.path.join(results_dir, name_file), index=False)
    print(f"Results exported to {name_file}")
    print(f"Results exported to {os.path.join(results_dir, name_file)}")
