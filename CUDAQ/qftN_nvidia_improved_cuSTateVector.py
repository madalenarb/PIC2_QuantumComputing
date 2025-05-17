"""
CUDAQ QFT with NVIDIA GPU benchmark (trying to improve performance)
- Switch to cuStateVector backend for better performance
- Write results incrementally and print each record as it's processed
"""
import math
import time
import os
import sys
import csv
import cudaq


def get_cpu_count():
    """
    Return the number of logical CPU cores available to this process.
    """
    return os.cpu_count()


def make_qft_kernel(n_bits):
    """
    Factory to generate an N-qubit QFT kernel.
    """
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
    """
    Convert raw counts to probabilities.
    """
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


def run_benchmark(n_bits, target, shots=1024, compute_l2=True):
    """
    Run a single benchmark and L2 measurement for given N and target.
    """
    cudaq.set_target(target)
    kernel = make_qft_kernel(n_bits)
    t_start = time.perf_counter()
    counts = cudaq.sample(kernel, shots_count=shots)
    t_end = time.perf_counter()

    l2_norm = 0.0
    if compute_l2:
        probs = get_probabilities(counts)
        l2_norm = get_l2_norm(probs, n_bits)
    return (t_end - t_start, l2_norm)


if __name__ == "__main__":
    # Number of shots (default 1024)
    shots = int(sys.argv[1]) if len(sys.argv) > 1 else 2048

    # Detect and report resources
    cpu_count = get_cpu_count()
    print(f"Detected logical CPU cores: {cpu_count}")
    cudaq.set_target("nvidia")
    gpu_count = cudaq.num_available_gpus()
    print(f"Detected NVIDIA GPUs   : {gpu_count}")

    # Choose your backend (e.g., cuStateVec for speed)
    target = "tensornet"  # "qpp-cpu", "qpp-gpu", "tensornet", "nvidia"

    cudaq.set_target(target)
    max_bits = 30
    compute_l2 = True

    print(f"\nBackend target: {target}")
    if target == "qpp-cpu":
        print(f"Using CPU cores: {cpu_count}")
    else:
        print(f"Using GPUs: {gpu_count}")

    # Prepare results CSV
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    name_file = f"benchmark_qft_{target}_incremental.csv"
    csv_path = os.path.join(results_dir, name_file)
    fieldnames = ["target", "n_bits", "shots", "sim_time_s", "l2_norm"]

    # Write header once
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    # Run benchmarks, appending and printing each record immediately
    for n_bits in range(3, max_bits + 1):
        sim_time, l2 = run_benchmark(n_bits, target, shots, compute_l2)
        record = {
            "target": target,
            "n_bits": n_bits,
            "shots": shots,
            "sim_time_s": sim_time,
            "l2_norm": l2
        }
        # Print the detailed record
        print(f"Processed {n_bits:2d}-qubit run → {record}")
        # Append to CSV
        with open(csv_path, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(record)

    print(f"All results written incrementally to {csv_path}")
