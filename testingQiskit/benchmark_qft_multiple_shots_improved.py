#!/usr/bin/env python3
"""
QFT Benchmark Script (Qiskit + AerSimulator)

This script benchmarks the Quantum Fourier Transform (QFT) circuit using
Qiskit AerSimulator, supporting both GPU and CPU backends. The user can customize:

- Number of qubits (range)
- Initialization method: zero or GHZ
- Backend simulation method (e.g., statevector, density_matrix)
- Shot counts (list of powers of 2)
- Output file name

Usage:
    python benchmark_qft.py --init ghz --min_qubits 4 --max_qubits 25 \
        --method statevector --shots 1024 2048 4096 --output results/ghz_test.csv

Author: Madalena Barros, PIC Quantum Simulation (2025)
"""

import os
import math
import time
import numpy as np
import pandas as pd

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

from qiskit.circuit.library import QFT


def build_qft(n: int, init_state: str = "zero"):
    """Return an n-qubit QFT circuit using Qiskit's built-in QFT class."""
    qc = QuantumCircuit(n,n)

    # Initialization
    if init_state == "ghz":
        qc.h(0)
        for i in range(1, n):
            qc.cx(0, i)
    elif init_state != "zero":
        raise ValueError(f"Unknown init_state '{init_state}' (choose 'zero' or 'ghz')")

    # Add QFT
    qft = QFT(num_qubits=n, approximation_degree=0, do_swaps=True, inverse=False, insert_barriers=False)
    qc.append(qft.to_instruction(), range(n))

    qc.measure(range(n), range(n))


    return qc



def compute_l2_error(counts, n_qubits, shots):
    """Compute the L2 norm between the observed and uniform distributions."""
    N = 2 ** n_qubits
    ideal = np.full(N, 1 / N)
    sampled = np.zeros(N)
    for bitstr, freq in counts.items():
        idx = int(bitstr[::-1], 2)
        sampled[idx] = freq / shots
    return np.linalg.norm(sampled - ideal, ord=2)


def make_backend(method: str = "statevector", device: str = "GPU"):
    """Create and return the appropriate backend simulator."""
    if method == "statevector":
        if device == "GPU":
            backend = AerSimulator(method='statevector', device='GPU')
            backend_type = "GPU"
        else:
            backend = AerSimulator(method='statevector')
            backend_type = "CPU"
    elif method == "density_matrix":
        if device == "GPU":
            backend = AerSimulator(method='density_matrix', device='GPU')
            backend_type = "GPU"
        else:
            backend = AerSimulator(method='density_matrix')
            backend_type = "CPU"
    else:
        raise ValueError(f"Unknown method '{method}' (choose 'statevector' or 'density_matrix')")

    return backend, backend_type

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--init", type=str, default="zero", choices=["zero", "ghz"],
                        help="Initial state before QFT (zero or ghz)")
    parser.add_argument("--method", type=str, default="statevector",
                        help="Backend simulation method (e.g., statevector, density_matrix)")
    parser.add_argument("--min_qubits", type=int, default=3, help="Minimum number of qubits")
    parser.add_argument("--max_qubits", type=int, default=27, help="Maximum number of qubits")
    parser.add_argument("--shots", type=int, nargs="+", default=[2**i for i in range(10, 20)],
                        help="List of shot counts (e.g., --shots 1024 2048 4096)")
    parser.add_argument("--output", type=str, default=None, help="Output CSV file path")
    parser.add_argument("--device", type=str, default="GPU", choices=["GPU", "CPU"],
                        help="Target device for simulation (default: GPU)")

    args = parser.parse_args()
    init_state = args.init
    method = args.method
    min_qubits = args.min_qubits
    max_qubits = args.max_qubits
    shot_counts = args.shots
    device_choice = args.device.upper()
    output_path = args.output

    backend, backend_type = make_backend(method, device_choice)

    records = []
    for shots in shot_counts:
        print(f"\n--- Benchmarking {shots} shots ---")
        for n in range(min_qubits, max_qubits + 1):
            qc = build_qft(n, init_state=init_state)
            tqc = transpile(qc, backend)

            _ = backend.run(tqc, shots=32).result()

            t0 = time.perf_counter()
            result = backend.run(tqc, shots=shots).result()
            elapsed = time.perf_counter() - t0

            counts = result.get_counts()
            l2_err = compute_l2_error(counts, n, shots)
            throughput = shots / elapsed

            print(f"{n:2d} qubits | {shots:7d} shots | {elapsed:6.3f}s | {throughput:8.1f} shots/s | "
                  f"L2 error={l2_err:.6f} | {backend_type} | init={init_state}")

            records.append({
                "n_bits":     n,
                "shots":      shots,
                "sim_time_s": elapsed,
                "throughput": throughput,
                "l2_error":   l2_err,
                "backend":    backend_type,
                "init":       init_state,
                "method":     method
            })

    os.makedirs("Results", exist_ok=True)
    if not output_path:
        output_path = os.path.join(
            "Results", f"multishot_qft_{backend_type.lower()}_{init_state}_{method}.csv"
        )

    pd.DataFrame(records).to_csv(output_path, index=False)
    print(f"\n📁 Results saved to {output_path}")
