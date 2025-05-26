#!/usr/bin/env python3
"""
Benchmark QFT circuit simulation performance (GPU or CPU) in Qiskit Aer.
Measures throughput (shots/s) for circuit sizes from START to STOP.
Requires qiskit-aer-gpu for GPU acceleration.
"""
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from time import perf_counter
from math import pi


def qft_circuit(n_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, n_qubits)
    # Build QFT
    for j in range(n_qubits):
        qc.h(j)
        for k in range(j+1, n_qubits):
            qc.cp(pi / (2 ** (k - j)), k, j)
    # Bit reversal
    for i in range(n_qubits // 2):
        qc.swap(i, n_qubits - 1 - i)
    # Measurement
    qc.measure_all()
    return qc

if __name__ == "__main__":
    START, STOP = 3, 28
    SHOTS = 131072

    # Initialize simulator (GPU if qiskit-aer-gpu is present)
    backend = AerSimulator(method='statevector')

    # Header
    print(f"  q |    shots    |   sec  |  shots/s  | backend")
    print("-" * 60)

    for n in range(START, STOP + 1):
        # Prepare circuit
        qc = qft_circuit(n)
        # Transpile for backend
        transpiled_qc = transpile(qc, backend, optimization_level=1)

        # Warm-up to remove JIT/context overhead
        backend.run(transpiled_qc, shots=32).result()

        # Timed run
        t0 = perf_counter()
        backend.run(transpiled_qc, shots=SHOTS).result()
        t1 = perf_counter()

        elapsed = t1 - t0
        throughput = SHOTS / elapsed if elapsed > 0 else 0.0

                # Identify backend device
        try:
            import importlib.util
            gpu_spec = importlib.util.find_spec('qiskit_aer_gpu')
            device = 'GPU' if gpu_spec else 'CPU'
        except ImportError:
            device = 'CPU'

        print(f"{n:3d} | {SHOTS:10d} | {elapsed:7.3f} | {throughput:9.1f} | {device}")
