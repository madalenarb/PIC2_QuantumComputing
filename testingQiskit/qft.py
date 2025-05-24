from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from time import perf_counter
from math import pi
import importlib.util

def qft_circuit(n_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits)
    for j in range(n_qubits):
        qc.h(j)
        for k in range(j+1, n_qubits):
            qc.cp(pi / (2 ** (k - j)), k, j)
    qc.reverse_bits()
    qc.measure_all()
    return qc

def is_aer_gpu_installed() -> bool:
    spec = importlib.util.find_spec("qiskit_aer")
    if spec and spec.submodule_search_locations:
        return any("gpu" in loc.lower() for loc in spec.submodule_search_locations)
    return False

def detect_gpu_backend(backend) -> str:
    config = backend.configuration()
    return getattr(config, "device", "CPU").upper()

if __name__ == "__main__":
    START, STOP = 3, 28
    SHOTS = 131072

    try:
        backend = AerSimulator(method="statevector", device="GPU")
        DEVICE = detect_gpu_backend(backend)
    except Exception:
        backend = AerSimulator(method="statevector")
        DEVICE = detect_gpu_backend(backend)

    if is_aer_gpu_installed():
        DEVICE = "GPU"

    # Print header
    print(f"  q | shots   |   sec  |  shots/s  | backend={DEVICE}")
    print("-" * 50)

    for n in range(START, STOP + 1):
        qc = qft_circuit(n)
        transpiled_qc = transpile(qc, backend, optimization_level=1)

        # Warm-up run to avoid JIT overhead in timing
        backend.run(transpiled_qc, shots=32).result()

        t0 = perf_counter()
        backend.run(transpiled_qc, shots=SHOTS).result()
        t1 = perf_counter()

        elapsed = t1 - t0
        throughput = SHOTS / elapsed if elapsed > 0 else 0
        print(f"{n:3d} | {SHOTS:7d} | {elapsed:6.3f} | {throughput:9.1f}")
