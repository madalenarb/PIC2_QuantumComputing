from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from time import time
from math import pi
import importlib.util

def qft_circuit(n_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits)
    for j in range(n_qubits):
        qc.h(j)
        for k in range(j+1, n_qubits):
            qc.cp(pi / (2 ** (k - j)), k, j)
    qc.reverse_bits()
    return qc

def is_aer_gpu_installed() -> bool:
    spec = importlib.util.find_spec("qiskit_aer")
    if spec and spec.submodule_search_locations:
        return any("gpu" in loc.lower() for loc in spec.submodule_search_locations)
    return False

def detect_gpu_backend(backend) -> str:
    config = backend.configuration()
    if hasattr(config, "device") and getattr(config, "device", "").upper() == "GPU":
        return "GPU"
    return "CPU"

if __name__ == "__main__":
    SHOTS = 2**18
    print(f"{'  q':>3} | {'shots':>5} | {'sec':>6} | {'shots/s':>10} | backend")
    print("-" * 50)

    for n in range(3, 29):
        qc = qft_circuit(n)
        backend = AerSimulator(method='statevector')
        backend_type = detect_gpu_backend(backend)
        if is_aer_gpu_installed():
            backend_type = "GPU"

        transpiled_qc = transpile(qc, backend)

        start = time()
        result = backend.run(transpiled_qc, shots=SHOTS).result()
        end = time()

        elapsed = end - start
        throughput = SHOTS / elapsed if elapsed > 0 else 0

        print(f"{n:>3} | {SHOTS:>5} | {elapsed:6.3f} | {throughput:10.1f} | backend={backend_type}")
