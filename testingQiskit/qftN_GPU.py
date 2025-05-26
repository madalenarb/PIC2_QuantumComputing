from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from time import perf_counter

# 1) Probe device support
probe = AerSimulator()
print("Supported devices:", probe.available_devices())

# 2) Build GPU-backed sim (this will fallback to CPU if unsupported)
sim = AerSimulator(method='statevector', device='GPU')
# 3) Prepare a small circuit
qc  = QuantumCircuit(5)
qc.h(range(5))
qc.measure_all()
tqc = transpile(qc, sim)

# 4) Run  
t0     = perf_counter()
result = sim.run(tqc, shots=1024).result()
elapsed = perf_counter() - t0

# 5) Inspect metadata
meta = result.results[0].metadata
print("Elapsed:", elapsed)
print("Metadata:", meta)
print("GPU shots parallelized:", meta.get("gpu_parallel_shots"))
