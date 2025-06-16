import cudaq

@cudaq.kernel
def bell_state(qubit_count: int):
    # Allocate qubits
    qvector = cudaq.qvector(qubit_count)
    
    # Place the first qubit in superposition
    h(qvector[0])
    
    # Apply controlled-X (CNOT) gates to entangle the qubits
    for qubit in range(qubit_count - 1):
        x.ctrl(qvector[qubit], qvector[qubit + 1])
    
    # Measure all qubits
    mz(qvector)

# Call the kernel and sample results
qubit_count = 2
# ── Warm-up run (32 shots) ──
_ = cudaq.sample(bell_state, qubit_count, shots_count=32)
print(cudaq.draw(bell_state, qubit_count))
results = cudaq.sample(bell_state, qubit_count)
# Should see a roughly 50/50 distribution between the |00> and
# |11> states. Example: {00: 505  11: 495}
print("Measurement distribution:" + str(results))

# Print the measurement results
print(results)


# Research papers in CudaQ simulators
# quantum FFT- explore these papers with this
