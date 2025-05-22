#!/usr/bin/env python3
"""
Benchmark QFT under simple noise models in Qiskit Aer:
  - population L2 error from sampling
  - Frobenius norm & fidelity from noisy density matrix
Saves results into a CSV.
"""
import math, time, os, sys
import numpy as np
import pandas as pd
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel,
    depolarizing_error,
    amplitude_damping_error,
    phase_damping_error,
    pauli_error
)

def build_qft(n: int, measure: bool):
    """Return an n-qubit QFT circuit.  If measure=False, inserts a save_density_matrix."""
    qc = QuantumCircuit(n, n if measure else 0)
    for k in range(n):
        qc.h(k)
        for j in range(1, n - k):
            qc.cp(math.pi / 2**j, k, k + j)
    for i in range(n // 2):
        qc.swap(i, n - 1 - i)
    if measure:
        qc.measure(range(n), range(n))
    else:
        # snapshot the full density matrix after the QFT
        qc.save_density_matrix(label='rho')
    return qc

def pop_l2_error(counts: dict, n: int, shots: int):
    """Compute L2 norm of (sampled_uniform – 1/2^n)."""
    N = 2**n
    ideal = np.full(N, 1/N)
    sample = np.zeros(N)
    for b, c in counts.items():
        idx = int(b[::-1], 2)
        sample[idx] = c / shots
    return np.linalg.norm(sample - ideal, 2)

def make_noise(name: str, p: float) -> NoiseModel:
    """Build a simple per-instruction NoiseModel."""
    if name == 'none':
        return None
    nm = NoiseModel()
    # reset & measure errors: bit-flip with prob p
    rm = pauli_error([('X', p), ('I', 1-p)])
    nm.add_all_qubit_quantum_error(rm, ['reset', 'measure'])
    # gate errors
    if name == 'Depolarizing':
        e1 = depolarizing_error(p, 1)
        e2 = depolarizing_error(p, 2)
    elif name == 'AmplitudeDamping':
        e1 = amplitude_damping_error(p)
        e2 = e1.tensor(e1)
    elif name == 'PhaseDamping':
        e1 = phase_damping_error(p)
        e2 = e1.tensor(e1)
    elif name == 'BitFlip':
        e1 = pauli_error([('X', p), ('I', 1-p)])
        e2 = e1.tensor(e1)
    else:
        raise ValueError(f"Unknown noise {name}")
    nm.add_all_qubit_quantum_error(e1, ['u1','u2','u3'])
    nm.add_all_qubit_quantum_error(e2, ['cx'])
    return nm

if __name__ == "__main__":
    max_qubits = 12
    shots      = 262144
    cpu_cores  = os.cpu_count() or 1

    # precompute ideal density matrices & statevectors
    IDEAL_RHO = {}
    IDEAL_PSI = {}
    for n in range(3, max_qubits+1):
        N = 2**n
        psi = np.full(N, 1/math.sqrt(N), dtype=complex)
        IDEAL_PSI[n] = psi
        IDEAL_RHO[n] = np.outer(psi, psi.conj())

    noise_types = ['none','Depolarizing','AmplitudeDamping','PhaseDamping','BitFlip']
    probs       = [0.01, 0.1, 0.5, 0.9]

    records = []
    none_flag = True
    # ─── Run remaining noise models with all p values ─────────────
    for prob in probs:
        for noise in noise_types:
            p = prob
            if noise == 'none' and not none_flag:
                continue
            none_flag = False
            if noise == 'none':
                p = 0.0
            nm = make_noise(noise, p)
            sampler = AerSimulator(noise_model=nm, max_parallel_threads=cpu_cores)
            densim  = AerSimulator(method='density_matrix', noise_model=nm, max_parallel_threads=cpu_cores)
            label = f"{noise}@{p}"

            for n in range(3, max_qubits+1):
                qc_m = build_qft(n, measure=True)
                qt_m = transpile(qc_m, sampler)
                sampler.run(qt_m, shots=32).result()
                t0 = time.perf_counter()
                res = sampler.run(qt_m, shots=shots).result()
                t_s = time.perf_counter() - t0
                l2 = pop_l2_error(res.get_counts(), n, shots)

                qc_d = build_qft(n, measure=False)
                qt_d = transpile(qc_d, densim)
                t0 = time.perf_counter()
                r2 = densim.run(qt_d).result()
                t_d = time.perf_counter() - t0
                dm = np.array(r2.data(0)['rho'])
                frob = np.linalg.norm(dm - IDEAL_RHO[n])
                fid  = float(np.real(IDEAL_PSI[n].conj() @ dm @ IDEAL_PSI[n]))

                print(f"{label:16s} n={n:2d}  pop→(t={t_s:.3f}s,L2={l2:.3e})  "
                    f"dm→(t={t_d:.3f}s,Fro={frob:.3e},F={fid:.4f})")

                records.append({
                    'n_bits':       n,
                    'shots':        shots,
                    'noise':        noise,
                    'probability':  p,
                    'time_sampling':t_s,
                    'l2_pop':       l2,
                    'time_density': t_d,
                    'fro_norm':     frob,
                    'fidelity':     fid
                })


    df = pd.DataFrame(records)
    os.makedirs("Results", exist_ok=True)
    csv_path = f"Results/qft_qiskit_noise_metrics_{shots}shots.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved all metrics to {csv_path}")
