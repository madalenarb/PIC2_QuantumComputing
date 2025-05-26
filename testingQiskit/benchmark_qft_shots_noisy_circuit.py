#!/usr/bin/env python3
"""
benchmark_qft_shots_noisy_circuit.py — QFT Benchmarking Under Noise (Qiskit Aer)
===================================================================

Simulates QFT circuits under various noise models using Qiskit Aer,
computing:

  • L2 error from sampled measurement distribution (optional)
  • Fidelity and Frobenius norm from the density matrix

Saves all results to a CSV.

──────────────────────────────────────────────────────────────
Usage:

  python benchmark_qft_shots_noisy_circuit.py [options]

To see available options:
  python benchmark_qft_shots_noisy_circuit.py --help

Examples:
  python benchmark_qft_shots_noisy_circuit.py
  python benchmark_qft_shots_noisy_circuit.py --shots 1024 --max_qubits 10
  python benchmark_qft_shots_noisy_circuit.py --noise Depolarizing AmplitudeDamping --probs 0.01 0.1
  python benchmark_qft_shots_noisy_circuit.py --device GPU
"""

import os, math, time, argparse
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
    """Return an n-qubit QFT circuit."""
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
        qc.save_density_matrix(label='rho')
    return qc


def pop_l2_error(counts: dict, n: int, shots: int):
    """Compute L2 norm of (sampled – uniform)."""
    N = 2**n
    ideal = np.full(N, 1/N)
    sample = np.zeros(N)
    for b, c in counts.items():
        idx = int(b[::-1], 2)
        sample[idx] = c / shots
    return np.linalg.norm(sample - ideal, 2)


def make_noise(name: str, p: float) -> NoiseModel:
    if name == 'none':
        return None
    nm = NoiseModel()
    rm = pauli_error([('X', p), ('I', 1-p)])
    nm.add_all_qubit_quantum_error(rm, ['reset', 'measure'])

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
        raise ValueError(f"Unknown noise type: {name}")

    nm.add_all_qubit_quantum_error(e1, ['u1', 'u2', 'u3'])
    nm.add_all_qubit_quantum_error(e2, ['cx'])
    return nm


def make_backend(method: str, noise_model: NoiseModel, device_choice: str):
    """Return AerSimulator with the requested method and device (auto/GPU/CPU)."""
    if device_choice.upper() == 'CPU':
        sim = AerSimulator(method=method, noise_model=noise_model)
        print(f"✅ Using CPU AerSimulator ({method})")
    elif device_choice.upper() == 'GPU':
        sim = AerSimulator(method=method, noise_model=noise_model, device='GPU')
        print(f"✅ Using GPU AerSimulator ({method})")
    else:  # auto
        try:
            sim = AerSimulator(method=method, noise_model=noise_model)
            sim.available_devices()  # trigger check
            print(f"✅ Using GPU AerSimulator ({method}) [auto]")
        except Exception:
            sim = AerSimulator(method=method, noise_model=noise_model)
            print(f"⚠️ GPU not available — falling back to CPU AerSimulator ({method}) [auto]")
    print(f"   Available devices: {sim.available_devices()}")
    return sim


def main():
    parser = argparse.ArgumentParser(description="Benchmark QFT under simple noise models.")
    parser.add_argument('--shots', type=int, default=1024, help='Number of shots for sampling')
    parser.add_argument('--max_qubits', type=int, default=12, help='Max number of qubits')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'GPU', 'CPU'],
                        help='Device selection: auto, GPU, or CPU (default: auto)')
    parser.add_argument('--noise', nargs='*',
                        default=['none', 'Depolarizing', 'AmplitudeDamping', 'PhaseDamping', 'BitFlip'],
                        help='Which noise types to run (default: all)')
    parser.add_argument('--probs', nargs='*', type=float, default=[0.01, 0.1, 0.5, 0.9],
                        help='Which probabilities to simulate')

    args = parser.parse_args()
    shots = args.shots
    max_qubits = args.max_qubits
    device = args.device
    noise_types = args.noise
    probs = args.probs

    print(f"\n▶ Starting QFT noise benchmark:")
    print(f"   → Shots       : {shots}")
    print(f"   → Max Qubits  : {max_qubits}")
    print(f"   → Device      : {device.upper()}")
    print(f"   → Noise types : {noise_types}")
    print(f"   → Probabilities: {probs}\n")

    IDEAL_PSI = {}
    IDEAL_RHO = {}
    for n in range(3, max_qubits+1):
        N = 2**n
        psi = np.full(N, 1/math.sqrt(N), dtype=complex)
        IDEAL_PSI[n] = psi
        IDEAL_RHO[n] = np.outer(psi, psi.conj())

    records = []
    for prob in probs:
        for noise in noise_types:
            p = 0.0 if noise == 'none' else prob
            nm = make_noise(noise, p)
            sampler = make_backend('automatic', noise_model=nm, device_choice=device)
            densim  = make_backend('density_matrix', noise_model=nm, device_choice=device)
            label = f"{noise}@{p}"

            for n in range(3, max_qubits + 1):
                # Measurement-based L2
                qc_m = build_qft(n, measure=True)
                qt_m = transpile(qc_m, sampler)
                sampler.run(qt_m, shots=32).result()  # warm-up
                t0 = time.perf_counter()
                res = sampler.run(qt_m, shots=shots).result()
                t_s = time.perf_counter() - t0
                l2 = pop_l2_error(res.get_counts(), n, shots)

                # Density matrix fidelity / frobenius
                qc_d = build_qft(n, measure=False)
                qt_d = transpile(qc_d, densim)
                t0 = time.perf_counter()
                r2 = densim.run(qt_d).result()
                t_d = time.perf_counter() - t0
                dm = np.array(r2.data(0)['rho'])
                frob = np.linalg.norm(dm - IDEAL_RHO[n])
                fid = float(np.real(IDEAL_PSI[n].conj() @ dm @ IDEAL_PSI[n]))

                print(f"{label:18s} n={n:2d} | L2={l2:.3e} | F={fid:.4f} | Fro={frob:.3e} | "
                      f"time_sample={t_s:.3f}s | time_dm={t_d:.3f}s")

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
    out_path = f"Results/qiskit_{device}_noise_{shots}.csv"
    df.to_csv(out_path, index=False)
    print(f"\n✅ Saved benchmark results to {out_path}")


if __name__ == "__main__":
    main()
