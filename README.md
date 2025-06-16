# Classical-Simulation Benchmarks for the Quantum Fourier Transform (QFT)

This repository hosts two complementary benchmark suites that explore how fast—and how accurately—modern classical simulators can execute the **Quantum Fourier Transform** across a wide range of qubit counts, shot numbers and noise settings.

---

## Suites

| Suite       | Stack                             | Focus                                                                                 | Docs                              |
|-------------|-----------------------------------|---------------------------------------------------------------------------------------|-----------------------------------|
| **CUDA-Q**  | NVIDIA CUDA-Quantum (`cudaq`)     | High-performance CPU/GPU: state-vector, density-matrix, noise models, GPU monitoring  | [`cudaq/README_cudaq.md`](cudaq/README_cudaq.md)   |
| **Qiskit**  | IBM Qiskit 2 + Aer (CPU & GPU)    | Reference sampling runs, noise models, throughput, accuracy comparisons               | [`qiskit/README_qiskit.md`](qiskit/README_qiskit.md) |

---

## Quick Start

1. **Clone the repo**  
   ```bash
   git clone <repo-url> QFT-Benchmarks
   cd QFT-Benchmarks
   ```

2. **Choose your suite**

   ```bash
   cd cudaq      # or: cd qiskit
   ```
3. **Create & activate the exact Conda env**

   ```bash
   # CUDA-Q
   conda env create -f environment.yml
   conda activate cuda-quantum

   # Qiskit (CPU)
   conda env create -f environment_qiskit.yml
   conda activate qiskit

   # Qiskit (GPU)
   conda env create -f environment_qiskit_gpu.yml
   conda activate qiskit-gpu
   ```
4. **Sanity-check the install**

   ```bash
   python - <<'PY'
   # CUDA-Q
   import cudaq
   print("CUDA-Q:", cudaq.__version__, "GPUs:", cudaq.num_available_gpus())

   # Qiskit
   import qiskit, qiskit_aer
   print("Qiskit:", qiskit.__version__, "Aer:", qiskit_aer.__version__)
   PY
   ```
5. **Run an example benchmark**

   ```bash
   python benchmark_QFT_multipleshots.py   # or any other script
   ```

> **Tip:**
> • CUDA-Q falls back to its `qpp-cpu` simulator if no NVIDIA GPU is available.
> • The Qiskit suite has separate CPU/GPU env files.

---

## Directory Layout

```
.
├── cudaq/       # CUDA-Q benchmarks & docs
│   └── README_cudaq.md
├── qiskit/      # Qiskit Aer benchmarks & docs
│   └── README_qiskit.md
└── results/     # Generated CSVs & PNGs (git-ignored)
```

---

## What We Measure

* **Runtime scaling** – seconds vs. qubits/shots
* **Throughput** – shots·s⁻¹ for sampling runs
* **Accuracy**

  * L₂-distance between empirical & ideal distributions
  * Fidelity & Frobenius-norm (density-matrix)
* **Noise sensitivity** (Depolarizing, amplitude/phase-damping, bit-flip)
* **GPU utilisation & memory footprint** (live monitoring in CUDA-Q)

*All scripts log tables to the console and save CSVs under `results/` for further analysis.*



  
Happy benchmarking! 🚀
