# Classical-Simulation Benchmarks for the Quantum Fourier Transform&nbsp;(QFT)

Two complementary benchmark suites explore **how fast** & **how accurately** modern
classical simulators can execute the **Quantum Fourier Transform** over a broad
range of qubit counts, shot numbers and noise settings.  
A third folder, **`data_graph/`**, turns the raw benchmark CSVs into tidy tables
and publication-ready figures.

---

## 1 · Suites & Tooling

| Suite / Folder | Tech stack | Core focus | Detailed docs |
|----------------|------------|------------|---------------|
| **CUDA-Q**     | NVIDIA **CUDA-Quantum** (`cudaq`) | High-performance CPU / GPU simulation: state-vector, density-matrix, noise models, **live GPU-utilisation & memory tracking** | [`cudaq/README_cudaq.md`](cudaq/README_cudaq.md) |
| **Qiskit**     | IBM **Qiskit 2** + **Aer** (CPU & GPU builds) | Reference sampling runs, Aer noise models, throughput & accuracy cross-checks | [`qiskit/README_qiskit.md`](qiskit/README_qiskit.md) |
| **data_graph** | Python + Pandas + Seaborn | **ETL + Plotting**: merges raw CSVs, generates composite plots for papers / talks | [`data_graph/README_data_graph.md`](data_graph/README_data_graph.md) |

---

## 2 · Quick Start

```bash
# 1) grab the repo
git clone <repo-url> QFT-Benchmarks
cd QFT-Benchmarks

# 2) choose a suite to start with
cd cudaq          # or: cd qiskit   |   cd data_graph (for plotting only)

# 3) create & activate the exact Conda env
conda env create -f environment.yml          # cudaq/
conda env create -f environment_qiskit.yml   # qiskit/ (CPU)
conda env create -f environment_qiskit_gpu.yml   # qiskit/ (GPU)
conda activate <env-name>

# 4) sanity-check the install
python - <<'PY'
import cudaq, qiskit, qiskit_aer, sys
print("CUDA-Q :", getattr(cudaq, "__version__", "n/a"),
      "| GPUs :", getattr(cudaq, "num_available_gpus", lambda: "n/a")())
print("Qiskit :", qiskit.__version__, "| Aer :", qiskit_aer.__version__)
PY

# 5) run a benchmark
python benchmark_QFT_multipleshots.py        # or any script in the chosen folder
````

> **Tips**
> • **CUDA-Q** automatically falls back to its `qpp-cpu` simulator when no NVIDIA
> GPU is present.
> • The **Qiskit** suite ships separate *CPU* and *GPU* environment files.
> • Use `--help` on any script for all available CLI options.

---

## 3 · Repository Layout

```
.
├── cudaq/               # CUDA-Q benchmarks & env file
│   ├── environment.yml
│   ├── benchmark_QFT_multipleshots.py
│   ├── …                # more CUDA-Q scripts
│   └── README_cudaq.md
├── qiskit/              # Qiskit-Aer benchmarks & env files
│   ├── environment_qiskit*.yml
│   ├── benchmark_qft_multiple_shots_improved.py
│   ├── …                # more Qiskit scripts
│   └── README_qiskit.md
├── data_graph/          # CSV merge + graph generation pipeline
│   ├── merge_csvs.py
│   ├── plot_benchmarking.ipynb
│   └── README_data_graph.md
└── results/             # all generated CSVs / PNGs (git-ignored)
```

---

## 4 · Metrics Captured

* **Runtime scaling** — seconds vs qubits / shots
* **Throughput** — shots · s⁻¹ for sampling runs
* **Accuracy**

  * L₂-distance vs ideal distribution
  * Fidelity & Frobenius norm (density-matrix)
* **Noise sensitivity** — depolarising, amplitude / phase damping, bit-flip
* **GPU stats** — live utilisation % & VRAM MiB (CUDA-Q on `nvidia` backend)

All scripts print human-readable tables **and** save tidy CSVs in
`results/`, ready for further analysis or the automated `data_graph`
plotting pipeline.
