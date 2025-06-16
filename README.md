# Classical-Simulation Benchmarks for the Quantum Fourier Transform (QFT)

Two complementary benchmark suites explore **how fast** & **how accurately** modern
classical simulators can execute the **Quantum Fourier Transform** over a broad
range of qubit counts, shot numbers and noise settings.  
A third folder, **`data_graph/`**, turns the raw benchmark CSVs into tidy tables
and publication-ready figures.

---

## 1 · Suites & Tooling

| Suite / Folder | Tech stack                                 | Core focus                                                                           | Detailed docs                                             |
|----------------|---------------------------------------------|---------------------------------------------------------------------------------------|-----------------------------------------------------------|
| **CUDA-Q**     | NVIDIA **CUDA-Quantum** (`cudaq`)           | High-performance CPU / GPU: state-vector, density-matrix, noise models, **live GPU-util**| [`cudaq/README_cudaq.md`](cudaq/README_cudaq.md)          |
| **Qiskit**     | IBM **Qiskit 2** + **Aer** (CPU & GPU)      | Reference sampling runs, Aer noise models, throughput & accuracy cross-checks         | [`qiskit/README_qiskit.md`](qiskit/README_qiskit.md)      |
| **data_graph** | Python + Pandas + Seaborn                   | **ETL + Plotting**: merges raw CSVs, generates composite plots for papers / talks      | [`data_graph/README_data_graph.md`](data_graph/README_data_graph.md) |

---

## 2 · Quick Start

```bash
# 1) grab the repo
git clone <repo-url> QFT-Benchmarks
cd QFT-Benchmarks

# 2) choose a suite
cd cudaq        # or: cd qiskit   |   cd data_graph

# 3) create & activate the Conda env
conda env create -f environment.yml                  # cudaq/
conda env create -f environment_qiskit.yml           # qiskit/ (CPU)
conda env create -f environment_qiskit_gpu.yml       # qiskit/ (GPU)
conda activate <env-name>

# 4) sanity-check
python - <<'PY'
import cudaq, qiskit, qiskit_aer, sys
print("CUDA-Q :", getattr(cudaq,"__version__","n/a"),
      "| GPUs :", getattr(cudaq,"num_available_gpus",lambda:0)())
print("Qiskit :", qiskit.__version__, "| Aer :", qiskit_aer.__version__)
PY

# 5) run a benchmark
python benchmark_QFT_multipleshots.py
````

> **Tip:**
> • CUDA-Q falls back to `qpp-cpu` if no GPU is present.
> • Qiskit has separate CPU/GPU env files.
> • Use `--help` on any script for full CLI options.

---

## 3 · Repository Layout

```
.
├── cudaq/
│   ├── environment.yml
│   ├── benchmark_QFT_multipleshots.py
│   ├── …  
│   └── README_cudaq.md
├── qiskit/
│   ├── environment_qiskit*.yml
│   ├── benchmark_qft_multiple_shots_improved.py
│   ├── …  
│   └── README_qiskit.md
├── data_graph/
│   ├── merge_csvs.py
│   ├── plot_benchmarking.ipynb
│   └── README_data_graph.md
├── organize_qft_outputs.py        ← helper to sort & rename CSVs
└── results/                       # CSVs & PNGs (git-ignored)
```

---

## 4 · Metrics Captured

* **Runtime scaling** — seconds vs. qubits/shots
* **Throughput** — shots·s⁻¹
* **Accuracy**

  * L₂-distance vs. ideal
  * Fidelity & Frobenius-norm
* **Noise sensitivity** — depolarizing, damping, bit-flip
* **GPU stats** — live % utilisation & VRAM MiB

All scripts print tables **and** save CSVs under `results/`, ready for the `data_graph/` pipeline.

---

## 5 · Organize & Rename Outputs

Once you’ve run your benchmarks, you can sort and consistently rename all CSV files
for both **CUDA-Q** and **Qiskit** with the provided helper:

```bash
python organize_qft_outputs.py
```

This will scan your `cudaq/results/` and `qiskit/Results/` folders, copy each CSV
into clean `noise/` and `non_noise/` subdirectories, and rename them to:

* `cudaq_qft_multishot_no_noise_{device}_{init}.csv`
* `cudaq_qft_noise_{init}_{device}_{shots}.csv`
* `qiskit_qft_multishot_no_noise_{device}_{init}.csv`
* `qiskit_qft_noise_{init}_{device}_{shots}.csv`

… with verbose console logs for every file processed.

---

Happy benchmarking! 🚀

```

