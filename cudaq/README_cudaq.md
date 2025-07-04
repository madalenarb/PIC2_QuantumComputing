# CUDA-Q QFT Simulation & Benchmark Suite 🚀

A curated set of **CUDA-Q** examples that showcase—and stress-test—the *Quantum Fourier Transform* (QFT) on both **CPU** (`qpp-cpu`) and **GPU** (`nvidia`) back-ends. The repo mirrors the Qiskit collection but every script here uses NVIDIA’s CUDA-Q Python API.

---

## 0 · Quick start

```bash
# 1) Clone or unzip this repository …
git clone <repo-url>
cd <repo-folder>

# 2) Create the exact Conda environment used below
conda env create -f environment.yml    # → “cuda-quantum”
conda activate cuda-quantum

# 3) Sanity‐check the install
python - <<'PY'
import cudaq
print("CUDA-Q version :", cudaq.__version__)
print("Detected GPUs   :", cudaq.num_available_gpus())
PY
````

> **Tip:**
> To run CPU-only, remove all `nvidia-*`, `cuda-*`, `cu*` lines from `environment.yml`, create a fresh env, and CUDA-Q will auto-fallback to its single-threaded `qpp-cpu` simulator.

---

## 1 · Environment overview

| Environment      | Purpose                                                        | Creation command                      |
| ---------------- | -------------------------------------------------------------- | ------------------------------------- |
| **cuda-quantum** | CUDA 11.8 + cuQuantum + cuTensor + CUDA-Q + plotting libraries | `conda env create -f environment.yml` |

All other dependencies (NumPy, pandas, Matplotlib, Seaborn, etc.) are pinned in `environment.yml`.

---

## 2 · Script catalogue

|  #  | Script                              | Purpose / highlights                                                            |
| :-: | ----------------------------------- | ------------------------------------------------------------------------------- |
|  1  | `1_MaximallyEntangled.py`           | Minimal Bell-pair kernel – sanity check                                         |
|  2  | `qft3.py`                           | 3–5-qubit GHZ→QFT state-vector demo + fidelity                                  |
|  3  | `qft3_noise_model.py`               | Interactive 3-qubit QFT with selectable noise model                             |
|  4  | `qft4.py`                           | 4-qubit GHZ→QFT + histogram (CSV/PNG)                                           |
|  5  | `qftN.py`                           | Run arbitrary-size QFT for quick experiments                                    |
|  6  | `benchmark_QFT_multipleshots.py`    | Main benchmark—CPU or GPU, sampling or state-vector, single or multi-shot sweep |
|  7  | `benchmark_qftN_cpu_noise_model.py` | CPU density-matrix engine, sweeps 4 noise channels & probabilities              |
|  8  | `benchmark_qftN_memory_track.py`    | GPU benchmark monitoring live utilization & VRAM (requires `pynvml`)            |

---

## 3 · Running the examples

### 3.1  Tiny sanity checks

```bash
python 1_MaximallyEntangled.py        # 2-qubit Bell pair
python qft3.py                        # 3–5-qubit GHZ→QFT demo
python qft3_noise_model.py            # choose a noise model interactively
python qft4.py                        # 4-qubit QFT + histogram CSV
```

---

### 3.2  `benchmark_QFT_multipleshots.py`

Benchmarks **sampling** *or* **state-vector** on CPU or GPU, with arbitrary shot counts.

```bash
# Sampling on GPU, 131072 shots
python benchmark_QFT_multipleshots.py \
  --target nvidia           \
  --init   zero             \
  --method sample           \
  --shots  131072

# State-vector sweep on CPU (up to 21 qubits)
python benchmark_QFT_multipleshots.py \
  --target qpp-cpu         \
  --init   ghz              \
  --method statevector     \
  --multi                   # runs shots = 2^12 … 2^19
```

**Key flags** (same as Qiskit version):

| Flag         | Meaning                                   | Default  |
| ------------ | ----------------------------------------- | -------- |
| `--target`   | `qpp-cpu` or `nvidia`                     | `nvidia` |
| `--init`     | `zero` or `ghz`                           | `zero`   |
| `--method`   | `sample` or `statevector`                 | `sample` |
| `--max-bits` | Max qubit register size (GPU 28 / CPU 21) | auto     |
| `--shots`    | Single shot count (mutually exclusive)    | 131072   |
| `--multi`    | Full shot sweep (2^12…2^19)               | –        |

---

### 3.3  `benchmark_qftN_cpu_noise_model.py`

Performs a **noiseless** CPU state-vector baseline, then CPU density-matrix sweeps over four noise channels:

```bash
python benchmark_qftN_cpu_noise_model.py \
  --init ghz  --shots 16384 \
  --max-bits 10 \
  --probs 0.01 0.05 0.1
```

Outputs → `results/qftN_<init>_<shots>_cpu.csv` with columns `Fidelity`, `Fro_norm`, `L2_pop`.

---

### 3.4  `benchmark_qftN_memory_track.py`

GPU-only benchmark that samples **pynvml** every 0.1 s to capture:

* average core %
* average memory %
* **delta** MiB used (above baseline)

```bash
python benchmark_qftN_memory_track.py \
  --shots    4096 \
  --max-bits 27   \
  --out-dir  results/GPUmonitored
```

Saves both CSV and XLSX for easy analysis.

---

## 4 · Outputs & cleanup

* All benchmarks save CSVs under **`results/`**
* Histograms & diagrams go under `results/qft4/` (created on demand)

To clean all generated data:

```bash
rm -rf results qft4
```

---

## 5 · Extra notes

* **Warm-up** every CUDA-Q kernel with 32 dummy shots—scripts handle this automatically.
* If you hit `RuntimeError: requested size is too big`, reduce `--max-bits` or switch to sampling.
* State-vector simulation requires **CUDA-Q ≥ 0.9.1** (`cudaq.simulate()` availability).
* `environment.yml` pins CUDA 11.8; adjust channels if your driver supports older toolkits.
* All plotting is headless via Matplotlib—no display required.



