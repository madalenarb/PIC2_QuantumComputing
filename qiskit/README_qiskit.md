# QFT Simulation & Benchmark Suite (Qiskit Edition)

Scripts for studying the Quantum Fourier Transform with **Qiskit Aer 2.x** on both CPU-only and CUDA-accelerated back-ends.

---

## 1. Quick start

```bash
# clone / unzip repository first …

# ── create & activate the exact envs used by these scripts ──
conda env create -f environment_qiskit.yml          # CPU-only
conda env create -f environment_qiskit_gpu.yml      # GPU-capable

conda activate qiskit                # or:  conda activate qiskit-gpu
```

<details>
<summary><strong>Verify installation</strong></summary>

```bash
python - <<'PY'
import qiskit, qiskit_aer as aer, json, subprocess, re
print("Qiskit version :", qiskit.__qiskit_version__['qiskit'])
print("Aer version    :", aer.__version__)
try:
    gpus = subprocess.check_output(["nvidia-smi","-L"], text=True)
    n = len(re.findall(r"GPU\s+\d+", gpus))
except Exception:
    n = 0
print("CUDA-capable GPUs detected :", n)
PY
```

If **0 GPUs** are detected you can still use all scripts from the `qiskit` CPU environment.

</details>

> **Need a leaner CPU-only setup?**
> Remove each `nvidia-*`, `cu*`, `cutensor*`, … line from `environment_qiskit_gpu.yml`, then create a fresh environment – Aer will silently fall back to its C++ CPU kernels.

---

## 2. Project layout

| Folder / file                              | Role                                                 |
| ------------------------------------------ | ---------------------------------------------------- |
| `environment_qiskit.yml`                   | Conda spec for the *CPU* environment                 |
| `environment_qiskit_gpu.yml`               | Conda spec for the *GPU* environment                 |
| **Demo / test scripts**                    |                                                      |
| `qft3Q.py`                                 | 3-qubit QFT walk-through, prints circuit & L₂ error  |
| `qft4.py`                                  | 4-qubit QFT (zero / GHZ), saves diagrams + histogram |
| `qft4_noise.py`                            | Same 4-qubit circuit but under strong noise models   |
| `qft4_print_diagram.py`                    | Renders the fully unrolled 4-qubit QFT diagram       |
| **Benchmarks**                             |                                                      |
| `benchmark_qft_multiple_shots_improved.py` | *No-noise* benchmark across (qubits × shot-counts)   |
| `benchmark_qft_shots_noisy_circuit.py`     | *Noise* benchmark (Depol, AmpDamp, Phase, BitFlip)   |

All outputs (PNGs & CSVs) are written into **`Results/`** or `qft4-results_*` sub-folders.

---

## 3. Running the examples

```bash
# 3-qubit tutorial
python qft3Q.py

# 4-qubit, GHZ initial state, histogram + diagrams
python qft4.py

# 4-qubit under several strong noise models
python qft4_noise.py

# Just the pretty circuit diagram (PNG)
python qft4_print_diagram.py
```

Diagrams are saved in `qft4-results_*/*.png`.

---

## 4. Benchmarks

### 4-a. `benchmark_qft_multiple_shots_improved.py`  – no noise

Measures runtime, throughput and L₂ error while sweeping **shot counts** and **qubit sizes**.

```bash
python benchmark_qft_multiple_shots_improved.py \
  --init zero \
  --method statevector \
  --device CPU \
  --min-qubits 3  --max-qubits 27 \
  --shots 4096 8192 16384 32768 65536 131072 \
  --output Results/non_noise.csv
```

<details>
<summary><strong>Main flags</strong></summary>

| Flag              | Meaning                           | Default        |
| ----------------- | --------------------------------- | -------------- |
| `-i/--init`       | `zero` or `ghz`                   | `zero`         |
| `-m/--method`     | `statevector` or `density_matrix` | `statevector`  |
| `-d/--device`     | `CPU` or `GPU`                    | `CPU`          |
| `-n/--min-qubits` | lower bound                       | 3              |
| `-N/--max-qubits` | upper bound                       | 27             |
| `-s/--shots`      | space-separated list              | 2¹⁰ … 2¹⁹      |
| `-o/--output`     | CSV path                          | auto-generated |

</details>

### 4-b. `benchmark_qft_shots_noisy_circuit.py` – noise sweep

Computes **sampling L₂**, **density-matrix fidelity** and **Frobenius norm** for several noise models.

```bash
python benchmark_qft_shots_noisy_circuit.py \
  --init ghz \
  --shots 16384 \
  --max-qubits 12 \
  --device GPU \
  --noise Depolarizing AmplitudeDamping \
  --probs 0.01 0.1 0.5 0.9
```

<details>
<summary><strong>Main flags</strong></summary>

| Flag              | Meaning              | Default            |
| ----------------- | -------------------- | ------------------ |
| `-i/--init`       | `zero` / `ghz`       | `zero`             |
| `-s/--shots`      | single integer       | `16384`            |
| `-q/--max-qubits` | upper bound          | 12                 |
| `-d/--device`     | `CPU`, `GPU`, `auto` | `CPU`              |
| `-n/--noise`      | noise types list     | all five           |
| `-p/--probs`      | error probabilities  | 0.01 0.1 0.5 0.9 1 |

</details>

---

## 5. Cleaning up

```bash
rm -rf Results qft4-results* qft4-*
```

---

## 6. Tips & notes

* **GPU runs:** make sure you activated `qiskit-gpu` *and* the machine exposes a compatible NVIDIA GPU (`GPU` should show at least one device).
* **CPU fallback:** every script also works inside the GPU environment even on machines without CUDA – Aer automatically selects its C++ reference kernel.
* All plotting uses a headless Matplotlib backend – the scripts can run on remote servers without an X-server.
* Add `--help` to any script for the exhaustive argument list.

