#!/usr/bin/env python3
"""
Organise QFT CSV outputs in both cudaq/ and qiskit/ folders.

Scans for CSVs in:
  ~/Documents/PIC/PIC2_QuantumComputing/cudaq
  ~/Documents/PIC/PIC2_QuantumComputing/qiskit

and sorts them into subfolders:
  non_noise/    noise/

while renaming to a consistent schema.

Usage:
  python organize_qft_outputs.py
"""

import re, shutil
from pathlib import Path

# adjust these to your local paths:
ROOT = Path.home() / "Documents/PIC/PIC2_QuantumComputing"
CUDAQ_DIR  = ROOT / "cudaq" / "results"
QISKIT_DIR = ROOT / "qiskit" / "Results"

def organise_cudaq():
    device_map = {'nvidia': 'gpu', 'qpp-cpu': 'cpu'}
    base = CUDAQ_DIR
    nn = base / "non_noise"
    no = base / "noise"
    nn.mkdir(exist_ok=True)
    no.mkdir(exist_ok=True)

    for f in base.glob("*.csv"):
        nm, tgt = None, None
        m = re.match(r"qftN_([^_]+)_([^_]+)_multiple_shots\.csv", f.name)
        if m:
            raw_dev, freq = m.groups()
            d = device_map.get(raw_dev, raw_dev)
            nm = f"cudaq_qft_multishot_no_noise_{d}_{freq}.csv"
            tgt = nn
        else:
            m = re.match(r"(?:cudaq_qft_)?([^_]+)_([0-9]+)(?:_([^\.]+))?\.csv", f.name)
            if m:
                freq, shots, raw_dev = m.group(1), m.group(2), m.group(3)
                dev = device_map.get(raw_dev, raw_dev or "cpu")
                nm = f"cudaq_qft_noise_{freq}_{dev}_{shots}.csv"
                tgt = no

        if nm:
            dest = tgt / nm
            if dest.exists():
                print(f"[cudaq] skip exists: {dest.relative_to(base)}")
            else:
                shutil.copy2(f, dest)
                print(f"[cudaq]  copied → {dest.relative_to(base)}")
        else:
            print(f"[cudaq]  skip unmatched: {f.name}")

def organise_qiskit():
    device_map = {'cpu': 'cpu', 'gpu': 'gpu'}
    base = QISKIT_DIR
    nn = base / "non_noise"
    no = base / "noise"
    nn.mkdir(exist_ok=True)
    no.mkdir(exist_ok=True)

    for f in base.glob("*.csv"):
        name = f.name.lower()
        nm, tgt = None, None
        m = re.match(r"qiskit_(cpu|gpu)_noise_([0-9]+)_(zero|ghz)\.csv", name)
        if m:
            raw, shots, freq = m.groups()
            nm = f"qiskit_qft_noise_{freq}_{device_map[raw]}_{shots}.csv"
            tgt = no
        else:
            m = re.match(r"(?:multishot_qft|qiskit)_(cpu|gpu)_(zero|ghz)_(?:statevector|multiple_shots)\.csv", name)
            if m:
                raw, freq = m.groups()
                nm = f"qiskit_qft_multishot_no_noise_{device_map[raw]}_{freq}.csv"
                tgt = nn

        if nm:
            dest = tgt / nm
            if dest.exists():
                print(f"[qiskit] skip exists: {dest.relative_to(base)}")
            else:
                shutil.copy2(f, dest)
                print(f"[qiskit]  copied → {dest.relative_to(base)}")
        else:
            print(f"[qiskit]  skip unmatched: {f.name}")

if __name__ == "__main__":
    print("Organising CUDA-Q outputs…")
    organise_cudaq()
    print("\nOrganising Qiskit outputs…")
    organise_qiskit()
    print("\nDone.")
