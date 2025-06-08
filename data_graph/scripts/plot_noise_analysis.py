#!/usr/bin/env python3
"""
plot_noise_analysis.py — QFT Benchmark Analysis (flexible per-simulator plots)
====================================================================
Generate A–D per-simulator figures on baseline (p=0) and noise data.

Usage examples:
  python plot_noise_analysis.py --shots_vs --shots_vs_sim --qubits_vs --init_compare --noise_4panel

Options:
  --shots_vs          Baseline: L2‐error & time vs shots per simulator/init
  --shots_vs_sim      Baseline: compare simulators L2‐error & time vs shots (fixed init)
  --qubits_vs         Baseline: L2‐error & time vs qubits per simulator (both inits)
  --init_compare      Baseline: compare CPU vs GPU for each vendor at fixed shots
  --noise_4panel      Noise: 2×2 grid L2, Fro, Fidelity, Time vs probability per simulator/init/shot
  --shots SHOT [SHOT ...]    Shots list (default: all)
  --init INIT [INIT ...]     Init styles zero,ghz (default: all)
  --targets TGT [TGT ...]    Simulator backends (default: all)
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# Load data
# ──────────────────────────────────────────────────────────────────────────────
BASE       = Path(__file__).parent
NOISE_CSV  = BASE / "data_csv/treated_data/qft_merged_noise.csv"
DF         = pd.read_csv(NOISE_CSV)
GRAPH_DIR  = BASE / "graphs/noise"
GRAPH_DIR.mkdir(exist_ok=True, parents=True)

# ──────────────────────────────────────────────────────────────────────────────
# Precompute
# ──────────────────────────────────────────────────────────────────────────────
INIT_LIST   = sorted(DF["init"].unique())
ALL_TARGETS = sorted(DF["target"].unique())
SHOT_LIST   = sorted(DF["shots"].unique())
VENDORS     = sorted({t.split('-')[0] for t in ALL_TARGETS})

# palettes
palette   = plt.rcParams['axes.prop_cycle'].by_key()['color']
MARKERS   = ['o','s','^','D','*','X','P','v','<','>']
COL_SHOT  = {s: palette[i%len(palette)] for i,s in enumerate(SHOT_LIST)}
COL_INIT  = {init: palette[i%len(palette)] for i,init in enumerate(INIT_LIST)}
COL_TGT   = {t: palette[i%len(palette)] for i,t in enumerate(ALL_TARGETS)}
NOISE_TYPES = sorted(DF['noise'].unique())
COL_NOISE = {n: palette[i%len(palette)] for i,n in enumerate(NOISE_TYPES)}

# split data
BASELINE_DF = DF[DF.probability == 0]
NOISY_DF    = DF[DF.probability > 0]

# helper

def normalize_noise_name(s: str) -> str:
    return re.sub(r'(?<=[a-z])([A-Z])', r' \1', s).strip()

# ──────────────────────────────────────────────────────────────────────────────
# A. Baseline: Shots vs L2 & Time per simulator/init
# ──────────────────────────────────────────────────────────────────────────────
def plot_shots_sweep(sim: str, init: str, df: pd.DataFrame):
    sub = df[(df.target==sim)&(df.init==init)]
    if sub.empty: return
    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(12,4),sharex=True)
    fig.suptitle(f"{sim}, init={init}: Shots Sweep",fontsize=14)
    for shot in sorted(sub.shots.unique()):
        d = sub[sub.shots==shot]
        ax1.plot(d.shots, d.l2_error, label=f"{shot:,}",
                 color=COL_SHOT[shot], marker=('o' if init=='zero' else 's'))
        ax2.plot(d.shots, d.time_sampling, label=f"{shot:,}",
                 color=COL_SHOT[shot], marker=('o' if init=='zero' else 's'))
    for ax,title,y in [(ax1,'L₂ Error vs Shots','l2_error'),(ax2,'Time vs Shots','time_sampling')]:
        ax.set_xscale('log'); ax.set(title=title, xlabel='Shots', ylabel=y); ax.grid();
    ax1.legend(title='Shots',loc='upper left',bbox_to_anchor=(1.05,1))
    fig.tight_layout(rect=[0,0,0.85,1])
    fig.savefig(GRAPH_DIR/f"{sim}_{init}_shots_sweep.png",dpi=300)
    plt.close(fig)

# ──────────────────────────────────────────────────────────────────────────────
# B. Baseline: Qubits vs L2 & Time per simulator
# ──────────────────────────────────────────────────────────────────────────────
def plot_qubits_sweep(sim: str, df: pd.DataFrame):
    sub = df[df.target==sim]
    if sub.empty: return
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,4),sharex=True)
    fig.suptitle(f"{sim}: Qubit Sweep",fontsize=14)
    for init in INIT_LIST:
        d = sub[sub.init==init]
        ax1.plot(d.n_bits, d.l2_error, label=init,
                 color=COL_INIT[init], marker=('o' if init=='zero' else 's'))
        ax2.plot(d.n_bits, d.time_sampling, label=init,
                 color=COL_INIT[init], marker=('o' if init=='zero' else 's'))
    for ax,title,y in [(ax1,'L₂ Error vs Qubits','l2_error'),(ax2,'Time vs Qubits','time_sampling')]:
        ax.set(xlabel='Qubits',ylabel=y); ax.grid()
    ax2.legend(title='Init',loc='upper left',bbox_to_anchor=(1.05,1))
    fig.tight_layout(rect=[0,0,0.85,1])
    fig.savefig(GRAPH_DIR/f"{sim}_qubits_sweep.png",dpi=300)
    plt.close(fig)

# ──────────────────────────────────────────────────────────────────────────────
# C. Baseline: Init & Platform Comparison per vendor
# ──────────────────────────────────────────────────────────────────────────────
def plot_init_platform_cmp(vendor: str, shot: int):
    platforms = [f"{vendor}-cpu", f"{vendor}-gpu"]
    sub = BASELINE_DF[(BASELINE_DF.shots==shot)&(BASELINE_DF.target.isin(platforms))]
    if sub.empty: return
    fig, ax1 = plt.subplots(figsize=(6,4))
    width=0.35; x=np.arange(len(INIT_LIST))
    for i,plat in enumerate(platforms):
        d= sub[sub.target==plat]
        vals_err = [d[d.init==init].l2_error.mean() for init in INIT_LIST]
        ax1.bar(x+i*width, vals_err, width, label=plat)
    ax1.set_xticks(x+width/2, INIT_LIST); ax1.set_ylabel('L₂ error')
    ax1.legend(title='Platform')
    ax2=ax1.twinx()
    for i,plat in enumerate(platforms):
        d=sub[sub.target==plat]
        vals_time=[d[d.init==init].time_sampling.mean() for init in INIT_LIST]
        ax2.bar(x+i*width, vals_time, width, alpha=0.3)
    ax2.set_ylabel('Time (s)')
    ax1.set_title(f"{vendor}: Init vs Platform (shots={shot})")
    fig.tight_layout(); fig.savefig(GRAPH_DIR/f"{vendor}_init_platform_cmp_{shot}.png",dpi=300); plt.close(fig)

# ──────────────────────────────────────────────────────────────────────────────
# D. Noise: 4-panel metrics vs noise probability per simulator/init/shot
# ──────────────────────────────────────────────────────────────────────────────
def plot_noise_4panel(sim: str, init: str, shot: int, df: pd.DataFrame):
    sub = df[(df.target==sim)&(df.init==init)&(df.shots==shot)]
    if sub.empty: return
    fig, axs = plt.subplots(2,2,figsize=(12,8),sharex=True)
    fig.suptitle(f"{sim}, init={init}, shots={shot}: Noise Analysis",fontsize=14)
    panels=[('l2_error','L₂ error'),('fro_norm','Fro norm'),('fidelity','Fidelity'),('time_sampling','Time')]
    for (col,label),ax in zip(panels, axs.flatten()):
        ax.set_xscale('log')
        for i,noise in enumerate(NOISE_TYPES):
            d=sub[sub.noise==noise]
            if d.empty: continue
            ax.plot(d.probability, d[col], label=noise,
                    color=COL_NOISE[noise], marker=MARKERS[i%len(MARKERS)])
        ax.set(xlabel='probability',ylabel=label); ax.grid(); ax.legend()
    fig.tight_layout(rect=[0,0,1,0.95])
    fig.savefig(GRAPH_DIR/f"{sim}_{init}_{shot}_noise_4panel.png",dpi=300)
    plt.close(fig)

# ──────────────────────────────────────────────────────────────────────────────
# Main CLI
# ──────────────────────────────────────────────────────────────────────────────
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--shots_vs', action='store_true')
    p.add_argument('--shots_vs_sim', action='store_true')
    p.add_argument('--qubits_vs', action='store_true')
    p.add_argument('--init_compare', action='store_true')
    p.add_argument('--noise_4panel', action='store_true')
    p.add_argument('--shots', nargs='+', type=int)
    p.add_argument('--inits', nargs='+', choices=INIT_LIST)
    p.add_argument('--targets', nargs='+', choices=ALL_TARGETS)
    args=p.parse_args()
    shots=args.shots or SHOT_LIST
    inits=args.inits or INIT_LIST
    tgts=args.targets or ALL_TARGETS

    if args.shots_vs:
        for sim in tgts:
            for init in inits:
                plot_shots_sweep(sim, init, BASELINE_DF)
    if args.shots_vs_sim:
        for init in inits:
            for sim in tgts:
                plot_shots_sweep(sim, init, BASELINE_DF)  # reuse
    if args.qubits_vs:
        for sim in tgts:
            plot_qubits_sweep(sim, BASELINE_DF)
    if args.init_compare:
        for vendor in VENDORS:
            for shot in shots:
                plot_init_platform_cmp(vendor, shot)
    if args.noise_4panel:
        for sim in tgts:
            for init in inits:
                for shot in shots:
                    plot_noise_4panel(sim, init, shot, NOISY_DF)

    print("\n✅ All requested plots written to", GRAPH_DIR)

if __name__=='__main__':
    main()
