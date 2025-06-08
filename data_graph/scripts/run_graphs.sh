#!/usr/bin/env bash
# generate_all_plots.sh — drive plot_noise_analysis.py with the correct flags

TARGETS=( cudaq-cpu cudaq-gpu qiskit-cpu qiskit-gpu )
INITS=( zero ghz )
SHOTS=( 131072 262144 )

# 1) A. Shots vs L2 & Time (per‐simulator, per‐init)
for tgt in "${TARGETS[@]}"; do
  for init in "${INITS[@]}"; do
    python3 plot_noise_analysis.py \
      --shots_vs \
      --targets "$tgt" \
      --inits "$init" \
      --shots "${SHOTS[@]}"
  done
done

# 2) A2. Shots vs L2 & Time comparison across simulators (fixed init)
#    Here we do one init at a time, comparing all TARGETS on one canvas:
for init in "${INITS[@]}"; do
  python3 plot_noise_analysis.py \
    --shots_vs_sim \
    --inits "$init" \
    --shots "${SHOTS[@]}"
done

# 3) B. Qubits vs L2 & Time (per‐simulator)
for tgt in "${TARGETS[@]}"; do
  python3 plot_noise_analysis.py \
    --qubits_vs \
    --targets "$tgt" \
    --inits "${INITS[@]}"
done

# 4) C. Init × Platform comparison (bar chart, at 131072 shots)
python3 plot_noise_analysis.py \
  --init_compare \
  --shots 131072 \
  --inits "${INITS[@]}" \
  --targets "${TARGETS[@]}"

# 5) D. Noise 4‐panel (L2/Fro/Fid/Time vs probability) 
#    one per (simulator, init, shot)
for tgt in "${TARGETS[@]}"; do
  for init in "${INITS[@]}"; do
    for shot in "${SHOTS[@]}"; do
      python3 plot_noise_analysis.py \
        --noise_4panel \
        --targets "$tgt" \
        --inits "$init" \
        --shots "$shot"
    done
  done
done

echo "✅ All plots generated under graphs/noise/"
