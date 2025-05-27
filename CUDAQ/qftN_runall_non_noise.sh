#!/bin/bash

echo "📡 Starting QFT benchmarks..."

init_states=("ghz" "zero")
targets=("qpp-cpu" "nvidia")
shots_list=(8192 16384 32768 65536 131072 262144 524288)

# ───────────── Noiseless Benchmarks ─────────────
for target in "${targets[@]}"; do
  for init_state in "${init_states[@]}"; do
    echo ""
    echo "🚀 Running NOISELESS benchmark for target='$target', init='$init_state', shots=$shots"
    python3 qftN_runall.py --target "$target" --init "$init_state"
    if [ $? -ne 0 ]; then
    echo "❌ Failed: $target | $init_state | $shots"
    else
    echo "✅ Done:   $target | $init_state | $shots"
    fi
  done
done

# ───────────── Noisy Model Benchmarks ─────────────
for init_state in "${init_states[@]}"; do
  for shots in "${shots_list[@]}"; do
    echo ""
    echo "🌩️  Running NOISY benchmark for init='$init_state', shots=$shots"
    python3 qftN_benchmark_cpu_noise_model.py --init "$init_state" --shots "$shots"
    if [ $? -ne 0 ]; then
      echo "❌ Failed: noisy | $init_state | $shots"
    else
      echo "✅ Done:   noisy | $init_state | $shots"
    fi
  done
done

echo -e "\n🎉 All QFT benchmarks completed."
