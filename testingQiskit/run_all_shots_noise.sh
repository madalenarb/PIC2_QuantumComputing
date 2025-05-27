#!/bin/bash

# List of 2^i shots: from 2^12 to 2^19
shot_list=(4096 8192 16384 32768 65536 131072 262144)
devices_list=("GPU" "CPU")
init_states=("ghz" "zero")

# Loop over device types
for device in "${devices_list[@]}"; do
  echo "🚀 Running benchmarks with device: $device"
  echo "-----------------------------------------"

  echo "📦 Running WITHOUT noise (if script available)"
  for init in "${init_states[@]}"; do
    echo "▶ Init: $init"
    # Uncomment the line below if you have the script for noiseless benchmarking
    #python3 benchmark_qft_multiple_shots_gpu.py --device "$device" --init "$init"
    echo "✅ Done → device=$device, init=$init, shots=32768 (noiseless)"
  done

  echo ""
  echo "🌩️ Running WITH noise"
  for shots in 524288; do
    echo "▶ Shots: $shots | Device: $device"
    for init in "${init_states[@]}"; do
      echo "• Init: $init"
      python3 benchmark_qft_shots_noisy_circuit_lib.py --device "$device" --shots "$shots" --init "$init"
      echo "✅ Done → device=$device, init=$init, shots=$shots (noisy)"
    done
  done

  echo "✅ Finished all runs for device: $device"
  echo ""
done

echo "🎉 All benchmarks completed!"
