#!/bin/bash

# List of 2^i shots: from 2^10 (1024) to 2^19 (524288)
shot_list=(1024 2048 4096 8192 16384 32768 65536 131072 262144 524288)

# Loop over device types
for device in GPU CPU
do
  echo "🚀 Running benchmarks with device: $device"
  echo "-----------------------------------"
  echo "running witout noise"
  python3 benchmark_qft_multiple_shots_gpu.py --device "$device" --shots "$shots" --init "ghz"

  # Loop over each shot count
  for shots in "${shot_list[@]}"
  do
    echo ""
    echo "with noise"
    echo "▶ Shots: $shots | Device: $device"
    python3 benchmark_qft_shots_noisy_circuit.py --device "$device" --shots "$shots" --init "ghz" 
  done

  echo "✅ Finished all runs for $device"
  echo ""
done
