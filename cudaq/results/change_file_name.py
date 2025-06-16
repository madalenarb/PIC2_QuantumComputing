import re
import shutil
from pathlib import Path

def copy_and_rename_qft_files_inplace():
    """
    In the script's directory, copy and rename QFT CSV files, preserving originals,
    creating 'noise' and 'non_noise' subfolders alongside.

    Handles these source patterns:
    - No-noise multishot:
      qftN_<raw_device>_<freq>_multiple_shots.csv
      -> non_noise/cudaq_qft_multishot_no_noise_<device>_<freq>.csv

    - Noise (raw qftN):
      qftN_<freq>_<num_shots>_<raw_device>.csv
      qftN_<freq>_<num_shots>.csv (assume CPU)
      -> noise/cudaq_qft_<num_shots>_noise_<device>_<freq>.csv

    - Pre-named noise without "noise" in name (cudaq_qft originals):
      cudaq_qft_<freq>_<num_shots>_<raw_dev>.csv
      -> noise/cudaq_qft_<num_shots>_noise_<device>_<freq>.csv

    raw_device mapping: 'nvidia' -> 'gpu', 'qpp-cpu' -> 'cpu'
    freq: 'zero' or 'ghz'

    If the renamed file already exists in the target folder, it will be skipped.
    """
    base_dir = Path(__file__).parent.resolve()
    device_map = {'nvidia': 'gpu', 'qpp-cpu': 'cpu'}

    # Create target subdirectories in base_dir
    non_noise_dir = base_dir / 'non_noise'
    noise_dir = base_dir / 'noise'
    non_noise_dir.mkdir(exist_ok=True)
    noise_dir.mkdir(exist_ok=True)

    # Process all CSV files in base_dir (ignore subdirectories)
    for file in base_dir.glob('*.csv'):
        if not file.is_file():
            continue
        name = file.name
        new_name = None
        target_dir = None

        # Pattern 1: cudaq_qft_<freq>_<num_shots>_<raw_dev>.csv (pre-named noise)
        m = re.match(r"cudaq_qft_([^_]+)_([0-9]+)_([^\.]+)\.csv", name)
        if m:
            freq, num_shots, raw_dev = m.groups()
            device = device_map.get(raw_dev, raw_dev)
            new_name = f"cudaq_qft_noise_{freq}_{device}_{num_shots}.csv"
            target_dir = noise_dir
        else:
            # Pattern 2: no-noise multishot
            m = re.match(r"qftN_([^_]+)_([^_]+)_multiple_shots\.csv", name)
            if m:
                raw_dev, freq = m.groups()
                device = device_map.get(raw_dev, raw_dev)
                new_name = f"cudaq_qft_multishot_no_noise_{device}_{freq}.csv"
                target_dir = non_noise_dir
            else:
                # Pattern 3: qftN_<freq>_<num_shots>_<raw_device>
                m = re.match(r"qftN_([^_]+)_([0-9]+)_([^_]+)\.csv", name)
                if m:
                    freq, num_shots, raw_dev = m.groups()
                    device = device_map.get(raw_dev, raw_dev)
                    new_name = f"cudaq_qft_noise_{freq}_{device}_{num_shots}.csv"
                    target_dir = noise_dir
                else:
                    # Pattern 4: qftN_<freq>_<num_shots>.csv (assume CPU noise)
                    m2 = re.match(r"qftN_([^_]+)_([0-9]+)\.csv", name)
                    if m2:
                        freq, num_shots = m2.groups()
                        device = 'cpu'
                        new_name = f"cudaq_qft_noise_{freq}_{device}_{num_shots}.csv"
                        target_dir = noise_dir

        # Copy if matched
        if new_name and target_dir:
            target = target_dir / new_name
            if target.exists():
                print(f"Skipped (already exists): {target.relative_to(base_dir)}")
            else:
                shutil.copy2(file, target)
                print(f"Copied: {name} -> {target.relative_to(base_dir)}")
        else:
            print(f"Skipped (no match): {name}")


if __name__ == "__main__":
    copy_and_rename_qft_files_inplace()