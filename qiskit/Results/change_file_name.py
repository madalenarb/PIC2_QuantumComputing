import re
import shutil
from pathlib import Path

def copy_and_rename_qiskit_files_inplace():
    """
    In the script’s directory, copy & rename Qiskit CSV results (and multishot statevector files),
    preserving the originals, and sort into 'noise/' and 'non_noise/' subfolders.

    1) Noise:
       qiskit_<CPU|GPU>_noise_<num_shots>_<freq>.csv
       -> noise/qiskit_qft_noise_<freq>_<device>_<num_shots>.csv

    2) Non-noise:
       a) multishot_qft_<cpu|gpu>_<freq>_statevector.csv
          -> non_noise/qiskit_qft_multishot_no_noise_<device>_<freq>.csv

       b) qiskit_<CPU|GPU>_<freq>_multiple_shots.csv (if present)
          -> non_noise/qiskit_qft_multishot_no_noise_<device>_<freq>.csv

    raw_device mapping: 'CPU' -> 'cpu', 'GPU' -> 'gpu'
    freq: 'ghz' or 'zero'
    """
    base_dir = Path(__file__).parent.resolve()
    device_map = {'CPU': 'cpu', 'GPU': 'gpu'}

    # Ensure subdirectories exist
    noise_dir = base_dir / 'noise'
    non_noise_dir = base_dir / 'non_noise'
    noise_dir.mkdir(exist_ok=True)
    non_noise_dir.mkdir(exist_ok=True)

    # Process all CSVs in the directory
    for file in base_dir.glob('*.csv'):
        name = file.name.lower()
        new_name = None
        target_dir = None

        # 1) Noise pattern
        m_noise = re.match(r"qiskit_(cpu|gpu)_noise_([0-9]+)_(ghz|zero)\.csv", name)
        if m_noise:
            raw_dev, shots, freq = m_noise.groups()
            device = device_map[raw_dev.upper()]
            new_name = f"qiskit_qft_noise_{freq}_{device}_{shots}.csv"
            target_dir = noise_dir

        else:
            # 2a) Multishot statevector non-noise
            m_state = re.match(r"multishot_qft_(cpu|gpu)_(ghz|zero)_statevector\.csv", name)
            if m_state:
                raw_dev, freq = m_state.groups()
                device = device_map[raw_dev.upper()]
                new_name = f"qiskit_qft_multishot_no_noise_{device}_{freq}.csv"
                target_dir = non_noise_dir
            else:
                # 2b) qiskit multishot non-noise pattern
                m_multi = re.match(r"qiskit_(cpu|gpu)_(ghz|zero)_multiple_shots\.csv", name)
                if m_multi:
                    raw_dev, freq = m_multi.groups()
                    device = device_map[raw_dev.upper()]
                    new_name = f"qiskit_qft_multishot_no_noise_{device}_{freq}.csv"
                    target_dir = non_noise_dir

        # Copy if we have a new name and directory
        if new_name and target_dir:
            target_path = target_dir / new_name
            if target_path.exists():
                print(f"Skipped (exists): {target_path.relative_to(base_dir)}")
            else:
                shutil.copy2(file, target_path)
                print(f"Copied: {file.name} -> {target_path.relative_to(base_dir)}")
        else:
            print(f"Skipped (no match): {file.name}")


if __name__ == "__main__":
    copy_and_rename_qiskit_files_inplace()

