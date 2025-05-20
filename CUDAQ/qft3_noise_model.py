import math
import numpy as np
import cudaq

# --------------------------------------------------------------------
# Noise‐model setup
# --------------------------------------------------------------------
p = 0.1
noise_names = ["depolarizing", "amplitude-damping", "phase-damping", "bit-flip"]
noise_channels = [
    cudaq.DepolarizationChannel(p),
    cudaq.AmplitudeDampingChannel(p),
    cudaq.PhaseFlipChannel(p),
    cudaq.BitFlipChannel(p)
]

# User selects which noise to apply
print("Select noise model:")
for i, name in enumerate(noise_names, start=1):
    print(f"{i}: {name}")
choice = int(input("Enter your choice (1-4): ")) - 1
if choice not in range(len(noise_names)):
    print("Invalid selection. Defaulting to depolarizing.")
    choice = 0

noise_name = noise_names[choice]
channel = noise_channels[choice]

# Build the NoiseModel by adding the chosen channel to each qubit
noise_model = cudaq.NoiseModel()
for q in range(3):
    for basis in ['x', 'y', 'z']:
        noise_model.add_channel(basis, [q], channel)

# --------------------------------------------------------------------
# 3-qubit QFT kernel
# --------------------------------------------------------------------
@cudaq.kernel
def qft3():
    q = cudaq.qvector(3)
    h(q[0])
    r1.ctrl(math.pi/2, q[1], q[0])
    r1.ctrl(math.pi/4, q[2], q[0])
    h(q[1])
    r1.ctrl(math.pi/2, q[2], q[1])
    h(q[2])
    swap(q[0], q[2])

# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------
def run_sampling(kernel, shots: int, noise_model=None):
    return cudaq.sample(kernel,
                        shots_count=shots,
                        noise_model=noise_model)

def get_probabilities(counts):
    total = sum(counts.values())
    return {s: c / total for s, c in counts.items()}

def print_distribution(probs, ideal):
    diffs = []
    for i in range(8):
        s = f"{i:03b}"
        p = probs.get(s, 0.0)
        d = abs(p - ideal)
        diffs.append(d)
        print(f"  |{s}⟩: {p:.4f} (ideal {ideal:.4f}, Δ={d:.4f})")
    return np.linalg.norm(diffs)

# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------
if __name__ == "__main__":
    TARGET = "qpp-cpu"
    SHOTS = 16384

    cudaq.set_target(TARGET)
    print("Target:", cudaq.get_target().name)
    print("Circuit:")
    print(cudaq.draw(qft3))

    # No‐noise run
    counts_i = run_sampling(qft3, SHOTS, noise_model=None)
    probs_i  = get_probabilities(counts_i)
    print("\n=== No Noise ===")
    L2_i = print_distribution(probs_i, 1/8)

    # With‐noise run
    counts_n = run_sampling(qft3, SHOTS, noise_model=noise_model)
    probs_n  = get_probabilities(counts_n)
    print(f"\n=== With Noise ({noise_name}, p={p}) ===")
    L2_n = print_distribution(probs_n, 1/8)

    print(f"\nL2 error (no noise)   : {L2_i:.4e}")
    print(f"L2 error (with noise) : {L2_n:.4e}")
