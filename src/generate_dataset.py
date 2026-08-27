import csv
import math
import random
from pathlib import Path

NUM_SAMPLES = 1000
OUTPUT_PATH = Path("data/sinr_dataset.csv")
SEED = 42

# https://en.wikipedia.org/wiki/Signal-to-interference-plus-noise_ratio
def calculate_sinr_db(
        signal_power: float,
        interference_power: float,
        noise_power: float
) -> float:
    sinr_linear = signal_power / (interference_power + noise_power)
    sinr_db = 10 * math.log10(sinr_linear)
    return sinr_db

def generate_sample():
    signal_power = random.uniform(0.1, 10.0)
    interference_power = random.uniform(0.01, 5.0)
    noise_power = random.uniform(0.001, 0.1)

    sinr_db = calculate_sinr_db(signal_power, interference_power, noise_power)

    return {
        "signal_power": signal_power,
        "interference_power": interference_power,
        "noise_power": noise_power,
        "sinr_db": sinr_db
    }

def main():
    random.seed(SEED)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    samples = [generate_sample() for _ in range(NUM_SAMPLES)]

    with OUTPUT_PATH.open(mode="w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["signal_power", "interference_power", "noise_power", "sinr_db"]
        )
        writer.writeheader()
        writer.writerows(samples)

    sinr_values = [sample["sinr_db"] for sample in samples]

    print(f"Generated {NUM_SAMPLES} samples.")
    print(f"Saved to {OUTPUT_PATH}")
    print(f"Minimum SINR (dB): {min(sinr_values)}")
    print(f"Maximum SINR (dB): {max(sinr_values)}")
    print(f"Average SINR (dB): {sum(sinr_values) / len(sinr_values)}")

if __name__ == "__main__":
    main()