import csv
import math
from pathlib import Path

DATASET_PATH = Path("data/sinr_dataset.csv")

def baseline_predict(
    signal_power: float, 
    interference_power: float
) -> float:
    """Deliberatly ignore noise power."""
    sinr_linear = signal_power / interference_power
    return 10 * math.log10(sinr_linear)


def main():
    true_values = []
    predicted_values = []

    with DATASET_PATH.open() as file:
        reader = csv.DictReader(file)

        for row in reader:
            signal_power = float(row["signal_power"])
            interference_power = float(row["interference_power"])
            true_sinr = float(row["sinr_db"])

            predicted_sinr = baseline_predict(signal_power, interference_power)
            true_values.append(true_sinr)
            predicted_values.append(predicted_sinr)

    errors = [
        prediction - true
        for prediction, true in zip(predicted_values, true_values)
    ]

    mae = sum(abs(error) for error in errors) / len(errors)
    mse = sum(error ** 2 for error in errors) / len(errors)
    rmse = math.sqrt(mse)

    print(f"Baseline MAE: {mae:.2f} dB")
    print(f"Baseline MSE: {mse:.2f} dB")
    print(f"Baseline RMSE: {rmse:.2f} dB")

if __name__ == "__main__":
    main()