
import csv
import math
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

DATASET_PATH = Path("data/sinr_dataset.csv")
SEED = 42

def load_dataset():
    features = []
    targets = []

    with DATASET_PATH.open() as file:
        reader = csv.DictReader(file)
        for row in reader:
            features.append([
                float(row["signal_power"]),
                float(row["interference_power"]),
                float(row["noise_power"])
            ])
            targets.append(float(row["sinr_db"]))

    return features, targets

def main():
    X, y = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
    model = RandomForestRegressor(n_estimators=100, random_state=SEED)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = math.sqrt(mse)

    print(f"Random Forest MAE: {mae:.2f} dB")
    print(f"Random Forest MSE: {mse:.2f} dB")
    print(f"Random Forest RMSE: {rmse:.2f} dB")

if __name__ == "__main__":
    main()