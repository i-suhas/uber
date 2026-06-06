import numpy as np
import joblib
import pandas as pd

from src.config import MODEL_PATH, FEATURES


def load_model(path: str = MODEL_PATH):
    return joblib.load(path)


def predict(input_data: dict) -> float:
    """
    input_data keys: lag_1hr_surge, lag_2hr_surge, hour (int 0-23), geohash
    """
    row = input_data.copy()
    hour = row.pop("hour", 0)
    row["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    row["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    df = pd.DataFrame([row])[FEATURES]
    model = load_model()
    return float(model.predict(df)[0])
