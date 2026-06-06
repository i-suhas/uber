import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_RAW       = os.path.join(BASE_DIR, "data", "raw", "surge_data.csv")
DATA_PROCESSED = os.path.join(BASE_DIR, "data", "processed", "processed.csv")
MODEL_PATH     = os.path.join(BASE_DIR, "models", "model.pkl")
METRICS_PATH   = os.path.join(BASE_DIR, "models", "metrics.json")

TARGET = "surge_multiplier"

NUMERIC_FEATURES = ["demand_vol", "active_drivers", "demand_supply_ratio",
                    "lag_1hr_surge", "lag_2hr_surge", "hour_sin", "hour_cos"]
CATEGORICAL_FEATURES = ["geohash"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

TEST_SIZE     = 0.2
RANDOM_STATE  = 42
N_ESTIMATORS  = 300
MAX_DEPTH     = 12
