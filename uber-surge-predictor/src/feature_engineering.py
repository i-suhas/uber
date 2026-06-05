import numpy as np
import pandas as pd


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Demand/supply ratio — primary driver of surge
    df["demand_supply_ratio"] = df["demand_vol"] / (df["active_drivers"] + 1e-6)

    # Lag features per geohash (time-series aware)
    df["lag_1hr_surge"] = df.groupby("geohash")["surge_multiplier"].shift(1)
    df["lag_2hr_surge"] = df.groupby("geohash")["surge_multiplier"].shift(2)

    # Cyclical hour encoding
    df["hour_sin"] = np.sin(2 * np.pi * df["timestamp"].dt.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["timestamp"].dt.hour / 24)

    df.dropna(inplace=True)
    return df
