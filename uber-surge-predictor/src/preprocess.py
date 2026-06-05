import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, OneHotEncoder

from src.config import DATA_RAW, DATA_PROCESSED, NUMERIC_FEATURES, CATEGORICAL_FEATURES


def load_data(path: str = DATA_RAW) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["timestamp"])


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(transformers=[
        ("num", RobustScaler(),    NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])


def save_processed(df: pd.DataFrame, path: str = DATA_PROCESSED) -> None:
    df.to_csv(path, index=False)
