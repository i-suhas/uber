import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import pytest

from src.feature_engineering import create_features


@pytest.fixture
def sample_df():
    dates = pd.date_range("2024-01-01", periods=5, freq="h")
    return pd.DataFrame({
        "timestamp": dates,
        "geohash": ["dw2b"] * 5,
        "demand_vol": [50, 55, 60, 65, 70],
        "active_drivers": [30, 32, 34, 36, 38],
        "surge_multiplier": [1.0, 1.2, 1.4, 1.6, 1.8],
    })


def test_lag_features_created(sample_df):
    result = create_features(sample_df)
    assert "lag_1hr_surge" in result.columns
    assert "lag_2hr_surge" in result.columns


def test_cyclical_features_created(sample_df):
    result = create_features(sample_df)
    assert "hour_sin" in result.columns
    assert "hour_cos" in result.columns


def test_no_nulls_after_features(sample_df):
    result = create_features(sample_df)
    assert result.isnull().sum().sum() == 0


def test_hour_sin_range(sample_df):
    result = create_features(sample_df)
    assert result["hour_sin"].between(-1, 1).all()
    assert result["hour_cos"].between(-1, 1).all()
