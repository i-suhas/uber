import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import patch, MagicMock


def test_predict_returns_float():
    mock_model = MagicMock()
    mock_model.predict.return_value = [1.75]
    with patch("src.predict.load_model", return_value=mock_model):
        from src.predict import predict
        result = predict({"lag_1hr_surge": 1.5, "lag_2hr_surge": 1.3, "hour": 8, "geohash": "dw2b"})
    assert isinstance(result, float)


def test_predict_output_in_valid_range():
    mock_model = MagicMock()
    mock_model.predict.return_value = [2.1]
    with patch("src.predict.load_model", return_value=mock_model):
        from src.predict import predict
        result = predict({"lag_1hr_surge": 2.0, "lag_2hr_surge": 1.8, "hour": 17, "geohash": "dw2c"})
    assert 1.0 <= result <= 4.0


def test_predict_different_geohashes():
    mock_model = MagicMock()
    mock_model.predict.return_value = [1.5]
    with patch("src.predict.load_model", return_value=mock_model):
        from src.predict import predict
        for gh in ["dw2b", "dw2c", "dw2e"]:
            r = predict({"lag_1hr_surge": 1.2, "lag_2hr_surge": 1.1, "hour": 12, "geohash": gh})
            assert isinstance(r, float)
