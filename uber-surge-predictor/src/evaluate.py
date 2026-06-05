import json
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import METRICS_PATH


def evaluate_model(model, X_test, y_test, save: bool = True) -> dict:
    preds = model.predict(X_test)
    metrics = {
        "mae":  round(mean_absolute_error(y_test, preds), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_test, preds))), 4),
        "r2":   round(r2_score(y_test, preds), 4),
    }
    print("Metrics:", metrics)
    if save:
        with open(METRICS_PATH, "w") as f:
            json.dump(metrics, f, indent=2)
    return metrics
