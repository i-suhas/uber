# Uber Surge Price Predictor

A production-style ML system that predicts Uber ride-share surge multipliers using time-series lag features, a scikit-learn pipeline, a FastAPI backend, and a Streamlit frontend.

<img width="1509" height="765" alt="Screenshot 2026-06-07 011437" src="https://github.com/user-attachments/assets/0696c736-64cd-4ad7-8a6a-54e7866267f8" />

<img width="1602" height="917" alt="Screenshot 2026-06-07 011525" src="https://github.com/user-attachments/assets/e0987895-b0d1-41a5-9b0f-537e8787f23d" />
<img width="1023" height="800" alt="Screenshot 2026-06-07 011534" src="https://github.com/user-attachments/assets/ed6d35ef-52aa-425a-a830-4639c0b59852" />



## Architecture

```
data/raw/surge_data.csv
        │
        ▼
src/preprocess.py   ──►  src/feature_engineering.py
        │
        ▼
src/train.py  ──►  models/model.pkl  +  models/metrics.json
                          │
                          ▼
                   api/main.py  (FastAPI)
                          │
                          ▼
                app/streamlit_app.py  (Streamlit UI)
```

---

## Quickstart

```bash
# 1. Create and activate virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model (generates models/model.pkl)
python -m src.train

# 4. Start the API  ← must be run from the project root
python -m uvicorn api.main:app --reload

# 5. Start the UI (open a second terminal, activate venv first)
streamlit run app/streamlit_app.py

# 6. Run tests
pytest tests/
```

> **Why `python -m uvicorn`?**  
> Running `uvicorn` directly requires it to be on your system PATH.  
> `python -m uvicorn` uses the uvicorn installed inside your active venv — always works.

API interactive docs: http://localhost:8000/docs

---

## File-by-file Explanation

### `data/`

| File | Description |
|---|---|
| `raw/surge_data.csv` | Raw historical Uber surge data with columns: `timestamp`, `geohash`, `surge_multiplier`. This is the only input the entire pipeline depends on. |
| `processed/.gitkeep` | Placeholder so git tracks the folder. Processed data is written here during training but excluded from version control. |

---

### `src/` — Core ML logic

#### `src/config.py`
Central configuration file. Defines all file paths (`DATA_RAW`, `MODEL_PATH`, …) and model hyper-parameters (`N_ESTIMATORS=300`, `MAX_DEPTH=12`). **Change paths or tuning parameters here**, not scattered across scripts.

#### `src/feature_engineering.py`
Transforms the raw DataFrame into model-ready features:
- **`lag_1hr_surge`** — surge multiplier 1 hour ago per geohash zone (strongest predictor).
- **`lag_2hr_surge`** — surge multiplier 2 hours ago per geohash zone (captures trend direction).
- **`hour_sin` / `hour_cos`** — cyclical encoding of the hour so the model knows midnight is adjacent to 1 am (not opposite).
- Rows with `NaN` lags (the first two timestamps per zone) are dropped.

#### `src/preprocess.py`
Handles data loading and the scikit-learn `ColumnTransformer`:
- `load_data()` — reads the CSV and parses the `timestamp` column.
- `build_preprocessor()` — returns a transformer that applies `RobustScaler` to numeric features (resistant to outlier surge spikes) and `OneHotEncoder` to the `geohash` column.

#### `src/train.py`
End-to-end training script:
1. Loads and engineers features.
2. Splits data **time-ordered** at 80% (no shuffle) — prevents future data leaking into training.
3. Builds a `Pipeline(preprocessor + RandomForestRegressor)`.
4. Fits, evaluates, and saves `model.pkl`.

Run with: `python -m src.train`

#### `src/evaluate.py`
Computes MAE, RMSE, and R² on the held-out test set and writes them to `models/metrics.json`. Called automatically at the end of `train.py`.

#### `src/utils.py`
Shared utility: `get_logger()` returns a configured Python logger used across scripts for consistent log formatting.

---

### `api/` — FastAPI backend

#### `api/main.py`
Exposes two endpoints:

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Returns `{"status": "ok"}` — used to check the server is alive. |
| `/predict` | POST | Accepts a JSON body and returns the predicted surge multiplier. |

Request body for `/predict`:
```json
{
  "lag_1hr_surge": 1.5,
  "lag_2hr_surge": 1.3,
  "hour": 8,
  "geohash": "dw2b"
}
```

The `sys.path.insert` at the top of the file ensures the `src/` module is importable when the server is started from the project root with `python -m uvicorn api.main:app`.

---

### `app/` — Streamlit frontend

#### `app/streamlit_app.py`
Interactive prediction UI:
- Input fields for all four model features with inline help text.
- Calls `POST /predict` on the local API.
- Displays the surge multiplier with a status label (Normal / Moderate / High / Very High).
- Shows a **"Why this prediction?"** table explaining how the trend, hour, zone, and model contributed.
- Collapsible **"How does the model work?"** section for technical detail.

---

### `models/`

| File | Description |
|---|---|
| `model.pkl` | Serialised scikit-learn pipeline (preprocessor + RandomForest). Generated by `src/train.py`. Not committed to git (large binary). |
| `metrics.json` | MAE, RMSE, R² from the last training run. |

---

### `tests/`

| File | What it tests |
|---|---|
| `tests/test_features.py` | Verifies feature engineering produces correct lag columns and cyclical hour encoding. |
| `tests/test_prediction.py` | Verifies the `predict()` function returns a float in a sensible range using the saved model. |

Run: `pytest tests/`

---

### Root files

| File | Description |
|---|---|
| `requirements.txt` | Pinned dependencies for reproducibility. |
| `Dockerfile` | Builds a container that serves the FastAPI app on port 8000. |
| `.gitignore` | Excludes `venv/`, `*.pkl`, `data/processed/`, and `__pycache__`. |

---

## Docker

```bash
docker build -t uber-surge .
docker run -p 8000:8000 uber-surge
```

---

## Model Summary

| Setting | Value |
|---|---|
| Algorithm | RandomForestRegressor |
| Estimators | 300 |
| Max depth | 12 |
| Train/test split | 80/20 time-ordered |
| Numeric scaling | RobustScaler |
| Categorical encoding | OneHotEncoder |

---

## Tech Stack

Python · pandas · scikit-learn · FastAPI · Uvicorn · Streamlit · Docker · pytest
