import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np

from src.predict import predict

app = FastAPI(title="Uber Surge Predictor API")


class SurgeRequest(BaseModel):
    lag_1hr_surge: float
    lag_2hr_surge: float
    hour: int          # 0-23
    geohash: str       # e.g. "dw2b"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict_surge(req: SurgeRequest):
    try:
        surge = predict(req.model_dump())
        return {"surge_multiplier": round(surge, 4)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
