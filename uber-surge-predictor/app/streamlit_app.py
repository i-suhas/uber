import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import streamlit as st
import joblib

from src.config import MODEL_PATH, FEATURES

st.set_page_config(page_title="Uber Surge Predictor", layout="centered")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

def predict(demand_vol, active_drivers, lag1, lag2, hour, geohash) -> float:
    ratio = demand_vol / (active_drivers + 1e-6)
    row = {
        "demand_vol": demand_vol,
        "active_drivers": active_drivers,
        "demand_supply_ratio": ratio,
        "lag_1hr_surge": lag1,
        "lag_2hr_surge": lag2,
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "geohash": geohash,
    }
    df = pd.DataFrame([row])[FEATURES]
    return float(load_model().predict(df)[0])

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("Uber Surge Price Predictor")
st.caption("Predicts surge multiplier directly from the trained model — no API required.")
st.divider()

st.subheader("Zone Conditions")

col1, col2 = st.columns(2)
demand_vol     = col1.number_input("Demand volume",     min_value=10.0, max_value=200.0, value=80.0, step=1.0,
                                    help="Number of ride requests in the zone right now.")
active_drivers = col2.number_input("Active drivers",    min_value=1.0,  max_value=100.0, value=30.0, step=1.0,
                                    help="Number of available drivers in the zone right now.")

col3, col4 = st.columns(2)
lag1 = col3.number_input("Surge 1 hour ago",  min_value=1.0, max_value=3.5, value=1.5, step=0.1,
                          help="The surge multiplier from 1 hour ago.")
lag2 = col4.number_input("Surge 2 hours ago", min_value=1.0, max_value=3.5, value=1.3, step=0.1,
                          help="The surge multiplier from 2 hours ago.")

hour    = st.slider("Hour of day (24h)", 0, 23, 8)
geohash = st.selectbox("Zone (Geohash)", ["dw2b", "dw2c", "dw2e"],
                        help="Geographic zone identifier (~5 km² cell).")

st.divider()

if st.button("Predict Surge", type="primary", use_container_width=True):
    surge = predict(demand_vol, active_drivers, lag1, lag2, hour, geohash)
    surge = round(surge, 4)

    st.subheader("Result")
    st.metric("Surge Multiplier", f"{surge}x")

    if surge < 1.5:
        st.success("Normal — good time to ride.")
    elif surge < 2.0:
        st.info("Moderate surge — slightly elevated demand.")
    elif surge < 2.5:
        st.warning("High surge — consider waiting 10-15 minutes.")
    else:
        st.error("Very high surge — peak demand, expect high fares.")

    # Explanation
    st.subheader("Why this prediction?")
    ratio = demand_vol / (active_drivers + 1e-6)
    trend = lag1 - lag2

    if ratio > 3.0:
        ratio_text = f"**High imbalance** — {demand_vol:.0f} requests vs {active_drivers:.0f} drivers (ratio {ratio:.1f}x). Strong upward pressure."
    elif ratio > 1.5:
        ratio_text = f"**Moderate imbalance** — {demand_vol:.0f} requests vs {active_drivers:.0f} drivers (ratio {ratio:.1f}x)."
    else:
        ratio_text = f"**Balanced** — {demand_vol:.0f} requests vs {active_drivers:.0f} drivers (ratio {ratio:.1f}x). Low surge pressure."

    if trend > 0.1:
        trend_text = f"**Rising** — was {lag2}x two hours ago, {lag1}x one hour ago."
    elif trend < -0.1:
        trend_text = f"**Falling** — was {lag2}x two hours ago, {lag1}x one hour ago."
    else:
        trend_text = f"**Stable** around {lag1}x."

    if 7 <= hour <= 9:
        hour_text = f"{hour}:00 — morning rush."
    elif 17 <= hour <= 20:
        hour_text = f"{hour}:00 — evening rush."
    elif hour >= 22 or hour <= 2:
        hour_text = f"{hour}:00 — late night."
    else:
        hour_text = f"{hour}:00 — off-peak."

    st.markdown(f"""
| Factor | Signal |
|---|---|
| Demand / Supply | {ratio_text} |
| Recent trend | {trend_text} |
| Time of day | {hour_text} |
| Zone | `{geohash}` |
| Model accuracy | R² = 1.0 · MAE = 0.0001 |
    """)

with st.expander("How does the model work?"):
    st.markdown("""
**Key features (in order of importance)**

| Feature | Role |
|---|---|
| `demand_supply_ratio` | Requests ÷ drivers — the primary surge driver |
| `demand_vol` | Raw request count in the zone |
| `active_drivers` | Available drivers in the zone |
| `lag_1hr_surge` | Surge 1 hour ago — captures recent momentum |
| `lag_2hr_surge` | Surge 2 hours ago — shows trend direction |
| `hour_sin / cos` | Cyclical time encoding (midnight connects to 1 am) |
| `geohash` | Zone identity, one-hot encoded |

**Pipeline:** RobustScaler → OneHotEncoder → RandomForest (300 trees, depth 12)  
**Split:** Time-ordered 80/20, no shuffle — no future data leakage  
**Metrics:** R² = 1.0 · MAE = 0.0001 · RMSE = 0.0003
    """)
