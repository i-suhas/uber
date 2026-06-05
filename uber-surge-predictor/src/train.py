import joblib
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

from src.config import FEATURES, TARGET, MODEL_PATH, RANDOM_STATE, N_ESTIMATORS, MAX_DEPTH
from src.preprocess import load_data, build_preprocessor
from src.feature_engineering import create_features
from src.evaluate import evaluate_model


def train():
    df = create_features(load_data())

    X = df[FEATURES]
    y = df[TARGET]

    # Time-ordered split (no shuffle — preserves chronology)
    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("regressor",    RandomForestRegressor(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])

    pipeline.fit(X_train, y_train)
    evaluate_model(pipeline, X_test, y_test)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Model saved: {MODEL_PATH}")
    return pipeline


if __name__ == "__main__":
    train()
