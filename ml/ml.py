"""Machine learning anomaly detection model for Automatic Weather Stations (AWS)."""

from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

FEATURES = ["temp", "humidity", "pressure", "wind", "rain"]
REQUIRED_COLUMNS = ["timestamp", "temp", "humidity", "pressure", "wind", "rain"]

MODEL_PATH = Path(__file__).parent / "ml_model.joblib"
DEFAULT_CSV_PATH = Path(__file__).resolve().parent / "data" / "history.csv"

_MODEL_CACHE = None


def _get_model():
    """Load the model and scaler lazily from disk and cache them in memory."""
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError("Run python ml.py first to train the model")
        _MODEL_CACHE = joblib.load(MODEL_PATH)
    return _MODEL_CACHE


def train(csv_path=DEFAULT_CSV_PATH, contamination=0.02):
    """Train StandardScaler and IsolationForest on historical data, then save model."""
    global _MODEL_CACHE
    df = pd.read_csv(csv_path)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[FEATURES])

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
    )
    model.fit(X_scaled)

    joblib.dump((scaler, model), MODEL_PATH)
    _MODEL_CACHE = (scaler, model)


def predict(df):
    """Detect anomalies in weather records and return copy with is_anomaly and score columns."""
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    for col in FEATURES:
        if pd.api.types.is_bool_dtype(df[col]) or not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"Feature '{col}' contains non-numeric values")
        if df[col].isna().any():
            raise ValueError(f"Feature '{col}' contains NaN values")

    scaler, model = _get_model()

    result_df = df.copy()
    X_scaled = scaler.transform(result_df[FEATURES])
    raw_predictions = model.predict(X_scaled)
    scores = -model.score_samples(X_scaled)

    result_df["is_anomaly"] = bool(False)  # initialize / type safety
    result_df["is_anomaly"] = raw_predictions == -1
    result_df["score"] = scores.astype(float)

    return result_df


if __name__ == "__main__":
    train()
    print(f"model saved to {MODEL_PATH}")
