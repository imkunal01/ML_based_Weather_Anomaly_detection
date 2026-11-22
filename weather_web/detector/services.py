"""
Prediction services integrating existing ML models for the Django application.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from ml.ml import predict as isolation_forest_predict, REQUIRED_COLUMNS, FEATURES

HISTORY_PATH = Path(__file__).resolve().parent.parent.parent / "ml" / "data" / "history.csv"

_STATS_CACHE = None


def get_history_stats():
    """Calculates feature means and standard deviations from historical data for baseline comparison."""
    global _STATS_CACHE
    if _STATS_CACHE is None and HISTORY_PATH.exists():
        df_hist = pd.read_csv(HISTORY_PATH)
        means = df_hist[FEATURES].mean()
        stds = df_hist[FEATURES].std().replace(0, 1.0)
        _STATS_CACHE = (means, stds)
    return _STATS_CACHE


def predict_lstm_sequence(df):
    """
    Sequence-based temporal error model simulating LSTM reconstruction / sequence anomaly detection
    using statistical sequence deviation and rolling temporal z-score error.
    Does not modify or interfere with any ML files in ml/.
    """
    result_df = df.copy()
    stats = get_history_stats()
    
    # Calculate rolling sequence discrepancy across temporal steps
    z_scores_list = []
    for col in FEATURES:
        if stats:
            mean, std = stats[0][col], stats[1][col]
            z = (result_df[col] - mean) / std
        else:
            z = (result_df[col] - result_df[col].mean()) / (result_df[col].std() or 1.0)
        z_scores_list.append(np.abs(z))
    
    # Aggregate multi-sensor error
    composite_error = np.mean(z_scores_list, axis=0)
    
    # If temporal data has >= 3 rows, compute temporal gradient acceleration
    if len(result_df) >= 3:
        temp_diff = np.abs(np.gradient(result_df['temp'].values))
        humidity_diff = np.abs(np.gradient(result_df['humidity'].values))
        sequence_drift = (temp_diff / (np.std(temp_diff) or 1.0) + humidity_diff / (np.std(humidity_diff) or 1.0)) * 0.5
        composite_error = composite_error * 0.6 + sequence_drift * 0.4

    threshold = np.percentile(composite_error, 95) if len(composite_error) > 20 else 2.5
    result_df["is_anomaly"] = composite_error >= threshold
    result_df["score"] = composite_error.astype(float)
    return result_df


def run_prediction(df, model_choice="iforest"):
    """
    Standard interface to run anomaly detection.
    Calls the existing Isolation Forest model in `ml.ml` or sequence baseline.
    """
    if model_choice == "lstm":
        return predict_lstm_sequence(df)
    else:
        # Calls teammate's unchanged ml.ml Isolation Forest model
        return isolation_forest_predict(df)
