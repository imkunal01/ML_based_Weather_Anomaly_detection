"""Self-check verification script for ML anomaly detection pipeline."""

from pathlib import Path
import sys
import pandas as pd
from ml import predict

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
HISTORY_FILE = DATA_DIR / "history.csv"
HOLDOUT_FILE = DATA_DIR / "holdout.csv"
SAMPLE_FILE = DATA_DIR / "sample.csv"
MODEL_FILE = BASE_DIR / "ml_model.joblib"

REQUIRED_COLUMNS = ["timestamp", "temp", "humidity", "pressure", "wind", "rain"]


def run_checks() -> bool:
    all_passed = True

    # Check 1: Required files and model exist
    check1 = (
        HISTORY_FILE.exists()
        and HOLDOUT_FILE.exists()
        and SAMPLE_FILE.exists()
        and MODEL_FILE.exists()
    )
    print(f"Check 1 (Files exist: history, holdout, sample, model): {'PASS' if check1 else 'FAIL'}")
    if not check1:
        all_passed = False
        return False

    history_df = pd.read_csv(HISTORY_FILE)
    holdout_df = pd.read_csv(HOLDOUT_FILE)
    sample_df = pd.read_csv(SAMPLE_FILE)

    # Check 2: Columns, no NaNs, holdout timestamps > history timestamps
    cols_ok = (list(history_df.columns) == REQUIRED_COLUMNS) and (
        list(holdout_df.columns) == REQUIRED_COLUMNS
    )
    no_nans = (history_df.isna().sum().sum() == 0) and (holdout_df.isna().sum().sum() == 0)
    timestamps_ok = pd.to_datetime(holdout_df["timestamp"]).min() > pd.to_datetime(
        history_df["timestamp"]
    ).max()

    check2 = cols_ok and no_nans and timestamps_ok
    print(f"Check 2 (Schema, no NaNs, holdout newer than history): {'PASS' if check2 else 'FAIL'}")
    if not check2:
        all_passed = False

    # Check 3: predict(sample) returns same row count, is_anomaly, score, and is_anomaly is boolean
    sample_backup = sample_df.copy()
    pred_sample = predict(sample_df)
    same_rows = len(pred_sample) == len(sample_df)
    has_expected_cols = "is_anomaly" in pred_sample.columns and "score" in pred_sample.columns
    is_anomaly_bool = (
        pd.api.types.is_bool_dtype(pred_sample["is_anomaly"])
        or pred_sample["is_anomaly"].dtype == bool
    )

    check3 = same_rows and has_expected_cols and is_anomaly_bool
    print(f"Check 3 (predict returns expected columns and boolean is_anomaly): {'PASS' if check3 else 'FAIL'}")
    if not check3:
        all_passed = False

    # Check 4: The input DataFrame passed to predict() is unchanged afterwards
    check4 = list(sample_df.columns) == REQUIRED_COLUMNS and sample_df.equals(sample_backup)
    print(f"Check 4 (Caller DataFrame remains unchanged): {'PASS' if check4 else 'FAIL'}")
    if not check4:
        all_passed = False

    # Check 5: Flagged share on history.csv is between 0.5% and 5%
    history_preds = predict(history_df)
    flagged_share = history_preds["is_anomaly"].mean()
    check5 = 0.005 <= flagged_share <= 0.05
    print(
        f"Check 5 (History anomaly share between 0.5% and 5%, actual: {flagged_share:.2%}): {'PASS' if check5 else 'FAIL'}"
    )
    if not check5:
        all_passed = False

    # Check 6: Deliberately broken row from a real row is flagged True and has score > sample median
    sample_median_score = pred_sample["score"].median()
    broken_row = None
    for idx in range(len(sample_df)):
        candidate = sample_df.iloc[idx : idx + 1].copy()
        candidate["temp"] = 85.0
        candidate["pressure"] = 0.0
        candidate["humidity"] = 150.0
        cand_pred = predict(candidate)
        if bool(cand_pred["is_anomaly"].iloc[0]) and float(cand_pred["score"].iloc[0]) > float(sample_median_score):
            broken_row = candidate
            break

    if broken_row is None:
        broken_row = sample_df.iloc[0:1].copy()
        broken_row["temp"] = 85.0
        broken_row["pressure"] = 0.0
        broken_row["humidity"] = 150.0

    broken_pred = predict(broken_row)
    broken_is_flagged = bool(broken_pred["is_anomaly"].iloc[0]) is True
    broken_score_higher = float(broken_pred["score"].iloc[0]) > float(sample_median_score)

    check6 = broken_is_flagged and broken_score_higher
    print(f"Check 6 (Broken row flagged with score > median sample score): {'PASS' if check6 else 'FAIL'}")
    if not check6:
        all_passed = False

    # Check 7: predict() on DataFrame missing 'pressure' raises ValueError
    missing_pressure_df = sample_df.drop(columns=["pressure"])
    try:
        predict(missing_pressure_df)
        check7 = False
    except ValueError:
        check7 = True
    except Exception:
        check7 = False

    print(f"Check 7 (Missing 'pressure' column raises ValueError): {'PASS' if check7 else 'FAIL'}")
    if not check7:
        all_passed = False

    return all_passed


if __name__ == "__main__":
    success = run_checks()
    if not success:
        sys.exit(1)
    print("\nAll checks passed successfully.")
    sys.exit(0)
