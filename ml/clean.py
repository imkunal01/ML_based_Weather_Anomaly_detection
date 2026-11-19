"""Clean raw weather data and split into history (training) and holdout (testing)."""

from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_FILE = DATA_DIR / "history_raw.csv"
HISTORY_FILE = DATA_DIR / "history.csv"
HOLDOUT_FILE = DATA_DIR / "holdout.csv"


def clean_and_split():
    df = pd.read_csv(RAW_FILE, parse_dates=["timestamp"])

    # Drop missing values, drop duplicate timestamps, and sort
    df = df.dropna().drop_duplicates(subset=["timestamp"]).sort_values("timestamp")

    # Filter physically impossible values
    valid_mask = (
        df["temp"].between(-60, 60)
        & df["humidity"].between(0, 100)
        & df["pressure"].between(850, 1100)
        & df["wind"].between(0, 200)
        & df["rain"].between(0, 200)
    )
    df = df[valid_mask]

    # Split by time: last 3 months to holdout, earlier to history
    max_timestamp = df["timestamp"].max()
    split_timestamp = max_timestamp - pd.DateOffset(months=3)

    holdout_df = df[df["timestamp"] > split_timestamp]
    history_df = df[df["timestamp"] <= split_timestamp]

    # Save both without index
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    history_df.to_csv(HISTORY_FILE, index=False)
    holdout_df.to_csv(HOLDOUT_FILE, index=False)

    print(f"Saved history data to {HISTORY_FILE}: {len(history_df)} rows")
    print(f"Saved holdout data to {HOLDOUT_FILE}: {len(holdout_df)} rows")
    print("\nTraining data description (df.describe()):")
    print(history_df.describe())


if __name__ == "__main__":
    clean_and_split()
