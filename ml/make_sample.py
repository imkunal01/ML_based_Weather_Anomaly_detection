"""Create sample test dataset from holdout data."""

from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
HOLDOUT_FILE = DATA_DIR / "holdout.csv"
SAMPLE_FILE = DATA_DIR / "sample.csv"


def make_sample():
    df = pd.read_csv(HOLDOUT_FILE)
    sample_df = df.head(200)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    sample_df.to_csv(SAMPLE_FILE, index=False)
    print(f"Saved sample data to {SAMPLE_FILE}. Shape: {sample_df.shape}")


if __name__ == "__main__":
    make_sample()
