"""Fetch 3 years of hourly historical weather data from Open-Meteo Archive API."""

from pathlib import Path
import sys
import time
import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_FILE = DATA_DIR / "history_raw.csv"

API_URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch_weather_data(latitude: float = 28.61, longitude: float = 77.21) -> pd.DataFrame:
    """Download historical weather data and return it as a pandas DataFrame."""
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": "2023-01-01",
        "end_date": "2025-12-31",
        "hourly": "temperature_2m,relative_humidity_2m,pressure_msl,wind_speed_10m,precipitation",
    }

    max_retries = 3
    response = None

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(API_URL, params=params, timeout=120)
            response.raise_for_status()
            break
        except requests.RequestException as exc:
            if attempt < max_retries:
                time.sleep(2)
            else:
                print(f"Error fetching data from Open-Meteo after {max_retries} attempts: {exc}", file=sys.stderr)
                sys.exit(1)

    payload = response.json()
    if "hourly" not in payload:
        print(f"Error: 'hourly' key missing from Open-Meteo response: {payload}", file=sys.stderr)
        sys.exit(1)

    hourly = payload["hourly"]

    # Open-Meteo keys -> target column names
    df = pd.DataFrame(
        {
            "timestamp": hourly["time"],
            "temp": hourly["temperature_2m"],
            "humidity": hourly["relative_humidity_2m"],
            "pressure": hourly["pressure_msl"],
            "wind": hourly["wind_speed_10m"],
            "rain": hourly["precipitation"],
        }
    )

    return df


def main():
    latitude = float(sys.argv[1]) if len(sys.argv) > 1 else 28.61
    longitude = float(sys.argv[2]) if len(sys.argv) > 2 else 77.21

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    df = fetch_weather_data(latitude=latitude, longitude=longitude)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved raw data to {OUTPUT_FILE}. Shape: {df.shape}")


if __name__ == "__main__":
    main()
