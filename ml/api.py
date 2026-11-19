"""FastAPI microservice for Automatic Weather Station (AWS) anomaly detection."""

from pathlib import Path
from typing import List, Union
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd

try:
    from ml.ml import predict
except ImportError:
    from ml import predict

app = FastAPI(
    title="AWS Weather Anomaly Detection API",
    description="Lightweight ML microservice for detecting weather station sensor anomalies",
    version="1.0.0",
)


class WeatherReading(BaseModel):
    timestamp: str = Field(..., example="2026-11-20 12:00:00")
    temp: float = Field(..., description="Temperature in °C", example=26.5)
    humidity: float = Field(..., description="Relative Humidity in %", example=65.0)
    pressure: float = Field(..., description="Pressure in hPa", example=1012.3)
    wind: float = Field(..., description="Wind speed in km/h", example=12.4)
    rain: float = Field(..., description="Precipitation in mm", example=0.0)


@app.get("/")
def health_check():
    """Health check endpoint to verify service is running."""
    return {"status": "ok", "service": "aws-anomaly-detector"}


@app.post("/predict")
def predict_anomalies(data: Union[WeatherReading, List[WeatherReading]]):
    """
    Accepts a single weather reading or a batch/list of readings.
    Returns the records with `is_anomaly` (boolean) and `score` (float).
    """
    try:
        if isinstance(data, WeatherReading):
            records = [data.model_dump()]
        else:
            records = [item.model_dump() for item in data]

        df = pd.DataFrame(records)
        result_df = predict(df)

        result_df["timestamp"] = result_df["timestamp"].astype(str)
        result_df["is_anomaly"] = result_df["is_anomaly"].astype(bool)

        results = result_df.to_dict(orient="records")
        return results if isinstance(data, list) else results[0]

    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err))
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Internal prediction error: {err}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8001, reload=True)
