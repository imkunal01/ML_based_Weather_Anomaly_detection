<div align="center">

<img src="ASSETS/banner.svg" alt="Weather Anomaly Detector" width="100%"/>

<br/>

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![Django](https://img.shields.io/badge/Django-web%20app-092E20?logo=django&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Isolation%20Forest-F7931E?logo=scikitlearn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-data-150458?logo=pandas&logoColor=white)
![Chart.js](https://img.shields.io/badge/Chart.js-dashboard-FF6384?logo=chartdotjs&logoColor=white)
![Status](https://img.shields.io/badge/status-working%20demo-success)

**Upload a CSV of weather sensor readings. Get back every reading that looks wrong, with a chart and a score for each one.**

[Overview](#-overview) · [How it works](#-how-it-works) · [The model](#-the-model) · [Evaluation](#-evaluation) · [Web app](#-the-web-app) · [Run it](#-getting-started) · [API Testing](#-ml-microservice--api-testing) · [Team](#-team)

</div>

---

## 🌦️ Overview

Weather sensors break. They get stuck, they spike, they report a humidity of 150%. If you're collecting thousands of hourly readings, nobody is going to spot those by eye.

This project does it automatically:

1. Trains an **Isolation Forest** on about 3 years of clean historical weather data.
2. Wraps it in a single `predict()` function that any app can call.
3. Plugs that into a **Django** site where you upload a CSV and see the flagged rows on a dashboard.
4. Tests the model against a simple **z-score baseline** on data with deliberately injected faults, so we know whether the ML is actually pulling its weight.

<div align="center">
<img src="ASSETS/anomaly-demo.svg" alt="Animated anomaly demo chart" width="100%"/>
<sub>Illustrative demo data. Your real chart appears on the dashboard after you upload a file.</sub>
</div>

---

## ⚙️ How it works

<div align="center">
<img src="ASSETS/pipeline.svg" alt="Animated pipeline" width="100%"/>
</div>

```mermaid
flowchart LR
    A[Open-Meteo API<br/>3 years hourly data] --> B[fetch_data.py]
    B --> C[clean.py<br/>drop bad rows + split]
    C -->|older data| D[(history.csv<br/>training)]
    C -->|last 3 months| E[(holdout.csv<br/>testing)]
    D --> F[ml.py<br/>train Isolation Forest]
    F --> G[[ml_model.joblib]]
    G --> H[predict df]
    E --> I[Ashish's tests<br/>faulty_sample.csv]
    I --> H
    H --> J[Django app]
    J --> K[(SQLite<br/>Reading table)]
    K --> L[Dashboard]
```

### What happens when you upload a file

```mermaid
sequenceDiagram
    actor U as User
    participant V as Django view
    participant M as ml.predict()
    participant DB as Database
    participant D as Dashboard

    U->>V: Upload CSV
    V->>V: Check: is it a CSV? empty? columns present?
    V->>V: Convert types, drop unreadable rows
    V->>M: DataFrame (6 columns)
    M-->>V: Same rows + is_anomaly + score
    V->>DB: Replace old readings with new ones
    V-->>U: Redirect to dashboard
    U->>D: Open page
    D->>DB: Count rows, find anomalies, top 20 by score
    D-->>U: Cards + chart + table
```

---

## 🧠 The model

We use **Isolation Forest**. The idea is simple and kind of clever: instead of learning what normal looks like and measuring distance from it, it just tries to *isolate* each point with random cuts. Weird points get isolated fast. Normal points take many cuts.

<div align="center">
<img src="ASSETS/isolation-forest.svg" alt="Isolation Forest intuition animation" width="100%"/>
</div>

**Why this model for this problem**

| Reason | What it means here |
|---|---|
| Looks at all sensors together | A row can be flagged because of an odd *combination* (e.g. hot but very high humidity and zero pressure) |
| No hand-set thresholds | We don't have to decide "above 50°C is bad" for every column |
| Fast | Trains in seconds on ~25,000 rows |
| Gives a score | Not just yes/no, so the dashboard can rank the worst rows first |

**Settings:** 200 trees, features scaled with `StandardScaler`, `contamination = 0.02`. That last one tells the model to expect about 2% of rows to be abnormal. So roughly 2% of normal data **will** get flagged. That's by design, not a bug.

### The `predict()` contract

This never changes, so all three parts of the project can be built at the same time.

| | Columns |
|---|---|
| **Input** | `timestamp`, `temp`, `humidity`, `pressure`, `wind`, `rain` |
| **Output** | everything above, plus `is_anomaly` (True/False) and `score` (higher = more abnormal) |

---

## 📊 Data

| | |
|---|---|
| **Source** | [Open-Meteo Historical Weather API](https://open-meteo.com/) (free, no key) |
| **Range** | 2023-01-01 to 2025-12-31, hourly |
| **Size** | about 26,000 rows, 6 columns |
| **Cleaning** | drop missing values, drop duplicate timestamps, remove impossible values |
| **Split** | last 3 months → `holdout.csv` (testing). Everything earlier → `history.csv` (training) |

Splitting by date and not randomly keeps test data fully separate from training data, so the evaluation is honest.

**Sanity ranges used when cleaning**

```mermaid
xychart-beta
    title "Allowed value ranges (upper limit used by clean.py)"
    x-axis ["temp °C", "humidity %", "pressure hPa", "wind km/h", "rain mm"]
    y-axis "Upper limit" 0 --> 1200
    bar [60, 100, 1100, 200, 200]
```

---

## 🧪 Evaluation

We don't just trust the model, we test it. Ashish takes 300 rows of holdout data and breaks 30 of them on purpose.

```mermaid
pie showData
    title The 30 injected faults
    "Temperature spike (85°C)" : 10
    "Dead sensor (pressure = 0)" : 10
    "Impossible value (humidity = 150)" : 10
```

Two methods go head to head:

| | Z-score baseline | Isolation Forest |
|---|---|---|
| **Idea** | Flag any value more than 3 standard deviations from the training mean | Learn normal from all sensors and flag rows that are easy to isolate |
| **Thresholds** | Fixed, per column | Learned |
| **Sees combinations** | No, one column at a time | Yes |

### Results

> Fill this in from `ml/data/results.csv` after running `evaluate.py`. Report whatever comes out, honestly.

| Method | Faults caught (of 30) | False alarms (of 270 normal rows) |
|---|---|---|
| Z-score (>3) | _?_ | _?_ |
| Isolation Forest | _?_ | _?_ |

Things to expect: Isolation Forest will flag around 5 or 6 normal rows because of the 2% contamination setting. If both methods catch all 30, that's fine, just say so. The point in that case is that Isolation Forest got there without any hand-picked thresholds.

Charts generated by the evaluation script land in `ml/data/`: `bar.png` (faults caught per method) and `line.png` (temperature with flagged points in red).

---

## 🖥️ The web app

**Upload page.** Pick a CSV and submit. Friendly error messages for non-CSV files, empty files, missing columns, or files with no usable rows left after cleaning.

**Dashboard.** Three parts:

| Part | What it shows |
|---|---|
| **Count cards** | Total readings, anomalies found, anomaly percentage |
| **Line chart** | Temperature over time (Chart.js). Flagged rows are drawn as bigger red dots |
| **Top 20 table** | The highest-scoring anomalies with *all* columns, so you can see why each row got flagged |

Each upload replaces the previous one, which keeps the demo simple.

### Database

```mermaid
erDiagram
    READING {
        datetime timestamp
        float temp
        float humidity
        float pressure
        float wind
        float rain
        bool is_anomaly
        float score
    }
```

### Try these files

| File | What you should see |
|---|---|
| `sample.csv` | Mostly normal data, maybe a few red points |
| `faulty_sample.csv` | About 30 red points (the injected faults) plus a few extras |
| a `.txt` file, an empty CSV, a CSV missing `pressure` | A clear error message, not a crash |

---

## 📁 Project structure

```
weather-anomaly/
├── ml/                        # Model, API microservice & validation
│   ├── fetch_data.py          # Download 3 years hourly data from Open-Meteo
│   ├── clean.py               # Clean + split into history / holdout
│   ├── ml.py                  # Train Isolation Forest + predict()
│   ├── api.py                 # FastAPI microservice for standalone deployment
│   ├── make_sample.py         # Generate sample.csv (first 200 rows of holdout)
│   ├── verify.py              # Self-check test suite (7 verification checks)
│   ├── requirements.txt       # Pinned ML dependencies
│   ├── ml_model.joblib        # Saved trained model + scaler
│   ├── data/                  # history, holdout, sample, faulty files, results
│   └── scripts/
│       ├── inject_faults.py   # Makes faulty_sample.csv + labels
│       └── evaluate.py        # Isolation Forest vs z-score
└── aws_anomaly/               # Django project
    ├── manage.py
    ├── requirements.txt
    └── detector/
        ├── models.py          # Reading table
        ├── views.py           # Upload + dashboard
        ├── urls.py
        └── templates/detector/
```

---

## 🚀 Getting started

### 1. Clone and set up virtual environment

```bash
git clone <your-repo-url>
cd weather-anomaly
python -m venv venv
```

Activate the virtual environment:
- **Windows (PowerShell):** `.\venv\Scripts\Activate.ps1`
- **Windows (cmd):** `venv\Scripts\activate.bat`
- **macOS / Linux:** `source venv/bin/activate`

Install dependencies:
```bash
pip install -r ml/requirements.txt fastapi uvicorn matplotlib django
```

> [!NOTE]
> Everyone on the team must use the **same scikit-learn version** (`scikit-learn==1.9.1`). A model saved with one version can warn or break on another. Check yours with `pip show scikit-learn`.

---

### 2. Build the dataset & train the model (from inside `ml/`)

```bash
cd ml

# Step 1: Download 3 years of hourly weather data from Open-Meteo
python fetch_data.py

# Step 2: Clean data, filter out impossible values, split history/holdout
python clean.py

# Step 3: Fit StandardScaler and IsolationForest, save ml_model.joblib
python ml.py

# Step 4: Generate sample test file (first 200 rows of holdout)
python make_sample.py

# Step 5: Run automated test suite to verify the pipeline
python verify.py
```

---

### 3. Run the evaluation benchmarks (optional)

```bash
python scripts/inject_faults.py
python scripts/evaluate.py
```

---

### 4. Start the ML API Microservice

Run the standalone FastAPI server on port `8001`:

```bash
# From inside the ml/ directory:
uvicorn api:app --host 0.0.0.0 --port 8001 --reload
```

The ML service will be online at:
- **Service root:** `http://127.0.0.1:8001/`
- **Interactive API Documentation:** `http://127.0.0.1:8001/docs`
- **Prediction Endpoint:** `http://127.0.0.1:8001/predict`

---

### 5. Start the Django website (in a separate terminal)

```bash
cd aws_anomaly
python manage.py migrate
python manage.py runserver
```

Open `http://127.0.0.1:8000`, upload `sample.csv`, then `faulty_sample.csv`.

---

## 🔌 ML Microservice & API Testing

The ML model can be run as an independent microservice separate from your web application. This allows your Django app (or any mobile/frontend client) to request real-time anomaly detection via standard HTTP JSON requests.

### Interactive Swagger UI (Browser)

1. Start the API server:
   ```bash
   cd ml
   uvicorn api:app --host 0.0.0.0 --port 8001 --reload
   ```
2. Open your browser and navigate to **`http://127.0.0.1:8001/docs`**.
3. Click on the **`POST /predict`** route $\rightarrow$ **Try it out**.
4. Paste a sample JSON payload and click **Execute** to see the response directly in the browser.

---

### Testing via Command Line / Terminal

#### Test 1: Single Normal Reading (cURL)
```bash
curl -X POST "http://127.0.0.1:8001/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "timestamp": "2026-11-20 12:00:00",
       "temp": 26.5,
       "humidity": 65.0,
       "pressure": 1012.3,
       "wind": 12.4,
       "rain": 0.0
     }'
```

**Expected Response (`is_anomaly: false`):**
```json
{
  "timestamp": "2026-11-20 12:00:00",
  "temp": 26.5,
  "humidity": 65.0,
  "pressure": 1012.3,
  "wind": 12.4,
  "rain": 0.0,
  "is_anomaly": false,
  "score": 0.41833169532328607
}
```

---

#### Test 2: Injected Fault / Broken Sensor Reading (cURL)
```bash
curl -X POST "http://127.0.0.1:8001/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "timestamp": "2026-11-20 12:00:00",
       "temp": 85.0,
       "humidity": 150.0,
       "pressure": 0.0,
       "wind": 1.4,
       "rain": 0.0
     }'
```

**Expected Response (`is_anomaly: true`, high score):**
```json
{
  "timestamp": "2026-11-20 12:00:00",
  "temp": 85.0,
  "humidity": 150.0,
  "pressure": 0.0,
  "wind": 1.4,
  "rain": 0.0,
  "is_anomaly": true,
  "score": 0.6038596226644785
}
```

---

#### Test 3: PowerShell (Windows)
```powershell
$body = @{
    timestamp = "2026-11-20 12:00:00"
    temp = 85.0
    humidity = 150.0
    pressure = 0.0
    wind = 1.4
    rain = 0.0
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://127.0.0.1:8001/predict" -Method Post -ContentType "application/json" -Body $body
```

---

#### Test 4: Batch of Readings (Multiple Sensors / Timeseries)
You can pass an array of readings to check multiple records simultaneously:

```bash
curl -X POST "http://127.0.0.1:8001/predict" \
     -H "Content-Type: application/json" \
     -d '[
       {
         "timestamp": "2026-11-20 12:00:00",
         "temp": 26.5,
         "humidity": 65.0,
         "pressure": 1012.3,
         "wind": 12.4,
         "rain": 0.0
       },
       {
         "timestamp": "2026-11-20 13:00:00",
         "temp": 85.0,
         "humidity": 150.0,
         "pressure": 0.0,
         "wind": 1.4,
         "rain": 0.0
       }
     ]'
```

---

### Calling the ML API from Django

In your Django app (e.g. `detector/services.py` or `views.py`), query the microservice with the standard `requests` library:

```python
import requests

ML_API_URL = "http://127.0.0.1:8001/predict"

def check_weather_anomalies(readings_list: list[dict]) -> list[dict]:
    """
    Sends weather readings to the ML microservice and returns predictions.
    Each item in readings_list must contain: timestamp, temp, humidity, pressure, wind, rain.
    """
    try:
        response = requests.post(ML_API_URL, json=readings_list, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as err:
        print(f"Error calling ML microservice: {err}")
        return []
```

---

## 👥 Team

| Person | Owns | Hands off |
|---|---|---|
| **Kunal** | Data, cleaning, model training, `predict()` | model file + holdout + sample file |
| **Ashish** | Fault injection, z-score baseline, evaluation, charts | `faulty_sample.csv` + results |
| **Sahasra** | Django app, upload page, dashboard | working site + requirements file |

```mermaid
gantt
    title Who does what (rough hours)
    dateFormat  HH:mm
    axisFormat  %H:%M
    section Kunal
    Setup + data + clean     :k1, 00:00, 75m
    Train + sanity check     :k2, after k1, 65m
    Handoff                  :k3, after k2, 15m
    section Ashish
    Wait for model           :crit, a0, 00:00, 155m
    Verify + inject faults   :a1, after a0, 45m
    Evaluate + charts        :a2, after a1, 65m
    section Sahasra
    Setup + DB               :s1, 00:00, 35m
    Upload + dashboard       :s2, after s1, 150m
    Final testing            :s3, after s2, 45m
```

---

## 🛠️ Troubleshooting

| Problem | Fix |
|---|---|
| `FileNotFoundError` for a data file | You ran a script from the wrong folder. Run ML scripts from inside `ml/` |
| Warning about scikit-learn versions when loading the model | Install the same version Kunal used, or retrain with `python ml.py` |
| Missing columns error on upload | CSV needs exactly: `timestamp, temp, humidity, pressure, wind, rain` |
| Timezone warnings in Django | Make sure `USE_TZ = False` in `settings.py` |
| Dashboard chart is empty | Upload a file first. The dashboard shows whatever the last upload saved |

---

## 🔮 Ideas for later

- Per-sensor anomaly explanations (which column pushed the score up)
- Keep upload history instead of replacing it
- Live data feed instead of CSV uploads
- Tune `contamination` from the dashboard
- Try other detectors (One-Class SVM, autoencoder) and add them to the comparison

---

<div align="center">

Built by Kunal, Ashish and Sahasra.

If a sensor says it's 85°C in Delhi in January, we'll notice.

</div>
