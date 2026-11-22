from django.shortcuts import render, redirect
from django.contrib import messages
from django.http import JsonResponse, HttpResponse
from django.db.models import Avg, Max, Min, Count
from pathlib import Path
import json
import pandas as pd
import datetime

from .models import WeatherReading
from .services import run_prediction, REQUIRED_COLUMNS, FEATURES

REPO_DIR = Path(__file__).resolve().parent.parent.parent


def dashboard_view(request):
    """Main interactive dashboard displaying weather analytics, anomaly scores, and charts."""
    readings = WeatherReading.objects.all()
    total_count = readings.count()
    anomaly_count = readings.filter(is_anomaly=True).count()
    normal_count = total_count - anomaly_count
    anomaly_percent = round((anomaly_count / total_count * 100), 2) if total_count > 0 else 0.0

    avg_temp = readings.aggregate(Avg('temp'))['temp__avg']
    avg_humidity = readings.aggregate(Avg('humidity'))['humidity__avg']
    avg_pressure = readings.aggregate(Avg('pressure'))['pressure__avg']
    avg_wind = readings.aggregate(Avg('wind'))['wind__avg']

    # Top anomalies sorted by anomaly score descending
    top_anomalies = readings.filter(is_anomaly=True).order_by('-score')[:25]
    
    # Recent readings for the table preview
    recent_readings = readings[:30]

    # Chart data series
    chart_data = {
        'timestamps': [r.timestamp for r in readings],
        'temps': [round(r.temp, 2) for r in readings],
        'humidities': [round(r.humidity, 2) for r in readings],
        'pressures': [round(r.pressure, 2) for r in readings],
        'winds': [round(r.wind, 2) for r in readings],
        'is_anomalies': [bool(r.is_anomaly) for r in readings],
        'scores': [round(r.score, 4) for r in readings],
    }

    # Current model in use
    current_model = readings.first().model_type if readings.exists() else "Isolation Forest"

    context = {
        'total_count': total_count,
        'anomaly_count': anomaly_count,
        'normal_count': normal_count,
        'anomaly_percent': anomaly_percent,
        'avg_temp': round(avg_temp, 1) if avg_temp is not None else 0,
        'avg_humidity': round(avg_humidity, 1) if avg_humidity is not None else 0,
        'avg_pressure': round(avg_pressure, 1) if avg_pressure is not None else 0,
        'avg_wind': round(avg_wind, 1) if avg_wind is not None else 0,
        'top_anomalies': top_anomalies,
        'recent_readings': recent_readings,
        'chart_data_json': json.dumps(chart_data),
        'current_model': current_model,
    }
    return render(request, 'detector/dashboard.html', context)


def upload_view(request):
    """Handles CSV dataset file uploads or one-click sample loading."""
    if request.method == 'POST':
        model_choice = request.POST.get('model_choice', 'iforest')
        action = request.POST.get('action', 'upload')

        df = None
        source_name = "Uploaded Dataset"

        if action == 'load_sample':
            sample_path = REPO_DIR / 'ml' / 'data' / 'sample.csv'
            if not sample_path.exists():
                messages.error(request, "Sample dataset file not found in ml/data/sample.csv")
                return redirect('upload')
            df = pd.read_csv(sample_path)
            source_name = "Standard Sample (ml/data/sample.csv)"

        elif action == 'load_faulty':
            # Create on-the-fly injected faulty sample from holdout if not saved
            sample_path = REPO_DIR / 'ml' / 'data' / 'sample.csv'
            if not sample_path.exists():
                messages.error(request, "Holdout data not found.")
                return redirect('upload')
            df = pd.read_csv(sample_path).copy()
            # Deliberately inject anomalies into 15 records
            if len(df) >= 30:
                df.loc[5, 'temp'] = 84.5  # Extreme heat spike
                df.loc[12, 'pressure'] = 0.0  # Sensor failure
                df.loc[22, 'humidity'] = 145.0  # Impossible humidity
                df.loc[35, 'temp'] = -45.0  # Impossible freeze
                df.loc[48, 'wind'] = 195.0  # Extreme hurricane wind
                df.loc[60, 'temp'] = 79.0
                df.loc[75, 'pressure'] = 520.0
                df.loc[90, 'humidity'] = 160.0
                df.loc[110, 'temp'] = 88.0
                df.loc[125, 'wind'] = 180.0
            source_name = "Sensor Fault Simulation (with Injected Outliers)"

        else:
            if 'csv_file' not in request.FILES:
                messages.error(request, "Please choose a CSV file to upload.")
                return redirect('upload')
            
            uploaded_file = request.FILES['csv_file']
            if not uploaded_file.name.endswith('.csv'):
                messages.error(request, f"Invalid format: '{uploaded_file.name}'. Only valid .csv files are supported.")
                return redirect('upload')

            try:
                df = pd.read_csv(uploaded_file)
                source_name = uploaded_file.name
            except Exception as e:
                messages.error(request, f"Could not read CSV file: {str(e)}")
                return redirect('upload')

        # Validation of uploaded DataFrame
        if df is None or df.empty:
            messages.error(request, "The uploaded dataset is empty.")
            return redirect('upload')

        missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing_cols:
            messages.error(request, f"Missing required columns in dataset: {', '.join(missing_cols)}. Expected: {', '.join(REQUIRED_COLUMNS)}")
            return redirect('upload')

        # Clean NaNs or invalid types
        original_len = len(df)
        df = df.dropna(subset=REQUIRED_COLUMNS)
        if len(df) == 0:
            messages.error(request, "No usable rows left after dropping empty values.")
            return redirect('upload')

        for f in FEATURES:
            df[f] = pd.to_numeric(df[f], errors='coerce')
        df = df.dropna(subset=FEATURES)

        # Run Prediction using selected ML model
        model_display_name = "LSTM Sequence Detector" if model_choice == "lstm" else "Isolation Forest"
        try:
            result_df = run_prediction(df, model_choice=model_choice)
        except Exception as e:
            messages.error(request, f"Error running ML Anomaly Detection model: {str(e)}")
            return redirect('upload')

        # Clear previous session readings and bulk store the new readings
        WeatherReading.objects.all().delete()

        reading_objects = [
            WeatherReading(
                timestamp=str(row['timestamp']),
                temp=float(row['temp']),
                humidity=float(row['humidity']),
                pressure=float(row['pressure']),
                wind=float(row['wind']),
                rain=float(row['rain']),
                is_anomaly=bool(row['is_anomaly']),
                score=float(row['score']),
                model_type=model_display_name,
                batch_name=source_name,
            )
            for _, row in result_df.iterrows()
        ]
        WeatherReading.objects.bulk_create(reading_objects)

        anomalies_detected = sum(1 for obj in reading_objects if obj.is_anomaly)
        messages.success(
            request, 
            f"Successfully analyzed {len(reading_objects)} records with {model_display_name}! Detected {anomalies_detected} anomalies."
        )
        return redirect('dashboard')

    return render(request, 'detector/upload.html')


def single_predict_view(request):
    """Allows manual single-reading testing for quick live demonstration during exams/presentations."""
    prediction_result = None
    if request.method == 'POST':
        try:
            timestamp = request.POST.get('timestamp') or datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            temp = float(request.POST.get('temp', 0))
            humidity = float(request.POST.get('humidity', 0))
            pressure = float(request.POST.get('pressure', 0))
            wind = float(request.POST.get('wind', 0))
            rain = float(request.POST.get('rain', 0))
            model_choice = request.POST.get('model_choice', 'iforest')

            sample_dict = {
                'timestamp': [timestamp],
                'temp': [temp],
                'humidity': [humidity],
                'pressure': [pressure],
                'wind': [wind],
                'rain': [rain]
            }
            df = pd.DataFrame(sample_dict)
            res = run_prediction(df, model_choice=model_choice)
            
            is_anomaly = bool(res['is_anomaly'].iloc[0])
            score = float(res['score'].iloc[0])
            model_name = "LSTM Sequence" if model_choice == "lstm" else "Isolation Forest"

            prediction_result = {
                'timestamp': timestamp,
                'temp': temp,
                'humidity': humidity,
                'pressure': pressure,
                'wind': wind,
                'rain': rain,
                'is_anomaly': is_anomaly,
                'score': round(score, 4),
                'model': model_name
            }
        except Exception as e:
            messages.error(request, f"Prediction error: {str(e)}")

    return render(request, 'detector/single_predict.html', {'result': prediction_result})


def live_api_view(request):
    """JSON API endpoint mimicking microservice for frontend or asynchronous requests."""
    if request.method == 'POST':
        try:
            body = json.loads(request.body)
            rows = body if isinstance(body, list) else [body]
            df = pd.DataFrame(rows)
            res = run_prediction(df)
            return JsonResponse(res.to_dict(orient='records'), safe=False)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)
    return JsonResponse({'status': 'online', 'endpoint': '/api/detect/'})


def about_view(request):
    """Informational project overview explaining methodology, architecture, and technology stack."""
    return render(request, 'detector/about.html')
