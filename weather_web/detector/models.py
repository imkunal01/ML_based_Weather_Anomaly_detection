from django.db import models


class WeatherReading(models.Model):
    timestamp = models.CharField(max_length=64)
    temp = models.FloatField()
    humidity = models.FloatField()
    pressure = models.FloatField()
    wind = models.FloatField()
    rain = models.FloatField()
    is_anomaly = models.BooleanField(default=False)
    score = models.FloatField(default=0.0)
    model_type = models.CharField(max_length=32, default="Isolation Forest")
    batch_name = models.CharField(max_length=128, default="default")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['id']

    def __str__(self):
        status = "Anomaly" if self.is_anomaly else "Normal"
        return f"[{self.timestamp}] Temp: {self.temp}°C | {status} (Score: {self.score:.3f})"
