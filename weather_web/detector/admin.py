from django.contrib import admin
from .models import WeatherReading


@admin.register(WeatherReading)
class WeatherReadingAdmin(admin.ModelAdmin):
    list_display = ('timestamp', 'temp', 'humidity', 'pressure', 'wind', 'rain', 'is_anomaly', 'score', 'model_type')
    list_filter = ('is_anomaly', 'model_type')
    search_fields = ('timestamp',)
