from django.apps import AppConfig
from django.db.models.signals import post_migrate
import sys

class DetectorConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'detector'

    def ready(self):
        # Auto-run migration checks when starting the app
        if 'runserver' in sys.argv or 'gunicorn' in sys.argv[0]:
            from django.core.management import call_command
            try:
                call_command('migrate', interactive=False)
            except Exception as e:
                print(f"Auto-migration notice: {e}")
