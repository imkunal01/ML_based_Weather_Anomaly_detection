from django.urls import path
from . import views

urlpatterns = [
    path('', views.dashboard_view, name='dashboard'),
    path('upload/', views.upload_view, name='upload'),
    path('single-test/', views.single_predict_view, name='single_predict'),
    path('about/', views.about_view, name='about'),
    path('api/detect/', views.live_api_view, name='api_detect'),
]
