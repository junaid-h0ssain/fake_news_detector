from django.urls import path
from . import views

urlpatterns = [
    path('', views.fake_news_detector, name='detector'),
]