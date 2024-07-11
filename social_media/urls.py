from django.urls import path
from . import views

# URLConf
urlpatterns = [
    path('', views.home, name="pad"),
    path('analysis/', views.analysis, name="pad-analysis"),
    path('entries/', views.entries, name="pad-entries"),
    path('analysis-process/', views.analysis_process, name="pad-analysis-process"),
    path('analysis-ready/', views.analysis_ready, name="pad-analysis-ready"),
    path('summary', views.summary, name="pad-summary"),
    path('analysis/<int:analysis_id>/delete/', views.delete_analysis, name='delete_analysis'),
    path('analysis/<int:analysis_id>/edit/', views.edit_analysis, name='edit_analysis'),
    path('analysis/<int:analysis_id>/summary/', views.analysis_summary, name='analysis_summary'),
]
