from django.conf.urls import url
from . import views
from django.urls import path
from django.contrib.auth import views as auth_views
urlpatterns = [
    url(r'^$', views.change_stat),
    path("home/change_stat", views.change_stat),
    path('monitor/home/alert.html',views.alert, name = 'alert'),
    path('audios/note.mp3',views.alert),
    path('login/',auth_views.LoginView.as_view(), name='login'),
    path('monitor/',views.livefe, name="livefe"),

]
