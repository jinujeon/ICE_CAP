from django.conf.urls import url
from . import views
from django.urls import path

urlpatterns = [
    url(r'^$', views.change_stat),
    path("home/change_stat", views.change_stat),
    path('home/alert.html',views.alert, name = 'alert'),
    path('audios/note.mp3',views.alert),

]
