from django.contrib import admin
from django.urls import path
from django.conf.urls import url,include
from notifier.views import HomeView

urlpatterns = [
    path('', HomeView.as_view()),
    #path('home/alert.html', AlertView.as_view()),
    path('admin/', admin.site.urls),
    url(r'^', include('notifier.urls')), 
    #path('notifier/', include('notifier.urls')),
]
