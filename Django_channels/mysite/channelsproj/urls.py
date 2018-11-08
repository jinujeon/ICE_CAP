from django.contrib import admin
from django.urls import path
from django.conf.urls import url,include
from notifier.views import HomeView
from django.contrib.auth import views as auth_views

urlpatterns = [
    path('monitor/', HomeView.as_view()),
    #path('home/alert.html', AlertView.as_view()),
    path('admin/', admin.site.urls),
    url(r'^', include('notifier.urls')),
    path('login/',auth_views.LoginView.as_view()),
    path('logout/', auth_views.LogoutView.as_view(), name='logout'),
    #path('notifier/', include('notifier.urls')),
]
