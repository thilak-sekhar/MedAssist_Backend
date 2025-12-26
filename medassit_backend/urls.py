from django.contrib import admin
from django.urls import path
from medassist_backend_app.views import ChatView

urlpatterns = [
    path('admin/', admin.site.urls),
    path('chat/', ChatView.as_view(), name='ChatView'),
]