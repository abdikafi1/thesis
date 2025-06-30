from django.urls import path
from django.contrib.auth import views as auth_views
from .forms import CustomLoginForm
from . import views

urlpatterns = [
    path('', views.landing, name='landing'),
    path('landing/', views.landing, name='landing'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('home/', views.home, name='home'),
    path('result/', views.result, name='result'),
    path('history/', views.history, name='history'),
    path('history/edit_note/<int:pk>/', views.edit_note, name='edit_note'),
    path('history/delete/<int:pk>/', views.delete_prediction, name='delete_prediction'),
    path('analytics/', views.analytics, name='analytics'),
    path('logout/', views.logout_view, name='logout'),
    path('login/', auth_views.LoginView.as_view(template_name='fraud_detection/login.html', authentication_form=CustomLoginForm), name='login'),
    path('signup/', views.signup_view, name='signup'),
    path('password_reset/', auth_views.PasswordResetView.as_view(template_name='fraud_detection/password_reset_form.html'), name='password_reset'),
    path('password_reset/done/', auth_views.PasswordResetDoneView.as_view(template_name='fraud_detection/password_reset_done.html'), name='password_reset_done'),
    path('reset/<uidb64>/<token>/', auth_views.PasswordResetConfirmView.as_view(template_name='fraud_detection/password_reset_confirm.html'), name='password_reset_confirm'),
    path('reset/done/', auth_views.PasswordResetCompleteView.as_view(template_name='fraud_detection/password_reset_complete.html'), name='password_reset_complete'),
    path('ead/', views.ead, name='ead'),
    path('predict/', views.predict, name='predict'),
    path('about/', views.about_view, name='about'),
] 