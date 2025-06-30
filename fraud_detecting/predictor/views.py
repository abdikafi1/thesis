from django.contrib.auth.decorators import login_required
from django.shortcuts import get_object_or_404, redirect
from django.views.decorators.http import require_POST
from django.shortcuts import render
from django.http import JsonResponse
from django.contrib.admin.views.decorators import staff_member_required
from django.db.models import Count
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth.models import User
from .forms import FraudPredictionForm, CustomLoginForm, CustomSignUpForm
from .models import Prediction
from .utils import preprocess_input, load_model, FEATURES
from django.views.decorators.csrf import csrf_exempt
from django.core.paginator import Paginator
from django.urls import reverse

import numpy as np

USER_LABELS = {
    'PolicyType': 'Policy Type',
    'VehiclePrice': 'Vehicle Price',
    'AgeOfVehicle': 'Age of Vehicle',
    'PastNumberOfClaims': 'Past Number of Claims',
    'Days_Policy_Accident': 'Days Policy to Accident',
    'PoliceReportFiled': 'Police Report Filed',
    'WitnessPresent': 'Witness Present',
    'NumberOfSuppliments': 'Number of Supplements',
    'AddressChange_Claim': 'Address Change at Claim'
}

@require_POST
@login_required
def delete_prediction(request, pk):
    prediction = get_object_or_404(Prediction, pk=pk, user=request.user)
    prediction.delete()
    messages.success(request, 'Prediction deleted successfully.')
    return redirect('history')

@require_POST
@login_required
def edit_note(request, pk):
    prediction = get_object_or_404(Prediction, pk=pk, user=request.user)
    note = request.POST.get('note', '')
    prediction.note = note[:255]
    prediction.save()
    messages.success(request, 'Note updated successfully.')
    return redirect('history')

@login_required
def history(request):
    predictions = Prediction.objects.filter(user=request.user).order_by('-created_at')
    result_filter = request.GET.get('result', '')
    search_query = request.GET.get('search', '')
    if result_filter in ['Fraud', 'Not Fraud']:
        predictions = predictions.filter(result=result_filter)
    if search_query:
        predictions = predictions.filter(note__icontains=search_query)
    paginator = Paginator(predictions, 10)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    return render(request, 'fraud_detection/history.html', {
        'predictions': page_obj.object_list,
        'page_obj': page_obj,
        'result_filter': result_filter,
        'search_query': search_query,
        'user_labels': USER_LABELS,
    })

@staff_member_required
def analytics(request):
    from .models import Prediction
    total = Prediction.objects.count()
    fraud_count = Prediction.objects.filter(result='Fraud').count()
    not_fraud_count = Prediction.objects.filter(result='Not Fraud').count()
    top_users = (
        Prediction.objects.values('user__username')
        .annotate(count=Count('id'))
        .order_by('-count')[:5]
    )
    recent = Prediction.objects.select_related('user').order_by('-created_at')[:10]
    # Bulk delete
    if request.method == 'POST' and 'delete_ids' in request.POST:
        ids = request.POST.getlist('delete_ids')
        Prediction.objects.filter(id__in=ids).delete()
        messages.success(request, 'Selected predictions deleted.')
        return redirect('analytics')
    # Chart data
    chart_data = {
        'labels': ['Fraud', 'Not Fraud'],
        'data': [fraud_count, not_fraud_count],
    }
    return render(request, 'fraud_detection/analytics.html', {
        'total': total,
        'fraud_count': fraud_count,
        'not_fraud_count': not_fraud_count,
        'top_users': top_users,
        'recent': recent,
        'chart_data': chart_data,
    })

# Public landing page
def landing(request):
    return render(request, 'fraud_detection/landing.html')

# Dashboard (login required)
@login_required
def dashboard(request):
    form = FraudPredictionForm()
    errors = {}
    if request.method == 'POST' and request.GET.get('predict'):
        form = FraudPredictionForm(request.POST)
        if form.is_valid():
            X, errors = preprocess_input(form.cleaned_data)
            if not errors:
                model = load_model()
                proba = model.predict_proba(X)[0]
                result = 'Fraud' if np.argmax(proba) == 1 else 'Not Fraud'
                probability = f"{100 * np.max(proba):.2f}%"
                pred = Prediction.objects.create(
                    user=request.user,
                    input_data=form.cleaned_data,
                    result=result,
                    probability=probability
                )
                request.session['last_prediction_id'] = pred.id
                messages.success(request, f'Prediction complete: {result} ({probability})')
                return redirect('result')
        else:
            errors = form.errors
            messages.error(request, 'Please correct the errors below.')
    return render(request, 'fraud_detection/dashboard.html', {'form': form, 'errors': errors})

# Home now redirects to landing
def home(request):
    return redirect('landing')

def result(request):
    pred_id = request.session.get('last_prediction_id')
    if not pred_id:
        messages.error(request, 'No prediction found.')
        return redirect('dashboard')
    pred = get_object_or_404(Prediction, pk=pred_id)
    return render(request, 'fraud_detection/result.html', {
        'result': pred.result,
        'probability': pred.probability,
        'input': pred.input_data,
        'user_labels': USER_LABELS,
    })

def login_view(request):
    if request.user.is_authenticated:
        return redirect('dashboard')
    
    next_page = request.GET.get('next', '')
    
    if request.method == 'POST':
        form = CustomLoginForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            messages.success(request, f'Welcome back, {user.username}!')
            
            # Redirect to next page if provided and valid, otherwise to dashboard
            if next_page and next_page != '/accounts/profile/':
                return redirect(next_page)
            return redirect('dashboard')
    else:
        form = CustomLoginForm(request)
    
    return render(request, 'fraud_detection/login.html', {
        'form': form,
        'next': next_page
    })

def logout_view(request):
    logout(request)
    messages.info(request, 'You have been logged out.')
    return redirect('login')

def signup_view(request):
    if request.user.is_authenticated:
        return redirect('dashboard')
        
    next_page = request.GET.get('next', '')
    
    if request.method == 'POST':
        form = CustomSignUpForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, f'Welcome, {user.username}! Your account has been created successfully.')
            
            # Redirect to next page if provided and valid, otherwise to dashboard
            if next_page and next_page != '/accounts/profile/':
                return redirect(next_page)
            return redirect('dashboard')
    else:
        form = CustomSignUpForm()
    
    return render(request, 'fraud_detection/signup.html', {
        'form': form,
        'next': next_page
    })

def ead(request):
    return render(request, 'fraud_detection/ead.html')

def predict(request):
    errors = {}
    form = FraudPredictionForm()
    if request.method == 'POST':
        form = FraudPredictionForm(request.POST)
        if form.is_valid():
            X, errors = preprocess_input(form.cleaned_data)
            if not errors:
                model = load_model()
                proba = model.predict_proba(X)[0]
                result = 'Fraud' if np.argmax(proba) == 1 else 'Not Fraud'
                probability = f"{100 * np.max(proba):.2f}%"
                pred = Prediction.objects.create(
                    user=request.user if request.user.is_authenticated else None,
                    input_data=form.cleaned_data,
                    result=result,
                    probability=probability
                )
                request.session['last_prediction_id'] = pred.id
                messages.success(request, f'Prediction complete: {result} ({probability})')
                return redirect('result')
        else:
            errors = form.errors
            messages.error(request, 'Please correct the errors below.')
    return render(request, 'fraud_detection/predict.html', {'form': form, 'errors': errors})

def about_view(request):
    return render(request, 'fraud_detection/about.html') 