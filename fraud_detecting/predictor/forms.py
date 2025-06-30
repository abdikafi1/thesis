from django import forms
from .utils import FEATURES, get_categories
from django.contrib.auth.forms import AuthenticationForm, UserCreationForm
from django.contrib.auth import authenticate
from django.contrib.auth.models import User

# Help texts for each field
HELP_TEXTS = {
    'PolicyType': 'Type of insurance policy.',
    'VehiclePrice': 'Price category of the vehicle.',
    'AgeOfVehicle': 'Age of the vehicle in years.',
    'PastNumberOfClaims': 'Number of past claims by the policyholder.',
    'Days_Policy_Accident': 'Days between policy start and accident.',
    'PoliceReportFiled': 'Was a police report filed? (Yes/No)',
    'WitnessPresent': 'Was a witness present? (Yes/No)',
    'NumberOfSuppliments': 'Number of supplementary documents submitted.',
    'AddressChange_Claim': 'Recent address change at time of claim.'
}

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

# FraudPredictionForm: Dynamically creates form fields based on FEATURES and available categories
class FraudPredictionForm(forms.Form):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        categories = get_categories()
        for f in FEATURES:
            label = USER_LABELS.get(f, f)
            if f in categories:
                self.fields[f] = forms.ChoiceField(
                    choices=[(v, v) for v in categories[f]],
                    label=label,
                    help_text=HELP_TEXTS.get(f, '')
                )
            else:
                self.fields[f] = forms.IntegerField(
                    label=label,
                    help_text=HELP_TEXTS.get(f, '')
                )

class CustomLoginForm(AuthenticationForm):
    username = forms.CharField(widget=forms.TextInput(attrs={
        'class': 'input-field',
        'placeholder': 'Email or Username'
    }))
    password = forms.CharField(widget=forms.PasswordInput(attrs={
        'class': 'input-field',
        'placeholder': 'Password'
    }))

    error_messages = {
        'invalid_login': 'Please enter a correct email and password.',
        'inactive': 'This account is inactive.',
    }

    def clean(self):
        username = self.cleaned_data.get('username')
        password = self.cleaned_data.get('password')

        if username and password:
            self.user_cache = authenticate(
                self.request, username=username, password=password
            )
            if self.user_cache is None:
                raise forms.ValidationError(
                    self.error_messages['invalid_login'],
                    code='invalid_login',
                )
            elif not self.user_cache.is_active:
                raise forms.ValidationError(
                    self.error_messages['inactive'],
                    code='inactive',
                )
        return self.cleaned_data

class CustomSignUpForm(UserCreationForm):
    email = forms.EmailField(required=True, widget=forms.EmailInput(attrs={
        'class': 'input-field',
        'placeholder': 'Email'
    }))
    username = forms.CharField(widget=forms.TextInput(attrs={
        'class': 'input-field',
        'placeholder': 'Username'
    }))
    password1 = forms.CharField(widget=forms.PasswordInput(attrs={
        'class': 'input-field',
        'placeholder': 'Password'
    }))
    password2 = forms.CharField(widget=forms.PasswordInput(attrs={
        'class': 'input-field',
        'placeholder': 'Confirm Password'
    }))

    class Meta:
        model = User
        fields = ('username', 'email', 'password1', 'password2')

    def save(self, commit=True):
        user = super().save(commit=False)
        user.email = self.cleaned_data['email']
        if commit:
            user.save()
        return user

