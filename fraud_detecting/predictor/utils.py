import os
import joblib
import numpy as np
from django.conf import settings

# FEATURES: List of all input features expected by the model
FEATURES = [
    'PolicyType',
    'VehiclePrice',
    'AgeOfVehicle',
    'PastNumberOfClaims',
    'Days_Policy_Accident',
    'PoliceReportFiled',
    'WitnessPresent',
    'NumberOfSuppliments',
    'AddressChange_Claim'
]

# ML_DIR: Path to the directory containing model and encoder files
ML_DIR = os.path.join(os.path.dirname(__file__), 'ml')

def load_model():
    # load_model: Loads the trained Random Forest model from disk
    return joblib.load(os.path.join(ML_DIR, 'rf_model.pkl'))

def load_label_encoder(col):
    # load_label_encoder: Loads a label encoder for a given column
    return joblib.load(os.path.join(ML_DIR, f'label_{col}.pkl'))

def get_categories():
    # get_categories: Returns available categories for categorical features using encoders
    categories = {}
    for col in FEATURES:
        encoder_path = os.path.join(ML_DIR, f'label_{col}.pkl')
        if os.path.exists(encoder_path):
            le = load_label_encoder(col)
            categories[col] = list(le.classes_)
    return categories

def preprocess_input(form_data):
    # preprocess_input: Transforms and validates input data for model prediction
    processed = []
    errors = {}
    for col in FEATURES:
        value = form_data.get(col)
        encoder_path = os.path.join(ML_DIR, f'label_{col}.pkl')
        if os.path.exists(encoder_path):
            le = load_label_encoder(col)
            if value not in le.classes_:
                errors[col] = f"'{value}' is not a valid option."
                continue
            value = le.transform([value])[0]
        else:
            # Numeric check
            try:
                value = int(value)
            except:
                errors[col] = "Enter a valid number."
                continue
        processed.append(value)
    return (np.array(processed).reshape(1, -1), errors)
