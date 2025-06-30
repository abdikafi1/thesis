# Machine Learning Models and Utilities

This folder contains machine learning models, label encoders, and utility scripts used for insurance fraud detection.

## Contents

- **insurance fraud detection Dataset.csv**: The dataset used for training and testing the models.
- **rf_model.pkl**: Trained Random Forest model for fraud detection.
- **xgboostModel.py**: Script for the XGBoost model implementation.
- **my_model.py**: Custom model or utility functions for model handling.
- **label_*.pkl**: Label encoders for categorical features (e.g., AgeOfPolicyHolder, Make, Month, etc.).

## Usage

- The `.pkl` files are pre-trained models and label encoders. Load them in your Python scripts using `pickle` or `joblib`.
- Use the encoders to transform categorical data before making predictions with the models.
- Refer to the model scripts (e.g., `xgboostModel.py`, `my_model.py`) for details on model usage and prediction functions.

## Example: Loading a Model

```python
import pickle

with open('rf_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Use `model.predict(X)` to make predictions
```

## Example: Loading a Label Encoder

```python
import pickle

with open('label_Make.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Use `label_encoder.transform(['Toyota', 'Honda'])`
```

## Notes
- Ensure all required `.pkl` files are present before running prediction scripts.
- For more details, see the main project documentation or the scripts in this folder. 