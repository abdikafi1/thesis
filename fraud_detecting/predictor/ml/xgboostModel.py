# Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Load Dataset
data = pd.read_csv('insurance fraud detection Dataset.csv')

# Selecting more features for better performance
selected_features = [
    'PolicyType', 'VehiclePrice', 'AgeOfVehicle', 'PastNumberOfClaims',
    'Days_Policy_Accident', 'PoliceReportFiled', 'WitnessPresent',
    'NumberOfSuppliments', 'AddressChange_Claim',
    # Add more features if available in your dataset
    'RepNumber', 'Age', 'Month', 'MonthClaimed', 'DayOfWeek', 'Make',
    'DayOfWeekClaimed', 'WeekOfMonth', 'WeekOfMonthClaimed', 'DriverRating',
    'AgeOfPolicyHolder'
]
# Only keep features that exist in the dataset
selected_features = [f for f in selected_features if f in data.columns]

# Target variable
y = data['FraudFound_P']
X = data[selected_features]

# Encoding categorical variables
X_encoded = X.apply(lambda col: LabelEncoder().fit_transform(col) if col.dtype == 'object' else col)

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Handling Class Imbalance with class_weight
scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]

# Model Training with tuned XGBoost
xgb_model = XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    scale_pos_weight=scale_pos_weight,
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    min_child_weight=1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
xgb_model.fit(X_train, y_train)

# Model Evaluation with custom threshold for balanced detection
probs = xgb_model.predict_proba(X_test)[:, 1]
threshold = 0.35  # Lower threshold to catch more fraud cases
custom_pred = (probs > threshold).astype(int)

accuracy = accuracy_score(y_test, custom_pred)
report = classification_report(y_test, custom_pred)

print(f'Accuracy (custom threshold): {accuracy:.2f}')
print('Classification Report (custom threshold):\n', report)
