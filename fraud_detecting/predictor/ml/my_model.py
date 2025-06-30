import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, accuracy_score

# Define features (must match backend/frontend)
selected_features = [
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

data = pd.read_csv('insurance fraud detection Dataset.csv')
y = data['FraudFound_P']
X = data[selected_features]

# Detect categorical fields
categorical_fields = [col for col in X.columns if X[col].dtype == 'object']
print('Categorical fields:', categorical_fields)

# Encode categorical fields
label_encoders = {}
for col in categorical_fields:
    le = LabelEncoder().fit(X[col])
    X[col] = le.transform(X[col])
    label_encoders[col] = le
    print(f'{col} categories:', list(le.classes_))
    joblib.dump(le, f'label_{col}.pkl')

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf_model = RandomForestClassifier(
    class_weight='balanced',
    n_estimators=300,      # Increased number of trees
    max_depth=12,          # Increased depth
    min_samples_split=4,   # Minimum samples to split
    min_samples_leaf=2,    # Minimum samples per leaf
    random_state=42,
    n_jobs=-1
)

sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)
rf_model.fit(X_res, y_res)

# Save model
# Save model and label encoders
joblib.dump(rf_model, 'rf_model.pkl')
for col in X.columns:
    if X[col].dtype == 'object':
        le = LabelEncoder().fit(X[col])
        joblib.dump(le, f'label_{col}.pkl')

# Evaluate model with custom threshold to catch more fraud cases
y_proba = rf_model.predict_proba(X_test)[:, 1]
threshold = 0.35  # Slightly higher threshold for better accuracy
y_pred = (y_proba > threshold).astype(int)
print('Classification Report (threshold=0.35):')
print(classification_report(y_test, y_pred))

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy (threshold=0.35): {accuracy:.2f}')

# # Libraries
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# from imblearn.over_sampling import SMOTE

# # Load Dataset
# data = pd.read_csv('insurance fraud detection Dataset.csv')

# # Selecting top 14 important features (same as before for comparison)
# selected_features = [
#     'RepNumber', 'Age', 'Month', 'MonthClaimed', 'DayOfWeek', 'Make',
#     'DayOfWeekClaimed', 'WeekOfMonth', 'WeekOfMonthClaimed', 'DriverRating',
#     'AgeOfVehicle', 'AgeOfPolicyHolder', 'PastNumberOfClaims', 'NumberOfSuppliments'
# ]

# # Target variable
# y = data['FraudFound_P']
# X = data[selected_features]

# # Encoding categorical variables
# X_encoded = X.apply(lambda col: LabelEncoder().fit_transform(col) if col.dtype == 'object' else col)

# # Splitting data
# X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# # Use SMOTE to balance the training data
# sm = SMOTE(random_state=42)
# X_res, y_res = sm.fit_resample(X_train, y_train)

# # Model Training with Random Forest
# rf_model = RandomForestClassifier(
#     class_weight='balanced',  # Automatically adjusts weights inverse of class frequencies
#     n_estimators=100,
#     max_depth=10,
#     random_state=42,
#     n_jobs=-1  # Use all available cores
# )
# rf_model.fit(X_res, y_res)

# # Model Evaluation with custom threshold for fraud detection
# y_proba = rf_model.predict_proba(X_test)[:, 1]
# threshold = 0.3  # Lower threshold to catch more frauds
# y_pred = (y_proba > threshold).astype(int)
# accuracy = accuracy_score(y_test, y_pred)
# report = classification_report(y_test, y_pred)
# conf_matrix = confusion_matrix(y_test, y_pred)

# print(f'Accuracy (threshold=0.3): {accuracy:.2f}')
# print('Classification Report (threshold=0.3):\n', report)
# print('Confusion Matrix:\n', conf_matrix)






