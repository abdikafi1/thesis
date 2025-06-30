import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import joblib
import lightgbm as lgb

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
    X.loc[:, col] = le.transform(X[col])
    label_encoders[col] = le
    print(f'{col} categories:', list(le.classes_))
    joblib.dump(le, f'label_{col}.pkl')

# Ensure all columns are numeric for LightGBM
X = X.apply(pd.to_numeric)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate LightGBM (no SMOTE)
lgbm_model = lgb.LGBMClassifier(is_unbalance=True, n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
lgbm_model.fit(X_train, y_train)

y_pred_lgbm = lgbm_model.predict(X_test)
print('LightGBM Classification Report:')
print(classification_report(y_test, y_pred_lgbm))

# Save LightGBM model
joblib.dump(lgbm_model, 'lgbm_model.pkl') 