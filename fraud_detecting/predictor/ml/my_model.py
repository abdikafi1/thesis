import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb

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

# Encode target if not already binary (0/1)
if y.dtype == 'object' or y.nunique() != 2:
    y_le = LabelEncoder().fit(y)
    y = y_le.transform(y)
    joblib.dump(y_le, 'label_FraudFound_P.pkl')

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)

xgb_model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=12,
    learning_rate=0.1,
    scale_pos_weight=(sum(y_res==0)/sum(y_res==1)),
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(X_res, y_res)

# Save model and label encoders
joblib.dump(xgb_model, 'xgb_model.pkl')
for col in X.columns:
    if col in categorical_fields:
        le = label_encoders[col]
        joblib.dump(le, f'label_{col}.pkl')

# Evaluate model with custom threshold to catch more fraud cases
y_proba = xgb_model.predict_proba(X_test)[:, 1]
threshold = 0.3  # Lower threshold for higher recall
y_pred = (y_proba > threshold).astype(int)
print('Classification Report (threshold=0.3):')
print(classification_report(y_test, y_pred))

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy (threshold=0.3): {accuracy:.2f}')






