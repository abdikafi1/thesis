import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

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

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate Logistic Regression (no SMOTE)
logreg_model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
logreg_model.fit(X_train, y_train)

y_pred_logreg = logreg_model.predict(X_test)
print('Logistic Regression Classification Report:')
print(classification_report(y_test, y_pred_logreg))

# Save Logistic Regression model
joblib.dump(logreg_model, 'logreg_model.pkl') 