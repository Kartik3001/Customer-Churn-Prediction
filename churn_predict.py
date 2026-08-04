import joblib
import pandas as pd
import numpy as np

# Load model and features
model = joblib.load('churn_model.pkl')
columns = joblib.load('feature_columns.pkl')

# Example input (must match training features)
sample = {
    'gender_Female': 1,
    'gender_Male': 0,
    'SeniorCitizen': 0,
    'tenure': 5,
    'MonthlyCharges': 75.5,
    'TotalCharges': 375.0,
    'Partner': 1,
    'Dependents': 0,
    'PhoneService': 1,
    'PaperlessBilling': 1,
    # Include all necessary one-hot encoded service/contract/payment fields...
    # For simplicity, you can generate it from training data
}

# Fill missing columns with 0
X_input = pd.DataFrame([sample])
X_input = X_input.reindex(columns=columns, fill_value=0)

# Predict
pred = model.predict(X_input)[0]
print("Prediction:", "Churn" if pred == 1 else "Not Churn")
