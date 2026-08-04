import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Load data
df = pd.read_csv('churn_data.csv')

# Drop customer ID
df.drop('customerID', axis=1, inplace=True)

# Replace blanks with NaN and convert to numeric
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

# Encode binary columns
binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
for col in binary_cols:
    df[col] = df[col].map({'Yes': 1, 'No': 0})

# One-hot encode remaining categorical features
df = pd.get_dummies(df)

# Features & Target
X = df.drop('Churn', axis=1)
y = df['Churn']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
lr_model = LogisticRegression(max_iter=1000)
rf_model = RandomForestClassifier(n_estimators=100)

lr_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

# Evaluate
print("Logistic Regression Report:\n", classification_report(y_test, lr_model.predict(X_test)))
print("Random Forest Report:\n", classification_report(y_test, rf_model.predict(X_test)))

# Save best model
joblib.dump(rf_model, 'churn_model.pkl')
joblib.dump(X.columns, 'feature_columns.pkl')
