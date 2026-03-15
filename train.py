import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import pickle

print("Loading data...")
# Load data
df = pd.read_csv('StudentPerformanceFactors.csv')
print(f"Data shape: {df.shape}")

# Fill missing values
df = df.fillna(df.mean(numeric_only=True))
print("Filled missing values")

# Encode categorical variables
categorical_cols = ['Parental_Involvement', 'Access_to_Resources', 'Extracurricular_Activities', 'Motivation_Level', 'Internet_Access', 'Family_Income', 'Teacher_Quality', 'School_Type', 'Peer_Influence', 'Physical_Activity', 'Learning_Disabilities', 'Parental_Education_Level', 'Distance_from_Home', 'Gender']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
print(f"After encoding: {df.shape}")
print("Columns:", df.columns.tolist())

# Features and target
X = df.drop('Exam_Score', axis=1)
y = df['Exam_Score']
print(f"X shape: {X.shape}")

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Split done")

# Polynomial features
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)
print(f"Poly shape: {X_train_poly.shape}")

# Model
model = LinearRegression()
model.fit(X_train_poly, y_train)
print("Model trained")

# Save
with open('poly.pkl', 'wb') as f:
    pickle.dump(poly, f)
print("Saved poly")

with open('poly_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Saved model")

# No scaler
scaler = None
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("Saved scaler as None")

print("Model trained and saved.")