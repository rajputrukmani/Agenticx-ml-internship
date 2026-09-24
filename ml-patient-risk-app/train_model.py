import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Sample Training Data (Age, BMI, SysBP, FBG, HbA1c -> Risk: 0=Low, 1=Moderate, 2=High)
data = {
    'age': [25, 45, 55, 60, 30, 50, 65],
    'bmi': [22.0, 28.5, 31.0, 34.5, 24.0, 29.0, 36.0],
    'sysBP': [120, 135, 142, 150, 115, 138, 160],
    'fbg': [85, 110, 128, 140, 90, 115, 150],
    'hba1c': [5.2, 6.2, 6.8, 7.5, 5.4, 6.3, 8.0],
    'target_risk': [0, 1, 2, 2, 0, 1, 2] # 0: Low, 1: Moderate, 2: High
}

df = pd.DataFrame(data)

X = df[['age', 'bmi', 'sysBP', 'fbg', 'hba1c']]
y = df['target_risk']

# Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save the trained model
joblib.dump(model, 'model.pkl')
print("Model trained and saved successfully as 'model.pkl'!")