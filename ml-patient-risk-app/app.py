from flask import Flask, render_template, request
import pandas as pd
import os

app = Flask(__name__)

# Aapke folder mein jo pehle se dataset.csv hai, uska naam yahan set kar diya hai
CSV_FILE = 'dataset.csv'

# Function to check if dataset.csv has headers, if empty, add headers
def init_csv():
    if not os.path.exists(CSV_FILE) or os.stat(CSV_FILE).st_size == 0:
        df = pd.DataFrame(columns=['Age', 'BMI', 'Systolic_BP', 'Glucose', 'HbA1c', 'Risk_Level'])
        df.to_csv(CSV_FILE, index=False)

init_csv()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get values from form
        age = float(request.form.get('age', 0))
        bmi = float(request.form.get('bmi', 0))
        sysBP = float(request.form.get('sysBP', 0))
        fbg = float(request.form.get('fbg', 0))
        hba1c = float(request.form.get('hba1c', 0))

        # Risk calculation logic
        factors = []
        risk_level = "Low"
        
        if fbg >= 126 or hba1c >= 6.5:
            risk_level = "High"
            factors.append(f"Fasting Blood Glucose ({fbg} mg/dL) or HbA1c ({hba1c}%) in diabetic range.")
        elif (100 <= fbg < 126) or (5.7 <= hba1c < 6.5):
            risk_level = "Moderate"
            factors.append(f"Pre-diabetic indicators: FBG ({fbg}), HbA1c ({hba1c}%).")
        
        if bmi >= 30:
            if risk_level == "Low": 
                risk_level = "Moderate"
            factors.append(f"Elevated BMI ({bmi} kg/m²).")
        
        if sysBP >= 140:
            risk_level = "High"
            factors.append(f"High Systolic Blood Pressure ({sysBP} mmHg).")

        if not factors:
            factors.append("All baseline metrics within standard limits.")

        # --- DATASET.CSV MEIN DATA SAVE KARNA ---
        new_data = pd.DataFrame([[age, bmi, sysBP, fbg, hba1c, risk_level]], 
                                columns=['Age', 'BMI', 'Systolic_BP', 'Glucose', 'HbA1c', 'Risk_Level'])
        new_data.to_csv(CSV_FILE, mode='a', header=False, index=False)
        # ----------------------------------------

        assessment = {
            "condition": "Type 2 Diabetes & Hypertension",
            "risk": risk_level,
            "factors": factors,
            "rationale": "Sustained elevation in metabolic/hemodynamic indices reflects increased health risk.",
            "steps": "Schedule an HbA1c/BP re-test and consult a medical specialist.",
            "disclaimer": "This is an AI-generated predictive estimation and must be verified by a licensed medical professional."
        }

        return render_template('index.html', assessment=assessment, show_results=True)

    except Exception as e:
        return render_template('index.html', error=str(e), show_results=False)

if __name__ == '__main__':
    app.run(debug=True)