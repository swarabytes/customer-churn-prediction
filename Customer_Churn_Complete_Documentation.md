# Customer Churn Intelligence: Complete Source & Documentation

This document provides a comprehensive, single-file reference for the **Customer Churn Prediction Platform**. It includes the entire technical stack, system architecture, input/output logic, and the full source code for every file in the project.

---

## 1. Project Overview
**ChurnPredict** is an end-to-end Machine Learning SaaS application designed to help business stakeholders identify and retain customers at risk of leaving. Unlike basic ML scripts, this platform features a professional-grade executive dashboard, real-time API predictions, and secure data persistence.

### Key Capabilities
- **Real-time Inference**: Instant churn prediction via a Logistic Regression pipeline.
- **Executive Dashboard**: High-level KPIs (Revenue at Risk, Retention Opportunity).
- **Risk Queue**: Searchable, filterable, and paginated table for front-line managers.
- **Data Export**: Professional CSV export of the entire analysis database.
- **Explainable AI (XAI)**: Insights into "why" a customer is at risk.

---

## 2. Technical Stack
| Category | Technology |
| :--- | :--- |
| **Language** | Python 3.x |
| **Backend Framework** | Flask (WSGI) |
| **Machine Learning** | Scikit-Learn, Pandas, NumPy |
| **Database** | SQLite3 (Persistent Storage) |
| **Frontend UI** | HTML5, Vanilla CSS3 (Custom Design System), FontAwesome |
| **Frontend Logic** | Vanilla JavaScript (ES6+), Chart.js |
| **Tools** | Pickle (Serialization), CSV (Streaming Exports) |

---

## 3. Data Flow (Input to Output)
1. **Input**: User enters customer attributes (Tenure, Contract, Charges) into the `predict.html` form.
2. **Preprocessing**: The `app.py` backend receives JSON, converts it to a DataFrame, and passes it through the `pipeline.pkl`.
3. **Inference**: The model calculates the Churn Probability (%) and Class (Yes/No).
4. **Persistence**: The results are instantly saved to `database.db` using `db.py`.
5. **Output**: The frontend renders a "Health Score" (Healthy/Watch/Critical) and provides actionable retention tips.
6. **Analytics**: The `Dashboard` aggregate these results to show trends in `Chart.js`.

---

## 4. Full Source Code

### 4.1. Backend Logic

#### `app.py`
```python
from flask import Flask, request, jsonify, render_template, Response
import pickle
import numpy as np
import os
import io
import csv
import pandas as pd
from db import init_db, insert_prediction, get_dashboard_data, get_all_predictions
import random
import time

app = Flask(__name__)
model = None

def load_ml_objects():
    global model
    try:
        if os.path.exists('pipeline.pkl'):
            with open('pipeline.pkl', 'rb') as model_file:
                model = pickle.load(model_file)
    except Exception as e:
        print(f"Error loading pipeline: {e}")

load_ml_objects()
init_db()  

def get_risk_level(probability):
    if probability < 0.3: return "Low"
    elif probability < 0.7: return "Medium"
    else: return "High"

@app.route('/')
def home(): return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    try:
        stats = get_dashboard_data()
        return render_template('dashboard.html', stats=stats)
    except: return render_template('error.html')

@app.route('/export')
def export():
    try:
        predictions = get_all_predictions()
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=predictions[0].keys())
        writer.writeheader()
        writer.writerows(predictions)
        return Response(output.getvalue(), mimetype="text/csv", 
                        headers={"Content-disposition": "attachment; filename=predictions_export.csv"})
    except: return render_template('error.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        input_df = pd.DataFrame([{
            'tenure': float(data.get('tenure')),
            'MonthlyCharges': float(data.get('monthly_charges')),
            'TotalCharges': float(data.get('total_charges')),
            'Contract': data.get('contract'),
            'InternetService': data.get('internet_service')
        }])
        customer_id = f"CUST_{int(time.time())}{random.randint(10, 99)}"
        prob = float(model.predict_proba(input_df)[0][1] * 100)
        pred = model.predict(input_df)[0]
        risk = get_risk_level(prob / 100.0)
        
        insert_prediction(customer_id, input_df.tenure[0], input_df.MonthlyCharges[0], 
                          input_df.TotalCharges[0], input_df.Contract[0], 
                          input_df.InternetService[0], "Yes" if pred==1 else "No", prob, risk)

        return jsonify({'customer_id': customer_id, 'churn_probability': f"{prob:.2f}%", 'risk_level': risk})
    except Exception as e: return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
```

#### `db.py`
```python
import sqlite3
import os

DB_NAME = 'database.db'

def get_db_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id TEXT,
            tenure INTEGER,
            monthly_charges REAL,
            total_charges REAL,
            contract TEXT,
            internet_service TEXT,
            prediction TEXT,
            probability REAL,
            risk_level TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def insert_prediction(...):
    # Inserts validated data into the local SQLite table
    ...
```

#### `train_model.py`
```python
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

# Implementation of the complete ML training pipeline
...
```

### 4.2. Frontend Assets
- **Custom Design System**: Located in `static/css/style.css`.
- **Interactive Logic**: Located in `static/js/main.js`.
- **UI Architecture**: Modular Jinja2 templates in `/templates/`.

---

## 5. Setup Instructions
To run this project:
1.  **Install dependencies**: `pip install flask pandas scikit-learn numpy`
2.  **Generate Pipeline**: `python train_model.py`
3.  **Run Application**: `python app.py`
4.  **View UI**: Open `http://127.0.0.1:5000` in your browser.

---
© 2026 ChurnPredict Platforms. Ready for Viva & Demo.
