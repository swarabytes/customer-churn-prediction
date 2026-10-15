# Churn Predict - Customer Churn Prediction Dashboard

## 🚀 Project Overview
**ChurnPredict** is an end-to-end Machine Learning web application designed to help business stakeholders and e-commerce platforms identify and retain customers at risk of leaving. Unlike basic ML scripts, this platform features a professional-grade executive dashboard, real-time API predictions, bulk data upload processing, and secure data persistence scoped per company.

It uses a machine learning model trained on historical customer data to calculate risk probabilities, segment customers into risk categories, and provide actionable retention playbooks.

## ✨ Features
- **Executive Dashboard**: KPI cards for "Revenue at Risk", "Retention Opportunity", and average risk percentages.
- **Real-time Inference**: Instant inference on customer behaviors through an interactive UI.
- **Bulk Processing**: Upload CSV/XLSX files for batch prediction on thousands of customers at once.
- **Explainable AI (XAI) & Actionable Insights**: Understand why a customer is at risk and get recommended actions (e.g., "Send re-engagement offer", "Direct VIP outreach").
- **Multi-Tenant Security**: User authentication and session management to ensure company data remains private.
- **Data Exporting**: Export analysis results and predictions to CSV for spreadsheet reporting.

## 🧠 Algorithms Implemented
- **Logistic Regression**: Used as the primary classification model due to its high interpretability and efficiency, paired with `class_weight='balanced'` to handle imbalanced churn datasets.
- **RFM Analysis (Recency, Frequency, Monetary)**: Implemented within the data pipeline to extract behavioral features from raw e-commerce transaction logs.
- **StandardScaler**: Used within a Scikit-Learn Pipeline to normalize feature distributions (mean=0, variance=1) for optimal model convergence.

## 🛠️ Tech Stack
- **Backend**: Python 3, Flask, SQLite (with Werkzeug Security)
- **Machine Learning**: Scikit-Learn, Pandas, NumPy, Joblib
- **Frontend**: HTML5, Vanilla CSS3, JavaScript (ES6), Chart.js

## 📂 Folder Structure
```text
project/
├── app.py                     # Flask Web Server & API routing
├── db.py                      # SQLite Database Layer & Authentication
├── train_model.py             # ML Training Pipeline script
├── data_pipeline.py           # Data processing and feature engineering script
├── requirements.txt           # Deployment Dependencies
├── README.md                  # Project Documentation
├── data/                      # Raw and processed datasets (ignored in git)
├── templates/                 # Jinja2 Layouts & UI HTML Pages
└── static/                    # Design System (CSS/JS assets)
```
*(Note: `database.db` and model artifacts like `pipeline.pkl` are generated dynamically when the scripts are run and are excluded from version control).*

## 🏃 How to Run

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Lokesh-CodeIt/churn-predict.git
   cd churn-predict
   ```

2. **Set up a Virtual Environment (Recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare the Data and Train the Model**
   - Ensure your raw data is placed in `data/olist_raw/`.
   - Process the raw data into features:
     ```bash
     python data_pipeline.py
     ```
   - Train the ML pipeline:
     ```bash
     python train_model.py
     ```

5. **Launch the Dashboard**
   ```bash
   python app.py
   ```

6. **Access Platform**
   - Open your browser and navigate to [http://127.0.0.1:5000](http://127.0.0.1:5000)
   - Register a new company account and log in.

## 📊 Sample Input/Output

**Sample API Request (`POST /predict`):**
```json
{
  "recency": 12.5,
  "frequency": 4.0,
  "monetary": 450.0,
  "average_order_value": 112.5,
  "category_diversity": 3.0,
  "average_review_score": 4.5,
  "payment_count": 4.0,
  "delivery_delay": -1.2,
  "inactivity_days": 12.5
}
```

**Sample API Response:**
```json
{
  "customer_id": "CUST_A1B2C3D4",
  "insight": "Stable behavior",
  "prediction": 0,
  "probability": 0.15,
  "recommended_action": "Standard engagement",
  "risk_level": "Low"
}
```

## 🔮 Future Improvements
- **Advanced Models**: Experiment with Random Forest or XGBoost for potentially higher accuracy.
- **Deep Learning**: Incorporate LSTM networks for time-series sequence modeling on customer purchases.
- **Automated Retraining**: Implement a background job to retrain the model weekly using new data.
- **Cloud Database**: Migrate from SQLite to PostgreSQL for better concurrency in production.
- **Third-Party Integrations**: Connect directly to Shopify/Stripe APIs for live data ingestion.

## 📄 License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
