# Project Status: Customer Churn Prediction System

## 🚀 System Health Overview
*   **Web Server:** ✅ **RUNNING** (Currently running `python app.py` on port 5000)
*   **ML Pipeline:** ✅ **LOADED** (`pipeline.pkl` contains the trained Logistic Regression model)
*   **Database:** ✅ **CONNECTED** (`database.db` is active and logging predictions)
*   **UI/UX:** ✅ **OPERATIONAL** (Premium Glassmorphic Dashboard with Dark/Light mode)

---

## 📂 Project Architecture

### 1. Backend Layer (Python / Flask)
*   **`app.py`:** The central controller handling routes for the Dashboard, Prediction API, and UI templates.
*   **`db.py`:** Manages the SQLite database connection, table initialization, and record insertion.
*   **`train_model.py`:** Automates the training process, comparing Logistic Regression and Random Forest, then saving the best-performing pipeline.

### 2. Machine Learning Pipeline
*   **Algorithm:** Logistic Regression (Selected for better accuracy and stability).
*   **Accuracy:** ~80.20%
*   **Preprocessing:** Includes automated `StandardScaler` for numerical data and `OneHotEncoder` for categorical data within a unified `sklearn.pipeline.Pipeline`.
*   **Features Used:** `tenure`, `MonthlyCharges`, `TotalCharges`, `Contract`, `InternetService`.

### 3. Frontend Layer (Modern UI)
*   **Design Paradigm:** Hand-crafted CSS using Glassmorphism (blur effects, subtle gradients).
*   **Interactive Features:**
    *   Dynamic Theme Toggle (Dark/Light mode) with persistent state in `localStorage`.
    *   Real-time predictions via AJAX (Fetch API) – no page refreshes.
    *   Data Visualizations using **Chart.js**.
*   **Pages:** Home, Dashboard, Predict, Analysis, Recommendations, About.

### 4. Data Layer (SQLite)
*   **Database File:** `database.db`
*   **Main Table:** `predictions`
*   **Schema:**
    *   `id`: Primary Key (Auto-increment)
    *   `tenure`, `monthly_charges`, `total_charges`: Customer metrics
    *   `contract`, `internet_service`: Service details
    *   `prediction`: Result (Yes/No)
    *   `probability`: Churn confidence level (%)
    *   `risk_level`: Classified as Low, Medium, or High
    *   `created_at`: Automatic timestamp

---

## 📈 Current Statistics & KPI Status
*   **Dataset:** IBM Telco Customer Churn (Professional Dataset).
*   **Status:** The system is fully ready for demonstration. The integration between the frontend form, backend model, and persistence database is complete.

---

## 🛠 Active Files Summary
| File | Responsibility | Status |
| :--- | :--- | :--- |
| `app.py` | Web API & Server | Active |
| `train_model.py` | Model Training | Ready |
| `db.py` | Database Management | Ready |
| `main.js` | UI Logic & Charts | Active |
| `style.css` | Premium Styling | Active |
| `pipeline.pkl` | Compiled ML Model | Active |

---

> [!NOTE]
> This status report was generated on **2026-04-14**. The system is stable and currently serving requests.
