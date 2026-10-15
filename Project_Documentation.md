# Customer Churn Prediction System - Complete Project Documentation

## 1. Project Overview
The **Customer Churn Prediction System** is an end-to-end Machine Learning web application designed to help businesses identify which customers are at risk of canceling their subscriptions (churning). 

By providing historical customer data to a predictive algorithm, the system calculates a "churn probability percentage" and assigns a visual risk level (Low, Medium, or High) to individual customers. This allows the Customer Success team to take proactive, actionable steps to retain them.

### What it actually does:
1. **Trains a Model:** A Logistic Regression algorithm learns patterns from historical data (who left and who stayed, based on how much they paid and how long they stayed).
2. **Serves an API:** A Python Flask backend hosts this model locally and opens an API endpoint taking real-time data inputs.
3. **Presents a Dashboard:** A visually stunning, resume-worthy, dark-mode ready web dashboard allows users to submit customer data and instantly view the prediction.
4. **Saves History:** Every prediction generated on the frontend is silently logged into a local SQLite database for future analysis.

---

## 2. The Dataset

We use the **IBM Telco Customer Churn dataset** (saved to `data/telco_churn.csv`). This is a real-world, industry-standard dataset built for subscription-based churn prediction. 

> [!IMPORTANT]
> The dataset is stored locally to ensure reliable deployment and avoid runtime dependency on external sources. The previous generic dummy CSV was strictly for initial functionality testing.

**Selected Features from Dataset:**
* **`tenure` (Numeric):** Number of months the customer has stayed.
* **`MonthlyCharges` (Numeric):** Amount billed to the customer monthly.
* **`TotalCharges` (Numeric):** Total amount billed to the customer.
* **`Contract` (Categorical):** Type of contract (Month-to-month, One year, Two year).
* **`InternetService` (Categorical):** Type of internet connection (Fiber optic, DSL, No).
* **`Churn` (Target):** Whether the customer left (Yes/1) or stayed (No/0).

---

## 3. Machine Learning Pipeline & Model Comparison

To make the system highly professional, we upgraded the training script (`train_model.py`) to construct an automated `sklearn.pipeline.Pipeline` with a `ColumnTransformer`. 

1. **Preprocessing:** It uses `StandardScaler` to scale continuous numeric features and `OneHotEncoder` to encode the categorical features automatically.
2. **Model Comparison:** The system trains both **Logistic Regression** and **Random Forest Classifier** side-by-side, evaluating them against Accuracy, Precision, Recall, and Confusion Matrix metrics.
3. **Selection:** It auto-selects the winner (Logistic Regression achieved ~80.2% accuracy whereas Random Forest achieved ~76.8%) and saves the combined preprocessing+model pipeline as `pipeline.pkl`.

---

## 3. What Outputs Are We Showing?

When a user interacts with the **Predict** page, they input the three features above. The model responds with:

1. **Churn Probability:** A direct percentage (e.g., `85.4%`) representing confidence that the user will cancel.
2. **Raw Verdict:** A simple text output: `Yes (Will Churn)` or `No (Stays)`.
3. **Risk Level:** 
   * **🔴 High Risk (>70%)**
   * **🟡 Medium Risk (30% - 70%)**
   * **🟢 Low Risk (<30%)**

In the **Dashboard** and **Analysis** tabs, we showcase Chart.js canvases (visualizing trends) and a breakdown of Feature Importance (showing that `Tenure` generally holds the heaviest weight in deciding if a user churns).

---

## 4. Complete Codebase Details

Below is the breakdown of the essential files that make this platform function.

### A. The Backend Pipeline (Python)

**`train_model.py`**
This script reads the new Telco dataset natively from the `data/` folder, builds a professional `ColumnTransformer` to handle both numeric scales and categorical encoding, and pits Logistic Regression against Random Forest. The winner is bundled with the `ColumnTransformer` into a single `pipeline.pkl`.

**`db.py`**
This script manages the SQLite local database. It contains `init_db()` to automatically generate a `predictions` table if one does not exist, and `insert_prediction()` to log whenever a dashboard user submits the form.

**`app.py`**
The heart of the web server. It loads the `.pkl` files, opens port `5000`, and manages routing. It intercepts standard API requests at `/predict`, parses JSON data, pushes it through the model, and responds with the probability. It also serves the frontend UI routes.

### B. The Frontend Display (HTML/CSS/JS)

For aesthetics, this project uses a premium, hand-coded, glassmorphic UI. No bulky frameworks like Bootstrap or Tailwind were used, ensuring total control and an elegant lightweight codebase.

**`templates/base.html`**
The unified shell. It constructs a modern layout utilizing a fixed left-aligned sidebar with FontAwesome icons, and a top header containing a Dark/Light mode toggle button.

**`static/css/style.css`**
Manages the visual fidelity of the project. It uses `[data-theme="dark"]` property selectors to completely swap the color palette of the application dynamically. It leans on an Indigo/Purple (`#4f46e5`) brand accent.

**`static/js/main.js`**
The brain of the interface. This file:
1. Attaches listeners to the Light/Dark mode toggle and saves user preference inside the browser's `localStorage`.
2. Intercepts the submit button on the Prediction Form. Instead of refreshing the page, it uses JavaScript `Fetch(...)` to quietly talk to `app.py`, waits for the prediction JSON result, and dynamically animates the Badge UI (Red/Yellow/Green) onto the screen immediately.
3. Instantiates `Chart.js` graphs to render analytical data visually.

### C. Other Core Template Pages

* **`index.html`:** The landing greeting page outlining the project.
* **`dashboard.html`:** Grid panels representing total users, average churn rates, and historical logs.
* **`predict.html`:** The form interface designed for data ingestion.
* **`analysis.html`:** Contains the Confusion Matrix and Model Metrics (Accuracy, Recall, Precision).
* **`recommendations.html`:** Suggests targeted marketing strategies based on the AI's risk level rating.
* **`about.html`:** Documentation identifying the underlying Logistic Regression logic.

---

## 5. How to Run Locally

If you are setting this up for a presentation or testing, simply execute in your terminal:

```bash
# 1. Train the initial model
python train_model.py

# 2. Start the web server
python app.py
```

Then, open a web browser and navigate to `http://127.0.0.1:5000`.
