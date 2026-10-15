from flask import Flask, request, jsonify, render_template, session, redirect, url_for, flash
import joblib
import pandas as pd
import uuid
import os
import json
from pathlib import Path
from functools import wraps
import db

app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / 'pipeline.pkl'
MODELS_DIR = BASE_DIR / 'models'
METRICS_PATH = BASE_DIR / 'model_metrics.json'
EVALUATION_CACHE_PATH = BASE_DIR / 'model_evaluation_cache.json'

# The 9 features in the exact order expected by all trained pipelines
FEATURE_ORDER = [
    'recency', 'frequency', 'monetary', 'average_order_value',
    'category_diversity', 'average_review_score', 'payment_count',
    'delivery_delay', 'inactivity_days'
]

# Secret key for secure session management (reads from SECRET_KEY or FLASK_SECRET_KEY)
app.secret_key = os.environ.get("SECRET_KEY") or os.environ.get("FLASK_SECRET_KEY") or "dev_secret_key_change_in_prod"

# ---------------------------------------------------------
# Load All Three Model Pipelines at Startup
# ---------------------------------------------------------
_model_files = {
    'logistic_regression': MODELS_DIR / 'logistic_regression.joblib',
    'random_forest':       MODELS_DIR / 'random_forest.joblib',
    'xgboost':             MODELS_DIR / 'xgboost.joblib',
}
MODEL_REGISTRY = {}
for _key, _path in _model_files.items():
    try:
        MODEL_REGISTRY[_key] = joblib.load(_path)
        print(f"Loaded model: {_key} from {_path}")
    except Exception as _e:
        print(f"Could not load {_key}: {_e}")
        MODEL_REGISTRY[_key] = None

# Default pipeline: Random Forest for highest accuracy across predictions
pipeline = MODEL_REGISTRY.get('random_forest') or MODEL_REGISTRY.get('logistic_regression') or next(
    (m for m in MODEL_REGISTRY.values() if m), None
)
if pipeline is None:
    print("WARNING: No trained model found. Run train_model.py first.")

# Load evaluation metrics JSON once at startup
MODEL_METRICS = {}
try:
    with open(METRICS_PATH) as _f:
        MODEL_METRICS = json.load(_f)
    print(f"Loaded model metrics from {METRICS_PATH}")
except Exception as _e:
    print(f"Could not load model_metrics.json: {_e}")

# Load comprehensive evaluation cache (ROC, PR curves, specificity) once at startup
MODEL_EVALUATION = {}
try:
    if EVALUATION_CACHE_PATH.exists():
        with open(EVALUATION_CACHE_PATH) as _f:
            MODEL_EVALUATION = json.load(_f)
        print(f"Loaded comprehensive evaluation cache from {EVALUATION_CACHE_PATH}")
except Exception as _e:
    print(f"Could not load model_evaluation_cache.json: {_e}")

try:
    db.init_db()
    print("Database initialized successfully.")
except Exception as e:
    print(f"Error initializing database: {e}")


# ---------------------------------------------------------
# Authentication Decorator
# ---------------------------------------------------------
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'company_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function


# ---------------------------------------------------------
# Authentication Routes
# ---------------------------------------------------------
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        company = db.authenticate_company(email, password)
        if company:
            session['company_id'] = company['id']
            session['company_name'] = company['company_name']
            return redirect(url_for('dashboard'))
        else:
            return render_template('login.html', error="Invalid email or password. Please try again.")
            
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        company_name = request.form.get('company_name')
        email = request.form.get('email')
        password = request.form.get('password')
        
        success, msg = db.create_company(company_name, email, password)
        if success:
            return redirect(url_for('login'))
        else:
            return render_template('register.html', error=msg)
            
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


# ---------------------------------------------------------
# Application Routes
# ---------------------------------------------------------

@app.route('/', methods=['GET'])
def index():
    """Landing page."""
    return render_template('index.html')

@app.route('/predict_ui', methods=['GET'])
@login_required
def predict_ui():
    """Prediction form page."""
    return render_template('predict.html')

@app.route('/dashboard', methods=['GET'])
@login_required
def dashboard():
    """Analytics dashboard page."""
    return render_template('dashboard.html')

@app.route('/bulk-upload', methods=['GET', 'POST'])
@login_required
def bulk_upload():
    """Bulk upload page and file handler scoped to company."""
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({"error": "No file part in the request"}), 400
            
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
            
        allowed_extensions = {'.csv', '.xlsx'}
        ext = os.path.splitext(file.filename)[1].lower()
        
        if ext not in allowed_extensions:
            return jsonify({"error": f"Unsupported file type. Allowed: CSV, XLSX"}), 400
            
        try:
            if ext == '.csv':
                df = pd.read_csv(file)
            else:
                df = pd.read_excel(file, engine="openpyxl")
                
            MAX_ROWS = 5000
            if len(df) > MAX_ROWS:
                return jsonify({
                    "error": "File too large",
                    "details": f"Maximum allowed is {MAX_ROWS} rows per batch."
                }), 400
                
            required_cols = [
                'recency', 'frequency', 'monetary', 'average_order_value',
                'category_diversity', 'average_review_score', 'payment_count',
                'delivery_delay', 'inactivity_days'
            ]
            missing_cols = [c for c in required_cols if c not in df.columns]
            if missing_cols:
                return jsonify({"error": "Missing required columns", "details": f"Missing: {', '.join(missing_cols)}"}), 400
                
            batch_id = f"BATCH_{str(uuid.uuid4())[:8].upper()}"
            
            total_rows = len(df)
            valid_rows = 0
            invalid_rows = 0
            high_risk = 0
            medium_risk = 0
            low_risk = 0
            total_prob = 0.0
            rev_at_risk = 0.0
            
            if pipeline is None:
                return jsonify({"error": "Model not loaded. Please ensure pipeline.pkl exists."}), 500
            
            company_id = session['company_id']
            
            for index, row in df.iterrows():
                try:
                    features = {col: float(row[col]) for col in required_cols}
                    df_input = pd.DataFrame([features])
                    prediction_array = pipeline.predict(df_input)
                    prediction = int(prediction_array[0])
                    probability = float(pipeline.predict_proba(df_input)[0][1])
                    
                    if probability < 0.3:
                        risk_level = "Low"
                        low_risk += 1
                    elif 0.3 <= probability <= 0.7:
                        risk_level = "Medium"
                        medium_risk += 1
                    else:
                        risk_level = "High"
                        high_risk += 1
                        rev_at_risk += features['monetary']
                        
                    total_prob += probability
                    
                    cust_id = str(row.get('customer_id', ''))
                    if not cust_id or pd.isna(row.get('customer_id')) or cust_id.strip() == 'nan':
                        cust_id = f"CUST_{str(uuid.uuid4())[:8].upper()}"
                        
                    health_score = round((1.0 - probability) * 100, 1)
                    
                    action = "Standard engagement"
                    if features['inactivity_days'] > 90:
                        action = "Send re-engagement offer"
                    elif features['monetary'] > 500 and probability > 0.7:
                        action = "Direct VIP outreach"
                    elif features['average_review_score'] < 3.0:
                        action = "Customer support follow-up"
                        
                    db.insert_prediction(
                        company_id=company_id,
                        customer_id=cust_id,
                        data=features,
                        prediction=prediction,
                        probability=probability,
                        risk_level=risk_level,
                        batch_id=batch_id,
                        source_file=file.filename,
                        upload_mode="bulk",
                        health_score=health_score,
                        recommendation=action
                    )
                    valid_rows += 1
                except Exception as row_error:
                    invalid_rows += 1
                    
            avg_prob = (total_prob / valid_rows) if valid_rows > 0 else 0
            
            return jsonify({
                "message": "Batch processed successfully",
                "batch_id": batch_id,
                "filename": file.filename,
                "total_rows": total_rows,
                "valid_rows": valid_rows,
                "invalid_rows": invalid_rows,
                "high_risk_count": high_risk,
                "medium_risk_count": medium_risk,
                "low_risk_count": low_risk,
                "average_probability": round(avg_prob, 3),
                "revenue_at_risk": round(rev_at_risk, 2)
            }), 200
            
        except Exception as e:
            return jsonify({"error": "Failed to process bulk upload.", "details": str(e)}), 400
            
    return render_template('bulk_upload.html')

@app.route('/export-batch/<batch_id>', methods=['GET'])
@login_required
def export_batch(batch_id):
    """Export predictions for a specific batch, scoped to company."""
    try:
        from flask import make_response
        import io
        company_id = session['company_id']
        predictions = db.get_predictions_by_batch(company_id, batch_id)
        if not predictions:
            return jsonify({"error": "No data found for this batch"}), 404

        df = pd.DataFrame(predictions)
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        output = make_response(csv_buffer.getvalue())
        output.headers["Content-Disposition"] = f"attachment; filename=batch_{batch_id}_export.csv"
        output.headers["Content-type"] = "text/csv"
        return output
    except Exception as e:
        return jsonify({"error": "Failed to export batch CSV", "details": str(e)}), 500

@app.route('/api/dashboard-data', methods=['GET'])
@login_required
def dashboard_data():
    """Return JSON metrics scoped to the logged-in company."""
    try:
        company_id = session['company_id']
        stats = db.get_advanced_stats(company_id)
        predictions = db.get_all_predictions_with_priority(company_id)
        alerts = db.get_alerts(company_id)
        
        # Historical dataset distribution & model evaluation context
        dataset_benchmark = {
            "historical_total": MODEL_METRICS.get('dataset_rows', 75320),
            "historical_churn_count": MODEL_METRICS.get('split', {}).get('train_churn', 59919) + MODEL_METRICS.get('split', {}).get('test_churn', 14980),
            "historical_retained_count": MODEL_METRICS.get('split', {}).get('train_retained', 337) + MODEL_METRICS.get('split', {}).get('test_retained', 84),
            "historical_churn_rate_pct": round(((MODEL_METRICS.get('split', {}).get('train_churn', 59919) + MODEL_METRICS.get('split', {}).get('test_churn', 14980)) / max(1, MODEL_METRICS.get('dataset_rows', 75320))) * 100, 2),
            "historical_retained_rate_pct": round(((MODEL_METRICS.get('split', {}).get('train_retained', 337) + MODEL_METRICS.get('split', {}).get('test_retained', 84)) / max(1, MODEL_METRICS.get('dataset_rows', 75320))) * 100, 2),
            "default_model": "Logistic Regression",
            "default_model_roc_auc": MODEL_METRICS.get('models', {}).get('Logistic Regression', {}).get('roc_auc', 0.6124),
            "default_model_retained_recall": MODEL_METRICS.get('models', {}).get('Logistic Regression', {}).get('retained_class_0', {}).get('recall', 0.5476)
        }
        
        return jsonify({
            "stats": stats,
            "recent_predictions": predictions,
            "alerts": alerts,
            "dataset_benchmark": dataset_benchmark
        })
    except Exception as e:
        print(f"Dashboard data error: {e}")
        return jsonify({"error": "Failed to load dashboard data"}), 500

@app.route('/export', methods=['GET'])
@login_required
def export_csv():
    """Export all company predictions as CSV."""
    try:
        from flask import make_response
        import io
        company_id = session['company_id']
        predictions = db.get_all_predictions_with_priority(company_id)
        if not predictions:
            return jsonify({"error": "No data to export yet"}), 404

        df = pd.DataFrame(predictions)
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        output = make_response(csv_buffer.getvalue())
        output.headers["Content-Disposition"] = "attachment; filename=predictions_export.csv"
        output.headers["Content-type"] = "text/csv"
        return output
    except Exception as e:
        print(f"Export error: {e}")
        return jsonify({"error": "CSV export failed", "details": str(e)}), 500

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    """Predict churn risk. Accepts optional 'model_type' in the JSON body to select among
    logistic_regression | random_forest | xgboost. Defaults to logistic_regression."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON payload provided"}), 400

        # Resolve which model to use (defaults to random_forest for highest accuracy)
        model_type = data.get('model_type', 'random_forest').lower().replace(' ', '_')
        # Normalise display name variants
        _alias = {
            'lr': 'logistic_regression',
            'rf': 'random_forest',
            'xgb': 'xgboost',
        }
        model_type = _alias.get(model_type, model_type)

        active_pipeline = MODEL_REGISTRY.get(model_type)
        if active_pipeline is None:
            return jsonify({"error": f"Model '{model_type}' is not available. Check that train_model.py has been run."}), 500

        # Build feature dict in canonical order to avoid column-order surprises
        try:
            features = {col: float(data.get(col, 0.0)) for col in FEATURE_ORDER}
        except (TypeError, ValueError) as fe:
            return jsonify({"error": f"Invalid feature value: {fe}"}), 400

        df_input = pd.DataFrame([features])[FEATURE_ORDER]
        prediction = int(active_pipeline.predict(df_input)[0])
        proba = active_pipeline.predict_proba(df_input)[0]  # [P(retained), P(churn)]
        prob_churn = float(proba[1])
        prob_retained = float(proba[0])

        if prob_churn < 0.3:
            risk_level = "Low"
        elif prob_churn <= 0.7:
            risk_level = "Medium"
        else:
            risk_level = "High"

        # Prediction=1 means Churned, =0 means Retained
        outcome_label = "Churned" if prediction == 1 else "Retained"

        customer_id = f"CUST_{str(uuid.uuid4())[:8].upper()}"
        company_id = session['company_id']

        # Insight + action logic (rule-based on feature values)
        insight = "Customer shows stable purchasing behaviour."
        action = "Standard engagement"
        if features['inactivity_days'] > 90:
            insight = "Customer has been inactive for an extended period."
            action = "Send re-engagement offer"
        elif features['monetary'] > 500 and prob_churn > 0.7:
            insight = "High-value customer identified as at risk."
            action = "Direct VIP outreach"
        elif features['average_review_score'] < 3.0:
            insight = "Poor review history — experience risk flagged."
            action = "Customer support follow-up"

        health_score = round((1.0 - prob_churn) * 100, 1)

        db.insert_prediction(
            company_id=company_id,
            customer_id=customer_id,
            data=features,
            prediction=prediction,
            probability=prob_churn,
            risk_level=risk_level,
            health_score=health_score,
            recommendation=action,
        )

        return jsonify({
            "customer_id": customer_id,
            "model_used": model_type,
            "prediction": prediction,
            "outcome_label": outcome_label,
            "probability": prob_churn,
            "probability_retained": prob_retained,
            "risk_level": risk_level,
            "health_score": health_score,
            "insight": insight,
            "recommended_action": action,
        }), 200

    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return jsonify({"error": "Failed to process prediction.", "details": str(e)}), 400


# ---------------------------------------------------------
# Model Comparison Routes
# ---------------------------------------------------------

@app.route('/models-comparison', methods=['GET'])
@login_required
def models_comparison():
    """Model comparison dashboard — reads real metrics from model_metrics.json."""
    return render_template('models_comparison.html')


@app.route('/api/models-metrics', methods=['GET'])
@login_required
def api_models_metrics():
    """Return normalised evaluation metrics keyed by snake_case model name.
    Includes Accuracy, Precision (Class 1 Churn), Recall (Class 1 Churn), F1-Score,
    Specificity (Class 0 Retained), ROC-AUC, Average Precision, Confusion Matrix,
    and Training Time."""
    if not MODEL_METRICS and not MODEL_EVALUATION:
        return jsonify({"error": "model_metrics.json not found. Run train_model.py first."}), 404

    raw_models = MODEL_METRICS.get('models', {})
    _name_map = {
        'Logistic Regression': 'logistic_regression',
        'Random Forest':       'random_forest',
        'XGBoost':             'xgboost',
    }
    split = MODEL_METRICS.get('split', {})
    cached_models = MODEL_EVALUATION.get('models', {})

    normalised = {}
    for display_name, data in raw_models.items():
        key = _name_map.get(display_name, display_name.lower().replace(' ', '_'))
        clf_report = data.get('classification_report', {})
        macro    = clf_report.get('macro avg', {})
        weighted = clf_report.get('weighted avg', {})
        cm_raw   = data.get('confusion_matrix', {})
        
        # Raw counts: Class 1 = Churned (Positive), Class 0 = Retained (Negative)
        # TN = correctly predicted retained, FP = retained predicted as churned
        # FN = churned predicted as retained, TP = correctly predicted churned
        tn = int(cm_raw.get('TN', 0))
        fp = int(cm_raw.get('FP', 0))
        fn = int(cm_raw.get('FN', 0))
        tp = int(cm_raw.get('TP', 0))
        cm = [[tn, fp], [fn, tp]]

        # Precise metric formulas treating Class 1 (Churned) as positive class
        total = tn + fp + fn + tp
        acc = (tp + tn) / total if total > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        c0 = data.get('retained_class_0', clf_report.get('0', {}))
        c1 = data.get('churn_class_1',    clf_report.get('1', {}))

        cached_item = cached_models.get(key, {})

        normalised[key] = {
            'name':               display_name,
            'accuracy':           round(acc, 4),
            'precision':          round(prec, 4),
            'recall':             round(rec, 4),
            'f1':                 round(f1, 4),
            'specificity':        round(spec, 4),
            'roc_auc':            data.get('roc_auc'),
            'average_precision':  data.get('pr_auc_churn'),
            'f1_macro':           data.get('f1_macro') or macro.get('f1-score'),
            'f1_weighted':        data.get('f1_weighted') or weighted.get('f1-score'),
            'precision_macro':    macro.get('precision'),
            'recall_macro':       macro.get('recall'),
            'precision_weighted': weighted.get('precision'),
            'recall_weighted':    weighted.get('recall'),
            'confusion_matrix':   cm,
            'confusion_counts':   {'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp},
            'training_time_seconds': data.get('training_time_seconds'),
            'train_set_size':     split.get('train_size'),
            'test_set_size':      split.get('test_size'),
            'roc_curve':          cached_item.get('roc_curve', []),
            'pr_curve':           cached_item.get('pr_curve', []),
            'class_0': {
                'name':      'Retained',
                'precision': c0.get('precision'),
                'recall':    c0.get('recall'),
                'f1':        c0.get('f1') or c0.get('f1-score'),
                'support':   c0.get('support'),
            },
            'class_1': {
                'name':      'Churned',
                'precision': c1.get('precision'),
                'recall':    c1.get('recall'),
                'f1':        c1.get('f1') or c1.get('f1-score'),
                'support':   c1.get('support'),
            },
        }

    return jsonify(normalised), 200


@app.route('/api/model-evaluation-detail', methods=['GET'])
@login_required
def api_model_evaluation_detail():
    """Return complete evaluation dataset statistics, ROC curves, PR curves,
    and 5 core metrics for the upgraded dashboard."""
    if MODEL_EVALUATION:
        return jsonify(MODEL_EVALUATION), 200
    
    # Fallback to api_models_metrics structure if cache file is somehow missing
    res, code = api_models_metrics()
    return res, code


# ---------------------------------------------------------
# Deletion & Reset Routes
# ---------------------------------------------------------

@app.route('/api/prediction/<int:prediction_id>', methods=['DELETE'])
@login_required
def delete_prediction(prediction_id):
    """Delete a single prediction, accessible only by owning company."""
    company_id = session['company_id']
    success = db.delete_prediction(company_id, prediction_id)
    if success:
        return jsonify({"status": "success", "message": "Prediction deleted."}), 200
    else:
        return jsonify({"status": "error", "message": "Failed to delete or not authorized."}), 400

@app.route('/api/batch/<batch_id>', methods=['DELETE'])
@login_required
def delete_batch(batch_id):
    """Delete a whole batch, accessible only by owning company."""
    company_id = session['company_id']
    success = db.delete_batch(company_id, batch_id)
    if success:
        return jsonify({"status": "success", "message": "Batch deleted."}), 200
    else:
        return jsonify({"status": "error", "message": "Failed to delete batch or not authorized."}), 400

@app.route('/api/reset', methods=['DELETE'])
@login_required
def reset_data():
    """Reset all predictions for the logged-in company."""
    company_id = session['company_id']
    success = db.reset_company_data(company_id)
    if success:
        return jsonify({"status": "success", "message": "All company data reset successfully."}), 200
    else:
        return jsonify({"status": "error", "message": "Failed to reset data."}), 400

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Route not found"}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    # use_reloader=False prevents Windows thread crash during background module checks
    app.run(debug=True, use_reloader=False, host='127.0.0.1', port=5000)
