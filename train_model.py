import pandas as pd
import numpy as np
import time
import json
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / 'data' / 'processed_ecommerce_data.csv'
MODELS_DIR = BASE_DIR / 'models'
MODELS_DIR.mkdir(exist_ok=True)
DEFAULT_PIPELINE_PATH = BASE_DIR / 'pipeline.pkl'
METRICS_SAVE_PATH = BASE_DIR / 'model_metrics.json'

FEATURES = [
    'recency',
    'frequency',
    'monetary',
    'average_order_value',
    'category_diversity',
    'average_review_score',
    'payment_count',
    'delivery_delay',
    'inactivity_days'
]
TARGET = 'churn'

def evaluate_predictions(y_true, y_pred, y_prob):
    """Computes comprehensive classification metrics for both classes."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Metrics focusing on positive class (churn=1) and minority retained class (churn=0)
    acc = accuracy_score(y_true, y_pred)
    
    # Class 1 (Churned)
    p_churn = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    r_churn = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1_churn = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    
    # Class 0 (Retained - Minority class)
    p_retained = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
    r_retained = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    f1_retained = f1_score(y_true, y_pred, pos_label=0, zero_division=0)
    
    # Macro / Weighted averages
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # ROC-AUC and PR-AUC (for minority retained class = 0 and churn = 1)
    # y_prob is probability of class 1 (churn)
    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc_churn = average_precision_score(y_true, y_prob)
    # PR-AUC for minority retained class (1 - y_true, with 1 - y_prob)
    pr_auc_retained = average_precision_score(1 - y_true, 1 - y_prob)
    
    report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    
    return {
        "accuracy": round(float(acc), 4),
        "roc_auc": round(float(roc_auc), 4),
        "pr_auc_churn": round(float(pr_auc_churn), 4),
        "pr_auc_retained": round(float(pr_auc_retained), 4),
        "f1_macro": round(float(f1_macro), 4),
        "f1_weighted": round(float(f1_weighted), 4),
        "churn_class_1": {
            "precision": round(float(p_churn), 4),
            "recall": round(float(r_churn), 4),
            "f1": round(float(f1_churn), 4),
            "support": int((y_true == 1).sum())
        },
        "retained_class_0": {
            "precision": round(float(p_retained), 4),
            "recall": round(float(r_retained), 4),
            "f1": round(float(f1_retained), 4),
            "support": int((y_true == 0).sum())
        },
        "confusion_matrix": {
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp)
        },
        "classification_report": report_dict
    }

def main():
    print("=" * 70)
    print("CUSTOMER CHURN PREDICTION — MULTI-MODEL TRAINING & EVALUATION")
    print("=" * 70)
    
    # 1. Load Data
    if not DATA_PATH.exists():
        print(f"Error: {DATA_PATH} not found. Please run data_pipeline.py first.")
        return
        
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset Loaded Successfully: {df.shape[0]:,} rows, {df.shape[1]} columns")
    
    # 2. Extract Features & Target
    X = df[FEATURES].copy()
    y = df[TARGET].astype(int)
    
    # Missing values safety
    X = X.fillna(X.mean())
    
    # 3. Stratified Train / Test Split
    # Test size 20%, random_state=42, strictly stratified
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    train_total = len(y_train)
    test_total = len(y_test)
    train_c1, train_c0 = int((y_train == 1).sum()), int((y_train == 0).sum())
    test_c1, test_c0 = int((y_test == 1).sum()), int((y_test == 0).sum())
    
    print(f"\n--- Split Distribution ---")
    print(f"Training Set: {train_total:,} rows | Churned (1): {train_c1:,} ({train_c1/train_total*100:.2f}%) | Retained (0): {train_c0} ({train_c0/train_total*100:.2f}%)")
    print(f"Testing Set:  {test_total:,} rows | Churned (1): {test_c1:,} ({test_c1/test_total*100:.2f}%) | Retained (0): {test_c0} ({test_c0/test_total*100:.2f}%)")
    
    # XGBoost scale_pos_weight
    # scale_pos_weight = count(negative class) / count(positive class)
    # Since class 1 is churn (99.4%) and class 0 is retained (0.56%),
    # scale_pos_weight for class 1 = count(0)/count(1) = 337 / 59919 = ~0.0056
    # Or to penalize misclassifying class 0 when predict_proba gives P(churn):
    xgb_scale_pos_weight = train_c0 / train_c1
    
    models = {
        "Logistic Regression": {
            "pipeline": Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', LogisticRegression(
                    class_weight='balanced',
                    random_state=42,
                    max_iter=1000,
                    solver='lbfgs'
                ))
            ]),
            "file_name": "logistic_regression.joblib",
            "strategy": "StandardScaler + class_weight='balanced'"
        },
        "Random Forest": {
            "pipeline": Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', RandomForestClassifier(
                    n_estimators=150,
                    class_weight='balanced',
                    max_depth=10,
                    min_samples_split=10,
                    min_samples_leaf=4,
                    random_state=42,
                    n_jobs=-1
                ))
            ]),
            "file_name": "random_forest.joblib",
            "strategy": "class_weight='balanced', constrained tree depth (max_depth=10)"
        },
        "XGBoost": {
            "pipeline": Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', XGBClassifier(
                    n_estimators=150,
                    max_depth=5,
                    learning_rate=0.05,
                    scale_pos_weight=xgb_scale_pos_weight,
                    eval_metric='logloss',
                    random_state=42,
                    n_jobs=-1
                ))
            ]),
            "file_name": "xgboost.joblib",
            "strategy": f"scale_pos_weight={xgb_scale_pos_weight:.6f} to balance class penalty"
        }
    }
    
    results = {}
    
    for name, config in models.items():
        print(f"\n" + "-" * 50)
        print(f"Training: {name}...")
        pipeline = config["pipeline"]
        
        start_time = time.time()
        pipeline.fit(X_train, y_train)
        training_time = round(time.time() - start_time, 3)
        print(f"{name} fitted in {training_time:.3f}s")
        
        # Inferences
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        
        metrics = evaluate_predictions(y_test, y_pred, y_prob)
        metrics["training_time_seconds"] = training_time
        metrics["balancing_strategy"] = config["strategy"]
        
        model_save_path = MODELS_DIR / config["file_name"]
        joblib.dump(pipeline, model_save_path)
        metrics["model_path"] = str(model_save_path)
        print(f"Saved model to: {model_save_path}")
        
        # Print summary
        print(f"  Accuracy:         {metrics['accuracy']}")
        print(f"  ROC-AUC:          {metrics['roc_auc']}")
        print(f"  PR-AUC (Retained):{metrics['pr_auc_retained']}")
        print(f"  F1 Macro:         {metrics['f1_macro']}")
        print(f"  Confusion Matrix: TN={metrics['confusion_matrix']['TN']}, FP={metrics['confusion_matrix']['FP']}, FN={metrics['confusion_matrix']['FN']}, TP={metrics['confusion_matrix']['TP']}")
        print(f"  Retained (0):     Precision={metrics['retained_class_0']['precision']}, Recall={metrics['retained_class_0']['recall']}, F1={metrics['retained_class_0']['f1']}")
        print(f"  Churned (1):      Precision={metrics['churn_class_1']['precision']}, Recall={metrics['churn_class_1']['recall']}, F1={metrics['churn_class_1']['f1']}")
        
        results[name] = metrics
        
    # Also save the primary default model as pipeline.pkl for Flask backward compatibility
    # Default is Logistic Regression as requested by base project architecture
    joblib.dump(models["Logistic Regression"]["pipeline"], DEFAULT_PIPELINE_PATH)
    print(f"\nDefault application model saved to: {DEFAULT_PIPELINE_PATH}")
    
    # Save structured metrics
    metadata = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_rows": len(df),
        "features": FEATURES,
        "split": {
            "train_size": train_total,
            "test_size": test_total,
            "train_churn": train_c1,
            "train_retained": train_c0,
            "test_churn": test_c1,
            "test_retained": test_c0
        },
        "models": results
    }
    
    with open(METRICS_SAVE_PATH, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Evaluation metrics structured JSON saved to: {METRICS_SAVE_PATH}")
    print("\n" + "=" * 70)
    print("ALL THREE MODELS TRAINED AND PERSISTED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == '__main__':
    main()
