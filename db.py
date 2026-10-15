import sqlite3
import datetime
from werkzeug.security import generate_password_hash, check_password_hash

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATABASE_NAME = os.environ.get('DATABASE_PATH', str(BASE_DIR / 'database.db'))

def get_db_connection():
    """Establish and return a connection to the SQLite database."""
    conn = sqlite3.connect(DATABASE_NAME)
    conn.row_factory = sqlite3.Row  # To get dict-like rows
    return conn

def init_db():
    """Initialize the database with companies and predictions tables safely."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # 1. Create companies table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS companies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            company_name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # 2. Create Legacy Company to avoid orphaned data
    cursor.execute("SELECT id FROM companies WHERE email = 'admin@legacy.com'")
    legacy_company = cursor.fetchone()
    if not legacy_company:
        pw_hash = generate_password_hash("legacy_admin_secret_change_me!")
        cursor.execute('''
            INSERT INTO companies (company_name, email, password_hash)
            VALUES (?, ?, ?)
        ''', ("Default Company", "admin@legacy.com", pw_hash))
        conn.commit()
        
    cursor.execute("SELECT id FROM companies WHERE email = 'admin@legacy.com'")
    legacy_company_id = cursor.fetchone()['id']

    # 3. Create predictions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id TEXT NOT NULL,
            recency REAL,
            frequency REAL,
            monetary REAL,
            average_order_value REAL,
            category_diversity REAL,
            average_review_score REAL,
            payment_count REAL,
            delivery_delay REAL,
            inactivity_days REAL,
            prediction INTEGER,
            probability REAL,
            risk_level TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # 4. Safely inject new columns (including company_id)
    new_columns = {
        'batch_id': 'TEXT',
        'source_file': 'TEXT',
        'upload_mode': 'TEXT DEFAULT "single"',
        'health_score': 'REAL',
        'recommendation': 'TEXT',
        'company_id': 'INTEGER REFERENCES companies(id)'
    }
    
    for col_name, col_type in new_columns.items():
        try:
            cursor.execute(f'ALTER TABLE predictions ADD COLUMN {col_name} {col_type}')
        except sqlite3.OperationalError:
            pass
            
    # 5. Migrate orphaned predictions safely
    cursor.execute("UPDATE predictions SET company_id = ? WHERE company_id IS NULL", (legacy_company_id,))
    
    conn.commit()
    conn.close()


# ---------------------------------------------------------
# Authentication / Company Handlers
# ---------------------------------------------------------

def create_company(company_name, email, password):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        pw_hash = generate_password_hash(password)
        cursor.execute("INSERT INTO companies (company_name, email, password_hash) VALUES (?, ?, ?)",
                       (company_name, email, pw_hash))
        conn.commit()
        return True, "Company registered successfully."
    except sqlite3.IntegrityError:
        return False, "Email already exists."
    finally:
        conn.close()

def authenticate_company(email, password):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT * FROM companies WHERE email = ?", (email,))
        company = cursor.fetchone()
        if company and check_password_hash(company['password_hash'], password):
            return dict(company)
        return None
    finally:
        conn.close()

def get_company_by_id(company_id):
    conn = get_db_connection()
    try:
        cursor = conn.execute("SELECT * FROM companies WHERE id = ?", (company_id,))
        row = cursor.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()

# ---------------------------------------------------------
# Prediction Queries (Scoped per company_id)
# ---------------------------------------------------------

def insert_prediction(company_id, customer_id, data, prediction, probability, risk_level, batch_id=None, source_file=None, upload_mode="single", health_score=None, recommendation=None):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('''
            INSERT INTO predictions (
                company_id, customer_id, recency, frequency, monetary, average_order_value,
                category_diversity, average_review_score, payment_count,
                delivery_delay, inactivity_days, prediction, probability, risk_level,
                batch_id, source_file, upload_mode, health_score, recommendation
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            company_id, customer_id, data.get('recency'), data.get('frequency'), data.get('monetary'), 
            data.get('average_order_value'), data.get('category_diversity'), data.get('average_review_score'), 
            data.get('payment_count'), data.get('delivery_delay'), data.get('inactivity_days'),
            int(prediction), float(probability), risk_level, batch_id, source_file, upload_mode, 
            health_score, recommendation
        ))
        conn.commit()
    except sqlite3.Error as e:
        print(f"Database error during insertion: {e}")
    finally:
        conn.close()

def insert_predictions_batch(rows_tuples):
    """Bulk insert predictions using executemany in a single transaction for maximum speed."""
    if not rows_tuples:
        return
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.executemany('''
            INSERT INTO predictions (
                company_id, customer_id, recency, frequency, monetary, average_order_value,
                category_diversity, average_review_score, payment_count,
                delivery_delay, inactivity_days, prediction, probability, risk_level,
                batch_id, source_file, upload_mode, health_score, recommendation
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', rows_tuples)
        conn.commit()
    except sqlite3.Error as e:
        print(f"Database error during batch insertion: {e}")
        raise e
    finally:
        conn.close()

def get_all_predictions(company_id):
    conn = get_db_connection()
    try:
        cursor = conn.execute("SELECT * FROM predictions WHERE company_id = ? ORDER BY created_at DESC", (company_id,))
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    except sqlite3.Error as e:
        return []
    finally:
        conn.close()

def get_dashboard_stats(company_id):
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as count FROM predictions WHERE company_id = ?", (company_id,))
        total_predictions = cursor.fetchone()['count']
        
        cursor.execute("SELECT COUNT(*) as count FROM predictions WHERE risk_level = 'High' AND company_id = ?", (company_id,))
        high_risk_count = cursor.fetchone()['count']
        
        cursor.execute("SELECT AVG(probability) as avg_prob FROM predictions WHERE company_id = ?", (company_id,))
        avg_prob_row = cursor.fetchone()
        avg_risk_percent = round((avg_prob_row['avg_prob'] or 0.0) * 100, 1)
        
        cursor.execute("SELECT SUM(monetary) as total_rev_risk FROM predictions WHERE risk_level = 'High' AND company_id = ?", (company_id,))
        rev_risk_row = cursor.fetchone()
        revenue_at_risk = round(rev_risk_row['total_rev_risk'] or 0.0, 2)

        return {
            "total_predictions": total_predictions,
            "high_risk_customers": high_risk_count,
            "avg_risk_percent": avg_risk_percent,
            "revenue_at_risk": revenue_at_risk
        }
    except sqlite3.Error as e:
        return {"total_predictions": 0, "high_risk_customers": 0, "avg_risk_percent": 0.0, "revenue_at_risk": 0.0}
    finally:
        conn.close()

def get_advanced_stats(company_id):
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as count FROM predictions WHERE company_id = ?", (company_id,))
        total_predictions = cursor.fetchone()['count']
        
        cursor.execute("SELECT COUNT(*) as count FROM predictions WHERE probability > 0.7 AND company_id = ?", (company_id,))
        high_risk_count = cursor.fetchone()['count']
        
        cursor.execute("SELECT AVG(probability) as avg_prob FROM predictions WHERE company_id = ?", (company_id,))
        avg_prob_row = cursor.fetchone()
        avg_risk_probability = avg_prob_row['avg_prob'] or 0.0
        
        cursor.execute("SELECT SUM(monetary * probability) as total_rev_risk FROM predictions WHERE company_id = ?", (company_id,))
        rev_risk_row = cursor.fetchone()
        revenue_at_risk = round(rev_risk_row['total_rev_risk'] or 0.0, 2)

        return {
            "total_predictions": total_predictions,
            "avg_risk_probability": avg_risk_probability,
            "high_risk_count": high_risk_count,
            "revenue_at_risk": revenue_at_risk
        }
    except sqlite3.Error as e:
        return {"total_predictions": 0, "avg_risk_probability": 0.0, "high_risk_count": 0, "revenue_at_risk": 0.0}
    finally:
        conn.close()

def get_risk_segmentation(probability, monetary):
    if probability > 0.8 and monetary > 500:
        return "Critical"
    elif 0.5 <= probability <= 0.8:
        return "Important"
    else:
        return "Monitor"

def get_alerts(company_id):
    conn = get_db_connection()
    alerts = []
    try:
        cursor = conn.execute("SELECT * FROM predictions WHERE company_id = ? ORDER BY created_at DESC LIMIT 100", (company_id,))
        rows = cursor.fetchall()
        for row in rows:
            if row['inactivity_days'] > 90:
                alerts.append({"customer_id": row['customer_id'], "message": "Customer inactive", "type": "warning"})
            if row['monetary'] > 500 and row['probability'] > 0.7:
                alerts.append({"customer_id": row['customer_id'], "message": "High value customer at risk", "type": "danger"})
            if row['average_review_score'] < 3.0:
                alerts.append({"customer_id": row['customer_id'], "message": "Bad experience risk", "type": "warning"})
    except sqlite3.Error:
        pass
    finally:
        conn.close()
    
    unique_alerts = []
    seen = set()
    for alert in alerts:
        identifier = f"{alert['customer_id']}-{alert['message']}"
        if identifier not in seen:
            seen.add(identifier)
            unique_alerts.append(alert)
            
    return unique_alerts[:10]

def get_all_predictions_with_priority(company_id):
    conn = get_db_connection()
    try:
        cursor = conn.execute("SELECT * FROM predictions WHERE company_id = ? ORDER BY created_at DESC", (company_id,))
        rows = cursor.fetchall()
        results = []
        for row in rows:
            d = dict(row)
            d['priority'] = get_risk_segmentation(d['probability'], d['monetary'])
            results.append(d)
        return results
    except sqlite3.Error:
        return []
    finally:
        conn.close()

def get_predictions_by_batch(company_id, batch_id):
    conn = get_db_connection()
    try:
        cursor = conn.execute("SELECT * FROM predictions WHERE company_id = ? AND batch_id = ? ORDER BY created_at DESC", (company_id, batch_id))
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    except sqlite3.Error:
        return []
    finally:
        conn.close()

# ---------------------------------------------------------
# Deletion & Reset Handlers 
# ---------------------------------------------------------

def delete_prediction(company_id, prediction_id):
    conn = get_db_connection()
    try:
        cursor = conn.execute("DELETE FROM predictions WHERE id = ? AND company_id = ?", (prediction_id, company_id))
        conn.commit()
        return cursor.rowcount > 0
    except sqlite3.Error:
        return False
    finally:
        conn.close()

def delete_batch(company_id, batch_id):
    conn = get_db_connection()
    try:
        cursor = conn.execute("DELETE FROM predictions WHERE batch_id = ? AND company_id = ?", (batch_id, company_id))
        conn.commit()
        return cursor.rowcount > 0
    except sqlite3.Error:
        return False
    finally:
        conn.close()

def reset_company_data(company_id):
    conn = get_db_connection()
    try:
        cursor = conn.execute("DELETE FROM predictions WHERE company_id = ?", (company_id,))
        conn.commit()
        return True
    except sqlite3.Error as e:
        print(f"Error resetting data: {e}")
        return False
    finally:
        conn.close()
