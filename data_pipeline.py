import os
import pandas as pd
import numpy as np
from pathlib import Path

# Configuration
BASE_DIR = Path(__file__).resolve().parent
# Check both possible locations for raw data
if (BASE_DIR / 'data' / 'olist_raw').exists():
    RAW_DATA_DIR = BASE_DIR / 'data' / 'olist_raw'
elif (BASE_DIR / 'olist_raw').exists():
    RAW_DATA_DIR = BASE_DIR / 'olist_raw'
else:
    RAW_DATA_DIR = BASE_DIR / 'data' / 'olist_raw'

PROCESSED_DATA_PATH = BASE_DIR / 'data' / 'processed_ecommerce_data.csv'
CHURN_THRESHOLD_DAYS = 90  # Prediction horizon window in days

def load_data():
    """Loads all necessary Olist CSV files into Pandas DataFrames."""
    print("Loading raw data files...")
    
    customers = pd.read_csv(os.path.join(RAW_DATA_DIR, 'olist_customers_dataset.csv'))
    orders = pd.read_csv(os.path.join(RAW_DATA_DIR, 'olist_orders_dataset.csv'))
    order_items = pd.read_csv(os.path.join(RAW_DATA_DIR, 'olist_order_items_dataset.csv'))
    payments = pd.read_csv(os.path.join(RAW_DATA_DIR, 'olist_order_payments_dataset.csv'))
    reviews = pd.read_csv(os.path.join(RAW_DATA_DIR, 'olist_order_reviews_dataset.csv'))
    products = pd.read_csv(os.path.join(RAW_DATA_DIR, 'olist_products_dataset.csv'))
    
    return customers, orders, order_items, payments, reviews, products

def clean_and_merge(customers, orders, order_items, payments, reviews, products):
    """Merges relational tables and processes date formats."""
    print("Cleaning and merging datasets...")
    
    # 1. Convert dates to datetime
    date_columns = ['order_purchase_timestamp', 'order_delivered_customer_date', 'order_estimated_delivery_date']
    for col in date_columns:
        orders[col] = pd.to_datetime(orders[col])
        
    # Merge orders with customers
    orders_customers = pd.merge(orders, customers, on='customer_id', how='inner')
    
    # Focus on delivered orders only to ensure valid purchases
    delivered_orders = orders_customers[orders_customers['order_status'] == 'delivered'].copy()
    
    # 2. Aggregate Payments to order level
    order_payments = payments.groupby('order_id').agg(
        total_payment=('payment_value', 'sum'),
        payment_count=('payment_sequential', 'count')
    ).reset_index()
    
    # 3. Aggregate Reviews to order level
    order_reviews = reviews.groupby('order_id').agg(
        review_score=('review_score', 'mean')
    ).reset_index()
    
    # 4. Bring items with product categories to order level
    order_items_prod = pd.merge(order_items[['order_id', 'product_id']], products[['product_id', 'product_category_name']], on='product_id', how='left')
    order_category_count = order_items_prod.groupby('order_id').agg(
        order_category_diversity=('product_category_name', 'nunique')
    ).reset_index()
    
    # 5. Bring everything together at the Order level
    df_merged = pd.merge(delivered_orders, order_payments, on='order_id', how='left')
    df_merged = pd.merge(df_merged, order_reviews, on='order_id', how='left')
    df_merged = pd.merge(df_merged, order_category_count, on='order_id', how='left')
    
    # Calculate delivery delay in days (actual - estimated)
    # Positive means delayed, negative means early
    df_merged['delivery_delay'] = (df_merged['order_delivered_customer_date'] - df_merged['order_estimated_delivery_date']).dt.total_seconds() / (24 * 3600)
    
    return df_merged, order_items_prod

def create_customer_features(df_merged, order_items_prod):
    """
    Aggregates order-level data into a single customer-level dataset using
    a scientifically valid observation window and future prediction window (cutoff date).
    Prevents target leakage by computing features strictly before the cutoff.
    """
    print("Building customer-level features with strict temporal cutoff to eliminate target leakage...")
    
    # Identify dataset temporal span
    min_date = df_merged['order_purchase_timestamp'].min()
    max_date = df_merged['order_purchase_timestamp'].max()
    print(f"Dataset date range: {min_date} to {max_date}")
    
    # Temporal Split: Cutoff separates observation window from future prediction window
    # Prediction window (horizon) = 90 days prior to dataset end
    prediction_window_days = CHURN_THRESHOLD_DAYS  # 90 days
    cutoff_date = max_date - pd.Timedelta(days=prediction_window_days)
    print(f"Observation window: [{min_date.date()} to {cutoff_date.date()}]")
    print(f"Prediction horizon: [{cutoff_date.date()} to {max_date.date()}] ({prediction_window_days} days)")
    
    # 1. Historical data strictly within the observation window (<= cutoff_date)
    df_obs = df_merged[df_merged['order_purchase_timestamp'] <= cutoff_date].copy()
    
    # Aggregate customer behavioral predictors available strictly at cutoff date
    customer_obs = df_obs.groupby('customer_unique_id').agg(
        first_purchase_date=('order_purchase_timestamp', 'min'),
        last_purchase_date=('order_purchase_timestamp', 'max'),
        total_orders=('order_id', 'nunique'),
        total_spend=('total_payment', 'sum'),
        average_review_score=('review_score', 'mean'),
        payment_count=('payment_count', 'sum'),
        delivery_delay=('delivery_delay', 'mean')
    ).reset_index()
    
    # Category diversity calculated strictly within the observation window
    obs_orders = df_obs[['order_id', 'customer_unique_id']].drop_duplicates()
    obs_items = pd.merge(obs_orders, order_items_prod, on='order_id', how='inner')
    obs_categories = obs_items.groupby('customer_unique_id').agg(
        category_diversity=('product_category_name', 'nunique')
    ).reset_index()
    customer_obs = pd.merge(customer_obs, obs_categories, on='customer_unique_id', how='left')
    
    # Calculate predictors as of the cutoff date
    # Recency: elapsed days from customer's latest purchase in observation window up to cutoff date
    customer_obs['recency'] = (cutoff_date - customer_obs['last_purchase_date']).dt.total_seconds() / (24 * 3600)
    customer_obs['frequency'] = customer_obs['total_orders']
    customer_obs['monetary'] = customer_obs['total_spend']
    
    # Safe division for average order value
    customer_obs['average_order_value'] = np.where(
        customer_obs['total_orders'] > 0,
        customer_obs['total_spend'] / customer_obs['total_orders'],
        0.0
    )
    
    # Impute missing values for observation metrics
    customer_obs['average_review_score'] = customer_obs['average_review_score'].fillna(customer_obs['average_review_score'].median())
    customer_obs['delivery_delay'] = customer_obs['delivery_delay'].fillna(0.0)
    customer_obs['category_diversity'] = customer_obs['category_diversity'].fillna(1.0)
    customer_obs['payment_count'] = customer_obs['payment_count'].fillna(1.0)

    
    # 2. Future activity during prediction window (> cutoff_date)
    df_future = df_merged[df_merged['order_purchase_timestamp'] > cutoff_date]
    future_active_customers = set(df_future['customer_unique_id'].unique())
    
    # 3. Target Definition:
    # A customer is churned (1) if they had NO purchases during the future 90-day prediction window
    # A customer is retained (0) if they made at least one purchase during the future 90-day prediction window
    customer_obs['churn'] = customer_obs['customer_unique_id'].apply(
        lambda cid: 0 if cid in future_active_customers else 1
    )
    
    # Inactivity days at prediction time is identical to recency (days inactive up to cutoff)
    # Keeping both preserves the existing 9-feature schema without data leakage
    customer_obs['inactivity_days'] = customer_obs['recency']
    
    customer_obs.rename(columns={'customer_unique_id': 'customer_id'}, inplace=True)
    customer_obs['first_purchase_date'] = customer_obs['first_purchase_date'].dt.date
    customer_obs['last_purchase_date'] = customer_obs['last_purchase_date'].dt.date
    
    cols = [
        'customer_id', 'first_purchase_date', 'last_purchase_date', 'inactivity_days',
        'recency', 'frequency', 'monetary', 'total_orders', 'total_spend', 'average_order_value',
        'category_diversity', 'average_review_score', 'payment_count', 'delivery_delay',
        'churn'
    ]
    
    return customer_obs[cols]

def main():
    print("--- Starting Olist Data Pipeline ---")
    
    # Ensure data directory exists
    if not os.path.exists(RAW_DATA_DIR):
        print(f"Error: Raw data directory '{RAW_DATA_DIR}' not found.")
        print("Please ensure the extracted Olist CSVs are placed there.")
        return
        
    try:
        # Pipeline Steps
        customers, orders, order_items, payments, reviews, products = load_data()
        
        df_merged, order_items_prod = clean_and_merge(
            customers, orders, order_items, payments, reviews, products
        )
        
        final_customer_df = create_customer_features(df_merged, order_items_prod)
        
        # Ensure data folder exists
        os.makedirs(PROCESSED_DATA_PATH.parent, exist_ok=True)
        final_customer_df.to_csv(PROCESSED_DATA_PATH, index=False)
        print(f"\nData successfully processed and saved to: {PROCESSED_DATA_PATH}")
        
        # Summary printing
        total_customers = len(final_customer_df)
        churned = final_customer_df['churn'].sum()
        active = total_customers - churned
        
        print("\n--- Data Summary ---")
        print(f"Total Cohort Customers: {total_customers:,}")
        print(f"Retained in Prediction Horizon: {active:,} ({active/total_customers*100:.1f}%)")
        print(f"Churned in Prediction Horizon: {churned:,} ({churned/total_customers*100:.1f}%)")
        print("------------------------------------")
        
    except Exception as e:
        print(f"\nPipeline failed due to an error: {str(e)}")

if __name__ == "__main__":
    main()

