import os
import pandas as pd

# Create data directory if it doesn't exist
os.makedirs("data", exist_ok=True)

print("Downloading IBM Telco Customer Churn dataset...")
url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
df = pd.read_csv(url)

# Save to local directory
df.to_csv("data/telco_churn.csv", index=False)
print("Dataset successfully saved to data/telco_churn.csv")
