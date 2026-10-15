import os
import zipfile
import glob

def extract_dataset():
    # Search for common zip names for Olist dataset
    target_zips = ["archive.zip", "brazilian-ecommerce.zip", "*.zip"]
    
    found_zip = None
    for pattern in target_zips:
        matches = glob.glob(pattern)
        if matches:
            found_zip = matches[0]
            break
            
    if not found_zip:
        # Check in data/ folder too
        matches = glob.glob("data/*.zip")
        if matches:
            found_zip = matches[0]
            
    if not found_zip:
        print("❌ Error: Olist dataset ZIP file not found in root or data/ directory.")
        print("Please ensure you have downloaded the Brazilian E-Commerce Dataset (Olist) and placed the ZIP in the project root.")
        return False

    extract_path = "data/olist_raw"
    os.makedirs(extract_path, exist_ok=True)
    
    print(f"📦 Extracting {found_zip} to {extract_path}...")
    try:
        with zipfile.ZipFile(found_zip, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print("✅ Extraction complete!")
        return True
    except Exception as e:
        print(f"❌ Failed to extract: {str(e)}")
        return False

if __name__ == "__main__":
    extract_dataset()
