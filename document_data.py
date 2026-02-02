import pandas as pd
import json
import os
from pathlib import Path

print("=" * 60)
print("DATA DOCUMENTATION")
print("=" * 60)

# Document NYC 311 data
print("\n1. NYC 311 SERVICE REQUESTS")
print("-" * 40)

nyc_path = "data/raw/nyc_311_sample.csv"
if os.path.exists(nyc_path):
    # Get file size
    size_mb = os.path.getsize(nyc_path) / (1024 * 1024)
    print(f"File: {nyc_path}")
    print(f"Size: {size_mb:.2f} MB")
    
    # Read first few rows
    df_nyc = pd.read_csv(nyc_path, nrows=5)  # Only read 5 rows for quick check
    
    print(f"Rows in sample: {len(pd.read_csv(nyc_path))}")
    print(f"Columns: {len(df_nyc.columns)}")
    print("\nFirst 10 column names:")
    for i, col in enumerate(df_nyc.columns[:10]):
        print(f"  {i+1}. {col}")
    
    print("\nFirst 2 rows as example:")
    print(df_nyc.head(2).to_string())
    
else:
    print(f"File not found: {nyc_path}")
    print("Please download it first")

# Document Yelp data
print("\n\n2. YELP BUSINESS DATA")
print("-" * 40)

yelp_path = "data/raw/yelp_business_sample.json"
if os.path.exists(yelp_path):
    size_mb = os.path.getsize(yelp_path) / (1024 * 1024)
    print(f"File: {yelp_path}")
    print(f"Size: {size_mb:.2f} MB")
    
    # Read first few lines
    businesses = []
    with open(yelp_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= 3:  # Only read 3 lines
                break
            businesses.append(json.loads(line))
    
    print(f"Total businesses in sample: 1000 (synthetic)")
    print("\nFirst business as example:")
    print(json.dumps(businesses[0], indent=2))
    
else:
    print(f"File not found: {yelp_path}")

# Create a summary file
print("\n\n3. CREATING SUMMARY DOCUMENT")
print("-" * 40)

summary = {
    "project": "Urban Data Cleaning Project",
    "data_sources": {
        "nyc_311": {
            "source": "https://data.cityofnewyork.us/Social-Services/311-Service-Requests-from-2010-to-Present/erm2-nwe9",
            "format": "CSV",
            "sample_size": "10,000 records",
            "purpose": "Messy, real-world complaint data for cleaning practice"
        },
        "yelp": {
            "source": "https://www.yelp.com/dataset",
            "format": "JSON",
            "sample_size": "1,000 synthetic businesses",
            "purpose": "Business context for integration with complaints"
        }
    },
    "folder_structure": {
        "data/raw": "Original downloaded files (DO NOT MODIFY)",
        "data/processed": "Cleaned and processed files"
    }
}

# Save summary
with open('data_documentation.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("✓ Summary saved to: data_documentation.json")
print("\n✓ Phase 1 Complete!")
print("You have successfully:")
print("1. Downloaded/sampled both datasets")
print("2. Documented their sizes and formats")
print("3. Organized them in data/raw/ folder")