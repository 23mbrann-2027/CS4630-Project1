import json
import pandas as pd
import numpy as np

rows = []

with open("data/raw/yelp_business_sample.json") as f:
    for line in f:
        b = json.loads(line)
        rows.append({
            "business_id": b.get("business_id"),
            "name": b.get("name"),
            "categories": b.get("categories"),
            "latitude": b.get("latitude"),
            "longitude": b.get("longitude"),
            "city": b.get("city"),
            "state": b.get("state"),
            "stars": b.get("stars")
        })

df = pd.DataFrame(rows)

# -------------------------
# Normalize categories
# -------------------------
df["categories"] = (
    df["categories"]
      .fillna("")
      .str.lower()
      .str.split(", ")
)

# Map similar categories
CATEGORY_MAP = {
    "restaurant": ["restaurants", "food", "dining"],
    "bar": ["bars", "pubs"],
    "retail": ["shopping", "store"]
}

def normalize_categories(cat_list):
    normalized = set()
    for c in cat_list:
        for group, aliases in CATEGORY_MAP.items():
            if c in aliases:
                normalized.add(group)
    return list(normalized)

df["normalized_categories"] = df["categories"].apply(normalize_categories)

# -------------------------
# Clean geo fields
# -------------------------
df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

df = df.dropna(subset=["latitude", "longitude"])

# -------------------------
# Save cleaned data
# -------------------------
df.to_csv("data/processed/yelp_business_clean.csv", index=False)
print("✓ Yelp business cleaning complete")
