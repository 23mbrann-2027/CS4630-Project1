import pandas as pd
import numpy as np
import re
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load data safely
df = pd.read_csv(
    "data/raw/nyc_311_full.csv",
    dtype={"incident_zip": str},
    low_memory=False
)

# Standardize column names
df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_")

# Drop unused/redundant columns
unused_cols = [
    "vehicle_type", "taxi_company_borough", "taxi_pick_up_location",
    "bridge_highway_name", "bridge_highway_direction", "road_ramp",
    "bridge_highway_segment", "due_date"
]
df.drop(columns=[c for c in unused_cols if c in df.columns], inplace=True)

# Normalize complaint types
df["complaint_type"] = df["complaint_type"].str.lower().str.strip()

# Clean location fields
for col in ["latitude", "longitude"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["zip_code"] = df["incident_zip"].str.extract(r"(\d{5})")
df["borough"] = df["borough"].str.upper().replace("UNSPECIFIED", np.nan)
df["has_coordinates"] = df[["latitude", "longitude"]].notna().all(axis=1)

# Date processing
df["created_date"] = pd.to_datetime(df["created_date"], errors="coerce")
df["closed_date"] = pd.to_datetime(df["closed_date"], errors="coerce")

# Resolution time in hours
df["resolution_time_hours"] = (
    (df["closed_date"] - df["created_date"]).dt.total_seconds() / 3600
)

# Status column
df["status"] = np.where(df["closed_date"].isna(), "Open", "Closed")

# Optionally, show 'Open' in closed_date column
df["closed_date"] = df["closed_date"].fillna("Open")

# Combine descriptors
if "descriptor_2" in df.columns:
    df["full_descriptor"] = (
        df["descriptor"].fillna("") + " " + df["descriptor_2"].fillna("")
    ).str.strip()
else:
    df["full_descriptor"] = df["descriptor"].fillna("")

# Drop original descriptor columns
df.drop(columns=[c for c in ["descriptor", "descriptor_2"] if c in df.columns], inplace=True)

# Combine intersection streets
if "intersection_street_1" in df.columns and "intersection_street_2" in df.columns:
    df["intersection_streets"] = (
        df["intersection_street_1"].fillna("") + " & " + df["intersection_street_2"].fillna("")
    ).str.strip(" &")
df.drop(columns=[c for c in ["intersection_street_1", "intersection_street_2"] if c in df.columns], inplace=True)

# Combine cross streets
if "cross_street_1" in df.columns and "cross_street_2" in df.columns:
    df["cross_streets"] = (
        df["cross_street_1"].fillna("") + " & " + df["cross_street_2"].fillna("")
    ).str.strip(" &")
df.drop(columns=[c for c in ["cross_street_1", "cross_street_2"] if c in df.columns], inplace=True)

# Deduplication
# Rule: same complaint_type, same address, same full_descriptor, same day = likely duplicate
df["created_day"] = df["created_date"].dt.date
df = df.drop_duplicates(
    subset=["complaint_type", "incident_address", "full_descriptor", "created_day"]
)

# Handle missing values
df["full_descriptor"] = df["full_descriptor"].replace("", "No description")
df["resolution_time_hours"] = df["resolution_time_hours"].fillna(df["resolution_time_hours"].median())

# Sentiment (lexicon-based)
sia = SentimentIntensityAnalyzer()
df["sentiment"] = df["full_descriptor"].apply(lambda x: sia.polarity_scores(str(x))["compound"])

# Severity heuristic
SEVERITY_KEYWORDS = {
    "high": ["emergency", "danger", "fire", "hazard"],
    "medium": ["noise", "leak", "odor"],
    "low": ["dirty", "broken", "missed"]
}

def estimate_severity(text):
    text = str(text).lower()
    for level, words in SEVERITY_KEYWORDS.items():
        if any(w in text for w in words):
            return level
    return "unknown"

df["severity"] = df["full_descriptor"].apply(estimate_severity)

# Keyword extraction (n-grams)
vectorizer = CountVectorizer(stop_words="english", max_features=20, ngram_range=(1,2))
ngrams = vectorizer.fit_transform(df["full_descriptor"])
keywords = vectorizer.get_feature_names_out()
df["top_keywords"] = [
    ", ".join([keywords[i] for i in row.nonzero()[1][:3]])
    for row in ngrams
]

# ML Complaint Classification (Traditional ML)
df_ml = df.dropna(subset=["complaint_type"])
X_train, X_test, y_train, y_test = train_test_split(
    df_ml["full_descriptor"], df_ml["complaint_type"], test_size=0.2, random_state=42
)
tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
X_train_vec = tfidf.fit_transform(X_train)
X_test_vec = tfidf.transform(X_test)
clf = LogisticRegression(max_iter=200)
clf.fit(X_train_vec, y_train)
df["predicted_complaint_type"] = clf.predict(tfidf.transform(df["full_descriptor"]))

# Yelp Category Normalization (Rule-based)
def normalize_yelp_categories(cat_string):
    if pd.isna(cat_string):
        return np.nan
    cat_string = cat_string.lower()
    if "pizza" in cat_string:
        return "pizza restaurant"
    if "coffee" in cat_string or "cafe" in cat_string:
        return "cafe"
    if "bar" in cat_string or "pub" in cat_string:
        return "bar"
    if "restaurant" in cat_string:
        return "restaurant"
    return "other"

if "categories" in df.columns:
    df["normalized_category"] = df["categories"].apply(normalize_yelp_categories)

# Save cleaned data
df.to_csv("data/processed/nyc_311_clean.csv", index=False)
print("✓ Phase 2 cleaning, enrichment, and ML classification complete")

