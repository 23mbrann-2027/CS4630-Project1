import pandas as pd
import numpy as np
import re
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load data

df = pd.read_csv("data/raw/nyc_311_full.csv")

# Standardize column names

df.columns = (
    df.columns.str.lower().str.strip().str.replace(" ", "_")
)

# Normalize complaint types

df["complaint_type"] = df["complaint_type"].str.lower().str.strip()

# Clean location fields

for col in ["latitude", "longitude"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["zip_code"] = df["incident_zip"].astype(str).str.extract(r"(\d{5})")

df["borough"] = (
    df["borough"].str.upper().replace("UNSPECIFIED", np.nan)
)

# KEEP rows even if missing location
df["has_coordinates"] = df[["latitude", "longitude"]].notna().all(axis=1)

# Date processing

df["created_date"] = pd.to_datetime(df["created_date"], errors="coerce")
df["closed_date"] = pd.to_datetime(df["closed_date"], errors="coerce")
df["resolution_time_hours"] = (
    (df["closed_date"] - df["created_date"]).dt.total_seconds() / 3600
)

# Deduplication
# Rule justification:
# Same complaint, same address, same description, same day = likely duplicate

df["created_day"] = df["created_date"].dt.date

df = df.drop_duplicates(
    subset=["complaint_type", "incident_address", "descriptor", "created_day"]
)

# Handle missing values

df["descriptor"] = df["descriptor"].fillna("No description")
df["resolution_time_hours"] = df["resolution_time_hours"].fillna(
    df["resolution_time_hours"].median()
)

# Sentiment (lexicon-based)

sia = SentimentIntensityAnalyzer()
df["sentiment"] = df["descriptor"].apply(
    lambda x: sia.polarity_scores(str(x))["compound"]
)

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

df["severity"] = df["descriptor"].apply(estimate_severity)

# Keyword extraction

vectorizer = CountVectorizer(stop_words="english", max_features=20, ngram_range=(1,2))
ngrams = vectorizer.fit_transform(df["descriptor"])
keywords = vectorizer.get_feature_names_out()

df["top_keywords"] = [
    ", ".join([keywords[i] for i in row.nonzero()[1][:3]])
    for row in ngrams
]

# ML Complaint Classification

# Use complaint_type as labels to train text classifier
df_ml = df.dropna(subset=["complaint_type"])

X_train, X_test, y_train, y_test = train_test_split(
    df_ml["descriptor"], df_ml["complaint_type"], test_size=0.2, random_state=42
)

tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
X_train_vec = tfidf.fit_transform(X_train)
X_test_vec = tfidf.transform(X_test)

clf = LogisticRegression(max_iter=200)
clf.fit(X_train_vec, y_train)

df["predicted_complaint_type"] = clf.predict(tfidf.transform(df["descriptor"]))

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

