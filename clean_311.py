import pandas as pd
import numpy as np
import re
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer

# -------------------------
# Load data
# -------------------------
df = pd.read_csv("data/raw/nyc_311_sample.csv")

# -------------------------
# Standardize column names
# -------------------------
df.columns = (
    df.columns
      .str.lower()
      .str.strip()
      .str.replace(" ", "_")
)

# -------------------------
# Normalize complaint types
# -------------------------
df["complaint_type"] = (
    df["complaint_type"]
      .str.lower()
      .str.strip()
)

# -------------------------
# Clean location fields
# -------------------------
for col in ["latitude", "longitude"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["zip_code"] = (
    df["incident_zip"]
      .astype(str)
      .str.extract(r"(\d{5})")
)

df["borough"] = (
    df["borough"]
      .str.upper()
      .replace("UNSPECIFIED", np.nan)
)

# Drop rows missing critical geo info
df = df.dropna(subset=["latitude", "longitude"])

# -------------------------
# Deduplication
# Rule: same complaint_type, same location, same day
# -------------------------
df["created_date"] = pd.to_datetime(df["created_date"], errors="coerce")
df["created_day"] = df["created_date"].dt.date

df = df.drop_duplicates(
    subset=["complaint_type", "latitude", "longitude", "created_day"]
)

# -------------------------
# Handle missing text
# -------------------------
df["descriptor"] = df["descriptor"].fillna("")

# -------------------------
# Sentiment (lexicon-based)
# -------------------------
sia = SentimentIntensityAnalyzer()

df["sentiment"] = df["descriptor"].apply(
    lambda x: sia.polarity_scores(x)["compound"]
)

# -------------------------
# Severity heuristic
# -------------------------
SEVERITY_KEYWORDS = {
    "high": ["emergency", "danger", "fire", "hazard"],
    "medium": ["noise", "leak", "odor"],
    "low": ["dirty", "broken", "missed"]
}

def estimate_severity(text):
    text = text.lower()
    for level, words in SEVERITY_KEYWORDS.items():
        if any(w in text for w in words):
            return level
    return "unknown"

df["severity"] = df["descriptor"].apply(estimate_severity)

# -------------------------
# Keyword extraction (n-grams)
# -------------------------
vectorizer = CountVectorizer(
    stop_words="english",
    max_features=20,
    ngram_range=(1, 2)
)

ngrams = vectorizer.fit_transform(df["descriptor"])
keywords = vectorizer.get_feature_names_out()

df["top_keywords"] = [
    ", ".join([keywords[i] for i in row.nonzero()[1][:3]])
    for row in ngrams
]

# -------------------------
# Save cleaned data
# -------------------------
df.to_csv("data/processed/nyc_311_clean.csv", index=False)
print("✓ NYC 311 cleaning complete")
