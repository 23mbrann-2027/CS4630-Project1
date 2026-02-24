import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder

# load data


df = pd.read_csv(
    "data/raw/philly_311_2025.csv",
    dtype={"zipcode": str},
    low_memory=False
)

# Standardize column names
df.columns = (
    df.columns
    .str.lower()
    .str.strip()
    .str.replace(" ", "_")
)


# rename columns


df.rename(columns={
    "requested_datetime": "created_date",
    "closed_datetime": "closed_date",
    "zipcode": "zip_code",
    "lat": "latitude",
    "lon": "longitude"
}, inplace=True)

# clean time
datetime_cols = ["created_date", "closed_date"]
for col in datetime_cols:
    if col in df.columns:
        df[col] = df[col].astype(str).str.replace(r"\+\d{2}(:\d{2})?$", "", regex=True)
        df[col] = pd.to_datetime(df[col], errors="coerce")

# clean text columns
df["service_name"] = (
    df["service_name"]
    .astype(str)
    .str.lower()
    .str.strip()
    .str.replace(r"[^a-z0-9\s]", "", regex=True)
)

df["status"] = df["status"].fillna("unknown")
df["subject"] = df["subject"].fillna("").astype(str)
df["status_notes"] = df["status_notes"].fillna("").astype(str)

# Standardize status values (fix capitalization + spacing)
df["status"] = df["status"].astype(str).str.lower().str.strip()

# Determine open/closed based on closed_date
has_closed_date = df["closed_date"].notna()

# Override status to ensure consistency
df.loc[has_closed_date, "status"] = "closed"
df.loc[~has_closed_date, "status"] = "open"

empty_notes = df["status_notes"].str.strip() == ""

# Fill missing status_notes
mask_closed = empty_notes & (df["status"].str.lower() == "closed")
df.loc[mask_closed, "status_notes"] = "Issue Resolved"

mask_open = empty_notes & (df["status"].str.lower() != "closed")
df.loc[mask_open, "status_notes"] = "Open"

# Clean ZIP and numeric coordinates
df["zip_code"] = df["zip_code"].str.extract(r"(\d{5})")
df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

# Flag missing lat/lon
df["latlon_missing"] = df["latitude"].isna() | df["longitude"].isna()

# Impute lat/lon using ZIP centroid
zip_centroids = df[df["latlon_missing"] == False].groupby("zip_code")[["latitude", "longitude"]].median()

mask_zip_fillable = df["zip_code"].isin(zip_centroids.index) & df["latlon_missing"]
if mask_zip_fillable.any():
    df.loc[mask_zip_fillable, ["latitude", "longitude"]] = df.loc[mask_zip_fillable].join(
        zip_centroids, on="zip_code", rsuffix="_centroid"
    )[["latitude_centroid", "longitude_centroid"]]

# Impute missing lat/lon and ZIP within groups defined by a single column
group_col = "agency_responsible" 
if group_col in df.columns:
    # Loop through each group
    for group_value in df[group_col].dropna().unique():
        group_rows = df[group_col] == group_value
        # Only use rows that already have lat/lon
        group_coords = df[group_rows & (~df["latlon_missing"])]
        if not group_coords.empty:
            group_median = group_coords[["latitude", "longitude"]].median()
            # Fill missing lat/lon for this group
            mask_missing = group_rows & df["latlon_missing"]
            df.loc[mask_missing, ["latitude", "longitude"]] = group_median.values

            # Fill missing ZIPs for this group using mode
            missing_zip_mask = group_rows & df["zip_code"].isna()
            if missing_zip_mask.any():
                common_zip = df.loc[group_rows, "zip_code"].mode()
                if not common_zip.empty:
                    df.loc[missing_zip_mask, "zip_code"] = common_zip[0]

# Flag which rows were imputed
df["latlon_imputed"] = df["latlon_missing"]

# Drop columns not important
df = df.drop(columns=["latitude_centroid", "longitude_centroid", "latlon_missing"], errors="ignore")
# Empty media
if "media_url" in df.columns:
    df["media_url"] = df["media_url"].fillna("").astype(str)
    df.loc[df["media_url"].str.strip() == "", "media_url"] = "no_media"

# Remove columns
cols_to_drop = ["updated_datetime", "expected_datetime"]
df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

# remove duplicates
if "service_request_id" in df.columns:
    df = df.drop_duplicates(subset=["service_request_id"])

df = df.sort_values("created_date", ascending=False)
df = df.drop_duplicates(subset=["address", "service_name", "created_date"], keep="first")

# Build text for ML
GENERIC_NOTES = {"issue resolved", "closed", "completed", "work completed"}

def build_text(row):
    subject = row["subject"].strip()
    notes = row["status_notes"].strip().lower()
    if notes in GENERIC_NOTES:
        return subject
    if subject and notes:
        return f"{subject} {notes}"
    return subject if subject else notes

df["full_text"] = df.apply(build_text, axis=1)
df.loc[df["full_text"] == "", "full_text"] = df["service_name"]

# RRule base
SEVERITY_KEYWORDS = {
    "high": ["emergency", "danger", "fire", "hazard"],
    "medium": ["noise", "leak", "blocked"],
    "low": ["broken", "request"]
}

def estimate_severity(text):
    text = str(text).lower()
    for level, words in SEVERITY_KEYWORDS.items():
        if any(word in text for word in words):
            return level
    return "unknown"

df["severity"] = df["full_text"].apply(estimate_severity)

# Keywords
vectorizer = CountVectorizer(stop_words="english", max_features=30, ngram_range=(1,2))
ngrams = vectorizer.fit_transform(df["full_text"])
keywords = vectorizer.get_feature_names_out()
df["top_keywords"] = [
    ", ".join([keywords[i] for i in row.nonzero()[1][:3]])
    for row in ngrams
]
df["top_keywords"] = df["top_keywords"].replace("", np.nan)
df["top_keywords"] = df["top_keywords"].fillna(df["service_name"])

# ML classification
df_ml = df.dropna(subset=["service_name", "full_text"])
X_train, X_test, y_train, y_test = train_test_split(
    df_ml["full_text"], df_ml["service_name"], test_size=0.2, random_state=42, stratify=df_ml["service_name"]
)

tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
X_train_vec = tfidf.fit_transform(X_train)
X_test_vec = tfidf.transform(X_test)

clf = LogisticRegression(max_iter=500, class_weight="balanced")
clf.fit(X_train_vec, y_train)
y_pred = clf.predict(X_test_vec)

print("\nModel Accuracy:", clf.score(X_test_vec, y_test))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

df["predicted_service_name"] = clf.predict(tfidf.transform(df["full_text"]))

# Save data
df.to_csv("data/processed/philly_311_clean.csv", index=False)
print("\n✓ Final 311 clean + formatted + ML dataset saved.")

