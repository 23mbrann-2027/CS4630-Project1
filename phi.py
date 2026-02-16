import pandas as pd
import numpy as np
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

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



df["service_name"] = (
    df["service_name"]
    .astype(str)
    .str.lower()
    .str.strip()
    .str.replace(r"[^a-z0-9\s]", "", regex=True)
)



# clean data

df["created_date"] = pd.to_datetime(df["created_date"], errors="coerce")
df["closed_date"] = pd.to_datetime(df["closed_date"], errors="coerce")
df["created_day"] = df["created_date"].dt.date

# location

df["zip_code"] = df["zip_code"].str.extract(r"(\d{5})")
df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

# missing values

df["status"] = df["status"].fillna("unknown")
df["subject"] = df["subject"].fillna("")
df["status_notes"] = df["status_notes"].fillna("")

df = df.dropna(subset=["service_name", "created_date"])

# 7. text construction

GENERIC_NOTES = [
    "issue resolved",
    "closed",
    "completed",
    "work completed"
]

def build_text(row):
    subject = row["subject"].strip()
    notes = row["status_notes"].strip().lower()

    if notes in GENERIC_NOTES:
        return subject

    if subject and notes:
        return subject + " " + notes

    return subject if subject else notes

df["full_text"] = df.apply(build_text, axis=1)
df.loc[df["full_text"] == "", "full_text"] = df["service_name"]



# remove exact duplicate request IDs
df = df.drop_duplicates(subset=["service_request_id"])

df = df.sort_values("created_date")
df = df.drop_duplicates(
    subset=["address", "service_name", "created_date"],
    keep="first"
)

# analysis

sia = SentimentIntensityAnalyzer()
df["sentiment"] = df["full_text"].apply(
    lambda x: sia.polarity_scores(str(x))["compound"]
)

# rule base severity

SEVERITY_KEYWORDS = {
    "high": ["emergency", "danger", "fire", "hazard"],
    "medium": ["noise", "leak", "odor", "blocked"],
    "low": ["dirty", "broken", "request"]
}

def estimate_severity(text):
    text = str(text).lower()
    for level, words in SEVERITY_KEYWORDS.items():
        if any(word in text for word in words):
            return level
    return "unknown"

df["severity"] = df["full_text"].apply(estimate_severity)


# Keyword

vectorizer = CountVectorizer(
    stop_words="english",
    max_features=30,
    ngram_range=(1, 2)
)

ngrams = vectorizer.fit_transform(df["full_text"])
keywords = vectorizer.get_feature_names_out()

df["top_keywords"] = [
    ", ".join([keywords[i] for i in row.nonzero()[1][:3]])
    for row in ngrams
]

df["top_keywords"] = df["top_keywords"].replace("", np.nan)
df["top_keywords"] = df["top_keywords"].fillna(df["service_name"])

# ML

df_ml = df.dropna(subset=["service_name", "full_text"])

X_train, X_test, y_train, y_test = train_test_split(
    df_ml["full_text"],
    df_ml["service_name"],
    test_size=0.2,
    random_state=42,
    stratify=df_ml["service_name"]
)

tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)

X_train_vec = tfidf.fit_transform(X_train)
X_test_vec = tfidf.transform(X_test)

clf = LogisticRegression(
    max_iter=500,
    class_weight="balanced"
)

clf.fit(X_train_vec, y_train)

y_pred = clf.predict(X_test_vec)

print("\nModel Accuracy:", clf.score(X_test_vec, y_test))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

df["predicted_service_name"] = clf.predict(
    tfidf.transform(df["full_text"])
)



# Save

df.to_csv("data/processed/philly_311_clean.csv", index=False)

print("\n✓ Final 311 cleaning + NLP + ML pipeline complete.")
