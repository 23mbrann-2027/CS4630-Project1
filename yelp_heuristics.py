from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pandas as pd
import re

def load_reviews(path):
    df = pd.read_parquet(path)

    print(f"Total Reviews loaded: {len(df)}")

    return df

def filter_complaints(df, max_stars = 2):

    complaint_df = df[df["stars"] <= max_stars].copy()

    print(f"Complaint Reviews: {len(complaint_df)}")

    return complaint_df

def extract_keywords(df, text_column = "text", max_features = 50):

    vectorizer = TfidfVectorizer(
        stop_words = "english",
        max_features = max_features
    )
    X = vectorizer.fit_transform(df[text_column])

    scores = X.sum(axis = 0).A1

    words = vectorizer.get_feature_names_out()
    keyword_df = pd.DataFrame({
        "keyword": words,
        "importance": scores
    })

    keyword_df = keyword_df.sort_values(
        by = "importance",
        ascending = False
    )

    return keyword_df

def extract_bigrams(df, text_column = "text", min_freq = 20):
    vectorizer = CountVectorizer(
        stop_words = "english",
        ngram_range = (2, 2),
        min_df = min_freq
    )

    X = vectorizer.fit_transform(df[text_column])

    counts = X.sum(axis = 0).A1
    phrases = vectorizer.get_feature_names_out()

    bigram_df = pd.DataFrame({
        "phrase": phrases,
        "count": counts
    })

    bigram_df = bigram_df.sort_values(
        by = "count",
        ascending = False
    )

    return bigram_df

def save_results(tfidf_df, bigram_df):
    tfidf_df.to_csv("data/processed/complaint_keywords.csv", index=False)

    bigram_df.to_csv("data/processed/complaint_phrases.csv", index=False)

    print("Saved complaint_keywords.csv")
    print("Saved complaint_phrases.csv")

if __name__ == "__main__":
    

    path = "data/processed/cleaned_reviews.parquet"

    df = load_reviews(path)

    complaint_df = filter_complaints(df)

    tfidf_df = extract_keywords(complaint_df)

    bigram_df = extract_bigrams(complaint_df)

    print("\nTop Keywords:")
    print(tfidf_df.head(20))

    print("\nTop Bigrams:")
    print(bigram_df.head(20))

    save_results(tfidf_df, bigram_df)