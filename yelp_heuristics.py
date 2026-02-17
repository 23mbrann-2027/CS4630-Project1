from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pandas as pd
import re

# Function to load the cleaned reviews
def load_reviews(path):
    df = pd.read_parquet(path)

    print(f"Total Reviews loaded: {len(df)}")

    return df
# Same filtering function used in yelp_training.py
def filter_complaints(df, max_stars = 2):

    complaint_df = df[df["stars"] <= max_stars].copy()

    print(f"Complaint Reviews: {len(complaint_df)}")

    return complaint_df

# Extracting common keywords from the reviews, as well as the importance score of the word
def extract_keywords(df, max_features = 50):

    vectorizer = TfidfVectorizer(
        stop_words = "english",
        max_features = max_features
    )
    X = vectorizer.fit_transform(df["text"])

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

# Extracts bigrams(Two word phrases), stores them into a new dataframe/file
def extract_bigrams(df, min_freq = 20):
    vectorizer = CountVectorizer(
        stop_words = "english",
        ngram_range = (2, 2),
        min_df = min_freq
    )

    X = vectorizer.fit_transform(df["text"])

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


# Same bigram map used in "yelp_training.py"
def create_bigram_category_map():

    return {

        # SERVICE
        "customer service": "Service",
        "service": "Service",
        "slow service": "Service",
        "terrible service": "Service",

        # STAFF
        "staff": "Staff Behavior",
        "staffs": "Staff Behavior",
        "employee": "Staff Behavior",
        "employees": "Staff Behavior",
        "rude staff": "Staff Behavior",
        "friendly staff": "Staff Behavior",

        # WAIT TIME
        "wait": "Wait Time",
        "wait time": "Wait Time",
        "long wait": "Wait Time",

        # FOOD
        "food": "Food Quality",
        "cold food": "Food Quality",
        "bad food": "Food Quality",

        # CLEANLINESS
        "dirty": "Cleanliness",
        "clean": "Cleanliness",
        "filthy": "Cleanliness",

        # PRICE
        "price": "Price",
        "expensive": "Price",
        "overpriced": "Price",

        # ORDER
        "order": "Order Accuracy",
        "wrong order": "Order Accuracy",

        # ENVIRONMENT
        "atmosphere": "Environment",
        "environment": "Environment"
    }

# Applies categories to bigrams, not using ML
def categorize_bigram(bigram, category_map):
    for keyword, category in category_map.items():
        if keyword in bigram:
            return category
    return "Other"



# Saves the common keywords & Bigrams to new files
def save_results(tfidf_df, bigram_df):
    tfidf_df.to_csv("data/processed/complaint_keywords.csv", index=False)

    bigram_df.to_csv("data/processed/complaint_phrases.csv", index=False)

    print("Saved complaint_keywords.csv")
    print("Saved complaint_phrases.csv")

if __name__ == "__main__":
    

    path = "data/processed/cleaned_reviews.parquet"

    df = load_reviews(path)

    # category_map = create_bigram_category_map()

    complaint_df = filter_complaints(df)

    tfidf_df = extract_keywords(complaint_df)

    bigram_df = extract_bigrams(complaint_df)
    print("\nTop bigrams with categories:\n", bigram_df.head(10))

 

    print("\nTop Keywords:")
    print(tfidf_df.head(20))

    print("\nTop Bigrams:")
    print(bigram_df.head(20))

    save_results(tfidf_df, bigram_df)