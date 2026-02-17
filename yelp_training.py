import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from tqdm import tqdm

# Function to load the cleaned reviews file
def load_reviews(path):
    df = pd.read_parquet(path)

    print(f"Total Reviews loaded: {len(df)}")

    return df

# Creates a map that is used to classify the complaints
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
        "frozen food": "Food Quality",
        

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

# Creates "Complaints" anything 2 stars or less
def filter_complaints(df, max_stars = 2):

    complaint_df = df[df["stars"] <= max_stars].copy()

    print(f"Complaint Reviews: {len(complaint_df)}")

    return complaint_df

# Returns the label that goes on the filtered review
def label_review(text, category_map):
    text = text.lower()
    for keyword, category in category_map.items():
        if keyword in text:
            return category
    return "Other"

# Puts the reviews in vector form
# Standard format for machine learning models
def vectorize_text(df, max_features = 2000):
    vectorizer = TfidfVectorizer(stop_words="english", max_features=max_features, ngram_range = (1,2))
    X = vectorizer.fit_transform(df["text"])
    y = df["category"]
    return X, y, vectorizer

# Defining the models used, as well as running the training loop on the data
def train_models(X_train, y_train):
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter = 1000,
            n_jobs = -1
        ),
        "SVM": LinearSVC(),
        "Random Forest": RandomForestClassifier(
            n_estimators = 50,
            max_depth = 20,
            n_jobs = -1,
            random_state = 42
        )
    }
    trained = {}
    for name, model, in models.items():
        model.fit(X_train, y_train)
        trained[name] = model
    return trained

# Testing the models to see if there is good generalization
# Also checks for under/overfitting
def evaluate_models(models, X_test, y_test):
    for name, model in models.items():
        print(f"\n=== {name} ===")
        preds = model.predict(X_test)
        print(classification_report(y_test, preds))


if __name__ == "__main__":
    tqdm.pandas()
    path = "data/processed/cleaned_reviews.parquet"
    df = load_reviews(path)

    complaint_df = filter_complaints(df)
    category_map = create_bigram_category_map()

    complaint_df["category"] = complaint_df["text"].progress_apply(lambda x: label_review(x, category_map))
    print("\nSample labeled reviews:\n", complaint_df[["text","category"]].head(5))

    complaint_df[["business_id","stars","text", "category"]].to_csv(
        "data/processed/complaint_phrases.csv",
        index = False
    )

    X,y, vectorizer = vectorize_text(complaint_df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = train_models(X_train, y_train)
    evaluate_models(models, X_test, y_test)