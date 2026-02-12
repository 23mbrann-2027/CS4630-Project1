import pandas as pd
import os


def standardize_columns(df):
    df.columns = (
        df.columns
        .str.lower()
        .str.strip()
        .str.replace(" ", "_")
    )
    return df

def clean_business_json(path):
    df = pd.read_json(path, lines = True)
    df = standardize_columns(df)

    #Business Category Normalization
    df["categories"] = (
        df["categories"]
        .str.lower()
        .str.strip()
    )

    #Check For valid Latitude and Longitude
    df = df[
        df["latitude"].between(-90, 90) &
        df["longitude"].between(-180, 180)
    ]
    df = df.dropna(subset = ["latitude", "longitude"])

    #Check Zip Codes
    df["postal_code"] = (
        df["postal_code"]
        .astype(str)
        .str.extract(r"(\d{5})")
    )

    df["city"] = df["city"].str.lower().str.strip()
    df["state"] = df["state"].str.lower().str.strip()

    # Removing Duplicate Records
    before = df.shape[0]

    df = df.drop_duplicates(subset = "business_id")

    after = df.shape[0]
    print(f"Removed {before - after} duplicate businesses")

    # Handling Missing Values
    df = df.dropna(subset = ["business_id"])

    df["categories"] = df["categories"].fillna("unknown") # Fills in the missing categories with 'unknown'

    return df


def clean_checkin_json(path):
    df = pd.read_json(path, lines = True)
    df = standardize_columns(df)

    df = df.drop_duplicates(subset = ["business_id", "date"]) # Removes Checkin Duplicates

    df = df.dropna(subset = ["date"]) # Drops rows with no checkin info

    return df


def clean_review_json(input_path, output_path, business_ids):

    chunk_iter = pd.read_json(
        input_path,
        lines = True,
        chunksize = 100_000
    )

    first_chunk = True

    for chunk in chunk_iter:
        chunk = standardize_columns(chunk)

        chunk = chunk.dropna(subset = ["text", "stars"])
        chunk = chunk.drop_duplicates(subset = "review_id") # Removes duplicate reviews

        chunk["text"] = (
            chunk["text"]
            .str.lower()
            .str.strip()
        )

        chunk = chunk[chunk["business_id"].isin(business_ids)]
        chunk["review_length"] = chunk["text"].str.len()
        chunk["word_count"] = chunk["text"].str.split().str.len()

        reduced = chunk[[
            "review_id",
            "business_id",
            "stars",
            "review_length",
            "word_count",
            "date"
        ]]

        reduced.to_csv(
            output_path,
            mode = 'w' if first_chunk else 'a',
            header = first_chunk,
            index = False
        )
        first_chunk = False

    print("Review Dataset Processed")


if __name__ == "__main__":
    business_path = "data/raw/yelp_academic_dataset_business.json"
    checkin_path = "data/raw/yelp_academic_dataset_checkin.json"
    review_path = "data/raw/yelp_academic_dataset_review.json"

    processed_review_path = "data/raw/cleaned_reviews.csv"

    df_business = clean_business_json(business_path)
    df_checkin = clean_checkin_json(checkin_path)

    df_business.to_csv(
        "data/processed/cleaned_business.csv",
        index = False
    )
    df_checkin.to_csv(
        "data/processed/cleaned_checkin.csv",
        index = False
    )

    # Filtering Business Dataset to NYC businesses
    nyc_business_ids = set(df_business["business_id"])

    clean_review_json(
        review_path,
        processed_review_path,
        nyc_business_ids
    )








