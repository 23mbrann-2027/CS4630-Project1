import pandas as pd

df_business = pd.read_json(
    "data/raw/yelp_academic_dataset_business.json",
    lines = True
)
df_checkin =pd.read_json(
    "data/raw/yelp_academic_dataset_checkin.json",
    lines = True
)
df_review = pd.read_json(
    "data/raw/yelp_academic_dataset_review.json",
    lines = True
)

#Testing Dataframes
#------------------------------------------------------------
df_business.head()
df_business.shape
df_business.columns

df_checkin.head()
df_checkin.shape
df_checkin.columns

df_review.head()
df_review.shape
df_review.columns
#------------------------------------------------------------

def standardize_columns(df):
    df.columns = (
        df.columns
        .str.lower()
        .str.strip()
        .str.replace(" ", "_")
    )
    return df

df_business = standardize_columns(df_business)
df_checkin = standardize_columns(df_checkin)
df_review = standardize_columns(df_review)

#Business Category Normalization
df_business["categories"] = (
    df_business["categories"]
    .str.lower()
    .str.strip()
)

df_review["text"] = (
    df_review["text"]
    .str.lower()
    .str.strip()
)

#Check For valid Latitude and Longitude
df_business = df_business[
    df_business["latitude"].between(-90, 90) &
    df_business["longitude"].between(-180, 180)
]
df_business = df_business.dropna(subset = ["latitude", "longitude"])

#Check Zip Codes
df_business["postal_code"] = (
    df_business["postal_code"]
    .astype(str)
    .str.extract(r"(\d{5})")
)

df_business["city"] = df_business["city"].str.lower().str.strip()
df_business["state"] = df_business["state"].str.lower().str.strip()

# Removing Duplicate Records
before = df_business.shape[0]

df_business = df_business.drop_duplicates(subset = "business_id")

after = df_business.shape[0]
print(f"Removed {before - after} duplicate businesses")

df_review = df_review.drop_duplicates(subset = "review_id") # Removes duplicate reviews

df_checkin = df_checkin.drop_duplicates(subset = ["business_id", "date"]) # Removes Checkin Duplicates

# Handling Missing Values
df_business = df_business.dropna(subset = ["business_id"])

df_business["categories"] = df_business["categories"].fillna("unknown") # Fills in the missing categories with 'unknown'

# Filtering Business Dataset to NYC businesses
nyc_business_ids = set(df_business["business_id"])



# Review Dataset
df_review = df_review.dropna(subset = ["text"]) # Drops reviews without any text

df_review = df_review.dropna(subset = ["stars"]) # Drops reviews missing ratings 


# Checkin Dataset
df_checkin = df_checkin.dropna(subset = ["date"]) # Drops rows with no checkin info

