import pandas as pd
import re

def load_businesses(path):
    df = pd.read_csv(path)

    print(f"Total Businesses: {len(df)}")

    return df

def primary_category(category_string):
    if pd.isna(category_string):
        return "Other"
    
    category_string = category_string.lower()


    #Create a map for the business and their respective categories

    if "restaurant" in category_string:
        return "Restaurant"
    
    elif "beauty" in category_string or "spa" in category_string or "barber" in category_string:
        return "Beauty"
    
    elif "bar" in category_string or "nightlife" in category_string:
        return "Bar/Nightlife"
    
    elif "coffee" in category_string or "tea" in category_string or "bakery" in category_string:
        return "Cafe/Bakery"
    
    elif "grocery" in category_string or "food" in category_string:
        return "Food Retail"
    
    elif "shopping" in category_string or "fashion" in category_string:
        return "Retail"
    
    elif "doctor" in category_string or "dentist" in category_string or "medical" in category_string:
        return "Healthcare"
    
    elif "automotive" in category_string or "gas station" in category_string:
        return "Automotive"
    
    elif "home service" in category_string or "cleaning" in category_string:
        return "Home Services"
    
    elif "hotel" in category_string or "travel" in category_string:
        return "Travel"
    
    elif "entertainment" in category_string or "museum" in category_string:
        return "Entertainment"
    
    else: 
        return "Other"
    
def category_collection(category_string):
    if pd.isna(category_string):
        return ["Other"]
    
    category_string = category_string.lower()
    categories = []

    if "restaurant" in category_string:
        categories.append("Restaurant")

    if "beauty" in category_string or "spa" in category_string or "barber" in category_string:
        categories.append("Beauty")

    if "bar" in category_string or "nightlife" in category_string:
        categories.append("Bar/Nightlife")

    if "coffee" in category_string or "tea" in category_string or "bakery" in category_string:
        categories.append("Cafe/Bakery")

    if "grocery" in category_string or "food" in category_string or "convenience" in category_string:
        categories.append("Food Retail")

    if "shopping" in category_string or "fashion" in category_string:
        categories.append("Retail")

    if "doctor" in category_string or "dentist" in category_string or "medical" in category_string:
        categories.append("Healthcare")

    if "automotive" in category_string or "gas station" in category_string:
        categories.append("Automotive")

    if "home service" in category_string or "cleaning" in category_string:
        categories.append("Home Services")

    if "hotel" in category_string or "travel" in category_string:
        categories.append("Travel")

    if "entertainment" in category_string or "museum" in category_string:
        categories.append("Entertainment")

    if len(categories) == 0:
        categories.append("Other")

    return categories
    
if __name__ == "__main__":

    path = "data/processed/cleaned_business.csv"

    df = load_businesses(path)

    df["primary_category"] = df["categories"].apply(primary_category)
    df["grouped_categories"] = df["categories"].apply(category_collection)

    df.to_csv("data/processed/normalized_businesses.csv", index = False)

    print(df["primary_category"].value_counts())