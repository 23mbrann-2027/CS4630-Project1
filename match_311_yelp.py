import pandas as pd
import geopandas as gpd
import numpy as np
from sklearn.neighbors import NearestNeighbors

# Load cleaned datasets
df_311 = pd.read_csv("data/processed/philly_311_clean.csv")
df_yelp = pd.read_csv("data/processed/normalized_businesses.csv")

# Use only rows with valid coordinates
df_311 = df_311[(df_311["latitude"] != -1) & (df_311["longitude"] != -1)].copy()
df_yelp = df_yelp[(df_yelp["latitude"] != -1) & (df_yelp["longitude"] != -1)].copy()

# Convert to GeoDataFrames
gdf_311 = gpd.GeoDataFrame(
    df_311,
    geometry=gpd.points_from_xy(df_311.longitude, df_311.latitude),
    crs="EPSG:4326"
)

gdf_yelp = gpd.GeoDataFrame(
    df_yelp,
    geometry=gpd.points_from_xy(df_yelp.longitude, df_yelp.latitude),
    crs="EPSG:4326"
)

# Convert degrees to radians for haversine
coords_311 = np.radians(gdf_311[["latitude", "longitude"]])
coords_yelp = np.radians(gdf_yelp[["latitude", "longitude"]])

# Nearest Neighbor Model (1 neighbor, haversine distance)
nn = NearestNeighbors(n_neighbors=1, metric="haversine")
nn.fit(coords_yelp)
distances, indices = nn.kneighbors(coords_311)

# Convert radians to miles
earth_radius_miles = 3958.8
distances_miles = distances.flatten() * earth_radius_miles

# Distance threshold for matching
radius_miles = 0.5

# Prepare lists for matched Yelp info
matched_business_ids = []
matched_categories = []
matched_names = []
matched_addresses = []
matched_stars = []
matched_review_counts = []

# Loop over 311 complaints and nearest Yelp businesses
for dist, idx in zip(distances_miles, indices.flatten()):
    if dist <= radius_miles:
        yelp_row = gdf_yelp.iloc[idx]
        matched_business_ids.append(yelp_row["business_id"])
        #cat_col = "normalized_category" if "normalized_category" in gdf_yelp.columns else "categories"
        #matched_categories.append(yelp_row.get(cat_col, "unknown"))
        matched_categories.append(yelp_row.get("primary_category","unknown"))
        matched_names.append(yelp_row.get("name", "unknown"))
        matched_addresses.append(yelp_row.get("address", "unknown"))
        matched_stars.append(yelp_row.get("stars", np.nan))
        matched_review_counts.append(yelp_row.get("review_count", np.nan))
    else:
        matched_business_ids.append(np.nan)
        matched_categories.append(np.nan)
        matched_names.append(np.nan)
        matched_addresses.append(np.nan)
        matched_stars.append(np.nan)
        matched_review_counts.append(np.nan)

# Add columns to 311 GeoDataFrame
gdf_311["nearest_yelp_business_id"] = matched_business_ids
gdf_311["nearest_yelp_primary_category"] = matched_categories
gdf_311["nearest_yelp_name"] = matched_names
gdf_311["nearest_yelp_address"] = matched_addresses
gdf_311["nearest_yelp_stars"] = matched_stars
gdf_311["nearest_yelp_review_count"] = matched_review_counts
gdf_311["distance_miles"] = distances_miles

# Keep only matched rows
gdf_matched = gdf_311.dropna(subset=["nearest_yelp_business_id"])

# Save final matched dataset
gdf_matched.to_csv(
    "data/processed/311_yelp_geospatial_matches_full.csv",
    index=False
)

print(f"✓ Matched {len(gdf_matched)} complaints within {radius_miles} mile radius.")
