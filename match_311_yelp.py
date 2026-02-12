import pandas as pd
import geopandas as gpd
import numpy as np
from sklearn.neighbors import NearestNeighbors

# Load cleaned datasets

df_311 = pd.read_csv("data/processed/nyc_311_clean.csv")
df_yelp = pd.read_csv("data/processed/cleaned_business.csv")


# Use data that have coordinates

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

# convert degrees to radians
coords_311 = np.radians(gdf_311[["latitude", "longitude"]])
coords_yelp = np.radians(gdf_yelp[["latitude", "longitude"]])

# 5. Nearest Neighbor Model (Only want 1 and use earth distance)
nn = NearestNeighbors(
    n_neighbors=1,
    metric="haversine"
)

nn.fit(coords_yelp)
distances, indices = nn.kneighbors(coords_311)

# Convert radians to miles
earth_radius_miles = 3958.8
distances_miles = distances.flatten() * earth_radius_miles

# Apply distance threshold
radius_miles = 0.5
matched_business_ids = []
matched_categories = []

for dist, idx in zip(distances_miles, indices.flatten()):
    if dist <= radius_miles:
        matched_business_ids.append(gdf_yelp.iloc[idx]["business_id"])
        cat_col = "normalized_category" if "normalized_category" in gdf_yelp.columns else "categories"
        matched_categories.append(gdf_yelp.iloc[idx].get(cat_col, "unknown"))
    else:
        matched_business_ids.append(np.nan)
        matched_categories.append(np.nan)

gdf_311["nearest_yelp_business_id"] = matched_business_ids
gdf_311["nearest_yelp_category"] = matched_categories
gdf_311["distance_miles"] = distances_miles

# Keep matched rows only
gdf_matched = gdf_311.dropna(subset=["nearest_yelp_business_id"])

# Save output
gdf_matched.to_csv(
    "data/processed/311_yelp_geospatial_matches.csv",
    index=False
)

print(f"Matched {len(gdf_matched)} complaints within {radius_miles} mile radius.")
