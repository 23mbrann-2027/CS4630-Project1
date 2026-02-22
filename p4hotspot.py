import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Better plot style
plt.style.use("ggplot")

# Load data
df = pd.read_csv("data/processed/311_yelp_matches_full.csv")

# Data Cleaning
df = df.dropna(subset=["latitude", "longitude"])

df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

df = df.dropna(subset=["latitude", "longitude"])

df["nearest_yelp_primary_category"] = df["nearest_yelp_primary_category"].fillna("unknown")
df["nearest_yelp_stars"] = df["nearest_yelp_stars"].fillna(df["nearest_yelp_stars"].median())
df["nearest_yelp_review_count"] = df["nearest_yelp_review_count"].fillna(0)

# Hotspot anaysis
df["lat_bin"] = pd.cut(df["latitude"], bins=20)
df["lon_bin"] = pd.cut(df["longitude"], bins=20)

hotspots = df.groupby(["lat_bin", "lon_bin"], observed=True).size().reset_index(name="count")
hotspots = hotspots[hotspots["count"] > 0]

top_hotspots = hotspots.sort_values(by="count", ascending=False).head(10)

# Create readable area labels
top_hotspots["area"] = top_hotspots.apply(
    lambda row: f"{round(row['lat_bin'].mid, 3)}, {round(row['lon_bin'].mid, 3)}",
    axis=1
)

print("\nTop Hotspots:")
print(top_hotspots[["area", "count"]])

# Plot hotspots
# plt.figure()
# plt.bar(top_hotspots["area"], top_hotspots["count"])
# plt.title("Top Complaint Hotspots (Areas)")
# plt.xlabel("Area (Lat, Lon)")
# plt.ylabel("Number of Complaints")
# plt.xticks(rotation=45)
# plt.grid(axis="y", alpha=0.3)
# plt.tight_layout()
# plt.show()


# Hexbin Hotspot Map (Geographic Heatmap)
plt.figure(figsize=(8, 6))

plt.hexbin(
    df["longitude"],
    df["latitude"],
    gridsize=60,        # resolution of the hex grid
    cmap="viridis",     # heatmap color scheme
    mincnt=1,            # only show bins with at least 1 complaint
    norm = LogNorm()
)

plt.colorbar(label="Complaint Count")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Complaint Hotspots (Hexbin Map)")
plt.tight_layout()
plt.show()


# Business vs complaints
business_counts = df.groupby("nearest_yelp_business_id").size().reset_index(name="complaint_count")
business_counts = business_counts.dropna(subset=["nearest_yelp_business_id"])

business_info = df[[
    "nearest_yelp_business_id",
    "nearest_yelp_stars",
    "nearest_yelp_review_count"
]].drop_duplicates()

business_df = business_counts.merge(business_info, on="nearest_yelp_business_id", how="left")

print("\nBusiness vs Complaints:")
print(business_df.head())

# Scatter plot 
plt.figure()

review_counts = np.log1p(business_df["nearest_yelp_review_count"])
complaints = np.log1p(business_df["complaint_count"])

plt.scatter(review_counts, complaints, alpha=0.5)

# Add trend line
z = np.polyfit(review_counts, complaints, 1)
p = np.poly1d(z)
plt.plot(review_counts, p(review_counts), color = "blue", linewidth = 2)

plt.xlabel("Business Popularity (Log Review Count)")
plt.ylabel("Nearby Complaints (Log Count)")
plt.title("Do Popular Businesses Have More Complaints Nearby?")
plt.grid(alpha=0.3)
plt.show()

# Category analysis
category_counts = df[
    (df["nearest_yelp_primary_category"] != "unknown") &
    (df["nearest_yelp_primary_category"].str.strip() != "")
]["nearest_yelp_primary_category"].value_counts().head(10)

category_counts.index = category_counts.index.str.title()

print("\nTop Business Categories Near Complaints:")
print(category_counts)

plt.figure()
category_counts.plot(kind="bar")
plt.title("Top Business Categories Near Complaints")
plt.xlabel("Category")
plt.ylabel("Number of Complaints")
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(axis="y", alpha=0.3)
plt.show()
# -----------------------
# Clustering (by location only)
# -----------------------
cluster_features = df[["latitude", "longitude"]].copy()  # only lat/lon

scaler = StandardScaler()
scaled_features = scaler.fit_transform(cluster_features)

# Optional: check inertia for 2–7 clusters
inertia = []
K_range = range(2, 8)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(scaled_features)
    inertia.append(km.inertia_)

# Final clustering (5 clusters)
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(scaled_features)

print("\nCluster Counts:")
print(df["cluster"].value_counts())

# Plot clusters (2D, geographic)
plt.figure()
scatter = plt.scatter(
    df["longitude"],
    df["latitude"],
    c=df["cluster"],
    alpha=0.6
)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Complaint Clusters (by Location)")

# Add legend
legend1 = plt.legend(*scatter.legend_elements(), title="Cluster")
plt.gca().add_artist(legend1)

plt.grid(alpha=0.3)
plt.show()


# Distance analysis
if "distance_miles" in df.columns:
    category_distance = df.groupby("nearest_yelp_primary_category")["distance_miles"].mean().sort_values()

    print("\nAvg Distance to Business by Category:")
    print(category_distance.head(10))

    plt.figure()
    category_distance.head(10).plot(kind="bar")
    plt.title("Average Distance to Business by Category")
    plt.ylabel("Miles")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.grid(axis="y", alpha=0.3)
    plt.show()

print("\nPhase 4 Analysis Complete!")
