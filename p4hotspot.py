import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import seaborn as sns
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
plt.figure()
plt.bar(top_hotspots["area"], top_hotspots["count"])
plt.title("Top Complaint Hotspots (Areas)")
plt.xlabel("Area (Lat, Lon)")
plt.ylabel("Number of Complaints")
plt.xticks(rotation=45)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()


# Hexbin Hotspot Map (Geographic Heatmap)
plt.figure(figsize=(8, 6))

plt.hexbin(
    df["longitude"],
    df["latitude"],
    gridsize=60,        # resolution of the hex grid (can)
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

# Apply log1p transformation
business_df['log_reviews'] = np.log1p(business_df["nearest_yelp_review_count"])
business_df['log_complaints'] = np.log1p(business_df["complaint_count"])

plt.figure(figsize=(10, 6))

# Use regplot for a much cleaner trend line + scatter combo
sns.regplot(
    data=business_df,
    x='log_reviews',
    y='log_complaints',
    scatter_kws={'alpha': 0.2, 's': 20}, # Lower alpha helps see density
    line_kws={'color': 'blue', 'lw': 3},
    x_jitter=0.1,  # Adds a tiny bit of horizontal "shake" to reveal overlaps
    y_jitter=0.1   # Adds a tiny bit of vertical "shake" to reveal overlaps
)

plt.xlabel("Business Popularity (Log Review Count)")
plt.ylabel("Nearby Complaints (Log Count)")
plt.title("Relationship Between Business Popularity and Nearby Complaints")
plt.grid(True, linestyle='--', alpha=0.5)

plt.show()


# Create spatial grid
# Round coordinates to create grid cells
df["lat_bin"] = df["latitude"].round(3)
df["lon_bin"] = df["longitude"].round(3)

# Count complaints and businesses per grid cell
area_df = df.groupby(["lat_bin", "lon_bin"]).agg(
    complaint_count=("service_request_id", "count"),
    business_count=("nearest_yelp_business_id", pd.Series.nunique)
).reset_index()

# Remove cells with 0 businesses
area_df = area_df[area_df["business_count"] > 0]

print("\nDensity vs Complaint Data:")
print(area_df.head())

# Log transform (helps visualization)
x = np.log1p(area_df["business_count"])
y = np.log1p(area_df["complaint_count"])

# Scatter plot
plt.figure(figsize=(8,6))

plt.scatter(x, y,alpha=0.5)

# Add trend line
z = np.polyfit(x, y, 1)
p = np.poly1d(z)
idx = np.argsort(x)

plt.plot(x.iloc[idx], p(x.iloc[idx]),color = "blue", linewidth=2)

# Labels and formatting
plt.xlabel("Business Density (Log Count)")
plt.ylabel("Complaint Frequency (Log Count)")
plt.title("Relationship Between Business Density and Complaint Frequency")
plt.grid(alpha=0.3)

plt.show()

# Correlation
corr = np.corrcoef(x, y)[0,1]

print("\nCorrelation between business density and complaints:", round(corr,3))

# Category analysis
# Count the top 10 Yelp business categories associated with nearby complaints
category_counts = df[
    (df["nearest_yelp_primary_category"] != "unknown") &
    (df["nearest_yelp_primary_category"].str.strip() != "")
]["nearest_yelp_primary_category"].value_counts().head(10)

# Clean up category labels for nicer plotting
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

# Clustering (by complaint type)
# Extract complaint text and fill missing values with empty strings
text_data = df["full_text"].fillna("")

# Convert complaint text into TF-IDF features
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=1000
)

X = vectorizer.fit_transform(text_data)

# Cluster complaints into 5 groups based on text similarity
kmeans = KMeans(
    n_clusters = 5,
    random_state = 42,
    n_init = 10
)

df["complaint_cluster"] = kmeans.fit_predict(X)

print("\nComplaint Clusters:")
print(df["complaint_cluster"].value_counts())

print("\nCluster vs Service Name:")

# For each cluster, show the top 5 most common service names
cluster_categories = (
    df.groupby("complaint_cluster")["service_name"]
    .value_counts()
    .groupby(level=0)
    .head(5)
)

print(cluster_categories)

import seaborn as sns

sns.countplot(data=df, x="complaint_cluster")
plt.title("Distribution of Complaint Clusters")
plt.show()

cluster_names = {
    0: "Infrastructure",
    1: "Maintenance",
    2: "Illegal Dumping",
    3: "Information Requests",
    4: "Waste Collection"
}

df["cluster_name"] = df["complaint_cluster"].map(cluster_names)

# Plot frequency of each cluster type
df["cluster_name"].value_counts().plot(kind="bar")

plt.title("Complaint Type Clusters")

plt.show()

# Create a table showing how complaint types vary by business category
pattern_table = pd.crosstab(
    df["nearest_yelp_primary_category"],
    df["cluster_name"]
)

print(pattern_table)

pattern_table.plot(
    kind="bar",
    stacked=True,
    figsize=(10,6)
)

plt.title("Complaint Patterns Near Different Business Categories")

plt.xlabel("Business Category")

plt.ylabel("Number of Complaints")

plt.legend(title="Complaint Type")

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()

# scaler = StandardScaler()
# scaled_features = scaler.fit_transform(cluster_features)

# # Optional: check inertia for 2–7 clusters
# inertia = []
# K_range = range(2, 8)
# for k in K_range:
#     km = KMeans(n_clusters=k, random_state=42, n_init=10)
#     km.fit(scaled_features)
#     inertia.append(km.inertia_)

# # Final clustering (5 clusters)
# kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
# df["cluster"] = kmeans.fit_predict(scaled_features)

# print("\nCluster Counts:")
# print(df["cluster"].value_counts())

# # Plot clusters (2D, geographic)
# plt.figure()
# scatter = plt.scatter(
#     df["longitude"],
#     df["latitude"],
#     c=df["cluster"],
#     alpha=0.6
# )
# plt.xlabel("Longitude")
# plt.ylabel("Latitude")
# plt.title("Complaint Clusters (by Location)")

# # Add legend
# legend1 = plt.legend(*scatter.legend_elements(), title="Cluster")
# plt.gca().add_artist(legend1)

# plt.grid(alpha=0.3)
# plt.show()

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
