import pandas as pd
import geopandas as gpd
from scipy.spatial.distance import cdist
import numpy as np
from sklearn.metrics import chi2_kernel, hellinger_kernel
from sklearn.preprocessing import MinMaxScaler

# Load neighborhood-level data
demographics = pd.read_csv("neighborhood_demographics.csv", index_col="neighborhood_id")
housing = pd.read_csv("neighborhood_housing.csv", index_col="neighborhood_id")
amenities = pd.read_csv("neighborhood_amenities.csv", index_col="neighborhood_id")
geo_boundaries = gpd.read_file("neighborhood_boundaries.shp")

# Custom distance metrics for different data types
def chi_square_dist(x, y):
    return chi2_kernel(x.values.reshape(1, -1), y.values.reshape(1, -1))

def hellinger_dist(x, y):
    return hellinger_kernel(x.values.reshape(1, -1), y.values.reshape(1, -1))

def gowers_dist(x, y):
    cols = x.columns
    weighted_dists = []
    for col in cols:
        if x[col].dtype == 'object':
            weighted_dists.append(hellinger_dist(x[col], y[col]))
        else:
            weighted_dists.append(np.abs(x[col] - y[col]))
    return np.mean(weighted_dists)

# Calculate component similarities  
demo_sim = cdist(demographics, demographics, metric=chi_square_dist)
race_sim = cdist(pd.get_dummies(demographics["race"]), pd.get_dummies(demographics["race"]), metric=hellinger_dist)    
housing_sim = cdist(MinMaxScaler().fit_transform(housing), MinMaxScaler().fit_transform(housing))
amenities_sim = cdist(amenities, amenities, metric=gowers_dist)

# Combine into holistic similarity score
weights = [0.3, 0.2, 0.3, 0.2]  # Example weights
similarity_matrix = weights[0] * demo_sim + weights[1] * race_sim + weights[2] * housing_sim + weights[3] * amenities_sim   

# Get top 10 most similar neighborhoods for a query
query_id = 125  # Lincoln Park
sorted_sims = sorted(list(enumerate(similarity_matrix[query_id])), key=lambda x: x[1])
top_10_ids = [x[0] for x in sorted_sims[:10]]
top_10_names = geo_boundaries.loc[top_10_ids, "neighborhood_name"].tolist()
print(f"For {geo_boundaries.loc[query_id, 'neighborhood_name']}, the top 10 most similar are:")
print(top_10_names)