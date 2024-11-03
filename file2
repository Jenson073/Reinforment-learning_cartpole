import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import mean_squared_error

# Generate synthetic data
X, _ = make_blobs(n_samples=200, centers=3, cluster_std=1.0, random_state=42)

# Function to visualize clusters
def plot_clusters(X, labels, title):
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o')
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

# DBSCAN Clustering
dbscan = DBSCAN(eps=0.5, min_samples=5)
initial_dbscan_labels = np.random.randint(0, 3, size=len(X))  # Random initial clusters
dbscan.fit(X)
dbscan_labels = dbscan.labels_

# For error rate, consider noise (-1) as a separate cluster
unique_labels = set(dbscan_labels)
centroids_dbscan = np.array([X[dbscan_labels == label].mean(axis=0) for label in unique_labels if label != -1])
error_rate_dbscan = mean_squared_error(X[dbscan_labels != -1], centroids_dbscan[dbscan_labels[dbscan_labels != -1]])

# Output for DBSCAN Clustering
print("DBSCAN Clustering")
print("1. Initial clusters:")
plot_clusters(X, initial_dbscan_labels, "Initial DBSCAN Clusters")

print("2. Final clusters:")
plot_clusters(X, dbscan_labels, "Final DBSCAN Clusters")

print(f"3. Final clusters with error rate: {error_rate_dbscan:.2f}")

# Gaussian Mixture Model (GMM) Clustering
gmm = GaussianMixture(n_components=3, random_state=42)
initial_gmm_labels = np.random.randint(0, 3, size=len(X))  # Random initial clusters
gmm.fit(X)
gmm_labels = gmm.predict(X)
error_rate_gmm = mean_squared_error(X, gmm.means_[gmm_labels])

# Output for GMM Clustering
print("\nGaussian Mixture Model (GMM) Clustering")
print("1. Initial clusters:")
plot_clusters(X, initial_gmm_labels, "Initial GMM Clusters")

print("2. Final clusters:")
plot_clusters(X, gmm_labels, "Final GMM Clusters")

print(f"3. Final clusters with error rate: {error_rate_gmm:.2f}")
