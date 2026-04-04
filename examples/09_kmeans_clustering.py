"""
09_kmeans_clustering.py
KMeans and DBSCAN clustering on synthetic blob data.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from numl.classical.cluster import KMeans, DBSCAN
from numl.metrics.clustering import (
    silhouette_score, davies_bouldin_score, adjusted_rand_score
)

print("=" * 50)
print("Example 9: KMeans & DBSCAN Clustering")
print("=" * 50)

# Generate blobs
np.random.seed(0)
centers = np.array([[0.0, 0.0], [5.0, 0.0], [2.5, 4.0]])
n_per = 100

X = np.vstack([
    np.random.randn(n_per, 2) * 0.5 + c
    for c in centers
])
y_true = np.repeat([0, 1, 2], n_per)

print(f"\nDataset: {len(X)} samples, 3 true clusters")

# --- KMeans ---
print("\n--- KMeans ---")
km = KMeans(n_clusters=3, max_iter=300, n_init=5, random_state=0)
km.fit(X)
km_labels = km.labels_

sil = silhouette_score(X, km_labels)
db = davies_bouldin_score(X, km_labels)
ari = adjusted_rand_score(y_true, km_labels)
print(f"  Inertia:          {km.inertia_:.4f}")
print(f"  Silhouette score: {sil:.4f}  (closer to 1 = better)")
print(f"  Davies-Bouldin:   {db:.4f}  (closer to 0 = better)")
print(f"  Adjusted Rand:    {ari:.4f}  (1 = perfect)")

# Effect of k
print("\nInertia vs number of clusters (elbow method):")
for k in range(1, 7):
    km_k = KMeans(n_clusters=k, n_init=3, random_state=0)
    km_k.fit(X)
    bar = '#' * int(km_k.inertia_ / 50)
    print(f"  k={k}: inertia={km_k.inertia_:8.2f}  {bar}")

# --- DBSCAN ---
print("\n--- DBSCAN ---")
db_alg = DBSCAN(eps=0.8, min_samples=5)
db_labels = db_alg.fit_predict(X)
n_clusters = len(set(db_labels) - {-1})
n_noise = np.sum(db_labels == -1)
print(f"  Clusters found: {n_clusters}  (expected 3)")
print(f"  Noise points:   {n_noise}")
if n_clusters > 1:
    mask = db_labels >= 0
    sil_db = silhouette_score(X[mask], db_labels[mask])
    ari_db = adjusted_rand_score(y_true[mask], db_labels[mask])
    print(f"  Silhouette score: {sil_db:.4f}")
    print(f"  Adjusted Rand:    {ari_db:.4f}")

# DBSCAN on rings (where KMeans fails)
print("\n--- DBSCAN on concentric rings (KMeans fails here) ---")
rng = np.random.RandomState(1)
theta = rng.uniform(0, 2 * np.pi, 200)
r_inner = 1.0 + rng.randn(100) * 0.1
r_outer = 3.0 + rng.randn(100) * 0.1
X_rings = np.vstack([
    np.column_stack([r_inner * np.cos(theta[:100]), r_inner * np.sin(theta[:100])]),
    np.column_stack([r_outer * np.cos(theta[100:]), r_outer * np.sin(theta[100:])]),
])
true_rings = np.array([0]*100 + [1]*100)

db_rings = DBSCAN(eps=0.5, min_samples=5)
ring_labels = db_rings.fit_predict(X_rings)
n_ring_clusters = len(set(ring_labels) - {-1})
ari_rings = adjusted_rand_score(true_rings, ring_labels)
print(f"  DBSCAN clusters: {n_ring_clusters}  (expected 2)")
print(f"  Adjusted Rand:   {ari_rings:.4f}")

km_rings = KMeans(n_clusters=2, random_state=0).fit(X_rings)
ari_km_rings = adjusted_rand_score(true_rings, km_rings.labels_)
print(f"  KMeans ARI:      {ari_km_rings:.4f}  (KMeans struggles with rings)")
