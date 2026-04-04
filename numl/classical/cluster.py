"""
numl/classical/cluster.py
--------------------------
Clustering algorithms:
  - KMeans (with kmeans++ initialization)
  - DBSCAN
"""

import numpy as np


class KMeans:
    """
    K-Means clustering with kmeans++ initialization.

    Parameters
    ----------
    n_clusters   : int (default 8)
    init         : 'kmeans++' or 'random' (default 'kmeans++')
    max_iter     : int (default 300)
    tol          : float — centroid movement convergence threshold (default 1e-4)
    n_init       : int — number of restarts; best (lowest inertia) is kept (default 10)
    random_state : int or None
    """

    def __init__(self, n_clusters: int = 8, init: str = 'kmeans++',
                 max_iter: int = 300, tol: float = 1e-4,
                 n_init: int = 10, random_state=None):
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = None

    def _init_centroids(self, X: np.ndarray) -> np.ndarray:
        n = len(X)
        if self.init == 'random':
            idx = np.random.choice(n, self.n_clusters, replace=False)
            return X[idx].copy()

        # kmeans++ initialization
        centroids = [X[np.random.randint(n)].copy()]
        for _ in range(1, self.n_clusters):
            # Distance from each point to nearest existing centroid
            dists = np.array([min(np.sum((x - c) ** 2) for c in centroids) for x in X])
            probs = dists / dists.sum()
            cumulative = np.cumsum(probs)
            r = np.random.rand()
            idx = np.searchsorted(cumulative, r)
            centroids.append(X[min(idx, n - 1)].copy())
        return np.array(centroids)

    def _run_once(self, X: np.ndarray):
        n = len(X)
        centroids = self._init_centroids(X)

        for it in range(self.max_iter):
            # Assignment step: find nearest centroid for each point
            dists = np.array([[np.sum((x - c) ** 2) for c in centroids] for x in X])
            labels = np.argmin(dists, axis=1)

            # Update step: recompute centroids
            new_centroids = np.zeros_like(centroids)
            for k in range(self.n_clusters):
                mask = labels == k
                if mask.sum() > 0:
                    new_centroids[k] = X[mask].mean(axis=0)
                else:
                    # Empty cluster: reinitialize to a random point
                    new_centroids[k] = X[np.random.randint(n)]

            # Check convergence
            shift = np.max(np.linalg.norm(new_centroids - centroids, axis=1))
            centroids = new_centroids
            if shift < self.tol:
                break

        # Compute inertia
        inertia = sum(np.sum((X[labels == k] - centroids[k]) ** 2)
                      for k in range(self.n_clusters))
        return centroids, labels, float(inertia), it + 1

    def fit(self, X: np.ndarray) -> "KMeans":
        if self.random_state is not None:
            np.random.seed(self.random_state)
        X = np.asarray(X, dtype=np.float64)

        best_inertia = np.inf
        best_centroids = None
        best_labels = None
        best_n_iter = 0

        for _ in range(self.n_init):
            centroids, labels, inertia, n_iter = self._run_once(X)
            if inertia < best_inertia:
                best_inertia = inertia
                best_centroids = centroids
                best_labels = labels
                best_n_iter = n_iter

        self.cluster_centers_ = best_centroids
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        self.n_iter_ = best_n_iter
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        dists = np.array([[np.sum((x - c) ** 2)
                           for c in self.cluster_centers_] for x in X])
        return np.argmin(dists, axis=1)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).labels_

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Return distances to each cluster center."""
        X = np.asarray(X, dtype=np.float64)
        return np.array([[np.linalg.norm(x - c) for c in self.cluster_centers_]
                         for x in X])

    def score(self, X: np.ndarray) -> float:
        """Negative inertia (for consistency with sklearn convention)."""
        X = np.asarray(X, dtype=np.float64)
        labels = self.predict(X)
        inertia = sum(np.sum((X[labels == k] - self.cluster_centers_[k]) ** 2)
                      for k in range(self.n_clusters))
        return -float(inertia)


class DBSCAN:
    """
    DBSCAN (Density-Based Spatial Clustering of Applications with Noise).

    Points are core points if they have >= min_samples neighbors within eps.
    Clusters are grown from core points; outliers are labeled -1.

    Parameters
    ----------
    eps        : float — radius for neighborhood (default 0.5)
    min_samples: int — minimum points for a core point (default 5)
    metric     : 'euclidean' or 'manhattan' (default 'euclidean')
    """

    def __init__(self, eps: float = 0.5, min_samples: int = 5,
                 metric: str = 'euclidean'):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.labels_ = None
        self.core_sample_indices_ = None
        self.n_features_in_ = None

    def _pairwise_distances(self, X: np.ndarray) -> np.ndarray:
        if self.metric == 'euclidean':
            sq = np.sum(X ** 2, axis=1)
            dists = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * X @ X.T, 0))
        elif self.metric == 'manhattan':
            dists = np.sum(np.abs(X[:, None] - X[None, :]), axis=2)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
        return dists

    def fit(self, X: np.ndarray) -> "DBSCAN":
        X = np.asarray(X, dtype=np.float64)
        n = len(X)
        self.n_features_in_ = X.shape[1]

        dists = self._pairwise_distances(X)
        neighbors = [np.where(dists[i] <= self.eps)[0] for i in range(n)]
        is_core = np.array([len(nb) >= self.min_samples for nb in neighbors])

        labels = np.full(n, -1, dtype=int)
        cluster_id = 0

        for i in range(n):
            if labels[i] != -1 or not is_core[i]:
                continue
            # Start a new cluster
            labels[i] = cluster_id
            queue = list(neighbors[i])
            while queue:
                j = queue.pop(0)
                if labels[j] == -1:
                    labels[j] = cluster_id
                    if is_core[j]:
                        queue.extend(neighbors[j].tolist())
                elif labels[j] == -2:  # previously labeled noise
                    labels[j] = cluster_id
            cluster_id += 1

        self.labels_ = labels
        self.core_sample_indices_ = np.where(is_core)[0]
        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).labels_
