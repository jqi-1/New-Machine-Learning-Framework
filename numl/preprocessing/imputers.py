"""
numl/preprocessing/imputers.py
--------------------------------
Missing value imputation.
"""

import numpy as np


class SimpleImputer:
    """
    Fill missing values (NaN) using a simple strategy.

    Parameters
    ----------
    strategy : 'mean', 'median', 'most_frequent', or 'constant'
    fill_value : used when strategy='constant'
    """

    def __init__(self, strategy: str = 'mean', fill_value=0.0):
        assert strategy in ('mean', 'median', 'most_frequent', 'constant')
        self.strategy = strategy
        self.fill_value = fill_value
        self.statistics_ = None

    def fit(self, X: np.ndarray) -> "SimpleImputer":
        X = np.asarray(X, dtype=np.float64)
        self.statistics_ = np.zeros(X.shape[1])
        for j in range(X.shape[1]):
            col = X[:, j]
            valid = col[~np.isnan(col)]
            if self.strategy == 'mean':
                self.statistics_[j] = valid.mean() if len(valid) > 0 else 0.0
            elif self.strategy == 'median':
                self.statistics_[j] = np.median(valid) if len(valid) > 0 else 0.0
            elif self.strategy == 'most_frequent':
                if len(valid) > 0:
                    vals, counts = np.unique(valid, return_counts=True)
                    self.statistics_[j] = vals[np.argmax(counts)]
                else:
                    self.statistics_[j] = 0.0
            else:  # constant
                self.statistics_[j] = self.fill_value
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64).copy()
        for j in range(X.shape[1]):
            mask = np.isnan(X[:, j])
            X[mask, j] = self.statistics_[j]
        return X

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class KNNImputer:
    """
    Fill missing values using k-Nearest Neighbor imputation.

    For each missing value, finds the k nearest non-missing rows (using
    available features) and uses their weighted mean to impute.

    Parameters
    ----------
    n_neighbors : int, default 5
    weights     : 'uniform' or 'distance'
    """

    def __init__(self, n_neighbors: int = 5, weights: str = 'uniform'):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.X_fit_ = None

    def fit(self, X: np.ndarray) -> "KNNImputer":
        self.X_fit_ = np.asarray(X, dtype=np.float64)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64).copy()
        n_rows, n_cols = X.shape

        for i in range(n_rows):
            missing_cols = np.where(np.isnan(X[i]))[0]
            if len(missing_cols) == 0:
                continue

            # Features available for distance computation
            avail_cols = np.where(~np.isnan(X[i]))[0]
            if len(avail_cols) == 0:
                # All features missing — use column mean from training
                for j in missing_cols:
                    valid = self.X_fit_[:, j]
                    valid = valid[~np.isnan(valid)]
                    X[i, j] = valid.mean() if len(valid) > 0 else 0.0
                continue

            # Compute distances from row i to all training rows
            # using only available columns
            dists = []
            for k, row in enumerate(self.X_fit_):
                both_avail = avail_cols[~np.isnan(row[avail_cols])]
                if len(both_avail) == 0:
                    dists.append(np.inf)
                else:
                    d = np.sqrt(np.sum((X[i, both_avail] - row[both_avail]) ** 2))
                    dists.append(d)

            dists = np.array(dists)
            # Find k nearest with non-inf distance
            sorted_idx = np.argsort(dists)
            neighbors = sorted_idx[:self.n_neighbors]

            for j in missing_cols:
                neighbor_vals = []
                neighbor_dists = []
                for nb in neighbors:
                    if not np.isnan(self.X_fit_[nb, j]):
                        neighbor_vals.append(self.X_fit_[nb, j])
                        neighbor_dists.append(dists[nb])

                if not neighbor_vals:
                    # Fallback to column mean
                    col_vals = self.X_fit_[:, j]
                    valid = col_vals[~np.isnan(col_vals)]
                    X[i, j] = valid.mean() if len(valid) > 0 else 0.0
                elif self.weights == 'distance' and any(d > 0 for d in neighbor_dists):
                    # Inverse distance weighting
                    nd = np.array(neighbor_dists)
                    nd = np.maximum(nd, 1e-10)
                    w = 1.0 / nd
                    X[i, j] = np.average(neighbor_vals, weights=w)
                else:
                    X[i, j] = np.mean(neighbor_vals)

        return X

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)
