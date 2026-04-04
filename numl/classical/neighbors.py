"""
numl/classical/neighbors.py
----------------------------
K-Nearest Neighbors with optional KD-Tree for faster queries.
"""

import numpy as np
import heapq


# ---------------------------------------------------------------------------
# KD-Tree (pure Python, for fast nearest-neighbor search)
# ---------------------------------------------------------------------------

class _KDNode:
    __slots__ = ['point', 'idx', 'left', 'right', 'axis']

    def __init__(self, point, idx, axis, left=None, right=None):
        self.point = point
        self.idx = idx
        self.axis = axis
        self.left = left
        self.right = right


class _KDTree:
    """Minimal pure-Python KD-tree for approximate k-nearest neighbor search."""

    def __init__(self, X: np.ndarray):
        self.n_features = X.shape[1]
        self.root = self._build(np.arange(len(X)), X, depth=0)
        self.X = X

    def _build(self, indices, X, depth):
        if len(indices) == 0:
            return None
        axis = depth % self.n_features
        sorted_idx = indices[np.argsort(X[indices, axis])]
        mid = len(sorted_idx) // 2
        node = _KDNode(
            point=X[sorted_idx[mid]],
            idx=int(sorted_idx[mid]),
            axis=axis,
        )
        node.left = self._build(sorted_idx[:mid], X, depth + 1)
        node.right = self._build(sorted_idx[mid + 1:], X, depth + 1)
        return node

    def query(self, x: np.ndarray, k: int):
        """Return (distances, indices) of k nearest neighbors to x."""
        # Max-heap: stores (-dist, idx)
        heap = []  # (-dist, idx)

        def _search(node):
            if node is None:
                return
            d = np.linalg.norm(x - node.point)
            if len(heap) < k:
                heapq.heappush(heap, (-d, node.idx))
            elif d < -heap[0][0]:
                heapq.heapreplace(heap, (-d, node.idx))

            axis = node.axis
            diff = x[axis] - node.point[axis]

            close, far = (node.left, node.right) if diff <= 0 else (node.right, node.left)
            _search(close)

            # Check if far side could contain closer points
            if len(heap) < k or abs(diff) < -heap[0][0]:
                _search(far)

        _search(self.root)
        results = sorted(heap, key=lambda h: -h[0])  # ascending distance
        dists = np.array([-d for d, _ in results])
        idxs = np.array([i for _, i in results])
        return dists, idxs


# ---------------------------------------------------------------------------
# KNN Classifier
# ---------------------------------------------------------------------------

class KNNClassifier:
    """
    K-Nearest Neighbors Classifier.

    Parameters
    ----------
    n_neighbors : int (default 5)
    metric      : 'euclidean', 'manhattan', or 'minkowski' (default 'euclidean')
    weights     : 'uniform' or 'distance' (default 'uniform')
    p           : float — Minkowski parameter (default 2 = euclidean)
    algorithm   : 'brute' or 'kd_tree' (default 'brute')
    """

    def __init__(self, n_neighbors: int = 5, metric: str = 'euclidean',
                 weights: str = 'uniform', p: float = 2, algorithm: str = 'brute'):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.weights = weights
        self.p = p
        self.algorithm = algorithm
        self.X_train_ = None
        self.y_train_ = None
        self.classes_ = None
        self._tree = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNClassifier":
        self.X_train_ = np.asarray(X, dtype=np.float64)
        self.y_train_ = np.asarray(y)
        self.classes_ = np.unique(y)
        if self.algorithm == 'kd_tree' and self.metric == 'euclidean':
            self._tree = _KDTree(self.X_train_)
        return self

    def _distances(self, x: np.ndarray) -> np.ndarray:
        if self.metric == 'euclidean':
            return np.sqrt(np.sum((self.X_train_ - x) ** 2, axis=1))
        elif self.metric == 'manhattan':
            return np.sum(np.abs(self.X_train_ - x), axis=1)
        elif self.metric == 'minkowski':
            return np.sum(np.abs(self.X_train_ - x) ** self.p, axis=1) ** (1.0 / self.p)
        raise ValueError(f"Unknown metric: {self.metric}")

    def _query(self, x: np.ndarray):
        if self._tree is not None:
            return self._tree.query(x, self.n_neighbors)
        dists = self._distances(x)
        idx = np.argsort(dists)[:self.n_neighbors]
        return dists[idx], idx

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        predictions = []
        for x in X:
            dists, idx = self._query(x)
            neighbors_y = self.y_train_[idx]
            if self.weights == 'uniform':
                # Majority vote
                vals, counts = np.unique(neighbors_y, return_counts=True)
                predictions.append(vals[np.argmax(counts)])
            else:
                # Inverse distance weighting
                dists = np.maximum(dists, 1e-10)
                w = 1.0 / dists
                vote = {}
                for yi, wi in zip(neighbors_y, w):
                    vote[yi] = vote.get(yi, 0) + wi
                predictions.append(max(vote, key=vote.get))
        return np.array(predictions)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        proba = []
        n_classes = len(self.classes_)
        for x in X:
            dists, idx = self._query(x)
            neighbors_y = self.y_train_[idx]
            p = np.zeros(n_classes)
            if self.weights == 'uniform':
                for yi in neighbors_y:
                    ci = np.searchsorted(self.classes_, yi)
                    p[ci] += 1.0
                p /= len(neighbors_y)
            else:
                dists = np.maximum(dists, 1e-10)
                w = 1.0 / dists
                for yi, wi in zip(neighbors_y, w):
                    ci = np.searchsorted(self.classes_, yi)
                    p[ci] += wi
                p /= p.sum()
            proba.append(p)
        return np.array(proba)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return np.mean(self.predict(X) == np.asarray(y))


# ---------------------------------------------------------------------------
# KNN Regressor
# ---------------------------------------------------------------------------

class KNNRegressor:
    """
    K-Nearest Neighbors Regressor.

    Parameters
    ----------
    n_neighbors : int (default 5)
    weights     : 'uniform' or 'distance'
    metric      : 'euclidean', 'manhattan', or 'minkowski'
    p           : float (default 2)
    algorithm   : 'brute' or 'kd_tree'
    """

    def __init__(self, n_neighbors: int = 5, weights: str = 'uniform',
                 metric: str = 'euclidean', p: float = 2, algorithm: str = 'brute'):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        self.p = p
        self.algorithm = algorithm
        self.X_train_ = None
        self.y_train_ = None
        self._tree = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNRegressor":
        self.X_train_ = np.asarray(X, dtype=np.float64)
        self.y_train_ = np.asarray(y, dtype=np.float64)
        if self.algorithm == 'kd_tree' and self.metric == 'euclidean':
            self._tree = _KDTree(self.X_train_)
        return self

    def _distances(self, x: np.ndarray) -> np.ndarray:
        if self.metric == 'euclidean':
            return np.sqrt(np.sum((self.X_train_ - x) ** 2, axis=1))
        elif self.metric == 'manhattan':
            return np.sum(np.abs(self.X_train_ - x), axis=1)
        return np.sum(np.abs(self.X_train_ - x) ** self.p, axis=1) ** (1.0 / self.p)

    def _query(self, x: np.ndarray):
        if self._tree is not None:
            return self._tree.query(x, self.n_neighbors)
        dists = self._distances(x)
        idx = np.argsort(dists)[:self.n_neighbors]
        return dists[idx], idx

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        predictions = []
        for x in X:
            dists, idx = self._query(x)
            neighbors_y = self.y_train_[idx]
            if self.weights == 'uniform':
                predictions.append(neighbors_y.mean())
            else:
                dists = np.maximum(dists, 1e-10)
                w = 1.0 / dists
                predictions.append(np.average(neighbors_y, weights=w))
        return np.array(predictions)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y = np.asarray(y, dtype=np.float64)
        pred = self.predict(X)
        ss_res = np.sum((y - pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return 1.0 - ss_res / (ss_tot + 1e-10)
