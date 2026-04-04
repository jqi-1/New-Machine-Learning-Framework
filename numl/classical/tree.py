"""
numl/classical/tree.py
-----------------------
Decision Tree Classifier and Regressor.

Uses recursive binary splitting with:
  - Gini impurity or entropy for classification
  - MSE or MAE for regression

Supports max_depth, min_samples_split, min_samples_leaf, max_features.
"""

import numpy as np
from dataclasses import dataclass, field


@dataclass
class _Node:
    feature_idx: int = None
    threshold: float = None
    left: "_Node" = None
    right: "_Node" = None
    value: np.ndarray = None      # leaf: class distribution (clf) or mean (reg)
    impurity: float = 0.0
    n_samples: int = 0
    impurity_decrease: float = 0.0


def _gini(y):
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    p = counts / len(y)
    return 1.0 - np.sum(p ** 2)


def _entropy(y):
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    p = counts / len(y)
    return -np.sum(p * np.log2(p + 1e-10))


def _mse(y):
    if len(y) == 0:
        return 0.0
    return np.var(y)


def _mae(y):
    if len(y) == 0:
        return 0.0
    return np.mean(np.abs(y - np.median(y)))


class DecisionTreeClassifier:
    """
    Decision Tree for classification.

    Parameters
    ----------
    criterion      : 'gini' or 'entropy' (default 'gini')
    max_depth      : int or None (default None — grow until pure)
    min_samples_split : int (default 2)
    min_samples_leaf  : int (default 1)
    max_features   : int, float, 'sqrt', 'log2', or None (default None)
    random_state   : int or None
    """

    def __init__(self, criterion: str = 'gini', max_depth=None,
                 min_samples_split: int = 2, min_samples_leaf: int = 1,
                 max_features=None, random_state=None):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.root_ = None
        self.classes_ = None
        self.n_features_ = None
        self.feature_importances_ = None

    def _impurity_fn(self):
        return _gini if self.criterion == 'gini' else _entropy

    def _n_features(self, n_total):
        if self.max_features is None:
            return n_total
        if self.max_features == 'sqrt':
            return max(1, int(np.sqrt(n_total)))
        if self.max_features == 'log2':
            return max(1, int(np.log2(n_total)))
        if isinstance(self.max_features, float):
            return max(1, int(self.max_features * n_total))
        return min(self.max_features, n_total)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeClassifier":
        if self.random_state is not None:
            np.random.seed(self.random_state)
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.n_features_ = X.shape[1]
        self._importances = np.zeros(self.n_features_)

        y_enc = np.searchsorted(self.classes_, y)
        self.root_ = self._grow(X, y_enc, depth=0)
        total = self._importances.sum()
        self.feature_importances_ = self._importances / (total + 1e-10)
        return self

    def _grow(self, X, y, depth):
        node = _Node(n_samples=len(y))
        n_classes = len(self.classes_)
        counts = np.bincount(y, minlength=n_classes)
        node.value = counts / counts.sum()
        node.impurity = self._impurity_fn()(y)

        # Stopping criteria
        if (len(y) < self.min_samples_split or
                (self.max_depth is not None and depth >= self.max_depth) or
                len(np.unique(y)) == 1):
            return node

        feat_idx = self._best_split(X, y, node.impurity)
        if feat_idx is None:
            return node

        feature, threshold, left_mask = feat_idx
        right_mask = ~left_mask

        # Impurity decrease for feature importance
        n = len(y)
        n_l, n_r = left_mask.sum(), right_mask.sum()
        imp_fn = self._impurity_fn()
        decrease = (node.impurity
                    - n_l / n * imp_fn(y[left_mask])
                    - n_r / n * imp_fn(y[right_mask]))
        self._importances[feature] += n * decrease

        node.feature_idx = feature
        node.threshold = threshold
        node.impurity_decrease = decrease
        node.left = self._grow(X[left_mask], y[left_mask], depth + 1)
        node.right = self._grow(X[right_mask], y[right_mask], depth + 1)
        return node

    def _best_split(self, X, y, parent_impurity):
        n_total = X.shape[1]
        n_try = self._n_features(n_total)
        features = np.random.choice(n_total, n_try, replace=False)
        imp_fn = self._impurity_fn()

        best_gain = 0.0
        best = None

        for feat in features:
            col = X[:, feat]
            thresholds = np.unique(col)
            if len(thresholds) <= 1:
                continue
            # Try midpoints
            mids = (thresholds[:-1] + thresholds[1:]) / 2
            for thr in mids:
                left_mask = col <= thr
                right_mask = ~left_mask
                n_l, n_r = left_mask.sum(), right_mask.sum()
                if n_l < self.min_samples_leaf or n_r < self.min_samples_leaf:
                    continue
                n = len(y)
                gain = parent_impurity - (n_l / n * imp_fn(y[left_mask])
                                         + n_r / n * imp_fn(y[right_mask]))
                if gain > best_gain:
                    best_gain = gain
                    best = (feat, thr, left_mask)

        return best

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        proba = np.array([self._traverse(self.root_, row) for row in X])
        return proba

    def _traverse(self, node: _Node, x: np.ndarray) -> np.ndarray:
        if node.left is None:  # leaf
            return node.value
        if x[node.feature_idx] <= node.threshold:
            return self._traverse(node.left, x)
        return self._traverse(node.right, x)

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return np.mean(self.predict(X) == np.asarray(y))


class DecisionTreeRegressor:
    """
    Decision Tree for regression.

    Parameters
    ----------
    criterion      : 'mse' or 'mae' (default 'mse')
    max_depth      : int or None
    min_samples_split : int (default 2)
    min_samples_leaf  : int (default 1)
    max_features   : int, float, 'sqrt', 'log2', or None
    random_state   : int or None
    """

    def __init__(self, criterion: str = 'mse', max_depth=None,
                 min_samples_split: int = 2, min_samples_leaf: int = 1,
                 max_features=None, random_state=None):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.root_ = None
        self.n_features_ = None
        self.feature_importances_ = None

    def _impurity_fn(self):
        return _mse if self.criterion == 'mse' else _mae

    def _n_features(self, n_total):
        if self.max_features is None:
            return n_total
        if self.max_features == 'sqrt':
            return max(1, int(np.sqrt(n_total)))
        if self.max_features == 'log2':
            return max(1, int(np.log2(n_total)))
        if isinstance(self.max_features, float):
            return max(1, int(self.max_features * n_total))
        return min(self.max_features, n_total)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeRegressor":
        if self.random_state is not None:
            np.random.seed(self.random_state)
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self.n_features_ = X.shape[1]
        self._importances = np.zeros(self.n_features_)
        self.root_ = self._grow(X, y, depth=0)
        total = self._importances.sum()
        self.feature_importances_ = self._importances / (total + 1e-10)
        return self

    def _grow(self, X, y, depth):
        node = _Node(n_samples=len(y))
        node.value = np.mean(y) if self.criterion == 'mse' else np.median(y)
        node.impurity = self._impurity_fn()(y)

        if (len(y) < self.min_samples_split or
                (self.max_depth is not None and depth >= self.max_depth) or
                len(y) == 1):
            return node

        result = self._best_split(X, y, node.impurity)
        if result is None:
            return node

        feature, threshold, left_mask = result
        right_mask = ~left_mask
        n = len(y)
        n_l, n_r = left_mask.sum(), right_mask.sum()
        imp_fn = self._impurity_fn()
        decrease = (node.impurity
                    - n_l / n * imp_fn(y[left_mask])
                    - n_r / n * imp_fn(y[right_mask]))
        self._importances[feature] += n * decrease

        node.feature_idx = feature
        node.threshold = threshold
        node.left = self._grow(X[left_mask], y[left_mask], depth + 1)
        node.right = self._grow(X[right_mask], y[right_mask], depth + 1)
        return node

    def _best_split(self, X, y, parent_impurity):
        n_total = X.shape[1]
        n_try = self._n_features(n_total)
        features = np.random.choice(n_total, n_try, replace=False)
        imp_fn = self._impurity_fn()

        best_gain = 0.0
        best = None

        for feat in features:
            col = X[:, feat]
            thresholds = np.unique(col)
            if len(thresholds) <= 1:
                continue
            mids = (thresholds[:-1] + thresholds[1:]) / 2
            for thr in mids:
                left_mask = col <= thr
                right_mask = ~left_mask
                n_l, n_r = left_mask.sum(), right_mask.sum()
                if n_l < self.min_samples_leaf or n_r < self.min_samples_leaf:
                    continue
                n = len(y)
                gain = parent_impurity - (n_l / n * imp_fn(y[left_mask])
                                         + n_r / n * imp_fn(y[right_mask]))
                if gain > best_gain:
                    best_gain = gain
                    best = (feat, thr, left_mask)

        return best

    def _traverse(self, node: _Node, x: np.ndarray):
        if node.left is None:
            return node.value
        if x[node.feature_idx] <= node.threshold:
            return self._traverse(node.left, x)
        return self._traverse(node.right, x)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return np.array([self._traverse(self.root_, row) for row in X])

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y = np.asarray(y, dtype=np.float64)
        pred = self.predict(X)
        ss_res = np.sum((y - pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return 1.0 - ss_res / (ss_tot + 1e-10)
