"""
numl/classical/ensemble.py
---------------------------
Ensemble methods:
  - RandomForestClassifier, RandomForestRegressor
  - GradientBoostingClassifier, GradientBoostingRegressor
"""

import numpy as np
from numl.classical.tree import DecisionTreeClassifier, DecisionTreeRegressor


class RandomForestClassifier:
    """
    Random Forest Classifier.

    Parameters
    ----------
    n_estimators   : int (default 100)
    max_depth      : int or None (default None)
    min_samples_split : int (default 2)
    min_samples_leaf  : int (default 1)
    max_features   : 'sqrt', 'log2', int, float, or None (default 'sqrt')
    bootstrap      : bool (default True)
    oob_score      : bool (default False)
    random_state   : int or None
    n_jobs         : unused (for API compatibility)
    """

    def __init__(self, n_estimators: int = 100, max_depth=None,
                 min_samples_split: int = 2, min_samples_leaf: int = 1,
                 max_features='sqrt', bootstrap: bool = True,
                 oob_score: bool = False, random_state=None, n_jobs=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.random_state = random_state
        self.estimators_ = []
        self.classes_ = None
        self.feature_importances_ = None
        self.oob_score_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestClassifier":
        if self.random_state is not None:
            np.random.seed(self.random_state)
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        n, n_features = X.shape
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        oob_counts = np.zeros((n, n_classes))
        oob_n = np.zeros(n, dtype=int)

        self.estimators_ = []
        importances = np.zeros(n_features)

        for i in range(self.n_estimators):
            if self.bootstrap:
                idx = np.random.choice(n, n, replace=True)
                oob_idx = np.setdiff1d(np.arange(n), idx)
            else:
                idx = np.arange(n)
                oob_idx = np.array([], dtype=int)

            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
            )
            tree.fit(X[idx], y[idx])
            self.estimators_.append(tree)
            importances += tree.feature_importances_

            if self.oob_score and len(oob_idx) > 0:
                oob_proba = tree.predict_proba(X[oob_idx])
                y_oob_int = np.searchsorted(self.classes_, y[oob_idx])
                oob_counts[oob_idx] += oob_proba
                oob_n[oob_idx] += 1

        self.feature_importances_ = importances / self.n_estimators

        if self.oob_score:
            valid = oob_n > 0
            oob_preds = self.classes_[np.argmax(oob_counts[valid], axis=1)]
            self.oob_score_ = np.mean(oob_preds == y[valid])

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        all_proba = np.array([tree.predict_proba(X) for tree in self.estimators_])
        return all_proba.mean(axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return np.mean(self.predict(X) == np.asarray(y))


class RandomForestRegressor:
    """
    Random Forest Regressor.

    Parameters
    ----------
    n_estimators : int (default 100)
    max_depth    : int or None
    max_features : 'sqrt', 'log2', int, float, or None (default None → n_features)
    bootstrap    : bool (default True)
    random_state : int or None
    """

    def __init__(self, n_estimators: int = 100, max_depth=None,
                 min_samples_split: int = 2, min_samples_leaf: int = 1,
                 max_features=None, bootstrap: bool = True,
                 random_state=None, n_jobs=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.estimators_ = []
        self.feature_importances_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestRegressor":
        if self.random_state is not None:
            np.random.seed(self.random_state)
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n = len(X)
        n_features = X.shape[1]
        importances = np.zeros(n_features)

        self.estimators_ = []
        for _ in range(self.n_estimators):
            if self.bootstrap:
                idx = np.random.choice(n, n, replace=True)
            else:
                idx = np.arange(n)
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
            )
            tree.fit(X[idx], y[idx])
            self.estimators_.append(tree)
            importances += tree.feature_importances_

        self.feature_importances_ = importances / self.n_estimators
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        preds = np.array([tree.predict(X) for tree in self.estimators_])
        return preds.mean(axis=0)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y = np.asarray(y, dtype=np.float64)
        pred = self.predict(X)
        ss_res = np.sum((y - pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return 1.0 - ss_res / (ss_tot + 1e-10)


# ---------------------------------------------------------------------------
# Gradient Boosting
# ---------------------------------------------------------------------------

def _sigmoid(x):
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


class GradientBoostingRegressor:
    """
    Gradient Boosting for regression (MSE loss).

    F_0 = mean(y)
    F_m = F_{m-1} + lr * tree_m  where tree fits pseudo-residuals r = y - F_{m-1}

    Parameters
    ----------
    n_estimators   : int (default 100)
    learning_rate  : float (default 0.1)
    max_depth      : int (default 3)
    subsample      : float in (0,1] (default 1.0) — fraction of training samples
    min_samples_split : int (default 2)
    random_state   : int or None
    """

    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1,
                 max_depth: int = 3, subsample: float = 1.0,
                 min_samples_split: int = 2, random_state=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.estimators_ = []
        self.F0_ = None
        self.train_loss_ = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientBoostingRegressor":
        if self.random_state is not None:
            np.random.seed(self.random_state)
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n = len(y)

        # Initialize with mean
        self.F0_ = y.mean()
        F = np.full(n, self.F0_)
        self.estimators_ = []

        for m in range(self.n_estimators):
            r = y - F  # pseudo-residuals for MSE

            if self.subsample < 1.0:
                idx = np.random.choice(n, int(n * self.subsample), replace=False)
            else:
                idx = np.arange(n)

            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
            )
            tree.fit(X[idx], r[idx])
            update = tree.predict(X)
            F += self.learning_rate * update
            self.estimators_.append(tree)
            self.train_loss_.append(np.mean(r ** 2))

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        F = np.full(len(X), self.F0_)
        for tree in self.estimators_:
            F += self.learning_rate * tree.predict(X)
        return F

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y = np.asarray(y, dtype=np.float64)
        pred = self.predict(X)
        ss_res = np.sum((y - pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return 1.0 - ss_res / (ss_tot + 1e-10)


class GradientBoostingClassifier:
    """
    Gradient Boosting for binary classification (log-loss).

    Uses log-odds parameterization:
      F_0 = log(p_0 / (1 - p_0))
      Pseudo-residuals: r = y - sigmoid(F)
      Leaf values: gamma_j = sum(r) / sum(p*(1-p)) in each region

    For multiclass, trains one model per class (one-vs-rest).

    Parameters
    ----------
    n_estimators  : int (default 100)
    learning_rate : float (default 0.1)
    max_depth     : int (default 3)
    subsample     : float (default 1.0)
    random_state  : int or None
    """

    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1,
                 max_depth: int = 3, subsample: float = 1.0,
                 min_samples_split: int = 2, random_state=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.estimators_ = []  # list of list (one per class)
        self.F0_ = None
        self.classes_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientBoostingClassifier":
        if self.random_state is not None:
            np.random.seed(self.random_state)
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n = len(y)

        y_int = np.searchsorted(self.classes_, y)

        if n_classes == 2:
            # Binary: single set of trees
            p0 = np.mean(y_int)
            p0 = np.clip(p0, 1e-6, 1 - 1e-6)
            self.F0_ = np.array([np.log(p0 / (1 - p0))])
            F = np.full(n, self.F0_[0])
            trees = []
            for m in range(self.n_estimators):
                p = _sigmoid(F)
                r = y_int - p
                if self.subsample < 1.0:
                    idx = np.random.choice(n, int(n * self.subsample), replace=False)
                else:
                    idx = np.arange(n)
                tree = DecisionTreeRegressor(max_depth=self.max_depth,
                                             min_samples_split=self.min_samples_split)
                tree.fit(X[idx], r[idx])
                F += self.learning_rate * tree.predict(X)
                trees.append(tree)
            self.estimators_ = [trees]
        else:
            # Multiclass one-vs-rest
            self.F0_ = []
            self.estimators_ = []
            for c in range(n_classes):
                y_bin = (y_int == c).astype(float)
                p0 = np.clip(y_bin.mean(), 1e-6, 1 - 1e-6)
                f0 = np.log(p0 / (1 - p0))
                self.F0_.append(f0)
                F = np.full(n, f0)
                trees = []
                for m in range(self.n_estimators):
                    p = _sigmoid(F)
                    r = y_bin - p
                    if self.subsample < 1.0:
                        idx = np.random.choice(n, int(n * self.subsample), replace=False)
                    else:
                        idx = np.arange(n)
                    tree = DecisionTreeRegressor(max_depth=self.max_depth,
                                                 min_samples_split=self.min_samples_split)
                    tree.fit(X[idx], r[idx])
                    F += self.learning_rate * tree.predict(X)
                    trees.append(tree)
                self.estimators_.append(trees)
            self.F0_ = np.array(self.F0_)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        n_classes = len(self.classes_)

        if n_classes == 2:
            F = np.full(len(X), self.F0_[0])
            for tree in self.estimators_[0]:
                F += self.learning_rate * tree.predict(X)
            p = _sigmoid(F)
            return np.column_stack([1 - p, p])
        else:
            scores = np.zeros((len(X), n_classes))
            for c in range(n_classes):
                F = np.full(len(X), self.F0_[c])
                for tree in self.estimators_[c]:
                    F += self.learning_rate * tree.predict(X)
                scores[:, c] = F
            # Softmax
            scores -= scores.max(axis=1, keepdims=True)
            e = np.exp(scores)
            return e / e.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return np.mean(self.predict(X) == np.asarray(y))
