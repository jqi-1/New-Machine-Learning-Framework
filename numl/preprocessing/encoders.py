"""
numl/preprocessing/encoders.py
--------------------------------
Categorical encoding transformers.
"""

import numpy as np


class LabelEncoder:
    """
    Encode labels as integers 0..n_classes-1.

    Attributes
    ----------
    classes_ : np.ndarray of unique classes, sorted
    """

    def __init__(self):
        self.classes_ = None

    def fit(self, y) -> "LabelEncoder":
        self.classes_ = np.unique(y)
        return self

    def transform(self, y) -> np.ndarray:
        y = np.asarray(y)
        mapping = {c: i for i, c in enumerate(self.classes_)}
        return np.array([mapping[v] for v in y])

    def fit_transform(self, y) -> np.ndarray:
        return self.fit(y).transform(y)

    def inverse_transform(self, y) -> np.ndarray:
        return self.classes_[np.asarray(y, dtype=int)]


class OneHotEncoder:
    """
    One-hot encode categorical features.

    Parameters
    ----------
    sparse      : bool — always False (we return dense arrays)
    drop        : None or 'first' — drop first category to avoid multicollinearity
    handle_unknown : 'error' or 'ignore'

    Example
    -------
    enc = OneHotEncoder()
    X_enc = enc.fit_transform(X[:, [col]])   # expects 2D input
    """

    def __init__(self, drop=None, handle_unknown: str = 'error'):
        self.drop = drop
        self.handle_unknown = handle_unknown
        self.categories_ = None  # list of arrays, one per feature column

    def fit(self, X) -> "OneHotEncoder":
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        self.categories_ = []
        for col in range(X.shape[1]):
            cats = np.unique(X[:, col])
            if self.drop == 'first':
                cats = cats[1:]
            self.categories_.append(cats)
        return self

    def transform(self, X) -> np.ndarray:
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        parts = []
        for col, cats in enumerate(self.categories_):
            col_data = X[:, col]
            ohe = np.zeros((len(col_data), len(cats)), dtype=np.float64)
            for i, val in enumerate(col_data):
                idx = np.where(cats == val)[0]
                if len(idx) == 0:
                    if self.handle_unknown == 'error':
                        raise ValueError(f"Unknown category '{val}' in column {col}.")
                    # ignore: leave zeros
                else:
                    ohe[i, idx[0]] = 1.0
            parts.append(ohe)
        return np.hstack(parts)

    def fit_transform(self, X) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray):
        """Decode one-hot back to category labels."""
        results = []
        offset = 0
        for cats in self.categories_:
            n = len(cats)
            block = X[:, offset:offset + n]
            idx = np.argmax(block, axis=1)
            results.append(cats[idx].reshape(-1, 1))
            offset += n
        return np.hstack(results)


class OrdinalEncoder:
    """
    Encode each categorical column as an integer ordinal.

    Parameters
    ----------
    categories : 'auto' or list of arrays — category order per column
    """

    def __init__(self, categories='auto'):
        self.categories_input = categories
        self.categories_ = None

    def fit(self, X) -> "OrdinalEncoder":
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        n_cols = X.shape[1]
        if self.categories_input == 'auto':
            self.categories_ = [np.unique(X[:, j]) for j in range(n_cols)]
        else:
            self.categories_ = [np.array(c) for c in self.categories_input]
        return self

    def transform(self, X) -> np.ndarray:
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        out = np.zeros(X.shape, dtype=np.float64)
        for j, cats in enumerate(self.categories_):
            mapping = {c: i for i, c in enumerate(cats)}
            out[:, j] = np.array([mapping[v] for v in X[:, j]], dtype=float)
        return out

    def fit_transform(self, X) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=int)
        out = np.empty(X.shape, dtype=object)
        for j, cats in enumerate(self.categories_):
            out[:, j] = cats[X[:, j]]
        return out
