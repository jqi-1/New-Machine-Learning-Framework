"""
numl/preprocessing/scalers.py
------------------------------
Feature scaling transformers.

All implement: fit(X), transform(X), fit_transform(X), inverse_transform(X).
"""

import numpy as np


class StandardScaler:
    """
    Standardize features by removing mean and scaling to unit variance.
    z = (x - mean) / std
    """

    def __init__(self, with_mean: bool = True, with_std: bool = True):
        self.with_mean = with_mean
        self.with_std = with_std
        self.mean_ = None
        self.scale_ = None

    def fit(self, X: np.ndarray) -> "StandardScaler":
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = X.mean(axis=0) if self.with_mean else np.zeros(X.shape[1])
        self.scale_ = X.std(axis=0) if self.with_std else np.ones(X.shape[1])
        self.scale_[self.scale_ == 0] = 1.0  # avoid division by zero
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return X * self.scale_ + self.mean_


class MinMaxScaler:
    """
    Scale features to a given range [feature_range[0], feature_range[1]].
    """

    def __init__(self, feature_range: tuple = (0, 1)):
        self.feature_range = feature_range
        self.min_ = None
        self.scale_ = None
        self.data_min_ = None
        self.data_max_ = None

    def fit(self, X: np.ndarray) -> "MinMaxScaler":
        X = np.asarray(X, dtype=np.float64)
        self.data_min_ = X.min(axis=0)
        self.data_max_ = X.max(axis=0)
        data_range = self.data_max_ - self.data_min_
        data_range[data_range == 0] = 1.0
        rng = self.feature_range[1] - self.feature_range[0]
        self.scale_ = rng / data_range
        self.min_ = self.feature_range[0] - self.data_min_ * self.scale_
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return X * self.scale_ + self.min_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return (X - self.min_) / self.scale_


class RobustScaler:
    """
    Scale using statistics that are robust to outliers (median and IQR).
    z = (x - median) / IQR
    """

    def __init__(self, with_centering: bool = True, with_scaling: bool = True,
                 quantile_range: tuple = (25.0, 75.0)):
        self.with_centering = with_centering
        self.with_scaling = with_scaling
        self.quantile_range = quantile_range
        self.center_ = None
        self.scale_ = None

    def fit(self, X: np.ndarray) -> "RobustScaler":
        X = np.asarray(X, dtype=np.float64)
        q_low, q_high = self.quantile_range
        self.center_ = np.median(X, axis=0) if self.with_centering else np.zeros(X.shape[1])
        if self.with_scaling:
            iqr = np.percentile(X, q_high, axis=0) - np.percentile(X, q_low, axis=0)
            iqr[iqr == 0] = 1.0
            self.scale_ = iqr
        else:
            self.scale_ = np.ones(X.shape[1])
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return (X - self.center_) / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return X * self.scale_ + self.center_


class MaxAbsScaler:
    """
    Scale each feature by its maximum absolute value.
    Preserves sparsity (no centering).
    """

    def __init__(self):
        self.scale_ = None
        self.max_abs_ = None

    def fit(self, X: np.ndarray) -> "MaxAbsScaler":
        X = np.asarray(X, dtype=np.float64)
        self.max_abs_ = np.abs(X).max(axis=0)
        self.max_abs_[self.max_abs_ == 0] = 1.0
        self.scale_ = self.max_abs_
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=np.float64) / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=np.float64) * self.scale_


class Normalizer:
    """
    Normalize samples (rows) to unit norm.

    Parameters
    ----------
    norm : 'l1', 'l2', or 'max'
    """

    def __init__(self, norm: str = 'l2'):
        assert norm in ('l1', 'l2', 'max')
        self.norm = norm

    def fit(self, X: np.ndarray) -> "Normalizer":
        return self  # stateless

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if self.norm == 'l2':
            norms = np.linalg.norm(X, axis=1, keepdims=True)
        elif self.norm == 'l1':
            norms = np.abs(X).sum(axis=1, keepdims=True)
        else:  # max
            norms = np.abs(X).max(axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return X / norms

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Normalizer is not invertible.")
