"""
numl/preprocessing/transforms.py
----------------------------------
Feature transformation utilities.
"""

import numpy as np
from itertools import combinations_with_replacement


class PolynomialFeatures:
    """
    Generate polynomial and interaction features up to degree d.

    For features [a, b], degree=2 gives [1, a, b, a^2, ab, b^2].

    Parameters
    ----------
    degree       : int (default 2)
    include_bias : bool — include constant term (default True)
    interaction_only : bool — only interaction terms, no powers (default False)
    """

    def __init__(self, degree: int = 2, include_bias: bool = True,
                 interaction_only: bool = False):
        self.degree = degree
        self.include_bias = include_bias
        self.interaction_only = interaction_only
        self.n_input_features_ = None
        self.powers_ = None  # list of tuples

    def fit(self, X: np.ndarray) -> "PolynomialFeatures":
        X = np.asarray(X)
        n_features = X.shape[1]
        self.n_input_features_ = n_features

        powers = []
        if self.include_bias:
            powers.append((0,) * n_features)

        for d in range(1, self.degree + 1):
            for combo in combinations_with_replacement(range(n_features), d):
                if self.interaction_only and len(set(combo)) < len(combo):
                    continue
                power = [0] * n_features
                for idx in combo:
                    power[idx] += 1
                powers.append(tuple(power))

        self.powers_ = powers
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        out = np.ones((X.shape[0], len(self.powers_)))
        for j, power in enumerate(self.powers_):
            for feat_idx, exp in enumerate(power):
                if exp != 0:
                    out[:, j] *= X[:, feat_idx] ** exp
        return out

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    @property
    def n_output_features_(self) -> int:
        return len(self.powers_) if self.powers_ is not None else 0


class Binarizer:
    """
    Binarize data (set feature values to 0 or 1 based on a threshold).

    Parameters
    ----------
    threshold : float (default 0.0)
    """

    def __init__(self, threshold: float = 0.0):
        self.threshold = threshold

    def fit(self, X: np.ndarray) -> "Binarizer":
        return self  # stateless

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X) > self.threshold).astype(np.float64)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Binarizer is not invertible.")


class FunctionTransformer:
    """
    Construct a transformer from an arbitrary callable.

    Parameters
    ----------
    func         : callable for transform (default identity)
    inverse_func : callable for inverse_transform (optional)
    validate     : bool — validate input as ndarray (default True)
    """

    def __init__(self, func=None, inverse_func=None, validate: bool = True):
        self.func = func
        self.inverse_func = inverse_func
        self.validate = validate

    def fit(self, X: np.ndarray) -> "FunctionTransformer":
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.validate:
            X = np.asarray(X)
        if self.func is None:
            return X
        return self.func(X)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        if self.inverse_func is None:
            raise NotImplementedError("No inverse_func provided.")
        return self.inverse_func(X)


class PowerTransformer:
    """
    Apply power transformation to make data more Gaussian-like.

    Parameters
    ----------
    method       : 'yeo-johnson' (default) or 'box-cox'
    standardize  : bool — standardize output (default True)
    """

    def __init__(self, method: str = 'yeo-johnson', standardize: bool = True):
        assert method in ('yeo-johnson', 'box-cox')
        self.method = method
        self.standardize = standardize
        self.lambdas_ = None
        self._scaler = None

    def fit(self, X: np.ndarray) -> "PowerTransformer":
        from scipy.stats import boxcox  # fallback: use numeric optimization
        X = np.asarray(X, dtype=np.float64)
        n_features = X.shape[1]
        self.lambdas_ = np.zeros(n_features)

        for j in range(n_features):
            col = X[:, j]
            self.lambdas_[j] = self._find_lambda(col)

        if self.standardize:
            from numl.preprocessing.scalers import StandardScaler
            X_t = self._transform_no_scale(X)
            self._scaler = StandardScaler()
            self._scaler.fit(X_t)

        return self

    def _find_lambda(self, x: np.ndarray, bounds=(-2, 2), n=100) -> float:
        """Find lambda by maximizing log-likelihood (grid search)."""
        best_lambda, best_ll = 0.0, -np.inf
        for lam in np.linspace(bounds[0], bounds[1], n):
            try:
                x_t = self._transform_col(x, lam)
                ll = self._log_likelihood(x, x_t, lam)
                if ll > best_ll:
                    best_ll = ll
                    best_lambda = lam
            except Exception:
                pass
        return best_lambda

    def _log_likelihood(self, x, x_t, lam) -> float:
        n = len(x)
        std = x_t.std()
        if std == 0:
            return -np.inf
        ll = -n * np.log(std)
        if self.method == 'yeo-johnson':
            ll += (lam - 1) * np.sum(np.sign(x) * np.log1p(np.abs(x)))
        else:  # box-cox
            ll += (lam - 1) * np.sum(np.log(x))
        return ll

    def _transform_col(self, x: np.ndarray, lam: float) -> np.ndarray:
        if self.method == 'yeo-johnson':
            return self._yeo_johnson(x, lam)
        else:
            return self._box_cox(x, lam)

    def _yeo_johnson(self, x: np.ndarray, lam: float) -> np.ndarray:
        out = np.zeros_like(x)
        pos = x >= 0
        if np.abs(lam) < 1e-10:
            out[pos] = np.log1p(x[pos])
        else:
            out[pos] = ((x[pos] + 1) ** lam - 1) / lam
        if np.abs(lam - 2) < 1e-10:
            out[~pos] = -np.log1p(-x[~pos])
        else:
            out[~pos] = -((-x[~pos] + 1) ** (2 - lam) - 1) / (2 - lam)
        return out

    def _box_cox(self, x: np.ndarray, lam: float) -> np.ndarray:
        if np.any(x <= 0):
            raise ValueError("Box-Cox requires strictly positive data.")
        if np.abs(lam) < 1e-10:
            return np.log(x)
        return (x ** lam - 1) / lam

    def _transform_no_scale(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        out = np.zeros_like(X)
        for j, lam in enumerate(self.lambdas_):
            out[:, j] = self._transform_col(X[:, j], lam)
        return out

    def transform(self, X: np.ndarray) -> np.ndarray:
        X_t = self._transform_no_scale(X)
        if self.standardize and self._scaler is not None:
            return self._scaler.transform(X_t)
        return X_t

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)
