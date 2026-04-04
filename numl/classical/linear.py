"""
numl/classical/linear.py
-------------------------
Linear regression models: LinearRegression, Ridge, Lasso, ElasticNet.

All use the sklearn-style fit/predict/score interface with raw NumPy arrays.
"""

import numpy as np


class LinearRegression:
    """
    Ordinary least squares linear regression.
    Solved via the normal equation: w = (X^T X)^{-1} X^T y
    Uses np.linalg.lstsq for numerical stability with rank-deficient inputs.

    Parameters
    ----------
    fit_intercept : bool (default True)
    """

    def __init__(self, fit_intercept: bool = True):
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegression":
        X, y = np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64)
        if self.fit_intercept:
            X = np.hstack([np.ones((len(X), 1)), X])
        w, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        if self.fit_intercept:
            self.intercept_ = w[0]
            self.coef_ = w[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = w
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=np.float64) @ self.coef_ + self.intercept_

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """R² score."""
        y = np.asarray(y, dtype=np.float64)
        pred = self.predict(X)
        ss_res = np.sum((y - pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return 1.0 - ss_res / (ss_tot + 1e-10)


class Ridge:
    """
    Ridge regression (L2-regularized).
    Closed form: w = (X^T X + alpha * I)^{-1} X^T y

    Parameters
    ----------
    alpha         : float — regularization strength (default 1.0)
    fit_intercept : bool (default True)
    """

    def __init__(self, alpha: float = 1.0, fit_intercept: bool = True):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Ridge":
        X, y = np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64)
        if self.fit_intercept:
            self.intercept_ = y.mean() - X.mean(axis=0) @ np.linalg.lstsq(
                X - X.mean(axis=0), y - y.mean(), rcond=None)[0]

        if self.fit_intercept:
            X_c = X - X.mean(axis=0)
            y_c = y - y.mean()
        else:
            X_c, y_c = X, y

        n_features = X_c.shape[1]
        A = X_c.T @ X_c + self.alpha * np.eye(n_features)
        b = X_c.T @ y_c
        self.coef_ = np.linalg.solve(A, b)

        if self.fit_intercept:
            self.intercept_ = y.mean() - X.mean(axis=0) @ self.coef_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=np.float64) @ self.coef_ + self.intercept_

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y = np.asarray(y, dtype=np.float64)
        pred = self.predict(X)
        ss_res = np.sum((y - pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return 1.0 - ss_res / (ss_tot + 1e-10)


class Lasso:
    """
    Lasso regression (L1-regularized) via coordinate descent.

    Parameters
    ----------
    alpha         : float — regularization strength (default 1.0)
    fit_intercept : bool (default True)
    max_iter      : int (default 1000)
    tol           : float — convergence tolerance (default 1e-4)
    """

    def __init__(self, alpha: float = 1.0, fit_intercept: bool = True,
                 max_iter: int = 1000, tol: float = 1e-4):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.coef_ = None
        self.intercept_ = 0.0
        self.n_iter_ = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Lasso":
        X, y = np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64)

        if self.fit_intercept:
            X_mean = X.mean(axis=0)
            y_mean = y.mean()
            X_c = X - X_mean
            y_c = y - y_mean
        else:
            X_c, y_c = X, y
            X_mean = np.zeros(X.shape[1])
            y_mean = 0.0

        n_samples, n_features = X_c.shape
        w = np.zeros(n_features)

        # Precompute column norms squared
        col_norms = (X_c ** 2).sum(axis=0)

        for it in range(self.max_iter):
            w_old = w.copy()
            for j in range(n_features):
                if col_norms[j] == 0:
                    continue
                # Partial residual
                r_j = y_c - X_c @ w + X_c[:, j] * w[j]
                rho_j = X_c[:, j] @ r_j
                # Soft threshold
                alpha_n = self.alpha * n_samples
                w[j] = np.sign(rho_j) * max(abs(rho_j) - alpha_n, 0) / col_norms[j]

            if np.max(np.abs(w - w_old)) < self.tol:
                self.n_iter_ = it + 1
                break
        else:
            self.n_iter_ = self.max_iter

        self.coef_ = w
        if self.fit_intercept:
            self.intercept_ = y_mean - X_mean @ w
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=np.float64) @ self.coef_ + self.intercept_

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y = np.asarray(y, dtype=np.float64)
        pred = self.predict(X)
        ss_res = np.sum((y - pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return 1.0 - ss_res / (ss_tot + 1e-10)


class ElasticNet:
    """
    ElasticNet regression (L1 + L2 penalty) via coordinate descent.

    Loss = 1/(2n)||y - Xw||^2 + alpha*l1_ratio*||w||_1
                               + alpha*(1-l1_ratio)/2 * ||w||^2

    Parameters
    ----------
    alpha         : float — total regularization strength (default 1.0)
    l1_ratio      : float in [0,1] — L1 / (L1+L2) mix (default 0.5)
    fit_intercept : bool (default True)
    max_iter      : int (default 1000)
    tol           : float (default 1e-4)
    """

    def __init__(self, alpha: float = 1.0, l1_ratio: float = 0.5,
                 fit_intercept: bool = True, max_iter: int = 1000, tol: float = 1e-4):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.coef_ = None
        self.intercept_ = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ElasticNet":
        X, y = np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64)

        if self.fit_intercept:
            X_mean = X.mean(axis=0)
            y_mean = y.mean()
            X_c = X - X_mean
            y_c = y - y_mean
        else:
            X_c, y_c = X, y
            X_mean = np.zeros(X.shape[1])
            y_mean = 0.0

        n, n_features = X_c.shape
        w = np.zeros(n_features)
        col_norms = (X_c ** 2).sum(axis=0)

        l1 = self.alpha * self.l1_ratio
        l2 = self.alpha * (1 - self.l1_ratio)

        for it in range(self.max_iter):
            w_old = w.copy()
            for j in range(n_features):
                if col_norms[j] == 0:
                    continue
                r_j = y_c - X_c @ w + X_c[:, j] * w[j]
                rho_j = X_c[:, j] @ r_j
                denom = col_norms[j] + l2 * n
                # Soft threshold for L1, then scale for L2
                w[j] = np.sign(rho_j) * max(abs(rho_j) - l1 * n, 0) / denom

            if np.max(np.abs(w - w_old)) < self.tol:
                break

        self.coef_ = w
        if self.fit_intercept:
            self.intercept_ = y_mean - X_mean @ w
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=np.float64) @ self.coef_ + self.intercept_

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y = np.asarray(y, dtype=np.float64)
        pred = self.predict(X)
        ss_res = np.sum((y - pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return 1.0 - ss_res / (ss_tot + 1e-10)
