"""
numl/classical/logistic.py
---------------------------
Logistic Regression via gradient descent with optional L1/L2 regularization.

Supports:
  - Binary classification (sigmoid + BCE)
  - Multiclass via multinomial softmax (default) or one-vs-rest (OvR)
"""

import numpy as np


def _sigmoid(x):
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


def _softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


class LogisticRegression:
    """
    Logistic Regression classifier.

    Parameters
    ----------
    C             : float — inverse regularization strength (default 1.0, higher = less reg)
    penalty       : 'l2', 'l1', or None
    multi_class   : 'multinomial' or 'ovr'
    max_iter      : int (default 1000)
    lr            : float — learning rate (default 0.1)
    tol           : float — convergence tolerance (default 1e-4)
    fit_intercept : bool (default True)
    """

    def __init__(self, C: float = 1.0, penalty: str = 'l2',
                 multi_class: str = 'multinomial',
                 max_iter: int = 1000, lr: float = 0.1,
                 tol: float = 1e-4, fit_intercept: bool = True):
        self.C = C
        self.penalty = penalty
        self.multi_class = multi_class
        self.max_iter = max_iter
        self.lr = lr
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = None
        self.classes_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegression":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_samples, n_features = X.shape

        # Encode labels
        y_int = np.searchsorted(self.classes_, y)

        if n_classes == 2:
            self._fit_binary(X, y_int, n_features, n_samples)
        elif self.multi_class == 'ovr':
            self._fit_ovr(X, y_int, n_features, n_samples, n_classes)
        else:
            self._fit_multinomial(X, y_int, n_features, n_samples, n_classes)

        return self

    def _fit_binary(self, X, y, n_features, n_samples):
        w = np.zeros(n_features)
        b = 0.0
        reg = 1.0 / (self.C * n_samples)

        for _ in range(self.max_iter):
            logits = X @ w + b
            p = _sigmoid(logits)
            err = p - y

            dw = X.T @ err / n_samples
            db = err.mean()

            if self.penalty == 'l2':
                dw += reg * w
            elif self.penalty == 'l1':
                dw += reg * np.sign(w)

            w_new = w - self.lr * dw
            b_new = b - self.lr * db

            if np.max(np.abs(w_new - w)) < self.tol:
                w, b = w_new, b_new
                break
            w, b = w_new, b_new

        self.coef_ = w.reshape(1, -1)
        self.intercept_ = np.array([b])

    def _fit_ovr(self, X, y, n_features, n_samples, n_classes):
        self.coef_ = np.zeros((n_classes, n_features))
        self.intercept_ = np.zeros(n_classes)
        for c in range(n_classes):
            y_bin = (y == c).astype(float)
            w = np.zeros(n_features)
            b = 0.0
            reg = 1.0 / (self.C * n_samples)
            for _ in range(self.max_iter):
                p = _sigmoid(X @ w + b)
                err = p - y_bin
                dw = X.T @ err / n_samples
                db = err.mean()
                if self.penalty == 'l2':
                    dw += reg * w
                elif self.penalty == 'l1':
                    dw += reg * np.sign(w)
                w_new = w - self.lr * dw
                b_new = b - self.lr * db
                if np.max(np.abs(w_new - w)) < self.tol:
                    w, b = w_new, b_new
                    break
                w, b = w_new, b_new
            self.coef_[c] = w
            self.intercept_[c] = b

    def _fit_multinomial(self, X, y, n_features, n_samples, n_classes):
        W = np.zeros((n_features, n_classes))
        b = np.zeros(n_classes)
        reg = 1.0 / (self.C * n_samples)

        # One-hot encode y
        Y = np.zeros((n_samples, n_classes))
        Y[np.arange(n_samples), y] = 1.0

        for _ in range(self.max_iter):
            logits = X @ W + b
            P = _softmax(logits)
            diff = P - Y

            dW = X.T @ diff / n_samples
            db = diff.mean(axis=0)

            if self.penalty == 'l2':
                dW += reg * W
            elif self.penalty == 'l1':
                dW += reg * np.sign(W)

            W_new = W - self.lr * dW
            b_new = b - self.lr * db

            if np.max(np.abs(W_new - W)) < self.tol:
                W, b = W_new, b_new
                break
            W, b = W_new, b_new

        self.coef_ = W.T  # (n_classes, n_features)
        self.intercept_ = b

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        logits = X @ self.coef_.T + self.intercept_
        if len(self.classes_) == 2:
            p = _sigmoid(logits[:, 0])
            return np.column_stack([1 - p, p])
        return _softmax(logits)

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        idx = np.argmax(proba, axis=1)
        return self.classes_[idx]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return np.mean(self.predict(X) == np.asarray(y))
