"""
numl/classical/svm.py
----------------------
Support Vector Machine: SVC and SVR using the SMO algorithm.

SMO (Sequential Minimal Optimization) alternately optimizes pairs of Lagrange
multipliers until convergence (Platt, 1998).
"""

import numpy as np


class SVC:
    """
    Support Vector Classifier.

    Parameters
    ----------
    C         : float — regularization parameter (default 1.0)
    kernel    : 'linear', 'rbf', 'poly', 'sigmoid' (default 'rbf')
    gamma     : 'scale', 'auto', or float (default 'scale')
    degree    : int — degree for 'poly' kernel (default 3)
    coef0     : float — independent term for 'poly'/'sigmoid' (default 0.0)
    tol       : float — tolerance for stopping criterion (default 1e-3)
    max_iter  : int (default 1000)

    For multiclass: one-vs-one (OvO) with majority vote.
    """

    def __init__(self, C: float = 1.0, kernel: str = 'rbf', gamma='scale',
                 degree: int = 3, coef0: float = 0.0, tol: float = 1e-3,
                 max_iter: int = 1000):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.tol = tol
        self.max_iter = max_iter
        # Fitted attributes
        self.support_vectors_ = None
        self.dual_coef_ = None
        self.intercept_ = None
        self.classes_ = None
        self._models = None  # list of (alpha, sv, sv_y, b, gamma) for OvO

    def _get_gamma(self, X):
        if self.gamma == 'scale':
            return 1.0 / (X.shape[1] * X.var())
        elif self.gamma == 'auto':
            return 1.0 / X.shape[1]
        return float(self.gamma)

    def _kernel_fn(self, X1, X2, gamma):
        if self.kernel == 'linear':
            return X1 @ X2.T
        elif self.kernel == 'rbf':
            # ||x1 - x2||^2 = ||x1||^2 + ||x2||^2 - 2 x1·x2
            sq1 = np.sum(X1 ** 2, axis=1, keepdims=True)
            sq2 = np.sum(X2 ** 2, axis=1, keepdims=True)
            dists = sq1 + sq2.T - 2 * X1 @ X2.T
            return np.exp(-gamma * np.maximum(dists, 0))
        elif self.kernel == 'poly':
            return (gamma * X1 @ X2.T + self.coef0) ** self.degree
        elif self.kernel == 'sigmoid':
            return np.tanh(gamma * X1 @ X2.T + self.coef0)
        raise ValueError(f"Unknown kernel: {self.kernel}")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SVC":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        gamma = self._get_gamma(X)

        if n_classes == 2:
            y_bin = np.where(y == self.classes_[0], -1.0, 1.0)
            alpha, b = self._smo(X, y_bin, gamma)
            self._models = [(alpha, X, y_bin, b, gamma)]
        else:
            # One-vs-one
            self._models = []
            pairs = [(i, j) for i in range(n_classes) for j in range(i + 1, n_classes)]
            for i, j in pairs:
                mask = (y == self.classes_[i]) | (y == self.classes_[j])
                X_sub = X[mask]
                y_sub = np.where(y[mask] == self.classes_[i], -1.0, 1.0)
                alpha, b = self._smo(X_sub, y_sub, gamma)
                self._models.append((alpha, X_sub, y_sub, b, gamma, i, j))

        return self

    def _smo(self, X: np.ndarray, y: np.ndarray, gamma: float):
        """Simplified SMO for binary SVM."""
        n = len(y)
        alpha = np.zeros(n)
        b = 0.0
        K = self._kernel_fn(X, X, gamma)

        for _ in range(self.max_iter):
            changed = 0
            for i in range(n):
                Ei = self._decision(alpha, y, K[:, i], b) - y[i]
                # KKT violation check
                if ((y[i] * Ei < -self.tol and alpha[i] < self.C) or
                        (y[i] * Ei > self.tol and alpha[i] > 0)):
                    # Choose j ≠ i randomly
                    j = i
                    while j == i:
                        j = np.random.randint(n)

                    Ej = self._decision(alpha, y, K[:, j], b) - y[j]
                    ai_old, aj_old = alpha[i], alpha[j]

                    # Bounds
                    if y[i] != y[j]:
                        L = max(0, alpha[j] - alpha[i])
                        H = min(self.C, self.C + alpha[j] - alpha[i])
                    else:
                        L = max(0, alpha[i] + alpha[j] - self.C)
                        H = min(self.C, alpha[i] + alpha[j])
                    if L >= H:
                        continue

                    eta = 2 * K[i, j] - K[i, i] - K[j, j]
                    if eta >= 0:
                        continue

                    alpha[j] -= y[j] * (Ei - Ej) / eta
                    alpha[j] = np.clip(alpha[j], L, H)

                    if abs(alpha[j] - aj_old) < 1e-5:
                        continue

                    alpha[i] += y[i] * y[j] * (aj_old - alpha[j])

                    # Update b
                    b1 = b - Ei - y[i] * (alpha[i] - ai_old) * K[i, i] \
                           - y[j] * (alpha[j] - aj_old) * K[i, j]
                    b2 = b - Ej - y[i] * (alpha[i] - ai_old) * K[i, j] \
                           - y[j] * (alpha[j] - aj_old) * K[j, j]
                    b = (b1 + b2) / 2
                    changed += 1

            if changed == 0:
                break

        return alpha, b

    def _decision(self, alpha, y, k_col, b):
        return float(np.sum(alpha * y * k_col) + b)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if len(self.classes_) == 2:
            alpha, sv, sv_y, b, gamma = self._models[0]
            K = self._kernel_fn(X, sv, gamma)
            return (K @ (alpha * sv_y)) + b
        raise NotImplementedError("decision_function not supported for multiclass OvO.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        n_classes = len(self.classes_)

        if n_classes == 2:
            scores = self.decision_function(X)
            return np.where(scores >= 0, self.classes_[1], self.classes_[0])
        else:
            # OvO voting
            votes = np.zeros((len(X), n_classes), dtype=int)
            for model in self._models:
                alpha, sv, sv_y, b, gamma, ci, cj = model
                K = self._kernel_fn(X, sv, gamma)
                scores = (K @ (alpha * sv_y)) + b
                pred = np.where(scores >= 0, cj, ci)
                for k, p in enumerate(pred):
                    votes[k, p] += 1
            return self.classes_[np.argmax(votes, axis=1)]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return np.mean(self.predict(X) == np.asarray(y))


class SVR:
    """
    Support Vector Regression with epsilon-insensitive loss.

    Simplified: uses the same SMO structure for regression.
    The dual problem involves alpha and alpha* (two sets of multipliers).

    For simplicity, we formulate it as a binary SVM on the augmented dataset:
    rows labeled +1 are y_i + epsilon, rows labeled -1 are y_i - epsilon.
    This is an approximation; a full SVR SMO is more complex.

    Parameters
    ----------
    C       : float (default 1.0)
    epsilon : float — epsilon in the epsilon-insensitive loss (default 0.1)
    kernel  : str (default 'rbf')
    gamma   : 'scale', 'auto', or float
    tol     : float (default 1e-3)
    max_iter: int (default 1000)
    """

    def __init__(self, C: float = 1.0, epsilon: float = 0.1, kernel: str = 'rbf',
                 gamma='scale', degree: int = 3, coef0: float = 0.0,
                 tol: float = 1e-3, max_iter: int = 1000):
        self.C = C
        self.epsilon = epsilon
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.tol = tol
        self.max_iter = max_iter
        self._alpha = None
        self._sv = None
        self._sv_y = None
        self._b = None
        self._gamma_val = None

    def _kernel_fn(self, X1, X2, gamma):
        if self.kernel == 'linear':
            return X1 @ X2.T
        elif self.kernel == 'rbf':
            sq1 = np.sum(X1 ** 2, axis=1, keepdims=True)
            sq2 = np.sum(X2 ** 2, axis=1, keepdims=True)
            return np.exp(-gamma * np.maximum(sq1 + sq2.T - 2 * X1 @ X2.T, 0))
        elif self.kernel == 'poly':
            return (gamma * X1 @ X2.T + self.coef0) ** self.degree
        raise ValueError(f"Unknown kernel: {self.kernel}")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SVR":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        if self.gamma == 'scale':
            self._gamma_val = 1.0 / (X.shape[1] * X.var())
        elif self.gamma == 'auto':
            self._gamma_val = 1.0 / X.shape[1]
        else:
            self._gamma_val = float(self.gamma)

        # Dual variable: alpha_i - alpha_i^*
        n = len(y)
        alpha = np.zeros(n)  # alpha_i - alpha_i^* (combined)
        b = 0.0
        K = self._kernel_fn(X, X, self._gamma_val)

        for _ in range(self.max_iter):
            changed = 0
            for i in range(n):
                f_i = np.sum(alpha * y * K[:, i]) + b  # prediction
                # Simplified SMO for regression
                err_i = f_i - y[i]
                if ((err_i > self.epsilon and alpha[i] > -self.C) or
                        (err_i < -self.epsilon and alpha[i] < self.C)):
                    j = np.random.randint(n)
                    if j == i:
                        continue
                    f_j = np.sum(alpha * y * K[:, j]) + b
                    err_j = f_j - y[j]
                    ai_old, aj_old = alpha[i], alpha[j]
                    eta = K[i, i] + K[j, j] - 2 * K[i, j]
                    if eta <= 0:
                        continue
                    alpha[j] -= (err_i - err_j) / eta
                    alpha[j] = np.clip(alpha[j], -self.C, self.C)
                    alpha[i] += aj_old - alpha[j]
                    alpha[i] = np.clip(alpha[i], -self.C, self.C)
                    b = b - err_i - (alpha[i] - ai_old) * K[i, i] \
                        - (alpha[j] - aj_old) * K[i, j]
                    changed += 1
            if changed == 0:
                break

        # Support vectors: where |alpha| > tol
        sv_mask = np.abs(alpha) > self.tol
        self._alpha = alpha[sv_mask]
        self._sv = X[sv_mask]
        self._sv_y = y[sv_mask]
        self._b = b
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        K = self._kernel_fn(X, self._sv, self._gamma_val)
        return K @ self._alpha + self._b

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y = np.asarray(y, dtype=np.float64)
        pred = self.predict(X)
        ss_res = np.sum((y - pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return 1.0 - ss_res / (ss_tot + 1e-10)
