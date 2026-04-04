"""
numl/decomposition/svd.py
--------------------------
Truncated SVD (useful for sparse/large matrices and text features).
"""

import numpy as np


class TruncatedSVD:
    """
    Dimensionality reduction via truncated SVD (LSA for text).

    Unlike PCA, does NOT center the data first, so it works well with
    sparse matrices (e.g., TF-IDF features).

    Parameters
    ----------
    n_components  : int — number of desired components (default 2)
    n_iter        : int — power iterations for randomized SVD (default 5)
    random_state  : int or None

    Attributes
    ----------
    components_           : (n_components, n_features)
    explained_variance_   : (n_components,)
    explained_variance_ratio_ : (n_components,)
    singular_values_      : (n_components,)
    """

    def __init__(self, n_components: int = 2, n_iter: int = 5,
                 random_state=None):
        self.n_components = n_components
        self.n_iter = n_iter
        self.random_state = random_state

        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.singular_values_ = None

    def _randomized_svd(self, X: np.ndarray):
        """
        Randomized SVD via power iteration (Halko et al., 2009).
        Much faster than full SVD for large matrices when n_components << min(n, p).
        """
        rng = np.random.RandomState(self.random_state)
        n, p = X.shape
        k = self.n_components

        # Random projection matrix
        Omega = rng.randn(p, k + 10)

        # Power iterations to improve accuracy
        Y = X @ Omega
        for _ in range(self.n_iter):
            Y = X @ (X.T @ Y)

        # QR decomposition to get orthonormal basis
        Q, _ = np.linalg.qr(Y)

        # Project X onto Q
        B = Q.T @ X  # (k+10, p)

        # Full SVD of small matrix B
        Uhat, S, Vt = np.linalg.svd(B, full_matrices=False)
        U = Q @ Uhat

        return U[:, :k], S[:k], Vt[:k]

    def fit(self, X: np.ndarray) -> "TruncatedSVD":
        X = np.asarray(X, dtype=np.float64)
        n, p = X.shape

        if self.n_components >= min(n, p) - 1:
            # Fall back to full SVD
            _, S, Vt = np.linalg.svd(X, full_matrices=False)
            k = self.n_components
            self.singular_values_ = S[:k]
            self.components_ = Vt[:k]
        else:
            _, S, Vt = self._randomized_svd(X)
            self.singular_values_ = S
            self.components_ = Vt

        self.explained_variance_ = self.singular_values_ ** 2 / (n - 1)
        total_var = np.var(X, axis=0).sum()
        self.explained_variance_ratio_ = self.explained_variance_ / (total_var + 1e-10)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=np.float64) @ self.components_.T

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=np.float64) @ self.components_
