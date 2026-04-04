"""
numl/decomposition/pca.py
--------------------------
Principal Component Analysis (PCA) and IncrementalPCA.
"""

import numpy as np


class PCA:
    """
    Principal Component Analysis via full SVD.

    Parameters
    ----------
    n_components : int, float (0, 1), or None
        Number of components. If float, selects enough to explain that fraction
        of variance. If None, keeps all components.
    whiten       : bool — divide scores by sqrt(eigenvalue) (default False)
    svd_solver   : 'full' (default) — always uses np.linalg.svd

    Attributes
    ----------
    components_           : (n_components, n_features) — principal axes
    explained_variance_   : (n_components,)
    explained_variance_ratio_ : (n_components,)
    singular_values_      : (n_components,)
    mean_                 : (n_features,)
    noise_variance_       : float
    """

    def __init__(self, n_components=None, whiten: bool = False,
                 svd_solver: str = 'full'):
        self.n_components = n_components
        self.whiten = whiten
        self.svd_solver = svd_solver

        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.singular_values_ = None
        self.mean_ = None
        self.noise_variance_ = 0.0

    def fit(self, X: np.ndarray) -> "PCA":
        X = np.asarray(X, dtype=np.float64)
        n_samples, n_features = X.shape

        # Center
        self.mean_ = X.mean(axis=0)
        X_c = X - self.mean_

        # Full SVD
        U, S, Vt = np.linalg.svd(X_c, full_matrices=False)

        # Explained variance
        explained_variance = S ** 2 / (n_samples - 1)
        total_var = explained_variance.sum()
        explained_variance_ratio = explained_variance / total_var

        # Determine n_components
        n_comp = self._resolve_n_components(n_features, explained_variance_ratio)

        self.components_ = Vt[:n_comp]
        self.explained_variance_ = explained_variance[:n_comp]
        self.explained_variance_ratio_ = explained_variance_ratio[:n_comp]
        self.singular_values_ = S[:n_comp]

        # Noise variance: average unexplained variance
        if n_comp < min(n_samples, n_features):
            self.noise_variance_ = explained_variance[n_comp:].mean()
        else:
            self.noise_variance_ = 0.0

        return self

    def _resolve_n_components(self, n_features, evr):
        n_comp = self.n_components
        if n_comp is None:
            return len(evr)
        if isinstance(n_comp, float) and 0.0 < n_comp < 1.0:
            cumsum = np.cumsum(evr)
            return int(np.searchsorted(cumsum, n_comp) + 1)
        return min(int(n_comp), len(evr))

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        X_c = X - self.mean_
        X_t = X_c @ self.components_.T
        if self.whiten:
            X_t /= np.sqrt(self.explained_variance_)
        return X_t

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if self.whiten:
            X = X * np.sqrt(self.explained_variance_)
        return X @ self.components_ + self.mean_

    def score(self, X: np.ndarray) -> float:
        """
        Average log-likelihood of X under the probabilistic PCA model.
        """
        X = np.asarray(X, dtype=np.float64)
        Xr = self.inverse_transform(self.transform(X))
        mse = np.mean((X - Xr) ** 2)
        return -0.5 * mse / (self.noise_variance_ + 1e-10)

    @property
    def n_components_(self) -> int:
        return len(self.components_) if self.components_ is not None else 0


class IncrementalPCA:
    """
    Incremental PCA — processes data in mini-batches using an online SVD update.

    Useful when the dataset is too large to fit in memory.

    Parameters
    ----------
    n_components : int — number of components (required)
    whiten       : bool (default False)
    batch_size   : int or None — if None, must call partial_fit() manually

    Usage
    -----
    ipca = IncrementalPCA(n_components=50)
    for batch in batches:
        ipca.partial_fit(batch)
    X_reduced = ipca.transform(X)
    """

    def __init__(self, n_components: int = None, whiten: bool = False,
                 batch_size: int = None):
        self.n_components = n_components
        self.whiten = whiten
        self.batch_size = batch_size

        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.singular_values_ = None
        self.mean_ = None
        self.var_ = None
        self.n_samples_seen_ = 0

    def partial_fit(self, X: np.ndarray) -> "IncrementalPCA":
        """Process one batch."""
        X = np.asarray(X, dtype=np.float64)
        n, n_features = X.shape
        n_comp = self.n_components or min(n, n_features)

        # Incremental mean and variance (Welford-style)
        batch_mean = X.mean(axis=0)
        batch_var = X.var(axis=0)

        if self.n_samples_seen_ == 0:
            self.mean_ = batch_mean
            self.var_ = batch_var
        else:
            # Update mean and variance
            total = self.n_samples_seen_ + n
            correction = (batch_mean - self.mean_) ** 2 * self.n_samples_seen_ * n / total
            self.var_ = (self.n_samples_seen_ * self.var_ + n * batch_var + correction) / total
            self.mean_ = (self.n_samples_seen_ * self.mean_ + n * batch_mean) / total

        self.n_samples_seen_ += n

        # Center this batch using updated mean
        X_c = X - self.mean_

        # Stack with previous components (if any) for an incremental update
        if self.components_ is not None:
            # Augment X_c with scaled previous singular vectors
            prev = self.components_ * np.sqrt(self.explained_variance_)[:, None]
            X_aug = np.vstack([prev, X_c])
        else:
            X_aug = X_c

        # SVD on augmented matrix
        _, S, Vt = np.linalg.svd(X_aug, full_matrices=False)

        self.components_ = Vt[:n_comp]
        self.singular_values_ = S[:n_comp]
        self.explained_variance_ = S[:n_comp] ** 2 / (self.n_samples_seen_ - 1)
        total_var = self.var_.sum() * (self.n_samples_seen_ - 1) / self.n_samples_seen_
        self.explained_variance_ratio_ = self.explained_variance_ / (total_var + 1e-10)

        return self

    def fit(self, X: np.ndarray) -> "IncrementalPCA":
        X = np.asarray(X, dtype=np.float64)
        bs = self.batch_size or len(X)
        for start in range(0, len(X), bs):
            self.partial_fit(X[start:start + bs])
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        X_c = X - self.mean_
        X_t = X_c @ self.components_.T
        if self.whiten:
            X_t /= np.sqrt(self.explained_variance_ + 1e-10)
        return X_t

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if self.whiten:
            X = X * np.sqrt(self.explained_variance_ + 1e-10)
        return X @ self.components_ + self.mean_
