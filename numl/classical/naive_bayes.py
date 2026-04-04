"""
numl/classical/naive_bayes.py
------------------------------
Naive Bayes classifiers:
  - GaussianNB   (continuous features, Gaussian likelihood)
  - MultinomialNB (count features, Multinomial likelihood)
  - BernoulliNB  (binary features)
"""

import numpy as np


class GaussianNB:
    """
    Gaussian Naive Bayes.

    Assumes each feature follows a Gaussian distribution within each class.
    P(x_j | y=k) = N(mu_kj, sigma_kj^2)

    Parameters
    ----------
    var_smoothing : float — smoothing to prevent zero variance (default 1e-9)
    """

    def __init__(self, var_smoothing: float = 1e-9):
        self.var_smoothing = var_smoothing
        self.classes_ = None
        self.class_prior_ = None
        self.theta_ = None   # (n_classes, n_features) per-class means
        self.var_ = None     # (n_classes, n_features) per-class variances

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianNB":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        self.theta_ = np.zeros((n_classes, n_features))
        self.var_ = np.zeros((n_classes, n_features))
        self.class_prior_ = np.zeros(n_classes)

        for k, cls in enumerate(self.classes_):
            X_k = X[y == cls]
            self.class_prior_[k] = len(X_k) / len(y)
            self.theta_[k] = X_k.mean(axis=0)
            self.var_[k] = X_k.var(axis=0)

        # Smooth variances
        max_var = self.var_.max()
        self.var_ += self.var_smoothing * max_var

        return self

    def _log_likelihood(self, X: np.ndarray) -> np.ndarray:
        """Returns (n_samples, n_classes) log-likelihoods."""
        n_samples = len(X)
        n_classes = len(self.classes_)
        log_ll = np.zeros((n_samples, n_classes))

        for k in range(n_classes):
            # Log Gaussian likelihood: -0.5*sum(log(2pi*var) + (x-mu)^2/var)
            log_ll[:, k] = -0.5 * np.sum(
                np.log(2 * np.pi * self.var_[k]) +
                (X - self.theta_[k]) ** 2 / self.var_[k],
                axis=1
            )

        return log_ll

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        log_prior = np.log(self.class_prior_)
        log_ll = self._log_likelihood(X)
        log_posterior = log_ll + log_prior
        # Normalize
        log_z = np.logaddexp.reduce(log_posterior, axis=1, keepdims=True)
        return log_posterior - log_z

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.exp(self.predict_log_proba(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_log_proba(X), axis=1)]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return np.mean(self.predict(X) == np.asarray(y))


class MultinomialNB:
    """
    Multinomial Naive Bayes for count/frequency data (e.g., text bag-of-words).

    P(x_j | y=k) is proportional to count of feature j in class k.
    Uses Laplace (additive) smoothing.

    Parameters
    ----------
    alpha : float — Laplace smoothing parameter (default 1.0)
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.classes_ = None
        self.class_log_prior_ = None
        self.feature_log_prob_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MultinomialNB":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        class_count = np.zeros(n_classes)
        feature_count = np.zeros((n_classes, n_features))

        for k, cls in enumerate(self.classes_):
            X_k = X[y == cls]
            class_count[k] = len(X_k)
            feature_count[k] = X_k.sum(axis=0)

        self.class_log_prior_ = np.log(class_count / class_count.sum())
        # Laplace smoothing
        smoothed = feature_count + self.alpha
        self.feature_log_prob_ = np.log(smoothed / smoothed.sum(axis=1, keepdims=True))

        return self

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        log_posterior = X @ self.feature_log_prob_.T + self.class_log_prior_
        log_z = np.logaddexp.reduce(log_posterior, axis=1, keepdims=True)
        return log_posterior - log_z

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.exp(self.predict_log_proba(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_log_proba(X), axis=1)]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return np.mean(self.predict(X) == np.asarray(y))


class BernoulliNB:
    """
    Bernoulli Naive Bayes for binary/boolean features.

    P(x_j=1 | y=k) = p_kj  (estimated from training data)
    P(x_j=0 | y=k) = 1 - p_kj

    Parameters
    ----------
    alpha    : float — Laplace smoothing (default 1.0)
    binarize : float or None — threshold for binarizing (default 0.0); None = no binarize
    """

    def __init__(self, alpha: float = 1.0, binarize: float = 0.0):
        self.alpha = alpha
        self.binarize = binarize
        self.classes_ = None
        self.class_log_prior_ = None
        self.feature_log_prob_ = None
        self.feature_log_prob_neg_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BernoulliNB":
        X = np.asarray(X, dtype=np.float64)
        if self.binarize is not None:
            X = (X > self.binarize).astype(float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        class_count = np.zeros(n_classes)
        feature_count = np.zeros((n_classes, n_features))

        for k, cls in enumerate(self.classes_):
            X_k = X[y == cls]
            class_count[k] = len(X_k)
            feature_count[k] = X_k.sum(axis=0)

        self.class_log_prior_ = np.log(class_count / class_count.sum())
        smoothed = feature_count + self.alpha
        denom = class_count + 2 * self.alpha
        p = smoothed / denom[:, None]
        self.feature_log_prob_ = np.log(p)
        self.feature_log_prob_neg_ = np.log(1 - p)

        return self

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if self.binarize is not None:
            X = (X > self.binarize).astype(float)
        # P(x|y) = prod_j p_kj^x_j * (1-p_kj)^(1-x_j)
        # log P = sum_j x_j*log(p) + (1-x_j)*log(1-p)
        #       = sum_j log(1-p) + x * (log(p) - log(1-p))
        neg_sum = self.feature_log_prob_neg_.sum(axis=1)  # (n_classes,)
        log_posterior = (
            X @ (self.feature_log_prob_ - self.feature_log_prob_neg_).T
            + neg_sum
            + self.class_log_prior_
        )
        log_z = np.logaddexp.reduce(log_posterior, axis=1, keepdims=True)
        return log_posterior - log_z

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.exp(self.predict_log_proba(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_log_proba(X), axis=1)]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return np.mean(self.predict(X) == np.asarray(y))
