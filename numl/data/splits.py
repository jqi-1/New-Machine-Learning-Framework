"""
numl/data/splits.py
-------------------
Data splitting utilities:
  - train_test_split
  - KFold
  - StratifiedKFold
  - LeaveOneOut
  - cross_val_score
"""

import numpy as np


def train_test_split(*arrays, test_size: float = 0.2, shuffle: bool = True,
                     random_state=None, stratify=None):
    """
    Split arrays into random train and test subsets.

    Parameters
    ----------
    *arrays      : one or more arrays of same first dimension
    test_size    : float — proportion for test set (default 0.2)
    shuffle      : bool (default True)
    random_state : int seed (optional)
    stratify     : array-like — if provided, do stratified split

    Returns
    -------
    list of split arrays: [X_train, X_test, y_train, y_test, ...]
    """
    if random_state is not None:
        np.random.seed(random_state)

    n = len(arrays[0])
    n_test = max(1, int(n * test_size))
    n_train = n - n_test

    if stratify is not None:
        stratify = np.array(stratify)
        classes = np.unique(stratify)
        train_idx, test_idx = [], []
        for cls in classes:
            cls_idx = np.where(stratify == cls)[0]
            if shuffle:
                np.random.shuffle(cls_idx)
            n_cls_test = max(1, int(len(cls_idx) * test_size))
            test_idx.extend(cls_idx[:n_cls_test].tolist())
            train_idx.extend(cls_idx[n_cls_test:].tolist())
        train_idx = np.array(train_idx)
        test_idx = np.array(test_idx)
        if shuffle:
            np.random.shuffle(train_idx)
            np.random.shuffle(test_idx)
    else:
        indices = np.arange(n)
        if shuffle:
            np.random.shuffle(indices)
        test_idx = indices[:n_test]
        train_idx = indices[n_test:]

    result = []
    for arr in arrays:
        arr = np.array(arr)
        result.append(arr[train_idx])
        result.append(arr[test_idx])
    return result


class KFold:
    """
    K-Fold cross-validator.

    Parameters
    ----------
    n_splits     : int — number of folds (default 5)
    shuffle      : bool (default False)
    random_state : int (optional)
    """

    def __init__(self, n_splits: int = 5, shuffle: bool = False, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X, y=None):
        """
        Yields (train_indices, val_indices) for each fold.
        """
        n = len(X)
        indices = np.arange(n)
        if self.shuffle:
            rng = np.random.RandomState(self.random_state)
            rng.shuffle(indices)

        fold_sizes = np.full(self.n_splits, n // self.n_splits)
        fold_sizes[: n % self.n_splits] += 1
        current = 0
        for fold_size in fold_sizes:
            val_idx = indices[current: current + fold_size]
            train_idx = np.concatenate([indices[:current], indices[current + fold_size:]])
            yield train_idx, val_idx
            current += fold_size

    def get_n_splits(self) -> int:
        return self.n_splits


class StratifiedKFold:
    """
    Stratified K-Fold: ensures each fold has the same class proportion.

    Parameters
    ----------
    n_splits     : int (default 5)
    shuffle      : bool (default False)
    random_state : int (optional)
    """

    def __init__(self, n_splits: int = 5, shuffle: bool = False, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X, y):
        """Yields (train_indices, val_indices) for each fold."""
        y = np.array(y)
        n = len(y)
        classes = np.unique(y)

        # Build per-class index lists
        class_indices = []
        for cls in classes:
            idx = np.where(y == cls)[0]
            if self.shuffle:
                rng = np.random.RandomState(self.random_state)
                rng.shuffle(idx)
            class_indices.append(idx)

        # Assign class indices to folds round-robin
        fold_idx = [[] for _ in range(self.n_splits)]
        for idx_list in class_indices:
            splits = np.array_split(idx_list, self.n_splits)
            for k, s in enumerate(splits):
                fold_idx[k].extend(s.tolist())

        fold_idx = [np.array(f) for f in fold_idx]

        for k in range(self.n_splits):
            val_idx = fold_idx[k]
            train_idx = np.concatenate([fold_idx[i] for i in range(self.n_splits) if i != k])
            yield train_idx, val_idx

    def get_n_splits(self) -> int:
        return self.n_splits


class LeaveOneOut:
    """Leave-One-Out cross-validator (n_splits = n_samples)."""

    def split(self, X, y=None):
        n = len(X)
        for i in range(n):
            val_idx = np.array([i])
            train_idx = np.concatenate([np.arange(i), np.arange(i + 1, n)])
            yield train_idx, val_idx

    def get_n_splits(self, X=None) -> int:
        return len(X) if X is not None else None


def cross_val_score(estimator, X, y, cv: int = 5, scoring: str = 'accuracy',
                    random_state=None) -> np.ndarray:
    """
    Evaluate an estimator by cross-validation.

    Parameters
    ----------
    estimator : object with fit(X, y) and predict(X) methods
    X         : np.ndarray of shape (n_samples, n_features)
    y         : np.ndarray of shape (n_samples,)
    cv        : int — number of folds (default 5)
    scoring   : str — 'accuracy', 'mse', 'mae', 'r2' (default 'accuracy')

    Returns
    -------
    scores : np.ndarray of shape (cv,)
    """
    kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state) \
        if scoring == 'accuracy' else KFold(n_splits=cv, shuffle=True, random_state=random_state)

    X = np.array(X)
    y = np.array(y)
    scores = []

    for train_idx, val_idx in kf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Clone by re-instantiating — estimator must support this
        import copy
        est = copy.deepcopy(estimator)
        est.fit(X_train, y_train)
        preds = est.predict(X_val)

        if scoring == 'accuracy':
            score = np.mean(preds == y_val)
        elif scoring == 'mse':
            score = np.mean((preds - y_val) ** 2)
        elif scoring == 'mae':
            score = np.mean(np.abs(preds - y_val))
        elif scoring == 'r2':
            ss_res = np.sum((y_val - preds) ** 2)
            ss_tot = np.sum((y_val - y_val.mean()) ** 2)
            score = 1.0 - ss_res / (ss_tot + 1e-10)
        else:
            raise ValueError(f"Unknown scoring: {scoring}")

        scores.append(score)

    return np.array(scores)
