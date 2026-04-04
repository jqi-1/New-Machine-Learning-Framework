"""
numl/data/dataset.py
--------------------
Abstract Dataset base class and concrete implementations.
"""

import numpy as np
from numl.core.tensor import Tensor


class Dataset:
    """Abstract base class for all datasets."""

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError


class ArrayDataset(Dataset):
    """
    Wraps one or more NumPy arrays into a Dataset.
    All arrays must have the same first dimension.

    Example
    -------
    ds = ArrayDataset(X_train, y_train)
    x, y = ds[0]
    """

    def __init__(self, *arrays: np.ndarray):
        assert all(len(a) == len(arrays[0]) for a in arrays), \
            "All arrays must have the same first dimension."
        self.arrays = arrays

    def __len__(self) -> int:
        return len(self.arrays[0])

    def __getitem__(self, idx):
        return tuple(a[idx] for a in self.arrays)


class TensorDataset(Dataset):
    """
    Wraps one or more Tensors into a Dataset.
    All tensors must have the same first dimension.
    """

    def __init__(self, *tensors: Tensor):
        assert all(len(t) == len(tensors[0]) for t in tensors), \
            "All tensors must have the same first dimension."
        self.tensors = tensors

    def __len__(self) -> int:
        return len(self.tensors[0])

    def __getitem__(self, idx):
        return tuple(Tensor(t.data[idx]) for t in self.tensors)


class SubsetDataset(Dataset):
    """A subset of a dataset at specified indices."""

    def __init__(self, dataset: Dataset, indices):
        self.dataset = dataset
        self.indices = list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]


class ConcatDataset(Dataset):
    """Concatenates multiple datasets."""

    def __init__(self, datasets: list):
        self.datasets = datasets
        self.cumulative_sizes = np.cumsum([len(d) for d in datasets])

    def __len__(self) -> int:
        return int(self.cumulative_sizes[-1])

    def __getitem__(self, idx):
        dataset_idx = np.searchsorted(self.cumulative_sizes, idx, side='right')
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]
