"""
numl/data/dataloader.py
-----------------------
DataLoader: batches a Dataset with optional shuffling and collation.
"""

import numpy as np
from numl.core.tensor import Tensor
from numl.data.dataset import Dataset


def default_collate(batch: list):
    """
    Collate a list of samples into a batch.

    Handles: tuples, ndarrays, Tensors, scalars, lists.
    """
    if not batch:
        return batch

    elem = batch[0]

    if isinstance(elem, tuple):
        return tuple(default_collate([sample[i] for sample in batch])
                     for i in range(len(elem)))

    if isinstance(elem, Tensor):
        return Tensor(np.stack([b.data for b in batch], axis=0))

    if isinstance(elem, np.ndarray):
        return np.stack(batch, axis=0)

    if isinstance(elem, (int, float, np.integer, np.floating)):
        return np.array(batch)

    if isinstance(elem, list):
        return [default_collate([sample[i] for sample in batch])
                for i in range(len(elem))]

    return batch


class DataLoader:
    """
    Iterates over a Dataset in mini-batches.

    Parameters
    ----------
    dataset    : Dataset
    batch_size : int, default 32
    shuffle    : bool, default False
    drop_last  : bool — drop last batch if smaller than batch_size (default False)
    collate_fn : callable — custom collation function (default: default_collate)

    Example
    -------
    loader = DataLoader(ArrayDataset(X, y), batch_size=64, shuffle=True)
    for X_batch, y_batch in loader:
        ...
    """

    def __init__(self, dataset: Dataset, batch_size: int = 32, shuffle: bool = False,
                 drop_last: bool = False, collate_fn=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.collate_fn = collate_fn or default_collate

    def __iter__(self):
        indices = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.shuffle(indices)

        for start in range(0, len(indices), self.batch_size):
            batch_idx = indices[start: start + self.batch_size]
            if self.drop_last and len(batch_idx) < self.batch_size:
                break
            batch = [self.dataset[int(i)] for i in batch_idx]
            yield self.collate_fn(batch)

    def __len__(self) -> int:
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size
