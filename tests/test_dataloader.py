"""Tests for Dataset, DataLoader, and data splits."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import unittest
import numpy as np


class TestArrayDataset(unittest.TestCase):

    def test_len_getitem(self):
        from numl.data.dataset import ArrayDataset
        X = np.ones((10, 3))
        y = np.arange(10)
        ds = ArrayDataset(X, y)
        self.assertEqual(len(ds), 10)
        xi, yi = ds[0]
        np.testing.assert_array_equal(xi, X[0])
        self.assertEqual(yi, y[0])

    def test_single_array(self):
        from numl.data.dataset import ArrayDataset
        X = np.random.randn(5, 2)
        ds = ArrayDataset(X)
        self.assertEqual(len(ds), 5)
        sample = ds[2]
        # Returns a tuple with one element
        np.testing.assert_array_equal(sample[0], X[2])


class TestTensorDataset(unittest.TestCase):

    def test_len_getitem(self):
        from numl.data.dataset import TensorDataset
        from numl.core.tensor import Tensor
        X = Tensor(np.ones((8, 4)))
        y = Tensor(np.arange(8.0))
        ds = TensorDataset(X, y)
        self.assertEqual(len(ds), 8)
        xi, yi = ds[3]
        np.testing.assert_array_equal(xi.data, X.data[3])


class TestDataLoader(unittest.TestCase):

    def test_batch_count(self):
        from numl.data.dataset import ArrayDataset
        from numl.data.dataloader import DataLoader
        X = np.ones((20, 3))
        y = np.arange(20)
        ds = ArrayDataset(X, y)
        loader = DataLoader(ds, batch_size=4, shuffle=False)
        batches = list(loader)
        self.assertEqual(len(batches), 5)
        for b_x, b_y in batches:
            self.assertEqual(b_x.shape[0], 4)

    def test_drop_last(self):
        from numl.data.dataset import ArrayDataset
        from numl.data.dataloader import DataLoader
        X = np.ones((21, 3))
        y = np.arange(21)
        ds = ArrayDataset(X, y)
        loader = DataLoader(ds, batch_size=4, shuffle=False, drop_last=True)
        batches = list(loader)
        self.assertEqual(len(batches), 5)  # 21 // 4 = 5

    def test_no_drop_last(self):
        from numl.data.dataset import ArrayDataset
        from numl.data.dataloader import DataLoader
        X = np.ones((21, 3))
        y = np.arange(21)
        ds = ArrayDataset(X, y)
        loader = DataLoader(ds, batch_size=4, shuffle=False, drop_last=False)
        batches = list(loader)
        self.assertEqual(len(batches), 6)  # 5 full + 1 partial
        # Last batch has 1 sample
        self.assertEqual(batches[-1][0].shape[0], 1)

    def test_shuffle_changes_order(self):
        from numl.data.dataset import ArrayDataset
        from numl.data.dataloader import DataLoader
        X = np.arange(20.0).reshape(20, 1)
        y = np.arange(20)
        ds = ArrayDataset(X, y)
        loader = DataLoader(ds, batch_size=20, shuffle=True)
        batches = list(loader)
        # The batch should contain all 20 elements but possibly in different order
        got = np.sort(batches[0][1].ravel())
        np.testing.assert_array_equal(got, np.arange(20))

    def test_all_samples_covered(self):
        """Every sample should appear exactly once per epoch."""
        from numl.data.dataset import ArrayDataset
        from numl.data.dataloader import DataLoader
        X = np.arange(30.0).reshape(30, 1)
        y = np.arange(30)
        ds = ArrayDataset(X, y)
        loader = DataLoader(ds, batch_size=7, shuffle=True, drop_last=False)
        all_labels = []
        for b_x, b_y in loader:
            all_labels.extend(b_y.ravel().tolist())
        self.assertEqual(sorted(all_labels), list(range(30)))


class TestSplits(unittest.TestCase):

    def test_train_test_split_sizes(self):
        from numl.data.splits import train_test_split
        X = np.ones((100, 3))
        y = np.arange(100)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)
        self.assertEqual(X_tr.shape[0], 80)
        self.assertEqual(X_te.shape[0], 20)
        self.assertEqual(len(y_tr), 80)
        self.assertEqual(len(y_te), 20)

    def test_train_test_split_no_overlap(self):
        from numl.data.splits import train_test_split
        idx = np.arange(50)
        tr, te, _, _ = train_test_split(idx, idx, test_size=0.4, random_state=1)
        self.assertEqual(len(set(tr) & set(te)), 0)

    def test_kfold(self):
        from numl.data.splits import KFold
        X = np.ones((30, 2))
        kf = KFold(n_splits=5)
        splits = list(kf.split(X))
        self.assertEqual(len(splits), 5)
        for train_idx, val_idx in splits:
            # no overlap
            self.assertEqual(len(set(train_idx) & set(val_idx)), 0)
            # full coverage
            self.assertEqual(len(train_idx) + len(val_idx), 30)

    def test_stratified_kfold(self):
        from numl.data.splits import StratifiedKFold
        X = np.ones((40, 2))
        y = np.array([0]*20 + [1]*20)
        skf = StratifiedKFold(n_splits=4)
        for train_idx, val_idx in skf.split(X, y):
            val_y = y[val_idx]
            # Each fold should have ~equal class distribution
            self.assertIn(val_y.sum(), [4, 5, 6])

    def test_cross_val_score(self):
        from numl.data.splits import cross_val_score
        from numl.classical.linear import LinearRegression
        rng = np.random.RandomState(0)
        X = rng.randn(50, 2)
        y = X[:, 0] * 2 + X[:, 1]
        scores = cross_val_score(LinearRegression(), X, y, cv=5, scoring='r2')
        self.assertEqual(len(scores), 5)
        self.assertGreater(np.mean(scores), 0.9)


if __name__ == '__main__':
    unittest.main()
