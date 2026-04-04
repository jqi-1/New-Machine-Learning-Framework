"""Tests for preprocessing: scalers, encoders, imputers, transforms."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import unittest
import numpy as np


class TestStandardScaler(unittest.TestCase):

    def test_fit_transform(self):
        from numl.preprocessing.scalers import StandardScaler
        rng = np.random.RandomState(0)
        X = rng.randn(100, 4) * 5 + 3
        sc = StandardScaler()
        Xt = sc.fit_transform(X)
        np.testing.assert_allclose(Xt.mean(axis=0), np.zeros(4), atol=1e-10)
        np.testing.assert_allclose(Xt.std(axis=0), np.ones(4), atol=1e-10)

    def test_inverse_transform(self):
        from numl.preprocessing.scalers import StandardScaler
        rng = np.random.RandomState(1)
        X = rng.randn(50, 3) * 2 + 1
        sc = StandardScaler()
        sc.fit(X)
        Xt = sc.transform(X)
        X_rec = sc.inverse_transform(Xt)
        np.testing.assert_allclose(X_rec, X, atol=1e-10)

    def test_no_std_copy_false(self):
        from numl.preprocessing.scalers import StandardScaler
        X = np.ones((10, 2))
        sc = StandardScaler()
        Xt = sc.fit_transform(X)
        # zero-std columns should be handled without error
        self.assertEqual(Xt.shape, X.shape)


class TestMinMaxScaler(unittest.TestCase):

    def test_range(self):
        from numl.preprocessing.scalers import MinMaxScaler
        rng = np.random.RandomState(0)
        X = rng.randn(50, 3)
        sc = MinMaxScaler()
        Xt = sc.fit_transform(X)
        np.testing.assert_allclose(Xt.min(axis=0), np.zeros(3), atol=1e-10)
        np.testing.assert_allclose(Xt.max(axis=0), np.ones(3), atol=1e-10)

    def test_custom_range(self):
        from numl.preprocessing.scalers import MinMaxScaler
        X = np.array([[0.0, 10.0], [5.0, 20.0], [10.0, 30.0]])
        sc = MinMaxScaler(feature_range=(-1, 1))
        Xt = sc.fit_transform(X)
        np.testing.assert_allclose(Xt.min(axis=0), [-1, -1], atol=1e-10)
        np.testing.assert_allclose(Xt.max(axis=0), [1, 1], atol=1e-10)

    def test_inverse_transform(self):
        from numl.preprocessing.scalers import MinMaxScaler
        rng = np.random.RandomState(2)
        X = rng.rand(40, 2) * 100
        sc = MinMaxScaler()
        sc.fit(X)
        X_rec = sc.inverse_transform(sc.transform(X))
        np.testing.assert_allclose(X_rec, X, atol=1e-10)


class TestRobustScaler(unittest.TestCase):

    def test_median_iqr(self):
        from numl.preprocessing.scalers import RobustScaler
        rng = np.random.RandomState(3)
        X = rng.randn(200, 2) * 5 + 10
        sc = RobustScaler()
        Xt = sc.fit_transform(X)
        np.testing.assert_allclose(np.median(Xt, axis=0), np.zeros(2), atol=1e-10)

    def test_inverse_transform(self):
        from numl.preprocessing.scalers import RobustScaler
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]])
        sc = RobustScaler()
        sc.fit(X)
        X_rec = sc.inverse_transform(sc.transform(X))
        np.testing.assert_allclose(X_rec, X, atol=1e-10)


class TestLabelEncoder(unittest.TestCase):

    def test_encode_decode(self):
        from numl.preprocessing.encoders import LabelEncoder
        y = np.array(['cat', 'dog', 'bird', 'cat', 'dog'])
        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        y_dec = le.inverse_transform(y_enc)
        np.testing.assert_array_equal(y_dec, y)

    def test_integer_labels(self):
        from numl.preprocessing.encoders import LabelEncoder
        y = np.array([2, 5, 2, 5, 5, 2])
        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        self.assertEqual(set(y_enc), {0, 1})


class TestOneHotEncoder(unittest.TestCase):

    def test_basic(self):
        from numl.preprocessing.encoders import OneHotEncoder
        X = np.array([[0], [1], [2], [1]])
        enc = OneHotEncoder()
        Xt = enc.fit_transform(X)
        self.assertEqual(Xt.shape, (4, 3))
        np.testing.assert_allclose(Xt.sum(axis=1), np.ones(4))

    def test_inverse_transform(self):
        from numl.preprocessing.encoders import OneHotEncoder
        X = np.array([[0], [1], [2]])
        enc = OneHotEncoder()
        Xt = enc.fit_transform(X)
        X_rec = enc.inverse_transform(Xt)
        np.testing.assert_array_equal(X_rec.ravel(), X.ravel())


class TestSimpleImputer(unittest.TestCase):

    def test_mean_imputation(self):
        from numl.preprocessing.imputers import SimpleImputer
        X = np.array([[1.0, np.nan], [3.0, 4.0], [np.nan, 6.0]])
        imp = SimpleImputer(strategy='mean')
        Xt = imp.fit_transform(X)
        self.assertFalse(np.any(np.isnan(Xt)))
        np.testing.assert_allclose(Xt[0, 1], 5.0, atol=1e-10)  # mean of 4, 6
        np.testing.assert_allclose(Xt[2, 0], 2.0, atol=1e-10)  # mean of 1, 3

    def test_constant_imputation(self):
        from numl.preprocessing.imputers import SimpleImputer
        X = np.array([[1.0, np.nan], [np.nan, 2.0]])
        imp = SimpleImputer(strategy='constant', fill_value=-999)
        Xt = imp.fit_transform(X)
        self.assertEqual(Xt[0, 1], -999)
        self.assertEqual(Xt[1, 0], -999)

    def test_median_imputation(self):
        from numl.preprocessing.imputers import SimpleImputer
        X = np.array([[1.0], [2.0], [np.nan], [100.0]])
        imp = SimpleImputer(strategy='median')
        Xt = imp.fit_transform(X)
        self.assertFalse(np.any(np.isnan(Xt)))
        np.testing.assert_allclose(Xt[2, 0], np.median([1.0, 2.0, 100.0]), atol=1e-10)


class TestPolynomialFeatures(unittest.TestCase):

    def test_degree2_shape(self):
        from numl.preprocessing.transforms import PolynomialFeatures
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        pf = PolynomialFeatures(degree=2, include_bias=False)
        Xt = pf.fit_transform(X)
        # For 2 features, degree 2, no bias: x1, x2, x1^2, x1*x2, x2^2 = 5 features
        self.assertEqual(Xt.shape[0], 2)
        self.assertGreater(Xt.shape[1], 2)

    def test_degree1_identity(self):
        from numl.preprocessing.transforms import PolynomialFeatures
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        pf = PolynomialFeatures(degree=1, include_bias=False)
        Xt = pf.fit_transform(X)
        np.testing.assert_array_equal(Xt, X)


class TestBinarizer(unittest.TestCase):

    def test_threshold(self):
        from numl.preprocessing.transforms import Binarizer
        X = np.array([[0.5, 1.5, -0.5], [2.0, 0.1, 0.9]])
        b = Binarizer(threshold=1.0)
        Xt = b.fit_transform(X)
        expected = np.array([[0, 1, 0], [1, 0, 0]], dtype=float)
        np.testing.assert_array_equal(Xt, expected)


class TestNormalizer(unittest.TestCase):

    def test_l2_norm(self):
        from numl.preprocessing.scalers import Normalizer
        X = np.array([[3.0, 4.0], [1.0, 0.0]])
        n = Normalizer(norm='l2')
        Xt = n.fit_transform(X)
        np.testing.assert_allclose(np.linalg.norm(Xt, axis=1), np.ones(2), atol=1e-10)

    def test_l1_norm(self):
        from numl.preprocessing.scalers import Normalizer
        X = np.array([[1.0, 2.0, 3.0]])
        n = Normalizer(norm='l1')
        Xt = n.fit_transform(X)
        np.testing.assert_allclose(np.abs(Xt).sum(axis=1), np.ones(1), atol=1e-10)


if __name__ == '__main__':
    unittest.main()
