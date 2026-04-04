"""Tests for classical ML algorithms."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import unittest
import numpy as np


def make_blobs(n=100, centers=None, std=0.5, seed=0):
    rng = np.random.RandomState(seed)
    if centers is None:
        centers = np.array([[0.0, 0.0], [4.0, 4.0]])
    X, y = [], []
    per = n // len(centers)
    for i, c in enumerate(centers):
        X.append(rng.randn(per, len(c)) * std + c)
        y.extend([i] * per)
    return np.vstack(X), np.array(y)


class TestLinearRegression(unittest.TestCase):

    def test_perfect_fit(self):
        from numl.classical.linear import LinearRegression
        X = np.linspace(0, 10, 50).reshape(-1, 1)
        y = 3.0 * X.ravel() + 2.0
        model = LinearRegression()
        model.fit(X, y)
        pred = model.predict(X)
        np.testing.assert_allclose(pred, y, atol=1e-8)

    def test_score(self):
        from numl.classical.linear import LinearRegression
        rng = np.random.RandomState(0)
        X = rng.randn(100, 3)
        y = X @ np.array([1.0, -2.0, 0.5]) + 0.5
        model = LinearRegression()
        model.fit(X, y)
        self.assertGreater(model.score(X, y), 0.99)


class TestRidge(unittest.TestCase):

    def test_shrinkage(self):
        from numl.classical.linear import Ridge, LinearRegression
        rng = np.random.RandomState(1)
        X = rng.randn(20, 5)
        y = X[:, 0] + rng.randn(20) * 0.1
        lr = LinearRegression().fit(X, y)
        ridge = Ridge(alpha=10.0).fit(X, y)
        # Ridge coefs should have smaller norm than OLS
        self.assertLess(np.linalg.norm(ridge.coef_), np.linalg.norm(lr.coef_))


class TestLasso(unittest.TestCase):

    def test_sparsity(self):
        from numl.classical.linear import Lasso
        rng = np.random.RandomState(2)
        X = rng.randn(50, 10)
        y = X[:, 0] * 2 + X[:, 1] * (-1) + rng.randn(50) * 0.1
        model = Lasso(alpha=0.5, max_iter=1000).fit(X, y)
        # Most coefficients should be zero
        n_zero = np.sum(np.abs(model.coef_) < 1e-3)
        self.assertGreater(n_zero, 5)


class TestLogisticRegression(unittest.TestCase):

    def test_binary_accuracy(self):
        from numl.classical.logistic import LogisticRegression
        X, y = make_blobs(n=200, std=0.5, seed=42)
        model = LogisticRegression(lr=0.1, max_iter=500)
        model.fit(X, y)
        acc = model.score(X, y)
        self.assertGreater(acc, 0.95)

    def test_predict_proba_shape(self):
        from numl.classical.logistic import LogisticRegression
        X, y = make_blobs(n=100, seed=0)
        model = LogisticRegression(max_iter=200)
        model.fit(X, y)
        proba = model.predict_proba(X)
        self.assertEqual(proba.shape, (100, 2))
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(100), atol=1e-6)


class TestDecisionTree(unittest.TestCase):

    def test_classifier_accuracy(self):
        from numl.classical.tree import DecisionTreeClassifier
        X, y = make_blobs(n=200, std=0.5, seed=10)
        tree = DecisionTreeClassifier(max_depth=4)
        tree.fit(X, y)
        acc = tree.score(X, y)
        self.assertGreater(acc, 0.9)

    def test_regressor_rmse(self):
        from numl.classical.tree import DecisionTreeRegressor
        from numl.metrics.regression import mean_squared_error
        rng = np.random.RandomState(5)
        X = rng.rand(100, 1)
        y = np.sin(2 * np.pi * X.ravel())
        tree = DecisionTreeRegressor(max_depth=6)
        tree.fit(X, y)
        pred = tree.predict(X)
        rmse = np.sqrt(mean_squared_error(y, pred))
        self.assertLess(rmse, 0.2)

    def test_feature_importances_sum(self):
        from numl.classical.tree import DecisionTreeClassifier
        X, y = make_blobs(n=100, seed=0)
        tree = DecisionTreeClassifier(max_depth=3)
        tree.fit(X, y)
        fi = tree.feature_importances_
        self.assertEqual(len(fi), X.shape[1])
        self.assertAlmostEqual(fi.sum(), 1.0, places=5)


class TestRandomForest(unittest.TestCase):

    def test_classifier_accuracy(self):
        from numl.classical.ensemble import RandomForestClassifier
        X, y = make_blobs(n=200, std=0.8, seed=7)
        rf = RandomForestClassifier(n_estimators=20, max_depth=4, random_state=0)
        rf.fit(X, y)
        acc = rf.score(X, y)
        self.assertGreater(acc, 0.90)

    def test_regressor(self):
        from numl.classical.ensemble import RandomForestRegressor
        rng = np.random.RandomState(0)
        X = rng.rand(80, 2)
        y = X[:, 0] ** 2 + X[:, 1]
        rf = RandomForestRegressor(n_estimators=10, max_depth=5, random_state=0)
        rf.fit(X, y)
        r2 = rf.score(X, y)
        self.assertGreater(r2, 0.8)


class TestKNN(unittest.TestCase):

    def test_classifier_accuracy(self):
        from numl.classical.neighbors import KNNClassifier
        X, y = make_blobs(n=200, std=0.5, seed=0)
        knn = KNNClassifier(k=5)
        knn.fit(X, y)
        acc = knn.score(X, y)
        self.assertGreater(acc, 0.95)

    def test_regressor(self):
        from numl.classical.neighbors import KNNRegressor
        rng = np.random.RandomState(1)
        X = rng.rand(50, 1)
        y = X.ravel() * 2 + 1
        knn = KNNRegressor(k=3)
        knn.fit(X, y)
        pred = knn.predict(X)
        r2 = 1 - np.sum((y - pred)**2) / np.sum((y - y.mean())**2)
        self.assertGreater(r2, 0.8)

    def test_k1_exact_memorize(self):
        from numl.classical.neighbors import KNNClassifier
        X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        y = np.array([0, 1, 2])
        knn = KNNClassifier(k=1)
        knn.fit(X, y)
        pred = knn.predict(X)
        np.testing.assert_array_equal(pred, y)


class TestNaiveBayes(unittest.TestCase):

    def test_gaussian(self):
        from numl.classical.naive_bayes import GaussianNB
        X, y = make_blobs(n=200, std=0.8, seed=3)
        nb = GaussianNB()
        nb.fit(X, y)
        acc = nb.score(X, y)
        self.assertGreater(acc, 0.85)

    def test_multinomial(self):
        from numl.classical.naive_bayes import MultinomialNB
        rng = np.random.RandomState(0)
        # Word counts for 2 topics
        X = np.vstack([
            rng.randint(0, 5, (50, 10)) + np.array([5, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            rng.randint(0, 5, (50, 10)) + np.array([0, 0, 0, 0, 0, 5, 0, 0, 0, 0]),
        ])
        y = np.array([0]*50 + [1]*50)
        nb = MultinomialNB()
        nb.fit(X, y)
        acc = nb.score(X, y)
        self.assertGreater(acc, 0.8)


class TestKMeans(unittest.TestCase):

    def test_cluster_centers(self):
        from numl.classical.cluster import KMeans
        centers = np.array([[0.0, 0.0], [10.0, 10.0], [-10.0, 10.0]])
        X, true_labels = make_blobs(n=150, centers=centers, std=0.5, seed=0)
        km = KMeans(n_clusters=3, random_state=0)
        km.fit(X)
        # Each predicted center should be close to a true center
        for c in km.cluster_centers_:
            dists = np.linalg.norm(centers - c, axis=1)
            self.assertLess(dists.min(), 2.0)

    def test_inertia_decreases(self):
        from numl.classical.cluster import KMeans
        X, _ = make_blobs(n=100, seed=42)
        km = KMeans(n_clusters=2, random_state=0)
        km.fit(X)
        self.assertIsNotNone(km.inertia_)
        self.assertGreater(km.inertia_, 0)


class TestDBSCAN(unittest.TestCase):

    def test_finds_clusters(self):
        from numl.classical.cluster import DBSCAN
        rng = np.random.RandomState(0)
        # Two tight clusters
        X = np.vstack([
            rng.randn(30, 2) * 0.3 + [0, 0],
            rng.randn(30, 2) * 0.3 + [5, 5],
        ])
        db = DBSCAN(eps=1.0, min_samples=3)
        labels = db.fit_predict(X)
        unique = set(labels) - {-1}
        self.assertEqual(len(unique), 2)

    def test_noise_label(self):
        from numl.classical.cluster import DBSCAN
        # A single outlier surrounded by nothing
        X = np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1],
                      [0.05, 0.05], [100.0, 100.0]])
        db = DBSCAN(eps=0.5, min_samples=3)
        labels = db.fit_predict(X)
        # Last point should be noise
        self.assertEqual(labels[-1], -1)


class TestSVM(unittest.TestCase):

    def test_linear_svc(self):
        from numl.classical.svm import SVC
        X, y = make_blobs(n=100, std=0.5, seed=0)
        svm = SVC(kernel='linear', C=1.0)
        svm.fit(X, y)
        acc = svm.score(X, y)
        self.assertGreater(acc, 0.9)

    def test_rbf_svc(self):
        from numl.classical.svm import SVC
        X, y = make_blobs(n=100, std=0.8, seed=1)
        svm = SVC(kernel='rbf', C=1.0)
        svm.fit(X, y)
        acc = svm.score(X, y)
        self.assertGreater(acc, 0.85)


class TestPCA(unittest.TestCase):

    def test_explained_variance(self):
        from numl.decomposition.pca import PCA
        rng = np.random.RandomState(0)
        X = rng.randn(100, 5)
        pca = PCA(n_components=3)
        X_t = pca.fit_transform(X)
        self.assertEqual(X_t.shape, (100, 3))
        self.assertAlmostEqual(sum(pca.explained_variance_ratio_), pca.explained_variance_ratio_.sum())
        self.assertLessEqual(pca.explained_variance_ratio_.sum(), 1.0 + 1e-6)

    def test_inverse_transform(self):
        from numl.decomposition.pca import PCA
        rng = np.random.RandomState(0)
        # 2D data lying in 1D subspace
        X = np.outer(rng.randn(50), np.array([1.0, 0.0, 0.0]))
        pca = PCA(n_components=1)
        X_t = pca.fit_transform(X)
        X_rec = pca.inverse_transform(X_t)
        np.testing.assert_allclose(X_rec, X, atol=1e-8)


if __name__ == '__main__':
    unittest.main()
