"""Tests for classification, regression, and clustering metrics."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import unittest
import numpy as np


class TestClassificationMetrics(unittest.TestCase):

    def test_accuracy_perfect(self):
        from numl.metrics.classification import accuracy_score
        y_true = np.array([0, 1, 2, 1, 0])
        y_pred = np.array([0, 1, 2, 1, 0])
        self.assertEqual(accuracy_score(y_true, y_pred), 1.0)

    def test_accuracy_partial(self):
        from numl.metrics.classification import accuracy_score
        y_true = np.array([0, 1, 2, 1, 0])
        y_pred = np.array([0, 0, 2, 1, 1])
        self.assertAlmostEqual(accuracy_score(y_true, y_pred), 3 / 5)

    def test_confusion_matrix(self):
        from numl.metrics.classification import confusion_matrix
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 1, 1, 1, 2, 0])
        cm = confusion_matrix(y_true, y_pred)
        self.assertEqual(cm.shape, (3, 3))
        self.assertEqual(cm[0, 0], 1)  # true 0, pred 0
        self.assertEqual(cm[0, 1], 1)  # true 0, pred 1
        self.assertEqual(cm[1, 1], 2)  # true 1, pred 1

    def test_precision_recall_f1_binary(self):
        from numl.metrics.classification import precision_score, recall_score, f1_score
        y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0])
        y_pred = np.array([1, 0, 1, 0, 0, 1, 1, 0])
        # TP=3, FP=1, FN=1, TN=3
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        self.assertAlmostEqual(precision, 3 / 4)
        self.assertAlmostEqual(recall, 3 / 4)
        self.assertAlmostEqual(f1, 3 / 4)

    def test_f1_macro(self):
        from numl.metrics.classification import f1_score
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 0, 1, 1, 2, 2])
        # Perfect predictions
        self.assertAlmostEqual(f1_score(y_true, y_pred, average='macro'), 1.0)

    def test_roc_auc(self):
        from numl.metrics.classification import roc_auc_score
        # Perfect classifier
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.1, 0.2, 0.8, 0.9])
        self.assertAlmostEqual(roc_auc_score(y_true, y_score), 1.0)

    def test_roc_auc_random(self):
        from numl.metrics.classification import roc_auc_score
        # Random classifier => ~0.5
        rng = np.random.RandomState(42)
        y_true = rng.randint(0, 2, 1000)
        y_score = rng.rand(1000)
        auc = roc_auc_score(y_true, y_score)
        self.assertAlmostEqual(auc, 0.5, delta=0.05)

    def test_roc_curve(self):
        from numl.metrics.classification import roc_curve
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.1, 0.4, 0.35, 0.8])
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        self.assertEqual(len(fpr), len(tpr))
        self.assertAlmostEqual(fpr[0], 0.0)
        self.assertAlmostEqual(tpr[-1], 1.0)


class TestRegressionMetrics(unittest.TestCase):

    def test_mse(self):
        from numl.metrics.regression import mean_squared_error
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 4.0])
        self.assertAlmostEqual(mean_squared_error(y_true, y_pred), 1 / 3)

    def test_rmse(self):
        from numl.metrics.regression import mean_squared_error
        y_true = np.array([0.0, 0.0])
        y_pred = np.array([3.0, 4.0])
        # MSE = (9+16)/2=12.5; RMSE = sqrt(12.5)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        self.assertAlmostEqual(rmse, np.sqrt(12.5))

    def test_mae(self):
        from numl.metrics.regression import mean_absolute_error
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 2.0, 2.0])
        self.assertAlmostEqual(mean_absolute_error(y_true, y_pred), 2 / 3)

    def test_r2_perfect(self):
        from numl.metrics.regression import r2_score
        y = np.array([1.0, 2.0, 3.0, 4.0])
        self.assertAlmostEqual(r2_score(y, y), 1.0)

    def test_r2_baseline(self):
        from numl.metrics.regression import r2_score
        y_true = np.array([1.0, 2.0, 3.0])
        # Predicting the mean gives R²=0
        y_pred = np.full(3, y_true.mean())
        self.assertAlmostEqual(r2_score(y_true, y_pred), 0.0, places=10)

    def test_r2_negative(self):
        from numl.metrics.regression import r2_score
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([3.0, 2.0, 1.0])
        self.assertLess(r2_score(y_true, y_pred), 0.0)

    def test_mape(self):
        from numl.metrics.regression import mean_absolute_percentage_error
        y_true = np.array([100.0, 200.0])
        y_pred = np.array([110.0, 190.0])
        # |10/100| + |10/200| = 0.1 + 0.05 = 0.075 mean
        self.assertAlmostEqual(mean_absolute_percentage_error(y_true, y_pred), 0.075)


class TestClusteringMetrics(unittest.TestCase):

    def test_silhouette_perfect(self):
        from numl.metrics.clustering import silhouette_score
        # Two tight, well-separated clusters
        X = np.vstack([np.zeros((10, 2)), np.ones((10, 2)) * 10])
        labels = np.array([0]*10 + [1]*10)
        score = silhouette_score(X, labels)
        self.assertGreater(score, 0.9)

    def test_adjusted_rand_perfect(self):
        from numl.metrics.clustering import adjusted_rand_score
        labels_true = np.array([0, 0, 1, 1, 2, 2])
        labels_pred = np.array([0, 0, 1, 1, 2, 2])
        self.assertAlmostEqual(adjusted_rand_score(labels_true, labels_pred), 1.0)

    def test_adjusted_rand_random(self):
        from numl.metrics.clustering import adjusted_rand_score
        rng = np.random.RandomState(0)
        labels_true = rng.randint(0, 3, 100)
        labels_pred = rng.randint(0, 3, 100)
        # Random labeling should give low ARI
        ari = adjusted_rand_score(labels_true, labels_pred)
        self.assertLess(abs(ari), 0.15)

    def test_davies_bouldin(self):
        from numl.metrics.clustering import davies_bouldin_score
        # Perfect clusters => low DB score
        X = np.vstack([np.zeros((10, 2)), np.ones((10, 2)) * 100])
        labels = np.array([0]*10 + [1]*10)
        db = davies_bouldin_score(X, labels)
        self.assertLess(db, 0.01)


if __name__ == '__main__':
    unittest.main()
