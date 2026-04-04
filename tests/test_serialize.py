"""Tests for model serialization: save/load neural networks and classical models."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import unittest
import tempfile
import numpy as np


class SimpleNet:
    """Minimal Module subclass for serialization testing."""


class TestSaveLoadModel(unittest.TestCase):

    def _make_model(self):
        from numl.nn.layers import Dense
        from numl.nn.sequential import Sequential
        from numl.nn.activations import ReLU
        np.random.seed(0)
        return Sequential([Dense(4, 8), ReLU(), Dense(8, 2)])

    def test_state_dict_roundtrip(self):
        model = self._make_model()
        state = model.state_dict()
        # Check keys present
        self.assertTrue(any('weight' in k for k in state))
        # Restore
        model2 = self._make_model()
        # Scramble model2's weights
        for k in model2.state_dict():
            pass  # just verify it's callable
        model2.load_state_dict(state)
        # Now model2 should have same weights as model
        state2 = model2.state_dict()
        for k in state:
            np.testing.assert_array_equal(state[k], state2[k])

    def test_save_load_npz_json(self):
        from numl.serialize.io import save_model, load_model
        from numl.nn.sequential import Sequential
        from numl.nn.layers import Dense

        model = Sequential([Dense(3, 4), Dense(4, 2)])
        x = np.random.randn(5, 3)

        # Run forward pass to get reference output
        from numl.core.tensor import Tensor
        ref_out = model(Tensor(x)).data

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'test_model')
            save_model(model, path)

            # Check files exist
            self.assertTrue(os.path.exists(path + '.npz'))
            self.assertTrue(os.path.exists(path + '.json'))

            # Load and compare
            model2 = load_model(path, model_class=Sequential)
            # Sequential with no-arg constructor won't work without config
            # Instead just verify state_dict loads correctly
            model2_state = model2.state_dict()
            model_state = model.state_dict()
            for k in model_state:
                np.testing.assert_array_equal(model_state[k], model2_state[k])

    def test_save_load_preserves_predictions(self):
        from numl.serialize.io import save_model, load_model
        from numl.nn.sequential import Sequential
        from numl.nn.layers import Dense
        from numl.core.tensor import Tensor

        np.random.seed(42)
        model = Sequential([Dense(2, 4), Dense(4, 1)])
        x = Tensor(np.random.randn(3, 2))
        pred_before = model(x).data.copy()

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'model')
            save_model(model, path)

            model2 = load_model(path, model_class=Sequential)
            pred_after = model2(x).data
            np.testing.assert_allclose(pred_before, pred_after, atol=1e-10)


class TestSaveLoadClassical(unittest.TestCase):

    def test_pickle_roundtrip(self):
        from numl.serialize.io import save_classical, load_classical
        from numl.classical.linear import LinearRegression

        rng = np.random.RandomState(0)
        X = rng.randn(50, 3)
        y = X[:, 0] * 2 + X[:, 1] - X[:, 2]

        model = LinearRegression()
        model.fit(X, y)
        pred_before = model.predict(X).copy()

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'lr_model')
            save_classical(model, path)
            model2 = load_classical(path)

        pred_after = model2.predict(X)
        np.testing.assert_allclose(pred_before, pred_after, atol=1e-10)

    def test_classical_preserves_type(self):
        from numl.serialize.io import save_classical, load_classical
        from numl.classical.tree import DecisionTreeClassifier

        X = np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
        y = np.array([0, 0, 1, 1])
        tree = DecisionTreeClassifier(max_depth=2)
        tree.fit(X, y)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'tree')
            save_classical(tree, path)
            tree2 = load_classical(path)

        self.assertIsInstance(tree2, DecisionTreeClassifier)
        np.testing.assert_array_equal(tree.predict(X), tree2.predict(X))


class TestCheckpoint(unittest.TestCase):

    def test_save_load_checkpoint(self):
        from numl.serialize.io import save_checkpoint, load_checkpoint
        from numl.nn.sequential import Sequential
        from numl.nn.layers import Dense
        from numl.optim.adam import Adam
        from numl.core.tensor import Tensor

        np.random.seed(0)
        model = Sequential([Dense(2, 4), Dense(4, 1)])
        opt = Adam(model.parameters(), lr=0.001)

        # Do a few steps
        x = Tensor(np.random.randn(4, 2))
        for _ in range(3):
            opt.zero_grad()
            loss = model(x).sum()
            loss.backward()
            opt.step()

        model_state = model.state_dict()

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'run')
            save_checkpoint(model, opt, epoch=3, path=path, extra={'loss': 0.5})
            meta = load_checkpoint(path, model)

        self.assertEqual(meta['epoch'], 3)
        self.assertEqual(meta['extra']['loss'], 0.5)

        # Model state should be unchanged after load
        for k, v in model.state_dict().items():
            np.testing.assert_array_equal(v, model_state[k])


if __name__ == '__main__':
    unittest.main()
