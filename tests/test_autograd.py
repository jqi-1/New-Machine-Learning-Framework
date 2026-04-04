"""End-to-end autograd tests with numerical gradient checks."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import unittest
import numpy as np
from numl.core.tensor import Tensor


def numerical_grad(f, x_data, eps=1e-5):
    grad = np.zeros_like(x_data)
    it = np.nditer(x_data, flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index
        xp = x_data.copy(); xp[idx] += eps
        xm = x_data.copy(); xm[idx] -= eps
        grad[idx] = (f(xp) - f(xm)) / (2 * eps)
        it.iternext()
    return grad


class TestEndToEndGrad(unittest.TestCase):

    def _check(self, fn, x_data, tol=1e-4):
        x = Tensor(x_data.copy(), requires_grad=True)
        out = fn(x)
        if out.data.ndim > 0:
            out = out.sum()
        out.backward()
        anal = x.grad.copy()
        num = numerical_grad(lambda v: fn(Tensor(v)).data.sum(), x_data)
        np.testing.assert_allclose(anal, num, rtol=tol, atol=tol,
                                   err_msg=f"Grad check failed for {fn}")

    def test_chain_exp_log(self):
        self._check(lambda x: (x.exp() * x).log(), np.array([0.5, 1.0, 1.5]))

    def test_matmul_chain(self):
        def fn(x):
            W = Tensor(np.random.randn(4, 3))
            return x @ W
        np.random.seed(0)
        self._check(fn, np.random.randn(2, 4))

    def test_two_layer_mlp(self):
        """Gradient check through a 2-layer network."""
        np.random.seed(42)
        W1 = np.random.randn(3, 4) * 0.1
        W2 = np.random.randn(4, 2) * 0.1

        def fn(x):
            from numl.core.ops import relu
            h = relu(Tensor(x) @ Tensor(W1))
            return h @ Tensor(W2)

        self._check(fn, np.random.randn(2, 3))

    def test_softmax_cross_entropy(self):
        """Combined log_softmax + NLL gradient."""
        from numl.core.ops import log_softmax

        def fn(x):
            lp = log_softmax(Tensor(x), axis=1)
            # NLL for class 0
            return -lp[:, 0]

        self._check(fn, np.random.randn(3, 4))

    def test_dense_layer_grad(self):
        from numl.nn.layers import Dense
        np.random.seed(7)
        layer = Dense(4, 3)

        def fn(x):
            return layer(Tensor(x))

        x_data = np.random.randn(2, 4)
        # Grad check w.r.t. input
        x = Tensor(x_data.copy(), requires_grad=True)
        out = layer(x).sum()
        out.backward()
        anal = x.grad.copy()
        num = numerical_grad(lambda v: layer(Tensor(v)).data.sum(), x_data)
        np.testing.assert_allclose(anal, num, rtol=1e-4, atol=1e-4)

    def test_loss_bce_with_logits(self):
        from numl.nn.loss import BCEWithLogitsLoss
        loss_fn = BCEWithLogitsLoss()
        y = Tensor(np.array([[1.0], [0.0], [1.0]]))

        def fn(x):
            return loss_fn(Tensor(x), y)

        self._check(fn, np.array([[1.5], [-0.5], [0.2]]))

    def test_loss_cross_entropy(self):
        from numl.nn.loss import CrossEntropyLoss
        loss_fn = CrossEntropyLoss()
        y = Tensor(np.array([0, 2, 1]))

        def fn(x):
            return loss_fn(Tensor(x), y)

        self._check(fn, np.random.randn(3, 3))

    def test_batch_norm_grad(self):
        from numl.nn.layers import BatchNorm1d
        bn = BatchNorm1d(4)
        bn.train()

        def fn(x):
            return bn(Tensor(x))

        x_data = np.random.randn(8, 4)
        x = Tensor(x_data.copy(), requires_grad=True)
        out = bn(x).sum()
        out.backward()
        anal = x.grad.copy()
        num = numerical_grad(lambda v: bn(Tensor(v)).data.sum(), x_data)
        np.testing.assert_allclose(anal, num, rtol=1e-3, atol=1e-3)


class TestXORConvergence(unittest.TestCase):

    def test_xor_converges(self):
        from numl.nn.layers import Dense
        from numl.nn.activations import ReLU
        from numl.nn.sequential import Sequential
        from numl.nn.loss import BCEWithLogitsLoss
        from numl.optim.adam import Adam

        np.random.seed(0)
        X = Tensor(np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64))
        y = Tensor(np.array([[0], [1], [1], [0]], dtype=np.float64))

        model = Sequential([Dense(2, 16), ReLU(), Dense(16, 1)])
        loss_fn = BCEWithLogitsLoss()
        opt = Adam(model.parameters(), lr=0.05)

        for _ in range(300):
            opt.zero_grad()
            loss = loss_fn(model(X), y)
            loss.backward()
            opt.step()

        self.assertLess(loss.item(), 0.05)


if __name__ == '__main__':
    unittest.main()
