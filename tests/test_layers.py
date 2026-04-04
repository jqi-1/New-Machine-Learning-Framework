"""Tests for neural network layers."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import unittest
import numpy as np
from numl.core.tensor import Tensor


class TestDense(unittest.TestCase):

    def test_output_shape(self):
        from numl.nn.layers import Dense
        layer = Dense(4, 3)
        x = Tensor(np.ones((5, 4)))
        out = layer(x)
        self.assertEqual(out.shape, (5, 3))

    def test_no_bias(self):
        from numl.nn.layers import Dense
        layer = Dense(4, 3, bias=False)
        self.assertFalse(hasattr(layer, 'bias') and layer.bias is not None)
        x = Tensor(np.ones((2, 4)))
        out = layer(x)
        self.assertEqual(out.shape, (2, 3))

    def test_weight_grad_populated(self):
        from numl.nn.layers import Dense
        from numl.nn.loss import MSELoss
        layer = Dense(3, 2)
        loss_fn = MSELoss()
        x = Tensor(np.random.randn(4, 3))
        target = Tensor(np.zeros((4, 2)))
        loss = loss_fn(layer(x), target)
        loss.backward()
        self.assertIsNotNone(layer.weight.grad)
        self.assertEqual(layer.weight.grad.shape, (3, 2))


class TestConv2d(unittest.TestCase):

    def test_output_shape(self):
        from numl.nn.layers import Conv2d
        conv = Conv2d(1, 4, kernel_size=3, padding=1)
        x = Tensor(np.ones((2, 1, 8, 8)))
        out = conv(x)
        self.assertEqual(out.shape, (2, 4, 8, 8))  # same-pad

    def test_stride(self):
        from numl.nn.layers import Conv2d
        conv = Conv2d(3, 8, kernel_size=3, stride=2)
        x = Tensor(np.ones((1, 3, 16, 16)))
        out = conv(x)
        self.assertEqual(out.shape, (1, 8, 7, 7))

    def test_grad_flows(self):
        from numl.nn.layers import Conv2d
        conv = Conv2d(1, 2, kernel_size=3)
        x = Tensor(np.random.randn(1, 1, 5, 5), requires_grad=True)
        out = conv(x).sum()
        out.backward()
        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, (1, 1, 5, 5))


class TestBatchNorm1d(unittest.TestCase):

    def test_output_normalized_train(self):
        from numl.nn.layers import BatchNorm1d
        bn = BatchNorm1d(4)
        x = Tensor(np.random.randn(16, 4) * 10 + 5)
        out = bn(x)
        # Output should be ~zero mean, ~unit std
        self.assertAlmostEqual(float(out.data.mean()), 0.0, places=4)
        self.assertAlmostEqual(float(out.data.std()), 1.0, places=2)

    def test_eval_uses_running_stats(self):
        from numl.nn.layers import BatchNorm1d
        bn = BatchNorm1d(4)
        x_train = Tensor(np.random.randn(32, 4) * 3 + 2)
        bn(x_train)  # update running stats
        bn.eval()
        x_test = Tensor(np.random.randn(4, 4))
        out_eval = bn(x_test)
        self.assertEqual(out_eval.shape, (4, 4))

    def test_grad_flows(self):
        from numl.nn.layers import BatchNorm1d
        bn = BatchNorm1d(3)
        x = Tensor(np.random.randn(8, 3), requires_grad=True)
        out = bn(x).sum()
        out.backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(bn.gamma.grad)
        self.assertIsNotNone(bn.beta.grad)


class TestDropout(unittest.TestCase):

    def test_zero_rate_in_train(self):
        from numl.nn.layers import Dropout
        drop = Dropout(p=0.0)
        x = Tensor(np.ones((100, 10)))
        out = drop(x)
        np.testing.assert_array_equal(out.data, x.data)

    def test_eval_identity(self):
        from numl.nn.layers import Dropout
        drop = Dropout(p=0.5)
        drop.eval()
        x = Tensor(np.ones((100, 10)))
        out = drop(x)
        np.testing.assert_array_equal(out.data, x.data)

    def test_approximate_rate(self):
        from numl.nn.layers import Dropout
        drop = Dropout(p=0.5)
        x = Tensor(np.ones((1000, 10)), requires_grad=True)
        # inverted dropout: output has same expected value as input
        out = drop(x)
        self.assertAlmostEqual(float(out.data.mean()), 1.0, delta=0.1)


class TestFlatten(unittest.TestCase):

    def test_shape(self):
        from numl.nn.layers import Flatten
        flat = Flatten()
        x = Tensor(np.ones((3, 2, 4, 4)))
        out = flat(x)
        self.assertEqual(out.shape, (3, 32))


class TestSequential(unittest.TestCase):

    def test_forward(self):
        from numl.nn.layers import Dense, BatchNorm1d, Dropout
        from numl.nn.activations import ReLU
        from numl.nn.sequential import Sequential
        model = Sequential([Dense(4, 8), BatchNorm1d(8), ReLU(), Dropout(0.0), Dense(8, 2)])
        x = Tensor(np.random.randn(5, 4))
        out = model(x)
        self.assertEqual(out.shape, (5, 2))

    def test_parameters_collected(self):
        from numl.nn.layers import Dense
        from numl.nn.sequential import Sequential
        model = Sequential([Dense(4, 8), Dense(8, 2)])
        params = model.parameters()
        # 2 weight + 2 bias = 4
        self.assertEqual(len(params), 4)


if __name__ == '__main__':
    unittest.main()
