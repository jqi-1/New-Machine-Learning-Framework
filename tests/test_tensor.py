"""Tests for numl/core/tensor.py and numl/core/ops.py."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import unittest
import numpy as np
from numl.core.tensor import Tensor
from numl.core import ops


def numerical_grad(f, x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Finite-difference numerical gradient of scalar f w.r.t. x."""
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index
        x_p = x.copy(); x_p[idx] += eps
        x_m = x.copy(); x_m[idx] -= eps
        grad[idx] = (f(x_p) - f(x_m)) / (2 * eps)
        it.iternext()
    return grad


class TestTensorBasic(unittest.TestCase):

    def test_shape_properties(self):
        t = Tensor(np.ones((3, 4)))
        self.assertEqual(t.shape, (3, 4))
        self.assertEqual(t.ndim, 2)
        self.assertEqual(t.size, 12)

    def test_repr(self):
        t = Tensor(np.array([1.0, 2.0]), requires_grad=True)
        self.assertIn("Tensor", repr(t))

    def test_item(self):
        t = Tensor(np.array(3.14))
        self.assertAlmostEqual(t.item(), 3.14, places=5)

    def test_numpy(self):
        arr = np.array([1.0, 2.0, 3.0])
        t = Tensor(arr)
        np.testing.assert_array_equal(t.numpy(), arr)

    def test_detach(self):
        t = Tensor(np.ones(3), requires_grad=True)
        d = t.detach()
        self.assertFalse(d.requires_grad)
        self.assertIsNone(d._bwd_fn)

    def test_zero_grad(self):
        t = Tensor(np.ones(3), requires_grad=True)
        t.grad = np.array([1.0, 2.0, 3.0])
        t.zero_grad()
        np.testing.assert_array_equal(t.grad, np.zeros(3))

    def test_integer_auto_promote(self):
        t = Tensor(np.array([1, 2, 3]))
        self.assertEqual(t.dtype, np.float64)


class TestArithmetic(unittest.TestCase):

    def test_add_grad(self):
        a = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
        b = Tensor(np.array([4.0, 5.0, 6.0]), requires_grad=True)
        c = (a + b).sum()
        c.backward()
        np.testing.assert_array_almost_equal(a.grad, np.ones(3))
        np.testing.assert_array_almost_equal(b.grad, np.ones(3))

    def test_mul_grad(self):
        a = Tensor(np.array([1.0, 2.0]), requires_grad=True)
        b = Tensor(np.array([3.0, 4.0]), requires_grad=True)
        c = (a * b).sum()
        c.backward()
        np.testing.assert_array_almost_equal(a.grad, b.data)
        np.testing.assert_array_almost_equal(b.grad, a.data)

    def test_sub_grad(self):
        a = Tensor(np.array([2.0, 3.0]), requires_grad=True)
        b = Tensor(np.array([1.0, 1.0]), requires_grad=True)
        c = (a - b).sum()
        c.backward()
        np.testing.assert_array_almost_equal(a.grad, np.ones(2))
        np.testing.assert_array_almost_equal(b.grad, -np.ones(2))

    def test_neg_grad(self):
        a = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
        c = (-a).sum()
        c.backward()
        np.testing.assert_array_almost_equal(a.grad, -np.ones(3))

    def test_pow_grad(self):
        a = Tensor(np.array([2.0, 3.0]), requires_grad=True)
        c = (a ** 2).sum()
        c.backward()
        np.testing.assert_array_almost_equal(a.grad, 2 * a.data)

    def test_div_grad(self):
        a = Tensor(np.array([4.0, 9.0]), requires_grad=True)
        b = Tensor(np.array([2.0, 3.0]), requires_grad=True)
        c = (a / b).sum()
        c.backward()
        np.testing.assert_array_almost_equal(a.grad, 1.0 / b.data)
        np.testing.assert_array_almost_equal(b.grad, -a.data / b.data ** 2)

    def test_matmul_grad(self):
        A = Tensor(np.random.randn(3, 4), requires_grad=True)
        B = Tensor(np.random.randn(4, 5), requires_grad=True)
        C = (A @ B).sum()
        C.backward()
        # dA = grad @ B.T = ones(3,5) @ B.T
        np.testing.assert_array_almost_equal(A.grad, np.ones((3, 5)) @ B.data.T)
        np.testing.assert_array_almost_equal(B.grad, A.data.T @ np.ones((3, 5)))


class TestBroadcastGrad(unittest.TestCase):
    """Gradients must reduce broadcast dims correctly."""

    def test_add_broadcast_bias(self):
        # x: (3,4)  b: (4,)
        x = Tensor(np.ones((3, 4)), requires_grad=True)
        b = Tensor(np.ones(4), requires_grad=True)
        c = (x + b).sum()
        c.backward()
        self.assertEqual(x.grad.shape, (3, 4))
        self.assertEqual(b.grad.shape, (4,))
        np.testing.assert_array_almost_equal(b.grad, np.full(4, 3.0))

    def test_mul_broadcast(self):
        x = Tensor(np.ones((2, 3)), requires_grad=True)
        s = Tensor(np.array([2.0]), requires_grad=True)
        c = (x * s).sum()
        c.backward()
        np.testing.assert_array_almost_equal(x.grad, np.full((2, 3), 2.0))
        np.testing.assert_array_almost_equal(s.grad, [6.0])  # sum of x


class TestGradAccumulation(unittest.TestCase):
    """A tensor used twice should accumulate gradients from both paths."""

    def test_reuse(self):
        a = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
        b = a + a  # both paths go through a
        c = b.sum()
        c.backward()
        np.testing.assert_array_almost_equal(a.grad, np.full(3, 2.0))


class TestReductionOps(unittest.TestCase):

    def test_sum_backward(self):
        a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), requires_grad=True)
        c = a.sum()
        c.backward()
        np.testing.assert_array_almost_equal(a.grad, np.ones((2, 2)))

    def test_sum_axis(self):
        a = Tensor(np.ones((3, 4)), requires_grad=True)
        c = a.sum(axis=0).sum()
        c.backward()
        np.testing.assert_array_almost_equal(a.grad, np.ones((3, 4)))

    def test_mean_backward(self):
        a = Tensor(np.array([1.0, 2.0, 3.0, 4.0]), requires_grad=True)
        c = a.mean()
        c.backward()
        np.testing.assert_array_almost_equal(a.grad, np.full(4, 0.25))


class TestMathOps(unittest.TestCase):

    def _grad_check(self, fn, x_data, tol=1e-4):
        x = Tensor(x_data.copy(), requires_grad=True)
        out = fn(x)
        if out.data.ndim > 0:
            out = out.sum()
        out.backward()
        analytical = x.grad.copy()
        numerical = numerical_grad(lambda v: fn(Tensor(v)).data.sum(), x_data)
        np.testing.assert_allclose(analytical, numerical, rtol=tol, atol=tol)

    def test_exp_grad(self):
        self._grad_check(lambda x: x.exp(), np.array([0.5, 1.0, 2.0]))

    def test_log_grad(self):
        self._grad_check(lambda x: x.log(), np.array([0.5, 1.0, 2.0]))

    def test_abs_grad(self):
        self._grad_check(lambda x: x.abs(), np.array([-1.0, 2.0, -3.0]))

    def test_relu_grad(self):
        from numl.core.ops import relu
        self._grad_check(lambda x: relu(x), np.array([-1.0, 0.5, 2.0]))

    def test_sigmoid_grad(self):
        from numl.core.ops import sigmoid
        self._grad_check(lambda x: sigmoid(x), np.array([-2.0, 0.0, 2.0]))

    def test_tanh_grad(self):
        from numl.core.ops import tanh_op
        self._grad_check(lambda x: tanh_op(x), np.array([-1.0, 0.0, 1.0]))


class TestShapeOps(unittest.TestCase):

    def test_reshape(self):
        a = Tensor(np.arange(6.0), requires_grad=True)
        b = a.reshape(2, 3).sum()
        b.backward()
        np.testing.assert_array_almost_equal(a.grad, np.ones(6))

    def test_transpose(self):
        a = Tensor(np.ones((2, 3)), requires_grad=True)
        b = a.T.sum()
        b.backward()
        np.testing.assert_array_almost_equal(a.grad, np.ones((2, 3)))

    def test_slice(self):
        a = Tensor(np.array([1.0, 2.0, 3.0, 4.0]), requires_grad=True)
        b = a[1:3].sum()
        b.backward()
        expected = np.array([0.0, 1.0, 1.0, 0.0])
        np.testing.assert_array_almost_equal(a.grad, expected)


if __name__ == '__main__':
    unittest.main()
