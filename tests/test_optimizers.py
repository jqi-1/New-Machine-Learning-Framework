"""Tests for optimizers and LR schedulers."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import unittest
import numpy as np
from numl.core.tensor import Tensor


class TestSGD(unittest.TestCase):

    def test_converges_quadratic(self):
        """SGD should minimize f(w) = ||w||^2 toward zero."""
        from numl.optim.sgd import SGD
        w = Tensor(np.array([3.0, -2.0, 1.0]), requires_grad=True)
        opt = SGD([w], lr=0.1)
        for _ in range(100):
            opt.zero_grad()
            loss = (w * w).sum()
            loss.backward()
            opt.step()
        self.assertLess(loss.item(), 0.01)

    def test_momentum(self):
        from numl.optim.sgd import SGD
        w = Tensor(np.array([5.0]), requires_grad=True)
        opt = SGD([w], lr=0.1, momentum=0.9)
        losses = []
        for _ in range(50):
            opt.zero_grad()
            loss = (w * w).sum()
            losses.append(loss.item())
            loss.backward()
            opt.step()
        self.assertLess(losses[-1], losses[0])

    def test_weight_decay(self):
        from numl.optim.sgd import SGD
        w = Tensor(np.array([1.0, 1.0]), requires_grad=True)
        opt = SGD([w], lr=0.1, weight_decay=0.5)
        opt.zero_grad()
        loss = w.sum()
        loss.backward()
        opt.step()
        # With WD, update = lr*(grad + wd*w) = 0.1*(1 + 0.5) = 0.15 per element
        np.testing.assert_allclose(w.data, np.array([0.85, 0.85]), rtol=1e-5)

    def test_zero_grad_clears(self):
        from numl.optim.sgd import SGD
        w = Tensor(np.ones(3), requires_grad=True)
        opt = SGD([w], lr=0.1)
        loss = (w * w).sum()
        loss.backward()
        self.assertIsNotNone(w.grad)
        opt.zero_grad()
        np.testing.assert_array_equal(w.grad, np.zeros(3))


class TestAdam(unittest.TestCase):

    def test_converges_quadratic(self):
        from numl.optim.adam import Adam
        w = Tensor(np.array([5.0, -3.0, 2.0]), requires_grad=True)
        opt = Adam([w], lr=0.1)
        for _ in range(200):
            opt.zero_grad()
            loss = (w * w).sum()
            loss.backward()
            opt.step()
        self.assertLess(loss.item(), 0.01)

    def test_bias_correction(self):
        """First step should use bias-corrected moments."""
        from numl.optim.adam import Adam
        w = Tensor(np.array([1.0]), requires_grad=True)
        opt = Adam([w], lr=0.001, betas=(0.9, 0.999))
        opt.zero_grad()
        w.grad = np.array([1.0])
        opt.step()
        # Expected: m=0.1, v=0.001; m_hat=1.0, v_hat=1.0; update=0.001
        self.assertAlmostEqual(float(w.data[0]), 1.0 - 0.001, places=3)

    def test_adamw_decoupled_wd(self):
        from numl.optim.adam import AdamW
        w = Tensor(np.array([2.0]), requires_grad=True)
        opt = AdamW([w], lr=0.1, weight_decay=0.1)
        opt.zero_grad()
        w.grad = np.array([0.0])  # zero gradient: only WD acts
        opt.step()
        # weight decay: w *= (1 - lr*wd) = (1 - 0.01)
        self.assertAlmostEqual(float(w.data[0]), 2.0 * (1 - 0.1 * 0.1), places=5)


class TestRMSProp(unittest.TestCase):

    def test_converges(self):
        from numl.optim.rmsprop import RMSProp
        w = Tensor(np.array([3.0, -2.0]), requires_grad=True)
        opt = RMSProp([w], lr=0.01)
        for _ in range(200):
            opt.zero_grad()
            loss = (w * w).sum()
            loss.backward()
            opt.step()
        self.assertLess(loss.item(), 0.01)


class TestAdagrad(unittest.TestCase):

    def test_converges(self):
        from numl.optim.adagrad import Adagrad
        w = Tensor(np.array([3.0, -2.0]), requires_grad=True)
        opt = Adagrad([w], lr=0.5)
        for _ in range(200):
            opt.zero_grad()
            loss = (w * w).sum()
            loss.backward()
            opt.step()
        self.assertLess(loss.item(), 0.01)


class TestSchedulers(unittest.TestCase):

    def test_step_lr(self):
        from numl.optim.sgd import SGD
        from numl.optim.schedulers import StepLR
        w = Tensor(np.ones(1), requires_grad=True)
        opt = SGD([w], lr=1.0)
        sched = StepLR(opt, step_size=2, gamma=0.5)
        # epoch 0: lr=1.0, epoch 1: lr=1.0, epoch 2: lr=0.5
        initial_lr = opt.param_groups[0]['lr']
        sched.step()  # epoch 1
        sched.step()  # epoch 2 -> reduce
        self.assertAlmostEqual(opt.param_groups[0]['lr'], initial_lr * 0.5, places=6)

    def test_cosine_annealing(self):
        from numl.optim.sgd import SGD
        from numl.optim.schedulers import CosineAnnealingLR
        w = Tensor(np.ones(1), requires_grad=True)
        opt = SGD([w], lr=1.0)
        sched = CosineAnnealingLR(opt, T_max=10, eta_min=0.0)
        lrs = []
        for _ in range(10):
            lrs.append(opt.param_groups[0]['lr'])
            sched.step()
        # LR should decrease monotonically (for first half of cosine)
        self.assertGreater(lrs[0], lrs[4])

    def test_reduce_on_plateau(self):
        from numl.optim.sgd import SGD
        from numl.optim.schedulers import ReduceLROnPlateau
        w = Tensor(np.ones(1), requires_grad=True)
        opt = SGD([w], lr=1.0)
        sched = ReduceLROnPlateau(opt, patience=3, factor=0.5)
        for _ in range(5):
            sched.step(1.0)  # no improvement
        self.assertLess(opt.param_groups[0]['lr'], 1.0)


if __name__ == '__main__':
    unittest.main()
