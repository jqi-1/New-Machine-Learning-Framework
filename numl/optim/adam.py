"""
numl/optim/adam.py
------------------
Adam and AdamW optimizers.

Adam  : Adaptive Moment Estimation (Kingma & Ba, 2014)
AdamW : Adam with decoupled weight decay (Loshchilov & Hutter, 2017)
"""

import numpy as np
from numl.optim.base import Optimizer


class Adam(Optimizer):
    """
    Adam optimizer.

    Parameters
    ----------
    params       : iterable of Tensor parameters
    lr           : float, default 1e-3
    betas        : (float, float), default (0.9, 0.999)
    eps          : float, default 1e-8
    weight_decay : float, default 0.0 (L2 regularization added to gradient)
    amsgrad      : bool, default False — use AMSGrad variant
    """

    def __init__(self, params, lr: float = 1e-3, betas: tuple = (0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.0, amsgrad: bool = False):
        defaults = {
            'lr': lr,
            'betas': betas,
            'eps': eps,
            'weight_decay': weight_decay,
            'amsgrad': amsgrad,
        }
        super().__init__(params, defaults)

    def step(self):
        self._step_count += 1
        t = self._step_count

        for group in self.param_groups:
            lr = group['lr']
            b1, b2 = group['betas']
            eps = group['eps']
            wd = group['weight_decay']
            amsgrad = group['amsgrad']

            for p in group['params']:
                if p.grad is None:
                    continue

                g = p.grad.copy()

                # L2 weight decay added to gradient (coupled)
                if wd != 0.0:
                    g = g + wd * p.data

                state = self.state.setdefault(id(p), {})
                m = state.get('m', np.zeros_like(p.data))
                v = state.get('v', np.zeros_like(p.data))

                # Biased first/second moment estimates
                m = b1 * m + (1 - b1) * g
                v = b2 * v + (1 - b2) * g ** 2

                state['m'] = m
                state['v'] = v

                # Bias correction
                m_hat = m / (1 - b1 ** t)
                v_hat = v / (1 - b2 ** t)

                if amsgrad:
                    v_max = state.get('v_max', np.zeros_like(p.data))
                    v_max = np.maximum(v_max, v_hat)
                    state['v_max'] = v_max
                    denom = np.sqrt(v_max) + eps
                else:
                    denom = np.sqrt(v_hat) + eps

                p.data -= lr * m_hat / denom


class AdamW(Optimizer):
    """
    AdamW optimizer — Adam with **decoupled** weight decay.

    Weight decay is applied directly to the parameters (not added to gradients),
    which is the mathematically correct form of L2 regularization with adaptive
    methods (Loshchilov & Hutter, 2017).

    Parameters
    ----------
    params       : iterable of Tensor parameters
    lr           : float, default 1e-3
    betas        : (float, float), default (0.9, 0.999)
    eps          : float, default 1e-8
    weight_decay : float, default 0.01
    """

    def __init__(self, params, lr: float = 1e-3, betas: tuple = (0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.01):
        defaults = {
            'lr': lr,
            'betas': betas,
            'eps': eps,
            'weight_decay': weight_decay,
        }
        super().__init__(params, defaults)

    def step(self):
        self._step_count += 1
        t = self._step_count

        for group in self.param_groups:
            lr = group['lr']
            b1, b2 = group['betas']
            eps = group['eps']
            wd = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                # Decoupled weight decay: shrink weights before update
                if wd != 0.0:
                    p.data *= (1 - lr * wd)

                g = p.grad.copy()

                state = self.state.setdefault(id(p), {})
                m = state.get('m', np.zeros_like(p.data))
                v = state.get('v', np.zeros_like(p.data))

                m = b1 * m + (1 - b1) * g
                v = b2 * v + (1 - b2) * g ** 2

                state['m'] = m
                state['v'] = v

                m_hat = m / (1 - b1 ** t)
                v_hat = v / (1 - b2 ** t)

                p.data -= lr * m_hat / (np.sqrt(v_hat) + eps)
