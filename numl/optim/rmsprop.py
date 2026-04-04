"""
numl/optim/rmsprop.py
---------------------
RMSProp optimizer.
"""

import numpy as np
from numl.optim.base import Optimizer


class RMSProp(Optimizer):
    """
    RMSProp optimizer.

    Parameters
    ----------
    params       : iterable of Tensor parameters
    lr           : float, default 1e-2
    alpha        : float — smoothing constant, default 0.99
    eps          : float, default 1e-8
    weight_decay : float, default 0.0
    momentum     : float, default 0.0
    centered     : bool — if True, normalize by estimated variance of gradient
    """

    def __init__(self, params, lr: float = 1e-2, alpha: float = 0.99,
                 eps: float = 1e-8, weight_decay: float = 0.0,
                 momentum: float = 0.0, centered: bool = False):
        defaults = {
            'lr': lr,
            'alpha': alpha,
            'eps': eps,
            'weight_decay': weight_decay,
            'momentum': momentum,
            'centered': centered,
        }
        super().__init__(params, defaults)

    def step(self):
        self._step_count += 1
        for group in self.param_groups:
            lr = group['lr']
            alpha = group['alpha']
            eps = group['eps']
            wd = group['weight_decay']
            mom = group['momentum']
            centered = group['centered']

            for p in group['params']:
                if p.grad is None:
                    continue

                g = p.grad.copy()
                if wd != 0.0:
                    g = g + wd * p.data

                state = self.state.setdefault(id(p), {})
                sq = state.get('sq', np.zeros_like(p.data))
                sq = alpha * sq + (1 - alpha) * g ** 2
                state['sq'] = sq

                if centered:
                    grad_avg = state.get('grad_avg', np.zeros_like(p.data))
                    grad_avg = alpha * grad_avg + (1 - alpha) * g
                    state['grad_avg'] = grad_avg
                    denom = np.sqrt(sq - grad_avg ** 2 + eps)
                else:
                    denom = np.sqrt(sq + eps)

                if mom != 0.0:
                    buf = state.get('buf', np.zeros_like(p.data))
                    buf = mom * buf + g / denom
                    state['buf'] = buf
                    p.data -= lr * buf
                else:
                    p.data -= lr * g / denom
