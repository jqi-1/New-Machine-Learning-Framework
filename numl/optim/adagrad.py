"""
numl/optim/adagrad.py
---------------------
Adagrad optimizer — adapts learning rates using accumulated squared gradients.
"""

import numpy as np
from numl.optim.base import Optimizer


class Adagrad(Optimizer):
    """
    Adagrad optimizer.

    Parameters
    ----------
    params        : iterable of Tensor parameters
    lr            : float, default 1e-2
    lr_decay      : float, default 0.0 — learning rate decay over steps
    eps           : float, default 1e-10
    weight_decay  : float, default 0.0
    initial_accumulator_value : float, default 0.0
    """

    def __init__(self, params, lr: float = 1e-2, lr_decay: float = 0.0,
                 eps: float = 1e-10, weight_decay: float = 0.0,
                 initial_accumulator_value: float = 0.0):
        defaults = {
            'lr': lr,
            'lr_decay': lr_decay,
            'eps': eps,
            'weight_decay': weight_decay,
            'initial_accumulator_value': initial_accumulator_value,
        }
        super().__init__(params, defaults)

    def step(self):
        self._step_count += 1
        for group in self.param_groups:
            lr = group['lr']
            lr_decay = group['lr_decay']
            eps = group['eps']
            wd = group['weight_decay']
            init_val = group['initial_accumulator_value']

            # Effective LR with decay
            clr = lr / (1 + (self._step_count - 1) * lr_decay)

            for p in group['params']:
                if p.grad is None:
                    continue

                g = p.grad.copy()
                if wd != 0.0:
                    g = g + wd * p.data

                state = self.state.setdefault(id(p), {})
                acc = state.get('acc', np.full_like(p.data, init_val))
                acc = acc + g ** 2
                state['acc'] = acc

                p.data -= clr * g / (np.sqrt(acc) + eps)
