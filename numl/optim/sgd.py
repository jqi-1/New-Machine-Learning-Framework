"""
numl/optim/sgd.py
-----------------
Stochastic Gradient Descent with optional momentum, Nesterov acceleration,
and weight decay (L2 regularization).
"""

import numpy as np
from numl.optim.base import Optimizer


class SGD(Optimizer):
    """
    Stochastic Gradient Descent.

    Parameters
    ----------
    params       : iterable of Tensor parameters
    lr           : float — learning rate
    momentum     : float — momentum factor (default 0.0)
    weight_decay : float — L2 penalty coefficient (default 0.0)
    nesterov     : bool — use Nesterov momentum (default False)
    dampening    : float — dampening for momentum (default 0.0)
    """

    def __init__(self, params, lr: float, momentum: float = 0.0,
                 weight_decay: float = 0.0, nesterov: bool = False,
                 dampening: float = 0.0):
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov requires momentum > 0 and dampening == 0.")
        defaults = {
            'lr': lr,
            'momentum': momentum,
            'weight_decay': weight_decay,
            'nesterov': nesterov,
            'dampening': dampening,
        }
        super().__init__(params, defaults)

    def step(self):
        self._step_count += 1
        for group in self.param_groups:
            lr = group['lr']
            mom = group['momentum']
            wd = group['weight_decay']
            nesterov = group['nesterov']
            dampening = group['dampening']

            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad.copy()

                # Weight decay: L2 reg gradient
                if wd != 0.0:
                    g = g + wd * p.data

                if mom != 0.0:
                    state = self.state.setdefault(id(p), {})
                    v = state.get('v', None)
                    if v is None:
                        v = g.copy()
                    else:
                        v = mom * v + (1 - dampening) * g
                    state['v'] = v

                    if nesterov:
                        g = g + mom * v
                    else:
                        g = v

                p.data -= lr * g
