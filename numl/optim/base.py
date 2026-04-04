"""
numl/optim/base.py
------------------
Base Optimizer class.
"""

import numpy as np
from numl.core.tensor import Tensor


class Optimizer:
    """
    Abstract base class for all optimizers.

    Subclasses must implement .step().

    Attributes
    ----------
    param_groups : list of dicts, each with 'params' key and hyperparameters.
    state        : dict keyed by id(param), stores per-parameter state (moments, etc.)
    """

    def __init__(self, params, defaults: dict):
        if isinstance(params, Tensor):
            params = [params]
        params = list(params)

        self.param_groups = [{'params': params, **defaults}]
        self.state: dict = {}
        self._step_count: int = 0

    def step(self):
        raise NotImplementedError

    def zero_grad(self):
        """Reset gradients on all tracked parameters to zero."""
        for group in self.param_groups:
            for p in group['params']:
                p.zero_grad()

    def add_param_group(self, param_group: dict):
        """
        Add a new parameter group (e.g., for fine-tuning specific layers).

        Parameters
        ----------
        param_group : dict with at least a 'params' key.
        """
        assert 'params' in param_group, "param_group must contain 'params'."
        self.param_groups.append(param_group)

    def state_dict(self) -> dict:
        """Return optimizer state for checkpointing."""
        return {
            'state': {str(k): v for k, v in self.state.items()},
            'param_groups': [{k: v for k, v in g.items() if k != 'params'}
                             for g in self.param_groups],
            'step_count': self._step_count,
        }

    def load_state_dict(self, state: dict):
        """Restore optimizer state from a checkpoint."""
        self._step_count = state.get('step_count', 0)
        # Restore per-param state
        for key, val in state.get('state', {}).items():
            self.state[int(key)] = val
        # Restore group hyperparameters (not params themselves)
        for g, saved_g in zip(self.param_groups, state.get('param_groups', [])):
            g.update(saved_g)
