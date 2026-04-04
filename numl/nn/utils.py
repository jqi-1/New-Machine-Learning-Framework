"""
numl/nn/utils.py
----------------
Utility functions for neural network training.
"""

import numpy as np
from numl.core.tensor import Tensor


def clip_grad_norm_(params: list, max_norm: float, norm_type: float = 2.0) -> float:
    """
    Clip gradients of parameters to a maximum L-norm, in-place.

    Parameters
    ----------
    params   : list of Tensor
    max_norm : float — maximum allowed norm
    norm_type: float — type of norm (2 for L2, 1 for L1, inf for max)

    Returns
    -------
    total_norm : float — the gradient norm before clipping
    """
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return 0.0

    if norm_type == float('inf'):
        total_norm = max(np.abs(g).max() for g in grads)
    else:
        total_norm = sum(np.sum(np.abs(g) ** norm_type) for g in grads) ** (1.0 / norm_type)

    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        for p in params:
            if p.grad is not None:
                p.grad *= clip_coef

    return float(total_norm)


def clip_grad_value_(params: list, clip_value: float):
    """
    Clip gradient values of parameters to [-clip_value, clip_value], in-place.

    Parameters
    ----------
    params     : list of Tensor
    clip_value : float
    """
    for p in params:
        if p.grad is not None:
            np.clip(p.grad, -clip_value, clip_value, out=p.grad)


def count_parameters(module) -> int:
    """Return total number of trainable parameters in a Module."""
    return sum(p.data.size for p in module.parameters())


def get_parameter_norms(module) -> dict:
    """Return {name: norm} for all named parameters."""
    norms = {}
    for name, p in module.named_parameters():
        norms[name] = float(np.linalg.norm(p.data))
    return norms


def get_gradient_norms(module) -> dict:
    """Return {name: grad_norm} for parameters that have gradients."""
    norms = {}
    for name, p in module.named_parameters():
        if p.grad is not None:
            norms[name] = float(np.linalg.norm(p.grad))
    return norms
