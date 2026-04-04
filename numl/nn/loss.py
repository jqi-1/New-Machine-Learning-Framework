"""
numl/nn/loss.py
---------------
Loss functions for neural networks.

All losses expect Tensor inputs and return scalar Tensors.
Where possible, numerically stable implementations are used.
"""

import numpy as np
from numl.core.tensor import Tensor
from numl.core.ops import _op, _accum
from numl.nn.module import Module


class MSELoss(Module):
    """
    Mean Squared Error: mean((pred - target)^2)

    Parameters
    ----------
    reduction : 'mean' or 'sum'
    """

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        diff = pred - target
        loss_sq = diff * diff
        if self.reduction == 'mean':
            return loss_sq.mean()
        return loss_sq.sum()


class MAELoss(Module):
    """Mean Absolute Error: mean(|pred - target|)"""

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        diff = pred - target
        loss_abs = diff.abs()
        if self.reduction == 'mean':
            return loss_abs.mean()
        return loss_abs.sum()


class CrossEntropyLoss(Module):
    """
    Cross entropy loss for multi-class classification.

    Expects raw **logits** (pre-softmax). Internally applies log_softmax
    for numerical stability.

    Parameters
    ----------
    reduction     : 'mean' or 'sum'
    label_smoothing: float in [0, 1), default 0.0
    """

    def __init__(self, reduction: str = 'mean', label_smoothing: float = 0.0):
        super().__init__()
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """
        Parameters
        ----------
        logits  : Tensor of shape (N, C) — raw scores
        targets : Tensor of shape (N,) (class indices) or (N, C) (one-hot / soft labels)
        """
        N = logits.data.shape[0]
        C = logits.data.shape[1]

        # Numerically stable log_softmax
        x = logits.data
        x_shifted = x - x.max(axis=1, keepdims=True)
        log_sum_exp = np.log(np.sum(np.exp(x_shifted), axis=1, keepdims=True))
        log_probs = x_shifted - log_sum_exp  # (N, C)
        probs = np.exp(log_probs)            # softmax

        # Build target distribution
        if targets.data.ndim == 1:
            # Integer class indices
            target_idx = targets.data.astype(int)
            y = np.zeros((N, C))
            y[np.arange(N), target_idx] = 1.0
        else:
            y = targets.data  # already one-hot or soft

        # Label smoothing
        if self.label_smoothing > 0:
            y = y * (1 - self.label_smoothing) + self.label_smoothing / C

        loss_per_sample = -np.sum(y * log_probs, axis=1)  # (N,)
        if self.reduction == 'mean':
            loss_val = loss_per_sample.mean()
        else:
            loss_val = loss_per_sample.sum()

        rg = logits.requires_grad
        y_copy = y.copy()
        probs_copy = probs.copy()

        def bwd(g):
            if logits.requires_grad:
                # d(CE)/d(logits) = (softmax - y) / N (for 'mean')
                if self.reduction == 'mean':
                    dlogits = (probs_copy - y_copy) / N
                else:
                    dlogits = probs_copy - y_copy
                _accum(logits, g * dlogits)

        return _op(np.array(loss_val), rg, (logits,), "cross_entropy", bwd if rg else None)


class BCELoss(Module):
    """
    Binary Cross Entropy: -mean(y*log(p) + (1-y)*log(1-p))

    Expects **probabilities** (post-sigmoid). Clamps predictions to avoid log(0).
    """

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        eps = 1e-7
        p = np.clip(pred.data, eps, 1 - eps)
        y = target.data
        loss_per = -(y * np.log(p) + (1 - y) * np.log(1 - p))

        if self.reduction == 'mean':
            loss_val = loss_per.mean()
        else:
            loss_val = loss_per.sum()

        rg = pred.requires_grad
        p_copy = p.copy()
        y_copy = y.copy()
        n = pred.data.size

        def bwd(g):
            if pred.requires_grad:
                dp = -(y_copy / p_copy - (1 - y_copy) / (1 - p_copy))
                if self.reduction == 'mean':
                    dp = dp / n
                _accum(pred, g * dp)

        return _op(np.array(loss_val), rg, (pred,), "bce", bwd if rg else None)


class BCEWithLogitsLoss(Module):
    """
    Combines sigmoid + BCE in a numerically stable way.

    Uses the identity: BCE(sigmoid(x), y) = max(x,0) - x*y + log(1+exp(-|x|))

    Expects raw **logits**.
    """

    def __init__(self, reduction: str = 'mean', pos_weight=None):
        super().__init__()
        self.reduction = reduction
        self.pos_weight = pos_weight  # optional class weight for positives

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        x = logits.data
        y = targets.data

        # Stable BCE with logits
        loss_per = (np.maximum(x, 0) - x * y + np.log(1 + np.exp(-np.abs(x))))

        if self.pos_weight is not None:
            pw = np.array(self.pos_weight)
            loss_per = loss_per * (y * (pw - 1) + 1)

        if self.reduction == 'mean':
            loss_val = loss_per.mean()
        else:
            loss_val = loss_per.sum()

        rg = logits.requires_grad
        x_copy = x.copy()
        y_copy = y.copy()
        n = logits.data.size

        def bwd(g):
            if logits.requires_grad:
                # d(BCE_logits)/dx = sigmoid(x) - y
                sig = 1.0 / (1.0 + np.exp(-x_copy))
                dl = sig - y_copy
                if self.reduction == 'mean':
                    dl = dl / n
                _accum(logits, g * dl)

        return _op(np.array(loss_val), rg, (logits,), "bce_with_logits", bwd if rg else None)


class NLLLoss(Module):
    """
    Negative Log Likelihood Loss.

    Expects **log-probabilities** (e.g., output of LogSoftmax).

    Parameters
    ----------
    reduction : 'mean' or 'sum'
    """

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, log_probs: Tensor, targets: Tensor) -> Tensor:
        N = log_probs.data.shape[0]

        if targets.data.ndim == 1:
            target_idx = targets.data.astype(int)
            loss_per = -log_probs.data[np.arange(N), target_idx]
        else:
            loss_per = -(targets.data * log_probs.data).sum(axis=1)

        if self.reduction == 'mean':
            loss_val = loss_per.mean()
        else:
            loss_val = loss_per.sum()

        rg = log_probs.requires_grad
        idx_copy = (targets.data.astype(int) if targets.data.ndim == 1
                    else targets.data.copy())
        is_idx = targets.data.ndim == 1

        def bwd(g):
            if log_probs.requires_grad:
                dl = np.zeros_like(log_probs.data)
                if is_idx:
                    dl[np.arange(N), idx_copy] = -1.0
                else:
                    dl = -idx_copy
                if self.reduction == 'mean':
                    dl /= N
                _accum(log_probs, g * dl)

        return _op(np.array(loss_val), rg, (log_probs,), "nll", bwd if rg else None)


class HuberLoss(Module):
    """
    Huber (Smooth L1) loss. Quadratic for |err| <= delta, linear otherwise.

    Parameters
    ----------
    delta     : float, transition point (default 1.0)
    reduction : 'mean' or 'sum'
    """

    def __init__(self, delta: float = 1.0, reduction: str = 'mean'):
        super().__init__()
        self.delta = delta
        self.reduction = reduction

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        err = pred.data - target.data
        abs_err = np.abs(err)
        quadratic = np.minimum(abs_err, self.delta)
        linear = abs_err - quadratic
        loss_per = 0.5 * quadratic ** 2 + self.delta * linear

        if self.reduction == 'mean':
            loss_val = loss_per.mean()
        else:
            loss_val = loss_per.sum()

        rg = pred.requires_grad
        err_copy = err.copy()
        delta = self.delta
        n = pred.data.size

        def bwd(g):
            if pred.requires_grad:
                dl = np.where(np.abs(err_copy) <= delta,
                              err_copy,
                              delta * np.sign(err_copy))
                if self.reduction == 'mean':
                    dl /= n
                _accum(pred, g * dl)

        return _op(np.array(loss_val), rg, (pred,), "huber", bwd if rg else None)


class KLDivLoss(Module):
    """
    KL Divergence: sum(y * (log(y) - log_probs)) where log_probs = log(p).

    Expects log-probabilities as input.
    """

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, log_probs: Tensor, target: Tensor) -> Tensor:
        y = target.data
        lp = log_probs.data
        loss_per = y * (np.log(np.maximum(y, 1e-8)) - lp)

        if self.reduction == 'mean':
            loss_val = loss_per.mean()
        else:
            loss_val = loss_per.sum()

        rg = log_probs.requires_grad
        y_copy = y.copy()
        n = log_probs.data.size

        def bwd(g):
            if log_probs.requires_grad:
                dl = -y_copy
                if self.reduction == 'mean':
                    dl = dl / n
                _accum(log_probs, g * dl)

        return _op(np.array(loss_val), rg, (log_probs,), "kldiv", bwd if rg else None)
