"""
numl/optim/schedulers.py
------------------------
Learning rate schedulers.

All schedulers wrap an Optimizer and adjust its learning rate(s) via .step().
"""

import numpy as np
from numl.optim.base import Optimizer


class LRScheduler:
    """Abstract base class for learning rate schedulers."""

    def __init__(self, optimizer: Optimizer, last_epoch: int = -1):
        self.optimizer = optimizer
        self.last_epoch = last_epoch
        self._base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.step()

    def get_lr(self) -> list:
        raise NotImplementedError

    def step(self, metrics=None):
        self.last_epoch += 1
        new_lrs = self.get_lr()
        for group, lr in zip(self.optimizer.param_groups, new_lrs):
            group['lr'] = lr

    def get_last_lr(self) -> list:
        return [group['lr'] for group in self.optimizer.param_groups]


class StepLR(LRScheduler):
    """
    Multiply LR by gamma every step_size epochs.

    Parameters
    ----------
    optimizer  : Optimizer
    step_size  : int — number of epochs between LR reductions
    gamma      : float — multiplicative factor (default 0.1)
    """

    def __init__(self, optimizer: Optimizer, step_size: int, gamma: float = 0.1):
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer)

    def get_lr(self) -> list:
        if self.last_epoch > 0 and self.last_epoch % self.step_size == 0:
            return [group['lr'] * self.gamma for group in self.optimizer.param_groups]
        return [group['lr'] for group in self.optimizer.param_groups]


class MultiStepLR(LRScheduler):
    """Multiply LR by gamma at each milestone epoch."""

    def __init__(self, optimizer: Optimizer, milestones: list, gamma: float = 0.1):
        self.milestones = set(milestones)
        self.gamma = gamma
        super().__init__(optimizer)

    def get_lr(self) -> list:
        if self.last_epoch in self.milestones:
            return [group['lr'] * self.gamma for group in self.optimizer.param_groups]
        return [group['lr'] for group in self.optimizer.param_groups]


class ExponentialLR(LRScheduler):
    """Multiply LR by gamma every epoch."""

    def __init__(self, optimizer: Optimizer, gamma: float):
        self.gamma = gamma
        super().__init__(optimizer)

    def get_lr(self) -> list:
        if self.last_epoch == 0:
            return self._base_lrs
        return [group['lr'] * self.gamma for group in self.optimizer.param_groups]


class CosineAnnealingLR(LRScheduler):
    """
    Cosine annealing: lr = eta_min + 0.5*(lr_0 - eta_min)*(1 + cos(pi*t/T_max))

    Parameters
    ----------
    optimizer : Optimizer
    T_max     : int — maximum number of iterations
    eta_min   : float — minimum LR (default 0)
    """

    def __init__(self, optimizer: Optimizer, T_max: int, eta_min: float = 0.0):
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer)

    def get_lr(self) -> list:
        t = self.last_epoch
        return [
            self.eta_min + 0.5 * (base_lr - self.eta_min) * (
                1 + np.cos(np.pi * t / self.T_max)
            )
            for base_lr in self._base_lrs
        ]


class CosineAnnealingWarmRestarts(LRScheduler):
    """Cosine annealing with warm restarts (SGDR)."""

    def __init__(self, optimizer: Optimizer, T_0: int, T_mult: int = 1,
                 eta_min: float = 0.0):
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self._T_cur = 0
        self._T_i = T_0
        super().__init__(optimizer)

    def get_lr(self) -> list:
        lrs = [
            self.eta_min + 0.5 * (base_lr - self.eta_min) * (
                1 + np.cos(np.pi * self._T_cur / self._T_i)
            )
            for base_lr in self._base_lrs
        ]
        return lrs

    def step(self, metrics=None):
        self.last_epoch += 1
        self._T_cur += 1
        if self._T_cur >= self._T_i:
            self._T_cur = 0
            self._T_i *= self.T_mult
        new_lrs = self.get_lr()
        for group, lr in zip(self.optimizer.param_groups, new_lrs):
            group['lr'] = lr


class ReduceLROnPlateau:
    """
    Reduce LR when a metric stops improving.

    Parameters
    ----------
    optimizer : Optimizer
    mode      : 'min' or 'max' (whether lower or higher metric is better)
    factor    : float — factor by which LR is reduced (default 0.1)
    patience  : int — epochs with no improvement before reducing (default 10)
    threshold : float — minimum change to qualify as improvement
    min_lr    : float — minimum LR (default 0)
    verbose   : bool
    """

    def __init__(self, optimizer: Optimizer, mode: str = 'min', factor: float = 0.1,
                 patience: int = 10, threshold: float = 1e-4,
                 min_lr: float = 0.0, verbose: bool = False):
        self.optimizer = optimizer
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.min_lr = min_lr
        self.verbose = verbose
        self._best = float('inf') if mode == 'min' else float('-inf')
        self._num_bad_epochs = 0

    def step(self, metrics: float):
        current = float(metrics)
        if self.mode == 'min':
            improved = current < self._best - self.threshold
        else:
            improved = current > self._best + self.threshold

        if improved:
            self._best = current
            self._num_bad_epochs = 0
        else:
            self._num_bad_epochs += 1

        if self._num_bad_epochs > self.patience:
            for group in self.optimizer.param_groups:
                new_lr = max(group['lr'] * self.factor, self.min_lr)
                if self.verbose:
                    print(f"ReduceLROnPlateau: reducing lr to {new_lr:.6f}")
                group['lr'] = new_lr
            self._num_bad_epochs = 0

    def get_last_lr(self) -> list:
        return [group['lr'] for group in self.optimizer.param_groups]


class WarmupScheduler(LRScheduler):
    """
    Linear warmup followed by another scheduler.

    Parameters
    ----------
    optimizer     : Optimizer
    warmup_epochs : int — number of warmup steps
    after_scheduler: LRScheduler — scheduler to use after warmup
    """

    def __init__(self, optimizer: Optimizer, warmup_epochs: int,
                 after_scheduler: LRScheduler | None = None):
        self.warmup_epochs = warmup_epochs
        self.after_scheduler = after_scheduler
        super().__init__(optimizer)

    def get_lr(self) -> list:
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            alpha = (self.last_epoch + 1) / self.warmup_epochs
            return [base_lr * alpha for base_lr in self._base_lrs]
        elif self.after_scheduler is not None:
            return self.after_scheduler.get_lr()
        return self._base_lrs

    def step(self, metrics=None):
        super().step(metrics)
        if self.last_epoch >= self.warmup_epochs and self.after_scheduler is not None:
            self.after_scheduler.step(metrics)


class LinearLR(LRScheduler):
    """Linearly decay LR from start_factor to end_factor over total_iters steps."""

    def __init__(self, optimizer: Optimizer, start_factor: float = 1.0 / 3,
                 end_factor: float = 1.0, total_iters: int = 5):
        self.start_factor = start_factor
        self.end_factor = end_factor
        self.total_iters = total_iters
        super().__init__(optimizer)

    def get_lr(self) -> list:
        t = min(self.last_epoch, self.total_iters)
        alpha = t / self.total_iters
        factor = self.start_factor + alpha * (self.end_factor - self.start_factor)
        return [base_lr * factor for base_lr in self._base_lrs]
