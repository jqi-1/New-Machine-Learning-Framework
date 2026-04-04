"""
numl/nn/activations.py
----------------------
Activation functions available as:
  1. Standalone functions: relu(x), sigmoid(x), ...
  2. Module subclasses: ReLU(), Sigmoid(), ... (for use in Sequential)
"""

import numpy as np
from numl.core.tensor import Tensor
from numl.core import ops
from numl.nn.module import Module


# ---------------------------------------------------------------------------
# Functional activations
# ---------------------------------------------------------------------------

def relu(x: Tensor) -> Tensor:
    return ops.relu(x)


def leaky_relu(x: Tensor, alpha: float = 0.01) -> Tensor:
    return ops.leaky_relu(x, alpha=alpha)


def elu(x: Tensor, alpha: float = 1.0) -> Tensor:
    return ops.elu(x, alpha=alpha)


def sigmoid(x: Tensor) -> Tensor:
    return ops.sigmoid(x)


def tanh(x: Tensor) -> Tensor:
    return ops.tanh_op(x)


def softmax(x: Tensor, axis: int = -1) -> Tensor:
    return ops.softmax(x, axis=axis)


def log_softmax(x: Tensor, axis: int = -1) -> Tensor:
    return ops.log_softmax(x, axis=axis)


def gelu(x: Tensor) -> Tensor:
    return ops.gelu(x)


def swish(x: Tensor) -> Tensor:
    return ops.swish(x)


def selu(x: Tensor) -> Tensor:
    """Scaled ELU activation."""
    alpha = 1.6732632423543772
    scale = 1.0507009873554805
    data = np.where(x.data > 0, x.data, alpha * (np.exp(np.minimum(x.data, 0)) - 1))
    data = scale * data
    rg = x.requires_grad
    x_data = x.data.copy()

    def bwd(g):
        from numl.core.ops import _accum
        grad_mask = np.where(x_data > 0, scale, scale * alpha * np.exp(np.minimum(x_data, 0)))
        _accum(x, g * grad_mask)

    from numl.core.ops import _op
    return _op(data, rg, (x,), "selu", bwd if rg else None)


def hardswish(x: Tensor) -> Tensor:
    """Hard Swish: x * relu6(x+3) / 6"""
    data = x.data * np.clip(x.data + 3, 0, 6) / 6
    rg = x.requires_grad
    x_data = x.data.copy()

    def bwd(g):
        from numl.core.ops import _accum
        inner = x_data + 3
        mask = (inner > 0) & (inner < 6)
        grad = np.where(inner <= 0, 0.0,
               np.where(inner >= 6, 1.0,
                        (2 * x_data + 3) / 6))
        _accum(x, g * grad)

    from numl.core.ops import _op
    return _op(data, rg, (x,), "hardswish", bwd if rg else None)


# ---------------------------------------------------------------------------
# Module wrappers
# ---------------------------------------------------------------------------

class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return relu(x)

    def __repr__(self):
        return "ReLU()"


class LeakyReLU(Module):
    def __init__(self, alpha: float = 0.01):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: Tensor) -> Tensor:
        return leaky_relu(x, self.alpha)

    def get_config(self):
        return {"alpha": self.alpha}

    def __repr__(self):
        return f"LeakyReLU(alpha={self.alpha})"


class ELU(Module):
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: Tensor) -> Tensor:
        return elu(x, self.alpha)

    def get_config(self):
        return {"alpha": self.alpha}

    def __repr__(self):
        return f"ELU(alpha={self.alpha})"


class Sigmoid(Module):
    def forward(self, x: Tensor) -> Tensor:
        return sigmoid(x)

    def __repr__(self):
        return "Sigmoid()"


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        return tanh(x)

    def __repr__(self):
        return "Tanh()"


class Softmax(Module):
    def __init__(self, axis: int = -1):
        super().__init__()
        self.axis = axis

    def forward(self, x: Tensor) -> Tensor:
        return softmax(x, self.axis)

    def get_config(self):
        return {"axis": self.axis}

    def __repr__(self):
        return f"Softmax(axis={self.axis})"


class LogSoftmax(Module):
    def __init__(self, axis: int = -1):
        super().__init__()
        self.axis = axis

    def forward(self, x: Tensor) -> Tensor:
        return log_softmax(x, self.axis)

    def get_config(self):
        return {"axis": self.axis}

    def __repr__(self):
        return f"LogSoftmax(axis={self.axis})"


class GELU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return gelu(x)

    def __repr__(self):
        return "GELU()"


class Swish(Module):
    def forward(self, x: Tensor) -> Tensor:
        return swish(x)

    def __repr__(self):
        return "Swish()"


class SELU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return selu(x)

    def __repr__(self):
        return "SELU()"


class Identity(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x

    def __repr__(self):
        return "Identity()"
