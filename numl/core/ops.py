"""
numl/core/ops.py
----------------
Differentiable primitive operations for the numl autograd engine.

Protocol:
  - _op(data, requires_grad, prev, op_name, bwd_fn) creates an output Tensor.
  - bwd_fn(grad_output) accumulates gradients into parent tensors via _accum().
  - _accum() reduces broadcast dimensions automatically.
  - All backward closures use += (accumulation) to handle reuse of tensors.
"""

import numpy as np
from numl.core.tensor import Tensor, _ensure_tensor


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _needs_grad(*tensors: Tensor) -> bool:
    return any(t.requires_grad for t in tensors)


def _reduce_broadcast(grad: np.ndarray, original_shape: tuple) -> np.ndarray:
    """
    Reduce a gradient that was broadcast-expanded back to original_shape.
    Sums over leading axes (broadcasting added them) and over size-1 axes.
    """
    while grad.ndim > len(original_shape):
        grad = grad.sum(axis=0)
    for i, (og, ng) in enumerate(zip(original_shape, grad.shape)):
        if og == 1 and ng > 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad


def _accum(t: Tensor, grad: np.ndarray):
    """Accumulate gradient into tensor t (only if requires_grad)."""
    if not t.requires_grad:
        return
    grad = _reduce_broadcast(grad, t.data.shape)
    if t.grad is None:
        t.grad = np.zeros_like(t.data)
    t.grad += grad


def _op(data: np.ndarray, requires_grad: bool, prev: tuple, op_name: str, bwd_fn) -> Tensor:
    """
    Factory for operation output tensors.

    Parameters
    ----------
    data         : np.ndarray    — forward result
    requires_grad: bool          — whether this output participates in autograd
    prev         : tuple[Tensor] — parent tensors in the computation graph
    op_name      : str           — label for debugging
    bwd_fn       : callable(grad) -> None  — backward function
    """
    out = Tensor.__new__(Tensor)
    out.data = data
    out.grad = None
    out.requires_grad = requires_grad
    out._prev = prev
    out._op = op_name
    out._bwd_fn = bwd_fn
    return out


# ---------------------------------------------------------------------------
# Arithmetic
# ---------------------------------------------------------------------------

def add(a: Tensor, b: Tensor) -> Tensor:
    data = a.data + b.data
    rg = _needs_grad(a, b)

    def bwd(g):
        if a.requires_grad:
            _accum(a, g)
        if b.requires_grad:
            _accum(b, g)

    return _op(data, rg, (a, b), "add", bwd if rg else None)


def subtract(a: Tensor, b: Tensor) -> Tensor:
    data = a.data - b.data
    rg = _needs_grad(a, b)

    def bwd(g):
        if a.requires_grad:
            _accum(a, g)
        if b.requires_grad:
            _accum(b, -g)

    return _op(data, rg, (a, b), "sub", bwd if rg else None)


def multiply(a: Tensor, b: Tensor) -> Tensor:
    data = a.data * b.data
    rg = _needs_grad(a, b)
    a_data = a.data.copy()
    b_data = b.data.copy()

    def bwd(g):
        if a.requires_grad:
            _accum(a, g * b_data)
        if b.requires_grad:
            _accum(b, g * a_data)

    return _op(data, rg, (a, b), "mul", bwd if rg else None)


def divide(a: Tensor, b: Tensor) -> Tensor:
    data = a.data / b.data
    rg = _needs_grad(a, b)
    a_data = a.data.copy()
    b_data = b.data.copy()

    def bwd(g):
        if a.requires_grad:
            _accum(a, g / b_data)
        if b.requires_grad:
            _accum(b, -g * a_data / (b_data ** 2))

    return _op(data, rg, (a, b), "div", bwd if rg else None)


def negate(a: Tensor) -> Tensor:
    data = -a.data
    rg = a.requires_grad

    def bwd(g):
        _accum(a, -g)

    return _op(data, rg, (a,), "neg", bwd if rg else None)


def power(a: Tensor, exp) -> Tensor:
    data = a.data ** exp
    rg = a.requires_grad
    a_data = a.data.copy()

    def bwd(g):
        _accum(a, g * exp * (a_data ** (exp - 1)))

    return _op(data, rg, (a,), "pow", bwd if rg else None)


def matmul(a: Tensor, b: Tensor) -> Tensor:
    data = a.data @ b.data
    rg = _needs_grad(a, b)
    a_data = a.data.copy()
    b_data = b.data.copy()

    def bwd(g):
        if a.requires_grad:
            if g.ndim == 0:
                ga = g * b_data
            elif b_data.ndim == 1:
                # g: (...,), b: (k,) => a: (..., k)
                ga = np.outer(g, b_data) if g.ndim == 1 else g[..., np.newaxis] * b_data
            else:
                ga = g @ b_data.swapaxes(-1, -2)
            _accum(a, ga)
        if b.requires_grad:
            if g.ndim == 0:
                gb = g * a_data
            elif a_data.ndim == 1:
                gb = np.outer(a_data, g) if g.ndim == 1 else a_data[..., np.newaxis] * g
            else:
                gb = a_data.swapaxes(-1, -2) @ g
            _accum(b, gb)

    return _op(data, rg, (a, b), "matmul", bwd if rg else None)


# ---------------------------------------------------------------------------
# Math
# ---------------------------------------------------------------------------

def exp(a: Tensor) -> Tensor:
    out_data = np.exp(a.data)
    rg = a.requires_grad
    out_copy = out_data.copy()

    def bwd(g):
        _accum(a, g * out_copy)

    return _op(out_data, rg, (a,), "exp", bwd if rg else None)


def log(a: Tensor) -> Tensor:
    data = np.log(a.data)
    rg = a.requires_grad
    a_data = a.data.copy()

    def bwd(g):
        _accum(a, g / a_data)

    return _op(data, rg, (a,), "log", bwd if rg else None)


def tensor_abs(a: Tensor) -> Tensor:
    data = np.abs(a.data)
    rg = a.requires_grad
    a_data = a.data.copy()

    def bwd(g):
        _accum(a, g * np.sign(a_data))

    return _op(data, rg, (a,), "abs", bwd if rg else None)


def clip(a: Tensor, a_min, a_max) -> Tensor:
    data = np.clip(a.data, a_min, a_max)
    rg = a.requires_grad
    mask = ((a.data >= a_min) & (a.data <= a_max)).astype(a.data.dtype)

    def bwd(g):
        _accum(a, g * mask)

    return _op(data, rg, (a,), "clip", bwd if rg else None)


# ---------------------------------------------------------------------------
# Reductions
# ---------------------------------------------------------------------------

def tensor_sum(a: Tensor, axis=None, keepdims: bool = False) -> Tensor:
    data = np.sum(a.data, axis=axis, keepdims=keepdims)
    rg = a.requires_grad
    a_shape = a.data.shape

    def bwd(g):
        if axis is None:
            grad = np.broadcast_to(g, a_shape).copy()
        else:
            g_exp = g if keepdims else np.expand_dims(g, axis=axis)
            grad = np.broadcast_to(g_exp, a_shape).copy()
        _accum(a, grad)

    return _op(data, rg, (a,), "sum", bwd if rg else None)


def tensor_mean(a: Tensor, axis=None, keepdims: bool = False) -> Tensor:
    data = np.mean(a.data, axis=axis, keepdims=keepdims)
    rg = a.requires_grad
    a_shape = a.data.shape

    if axis is None:
        n = a.data.size
    elif isinstance(axis, (list, tuple)):
        n = 1
        for ax in axis:
            n *= a_shape[ax]
    else:
        n = a_shape[axis]

    def bwd(g):
        if axis is None:
            grad = np.broadcast_to(g / n, a_shape).copy()
        else:
            g_exp = g if keepdims else np.expand_dims(g, axis=axis)
            grad = np.broadcast_to(g_exp / n, a_shape).copy()
        _accum(a, grad)

    return _op(data, rg, (a,), "mean", bwd if rg else None)


def tensor_max(a: Tensor, axis=None, keepdims: bool = False) -> Tensor:
    data = np.max(a.data, axis=axis, keepdims=keepdims)
    rg = a.requires_grad
    a_data = a.data.copy()
    d_copy = data.copy()

    def bwd(g):
        if axis is None:
            mask = (a_data == d_copy).astype(a_data.dtype)
            total = mask.sum()
            mask = mask / np.maximum(total, 1)
            _accum(a, mask * g)
        else:
            d_exp = d_copy if keepdims else np.expand_dims(d_copy, axis=axis)
            mask = (a_data == d_exp).astype(a_data.dtype)
            total = mask.sum(axis=axis, keepdims=True)
            mask = mask / np.maximum(total, 1)
            g_exp = g if keepdims else np.expand_dims(g, axis=axis)
            _accum(a, mask * np.broadcast_to(g_exp, a_data.shape))

    return _op(data, rg, (a,), "max", bwd if rg else None)


# ---------------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------------

def reshape(a: Tensor, shape) -> Tensor:
    data = a.data.reshape(shape)
    rg = a.requires_grad
    a_shape = a.data.shape

    def bwd(g):
        _accum(a, g.reshape(a_shape))

    return _op(data, rg, (a,), "reshape", bwd if rg else None)


def transpose(a: Tensor, axes=None) -> Tensor:
    data = np.transpose(a.data, axes)
    rg = a.requires_grad
    inv_axes = None if axes is None else tuple(np.argsort(axes))

    def bwd(g):
        _accum(a, np.transpose(g, inv_axes))

    return _op(data, rg, (a,), "transpose", bwd if rg else None)


def tensor_slice(a: Tensor, idx) -> Tensor:
    data = a.data[idx]
    rg = a.requires_grad
    a_shape = a.data.shape

    def bwd(g):
        if a.grad is None:
            a.grad = np.zeros(a_shape, dtype=a.data.dtype)
        np.add.at(a.grad, idx, g)

    return _op(data, rg, (a,), "slice", bwd if rg else None)


def concat(tensors: list, axis: int = 0) -> Tensor:
    arrays = [t.data for t in tensors]
    data = np.concatenate(arrays, axis=axis)
    rg = _needs_grad(*tensors)
    sizes = [t.data.shape[axis] for t in tensors]

    def bwd(g):
        splits = np.split(g, np.cumsum(sizes[:-1]), axis=axis)
        for t, s in zip(tensors, splits):
            if t.requires_grad:
                _accum(t, s)

    return _op(data, rg, tuple(tensors), "concat", bwd if rg else None)


def stack(tensors: list, axis: int = 0) -> Tensor:
    arrays = [t.data for t in tensors]
    data = np.stack(arrays, axis=axis)
    rg = _needs_grad(*tensors)

    def bwd(g):
        for i, t in enumerate(tensors):
            if t.requires_grad:
                _accum(t, np.take(g, i, axis=axis))

    return _op(data, rg, tuple(tensors), "stack", bwd if rg else None)


# ---------------------------------------------------------------------------
# Activation ops (also available as nn.activations modules)
# ---------------------------------------------------------------------------

def relu(a: Tensor) -> Tensor:
    mask = (a.data > 0).astype(a.data.dtype)
    data = a.data * mask
    rg = a.requires_grad

    def bwd(g):
        _accum(a, g * mask)

    return _op(data, rg, (a,), "relu", bwd if rg else None)


def leaky_relu(a: Tensor, alpha: float = 0.01) -> Tensor:
    mask = np.where(a.data > 0, 1.0, alpha)
    data = a.data * mask
    rg = a.requires_grad

    def bwd(g):
        _accum(a, g * mask)

    return _op(data, rg, (a,), "leaky_relu", bwd if rg else None)


def elu(a: Tensor, alpha: float = 1.0) -> Tensor:
    data = np.where(a.data > 0, a.data, alpha * (np.exp(np.minimum(a.data, 0)) - 1))
    rg = a.requires_grad
    a_data = a.data.copy()

    def bwd(g):
        grad_mask = np.where(a_data > 0, 1.0, alpha * np.exp(np.minimum(a_data, 0)))
        _accum(a, g * grad_mask)

    return _op(data, rg, (a,), "elu", bwd if rg else None)


def sigmoid(a: Tensor) -> Tensor:
    # Numerically stable sigmoid
    x = a.data
    s = np.where(x >= 0,
                 1.0 / (1.0 + np.exp(-x)),
                 np.exp(x) / (1.0 + np.exp(x)))
    rg = a.requires_grad
    s_copy = s.copy()

    def bwd(g):
        _accum(a, g * s_copy * (1.0 - s_copy))

    return _op(s, rg, (a,), "sigmoid", bwd if rg else None)


def tanh_op(a: Tensor) -> Tensor:
    data = np.tanh(a.data)
    rg = a.requires_grad
    d_copy = data.copy()

    def bwd(g):
        _accum(a, g * (1.0 - d_copy ** 2))

    return _op(data, rg, (a,), "tanh", bwd if rg else None)


def softmax(a: Tensor, axis: int = -1) -> Tensor:
    shifted = a.data - np.max(a.data, axis=axis, keepdims=True)
    e = np.exp(shifted)
    s = e / np.sum(e, axis=axis, keepdims=True)
    rg = a.requires_grad
    s_copy = s.copy()

    def bwd(g):
        # Jacobian-vector product: (diag(s) - s*s^T) * g
        dot = np.sum(g * s_copy, axis=axis, keepdims=True)
        _accum(a, s_copy * (g - dot))

    return _op(s, rg, (a,), "softmax", bwd if rg else None)


def log_softmax(a: Tensor, axis: int = -1) -> Tensor:
    shifted = a.data - np.max(a.data, axis=axis, keepdims=True)
    log_sum_exp = np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))
    data = shifted - log_sum_exp
    rg = a.requires_grad
    s = np.exp(data)  # softmax values

    def bwd(g):
        sum_g = np.sum(g, axis=axis, keepdims=True)
        _accum(a, g - s * sum_g)

    return _op(data, rg, (a,), "log_softmax", bwd if rg else None)


def gelu(a: Tensor) -> Tensor:
    x = a.data
    c = np.sqrt(2.0 / np.pi)
    inner = c * (x + 0.044715 * x ** 3)
    tanh_val = np.tanh(inner)
    data = 0.5 * x * (1.0 + tanh_val)
    rg = a.requires_grad
    x_c = x.copy()
    t_c = tanh_val.copy()

    def bwd(g):
        c2 = np.sqrt(2.0 / np.pi)
        sech2 = 1.0 - t_c ** 2
        dtanh = c2 * (1.0 + 3.0 * 0.044715 * x_c ** 2)
        dgelu = 0.5 * (1.0 + t_c) + 0.5 * x_c * sech2 * dtanh
        _accum(a, g * dgelu)

    return _op(data, rg, (a,), "gelu", bwd if rg else None)


def swish(a: Tensor) -> Tensor:
    x = a.data
    s = np.where(x >= 0,
                 1.0 / (1.0 + np.exp(-x)),
                 np.exp(x) / (1.0 + np.exp(x)))
    data = x * s
    rg = a.requires_grad
    x_c = x.copy()
    s_c = s.copy()

    def bwd(g):
        dswish = s_c + x_c * s_c * (1.0 - s_c)
        _accum(a, g * dswish)

    return _op(data, rg, (a,), "swish", bwd if rg else None)
