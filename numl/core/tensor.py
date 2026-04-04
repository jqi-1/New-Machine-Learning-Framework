"""
numl/core/tensor.py
-------------------
The core Tensor class with reverse-mode automatic differentiation.

Design:
  - Each tensor holds .data (np.ndarray) and optionally .grad.
  - Operations create new Tensors and attach a _bwd_fn closure.
  - .backward() does a topological sort and walks in reverse, calling
    each node's _bwd_fn(node.grad) to propagate gradients.
  - Gradients accumulate (+= not =) so shared tensors work correctly.
"""

import numpy as np


class Tensor:
    """
    A NumPy-backed tensor with automatic differentiation support.
    """

    def __init__(self, data, requires_grad: bool = False, dtype=None):
        if isinstance(data, Tensor):
            data = data.data
        if isinstance(data, (int, float)):
            data = np.array(data, dtype=np.float64)
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        if dtype is not None:
            data = data.astype(dtype)
        elif data.dtype.kind in ('i', 'u'):
            data = data.astype(np.float64)

        self.data: np.ndarray = data
        self.grad: np.ndarray | None = None
        self.requires_grad: bool = requires_grad

        # Computation graph
        self._prev: tuple["Tensor", ...] = ()
        self._op: str = ""
        self._bwd_fn = None  # Callable[[np.ndarray], None] | None

    # ------------------------------------------------------------------
    # Backpropagation
    # ------------------------------------------------------------------

    def backward(self, gradient: np.ndarray | None = None):
        """
        Compute gradients via reverse-mode autodiff.

        Parameters
        ----------
        gradient : optional np.ndarray
            The upstream gradient. Defaults to ones for scalars.
        """
        if not self.requires_grad:
            return

        if gradient is None:
            if self.data.size == 1:
                gradient = np.ones_like(self.data)
            else:
                raise RuntimeError(
                    "backward() requires an explicit gradient for non-scalar tensors."
                )

        # Seed this tensor's gradient
        if self.grad is None:
            self.grad = np.zeros_like(self.data)
        self.grad += gradient

        # Build topological ordering (leaves last)
        topo: list["Tensor"] = []
        visited: set[int] = set()

        def build(t: "Tensor"):
            if id(t) not in visited:
                visited.add(id(t))
                for parent in t._prev:
                    build(parent)
                topo.append(t)

        build(self)

        # Walk in reverse (from output toward leaves), propagate grads
        for node in reversed(topo):
            if node.grad is not None and node._bwd_fn is not None:
                node._bwd_fn(node.grad)

    # ------------------------------------------------------------------
    # Gradient utilities
    # ------------------------------------------------------------------

    def zero_grad(self):
        """Reset gradient to zero."""
        if self.grad is not None:
            self.grad = np.zeros_like(self.data)

    def detach(self) -> "Tensor":
        """Return a new Tensor with the same data but no gradient tracking."""
        t = Tensor.__new__(Tensor)
        t.data = self.data
        t.grad = None
        t.requires_grad = False
        t._prev = ()
        t._op = ""
        t._bwd_fn = None
        return t

    def numpy(self) -> np.ndarray:
        """Return the underlying NumPy array."""
        return self.data

    def item(self):
        """Return Python scalar for single-element tensors."""
        return self.data.item()

    # ------------------------------------------------------------------
    # Shape / property delegation
    # ------------------------------------------------------------------

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def T(self) -> "Tensor":
        return self.transpose()

    def reshape(self, *shape) -> "Tensor":
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        from numl.core.ops import reshape
        return reshape(self, shape)

    def transpose(self, axes=None) -> "Tensor":
        from numl.core.ops import transpose
        return transpose(self, axes)

    def sum(self, axis=None, keepdims: bool = False) -> "Tensor":
        from numl.core.ops import tensor_sum
        return tensor_sum(self, axis=axis, keepdims=keepdims)

    def mean(self, axis=None, keepdims: bool = False) -> "Tensor":
        from numl.core.ops import tensor_mean
        return tensor_mean(self, axis=axis, keepdims=keepdims)

    def max(self, axis=None, keepdims: bool = False) -> "Tensor":
        from numl.core.ops import tensor_max
        return tensor_max(self, axis=axis, keepdims=keepdims)

    def exp(self) -> "Tensor":
        from numl.core.ops import exp
        return exp(self)

    def log(self) -> "Tensor":
        from numl.core.ops import log
        return log(self)

    def abs(self) -> "Tensor":
        from numl.core.ops import tensor_abs
        return tensor_abs(self)

    def clip(self, a_min, a_max) -> "Tensor":
        from numl.core.ops import clip
        return clip(self, a_min, a_max)

    def sqrt(self) -> "Tensor":
        return self ** 0.5

    def __getitem__(self, idx) -> "Tensor":
        from numl.core.ops import tensor_slice
        return tensor_slice(self, idx)

    # ------------------------------------------------------------------
    # Arithmetic operator overloads
    # ------------------------------------------------------------------

    def __add__(self, other) -> "Tensor":
        from numl.core.ops import add
        return add(self, _ensure_tensor(other))

    def __radd__(self, other) -> "Tensor":
        from numl.core.ops import add
        return add(_ensure_tensor(other), self)

    def __sub__(self, other) -> "Tensor":
        from numl.core.ops import subtract
        return subtract(self, _ensure_tensor(other))

    def __rsub__(self, other) -> "Tensor":
        from numl.core.ops import subtract
        return subtract(_ensure_tensor(other), self)

    def __mul__(self, other) -> "Tensor":
        from numl.core.ops import multiply
        return multiply(self, _ensure_tensor(other))

    def __rmul__(self, other) -> "Tensor":
        from numl.core.ops import multiply
        return multiply(_ensure_tensor(other), self)

    def __truediv__(self, other) -> "Tensor":
        from numl.core.ops import divide
        return divide(self, _ensure_tensor(other))

    def __rtruediv__(self, other) -> "Tensor":
        from numl.core.ops import divide
        return divide(_ensure_tensor(other), self)

    def __neg__(self) -> "Tensor":
        from numl.core.ops import negate
        return negate(self)

    def __pow__(self, exp) -> "Tensor":
        from numl.core.ops import power
        return power(self, exp)

    def __matmul__(self, other) -> "Tensor":
        from numl.core.ops import matmul
        return matmul(self, _ensure_tensor(other))

    def __rmatmul__(self, other) -> "Tensor":
        from numl.core.ops import matmul
        return matmul(_ensure_tensor(other), self)

    # Comparisons (no gradient)
    def __gt__(self, other):
        return Tensor(self.data > _ensure_tensor(other).data)

    def __lt__(self, other):
        return Tensor(self.data < _ensure_tensor(other).data)

    def __ge__(self, other):
        return Tensor(self.data >= _ensure_tensor(other).data)

    def __le__(self, other):
        return Tensor(self.data <= _ensure_tensor(other).data)

    def __eq__(self, other):
        return Tensor(self.data == _ensure_tensor(other).data)

    def __repr__(self) -> str:
        grad_fn = f", grad_fn=<{self._op}>" if self._op else ""
        rg = ", requires_grad=True" if self.requires_grad else ""
        return f"Tensor({self.data}{grad_fn}{rg})"

    def __len__(self):
        return len(self.data)

    def __hash__(self):
        return id(self)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _ensure_tensor(x) -> Tensor:
    if isinstance(x, Tensor):
        return x
    return Tensor(np.array(x, dtype=np.float64))


# ------------------------------------------------------------------
# Convenience constructors
# ------------------------------------------------------------------

def tensor(data, requires_grad: bool = False, dtype=None) -> Tensor:
    return Tensor(data, requires_grad=requires_grad, dtype=dtype)


def zeros(*shape, requires_grad: bool = False) -> Tensor:
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])
    return Tensor(np.zeros(shape), requires_grad=requires_grad)


def ones(*shape, requires_grad: bool = False) -> Tensor:
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])
    return Tensor(np.ones(shape), requires_grad=requires_grad)


def randn(*shape, requires_grad: bool = False) -> Tensor:
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])
    return Tensor(np.random.randn(*shape), requires_grad=requires_grad)


def concatenate(tensors: list, axis: int = 0) -> Tensor:
    from numl.core.ops import concat
    return concat(tensors, axis=axis)
