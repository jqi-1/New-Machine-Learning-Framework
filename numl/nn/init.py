"""
numl/nn/init.py
---------------
Weight initialization functions for neural network parameters.
All functions operate in-place on a Tensor's .data array.
"""

import numpy as np
from numl.core.tensor import Tensor


def _calculate_fan(tensor: Tensor):
    """Compute fan_in and fan_out for a weight tensor."""
    ndim = tensor.data.ndim
    if ndim < 2:
        raise ValueError("Fan can only be calculated for tensors with 2+ dimensions.")
    if ndim == 2:
        fan_in = tensor.data.shape[0]
        fan_out = tensor.data.shape[1]
    else:
        # Convolutional: (out_channels, in_channels, *kernel)
        receptive_field = 1
        for s in tensor.data.shape[2:]:
            receptive_field *= s
        fan_in = tensor.data.shape[1] * receptive_field
        fan_out = tensor.data.shape[0] * receptive_field
    return fan_in, fan_out


def kaiming_uniform(tensor: Tensor, a: float = 0.0, mode: str = "fan_in",
                    nonlinearity: str = "leaky_relu") -> Tensor:
    """
    Kaiming (He) uniform initialization.
    Recommended for ReLU/LeakyReLU networks.
    """
    fan_in, fan_out = _calculate_fan(tensor)
    fan = fan_in if mode == "fan_in" else fan_out

    if nonlinearity == "relu":
        gain = np.sqrt(2.0)
    elif nonlinearity == "leaky_relu":
        gain = np.sqrt(2.0 / (1 + a ** 2))
    elif nonlinearity in ("sigmoid", "tanh"):
        gain = 1.0
    else:
        gain = 1.0

    std = gain / np.sqrt(fan)
    bound = np.sqrt(3.0) * std
    tensor.data[:] = np.random.uniform(-bound, bound, tensor.data.shape)
    return tensor


def kaiming_normal(tensor: Tensor, a: float = 0.0, mode: str = "fan_in",
                   nonlinearity: str = "leaky_relu") -> Tensor:
    """Kaiming (He) normal initialization."""
    fan_in, fan_out = _calculate_fan(tensor)
    fan = fan_in if mode == "fan_in" else fan_out

    if nonlinearity == "relu":
        gain = np.sqrt(2.0)
    elif nonlinearity == "leaky_relu":
        gain = np.sqrt(2.0 / (1 + a ** 2))
    else:
        gain = 1.0

    std = gain / np.sqrt(fan)
    tensor.data[:] = np.random.randn(*tensor.data.shape) * std
    return tensor


def xavier_uniform(tensor: Tensor, gain: float = 1.0) -> Tensor:
    """
    Xavier (Glorot) uniform initialization.
    Recommended for tanh/sigmoid networks.
    """
    fan_in, fan_out = _calculate_fan(tensor)
    std = gain * np.sqrt(2.0 / (fan_in + fan_out))
    bound = np.sqrt(3.0) * std
    tensor.data[:] = np.random.uniform(-bound, bound, tensor.data.shape)
    return tensor


def xavier_normal(tensor: Tensor, gain: float = 1.0) -> Tensor:
    """Xavier (Glorot) normal initialization."""
    fan_in, fan_out = _calculate_fan(tensor)
    std = gain * np.sqrt(2.0 / (fan_in + fan_out))
    tensor.data[:] = np.random.randn(*tensor.data.shape) * std
    return tensor


def orthogonal(tensor: Tensor, gain: float = 1.0) -> Tensor:
    """
    Orthogonal initialization via SVD.
    Useful for RNNs to prevent vanishing/exploding gradients at init.
    """
    shape = tensor.data.shape
    flat_shape = (shape[0], int(np.prod(shape[1:])))
    a = np.random.randn(*flat_shape)
    u, _, vt = np.linalg.svd(a, full_matrices=False)
    # Pick the matrix with the correct shape
    q = u if u.shape == flat_shape else vt
    q = q.reshape(shape)
    tensor.data[:] = gain * q
    return tensor


def normal(tensor: Tensor, mean: float = 0.0, std: float = 1.0) -> Tensor:
    """Fill tensor with samples from N(mean, std)."""
    tensor.data[:] = np.random.randn(*tensor.data.shape) * std + mean
    return tensor


def uniform(tensor: Tensor, a: float = 0.0, b: float = 1.0) -> Tensor:
    """Fill tensor with samples from U(a, b)."""
    tensor.data[:] = np.random.uniform(a, b, tensor.data.shape)
    return tensor


def zeros(tensor: Tensor) -> Tensor:
    """Fill tensor with zeros."""
    tensor.data[:] = 0.0
    return tensor


def ones(tensor: Tensor) -> Tensor:
    """Fill tensor with ones."""
    tensor.data[:] = 1.0
    return tensor


def constant(tensor: Tensor, val: float) -> Tensor:
    """Fill tensor with a constant value."""
    tensor.data[:] = val
    return tensor


def eye(tensor: Tensor) -> Tensor:
    """Fill square tensor with identity matrix."""
    assert tensor.data.ndim == 2 and tensor.data.shape[0] == tensor.data.shape[1]
    tensor.data[:] = np.eye(tensor.data.shape[0])
    return tensor
