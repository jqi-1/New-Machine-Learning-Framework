"""
numl/nn/layers.py
-----------------
Core neural network layers:
  - Dense (fully connected)
  - Conv2d (using im2col)
  - MaxPool2d
  - BatchNorm1d, BatchNorm2d
  - Dropout
  - Flatten
  - Embedding
"""

import numpy as np
from numl.core.tensor import Tensor
from numl.core.ops import _op, _accum
from numl.nn.module import Module


# ---------------------------------------------------------------------------
# Dense (Linear / Fully Connected)
# ---------------------------------------------------------------------------

class Dense(Module):
    """
    Fully connected layer: y = x @ W + b

    Parameters
    ----------
    in_features  : int
    out_features : int
    bias         : bool, default True
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias

        # Kaiming uniform initialization
        w_data = np.random.randn(in_features, out_features).astype(np.float64)
        bound = np.sqrt(1.0 / in_features)
        w_data = np.random.uniform(-bound, bound, (in_features, out_features))
        self.weight = Tensor(w_data, requires_grad=True)

        if bias:
            self.bias = Tensor(np.zeros(out_features), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        out = x @ self.weight
        if self.use_bias:
            out = out + self.bias
        return out

    def get_config(self):
        return {
            "in_features": self.in_features,
            "out_features": self.out_features,
            "bias": self.use_bias,
        }

    def __repr__(self):
        return f"Dense({self.in_features}, {self.out_features}, bias={self.use_bias})"


# ---------------------------------------------------------------------------
# Conv2d (im2col based)
# ---------------------------------------------------------------------------

def _im2col(x: np.ndarray, kH: int, kW: int, stride: int, pad: int) -> np.ndarray:
    """
    Convert image patches to columns for efficient convolution.

    Parameters
    ----------
    x      : (N, C, H, W)
    Returns : (N, C*kH*kW, H_out*W_out)
    """
    N, C, H, W = x.shape
    H_out = (H + 2 * pad - kH) // stride + 1
    W_out = (W + 2 * pad - kW) // stride + 1

    if pad > 0:
        x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
    else:
        x_pad = x

    col = np.zeros((N, C, kH, kW, H_out, W_out), dtype=x.dtype)
    for h in range(kH):
        h_max = h + stride * H_out
        for w in range(kW):
            w_max = w + stride * W_out
            col[:, :, h, w, :, :] = x_pad[:, :, h:h_max:stride, w:w_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3)  # (N, H_out, W_out, C, kH, kW)
    col = col.reshape(N, H_out * W_out, C * kH * kW)
    return col


def _col2im(col: np.ndarray, x_shape: tuple, kH: int, kW: int,
            stride: int, pad: int) -> np.ndarray:
    """
    Inverse of im2col: accumulate gradients back into (N, C, H, W).

    col : (N, H_out*W_out, C*kH*kW)
    """
    N, C, H, W = x_shape
    H_out = (H + 2 * pad - kH) // stride + 1
    W_out = (W + 2 * pad - kW) // stride + 1

    col_reshaped = col.reshape(N, H_out, W_out, C, kH, kW)
    col_transposed = col_reshaped.transpose(0, 3, 4, 5, 1, 2)  # (N, C, kH, kW, H_out, W_out)

    H_pad = H + 2 * pad
    W_pad = W + 2 * pad
    x_pad = np.zeros((N, C, H_pad, W_pad), dtype=col.dtype)

    for h in range(kH):
        h_max = h + stride * H_out
        for w in range(kW):
            w_max = w + stride * W_out
            x_pad[:, :, h:h_max:stride, w:w_max:stride] += col_transposed[:, :, h, w, :, :]

    if pad > 0:
        return x_pad[:, :, pad:-pad, pad:-pad]
    return x_pad


class Conv2d(Module):
    """
    2D convolutional layer using im2col transformation.

    Parameters
    ----------
    in_channels  : int
    out_channels : int
    kernel_size  : int or (int, int)
    stride       : int, default 1
    padding      : int, default 0
    bias         : bool, default True
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size,
                 stride: int = 1, padding: int = 0, bias: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if isinstance(kernel_size, int):
            self.kH = self.kW = kernel_size
        else:
            self.kH, self.kW = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_bias = bias

        # Kaiming uniform init
        fan_in = in_channels * self.kH * self.kW
        bound = np.sqrt(1.0 / fan_in)
        w_data = np.random.uniform(-bound, bound,
                                   (out_channels, in_channels, self.kH, self.kW))
        self.weight = Tensor(w_data, requires_grad=True)

        if bias:
            self.bias = Tensor(np.zeros(out_channels), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        N, C, H, W = x.data.shape
        kH, kW = self.kH, self.kW
        stride, pad = self.stride, self.padding

        H_out = (H + 2 * pad - kH) // stride + 1
        W_out = (W + 2 * pad - kW) // stride + 1

        # im2col: (N, H_out*W_out, C*kH*kW)
        col = _im2col(x.data, kH, kW, stride, pad)
        # weight: (out_channels, C*kH*kW)
        w_flat = self.weight.data.reshape(self.out_channels, -1)
        # out: (N, H_out*W_out, out_channels)
        out_col = col @ w_flat.T  # (N, H_out*W_out, out_channels)
        # Reshape to (N, out_channels, H_out, W_out)
        out_data = out_col.reshape(N, H_out, W_out, self.out_channels)
        out_data = out_data.transpose(0, 3, 1, 2)

        rg = x.requires_grad or self.weight.requires_grad
        x_data = x.data.copy()
        col_copy = col.copy()
        w_copy = w_flat.copy()
        x_shape = x.data.shape

        def bwd(g):
            # g: (N, out_channels, H_out, W_out)
            dout = g.transpose(0, 2, 3, 1)  # (N, H_out, W_out, out_channels)
            dout_flat = dout.reshape(N, H_out * W_out, self.out_channels)

            if self.weight.requires_grad:
                # dW = sum over N and spatial: col.T @ dout_flat
                # col: (N, H_out*W_out, C*kH*kW)
                dw_flat = np.sum(
                    np.matmul(col_copy.transpose(0, 2, 1), dout_flat), axis=0
                ).T  # (out_channels, C*kH*kW)
                _accum(self.weight, dw_flat.reshape(self.weight.data.shape))

            if self.use_bias and self.bias.requires_grad:
                db = g.sum(axis=(0, 2, 3))  # (out_channels,)
                _accum(self.bias, db)

            if x.requires_grad:
                # dcol = dout_flat @ w_flat: (N, H_out*W_out, C*kH*kW)
                dcol = dout_flat @ w_copy
                dx = _col2im(dcol, x_shape, kH, kW, stride, pad)
                _accum(x, dx)

        prev = (x, self.weight) + ((self.bias,) if self.use_bias else ())
        return _op(out_data, rg, prev, "conv2d", bwd if rg else None)

    def get_config(self):
        return {
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "kernel_size": (self.kH, self.kW),
            "stride": self.stride,
            "padding": self.padding,
            "bias": self.use_bias,
        }

    def __repr__(self):
        return (f"Conv2d({self.in_channels}, {self.out_channels}, "
                f"kernel_size=({self.kH},{self.kW}), stride={self.stride}, "
                f"padding={self.padding})")


# ---------------------------------------------------------------------------
# MaxPool2d
# ---------------------------------------------------------------------------

class MaxPool2d(Module):
    """2D max pooling."""

    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        if isinstance(kernel_size, int):
            self.kH = self.kW = kernel_size
        else:
            self.kH, self.kW = kernel_size
        self.stride = stride if stride is not None else self.kH
        self.padding = padding

    def forward(self, x: Tensor) -> Tensor:
        N, C, H, W = x.data.shape
        kH, kW = self.kH, self.kW
        s, pad = self.stride, self.padding

        H_out = (H + 2 * pad - kH) // s + 1
        W_out = (W + 2 * pad - kW) // s + 1

        if pad > 0:
            xp = np.pad(x.data, ((0,0),(0,0),(pad,pad),(pad,pad)), mode='constant',
                        constant_values=-np.inf)
        else:
            xp = x.data

        out = np.zeros((N, C, H_out, W_out), dtype=x.data.dtype)
        mask = np.zeros_like(xp, dtype=bool)

        for h in range(H_out):
            for w in range(W_out):
                patch = xp[:, :, h*s:h*s+kH, w*s:w*s+kW]
                max_val = patch.max(axis=(2, 3), keepdims=True)
                out[:, :, h, w] = max_val[:, :, 0, 0]
                m = (patch == max_val)
                mask[:, :, h*s:h*s+kH, w*s:w*s+kW] |= m

        rg = x.requires_grad
        xp_copy = xp.copy()

        def bwd(g):
            if not x.requires_grad:
                return
            dx_pad = np.zeros_like(xp_copy)
            for h in range(H_out):
                for w in range(W_out):
                    patch = xp_copy[:, :, h*s:h*s+kH, w*s:w*s+kW]
                    max_val = patch.max(axis=(2, 3), keepdims=True)
                    m = (patch == max_val).astype(float)
                    m /= m.sum(axis=(2, 3), keepdims=True)  # normalize ties
                    dx_pad[:, :, h*s:h*s+kH, w*s:w*s+kW] += m * g[:, :, h:h+1, w:w+1]
            if pad > 0:
                dx = dx_pad[:, :, pad:-pad, pad:-pad]
            else:
                dx = dx_pad
            _accum(x, dx)

        return _op(out, rg, (x,), "maxpool2d", bwd if rg else None)

    def __repr__(self):
        return f"MaxPool2d(kernel_size=({self.kH},{self.kW}), stride={self.stride})"


# ---------------------------------------------------------------------------
# BatchNorm1d
# ---------------------------------------------------------------------------

class BatchNorm1d(Module):
    """
    Batch normalization for 2D inputs (N, features).

    During training: normalizes over the batch dimension.
    During eval: uses running mean/var.
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        self.gamma = Tensor(np.ones(num_features), requires_grad=True)
        self.beta = Tensor(np.zeros(num_features), requires_grad=True)

        # Running stats — not Tensors, not differentiated
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

    def forward(self, x: Tensor) -> Tensor:
        if self._training:
            mean = x.data.mean(axis=0)
            var = x.data.var(axis=0)
            # Update running stats (EMA)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var

        inv_std = 1.0 / np.sqrt(var + self.eps)
        x_hat_data = (x.data - mean) * inv_std
        out_data = self.gamma.data * x_hat_data + self.beta.data

        rg = x.requires_grad or self.gamma.requires_grad or self.beta.requires_grad
        x_data = x.data.copy()
        gamma_data = self.gamma.data.copy()
        N = x.data.shape[0]
        mean_c = mean.copy()
        inv_std_c = inv_std.copy()
        x_hat_c = x_hat_data.copy()
        training = self._training

        def bwd(g):
            if self.gamma.requires_grad:
                _accum(self.gamma, (g * x_hat_c).sum(axis=0))
            if self.beta.requires_grad:
                _accum(self.beta, g.sum(axis=0))

            if x.requires_grad:
                if training:
                    # Exact BN backward
                    dxhat = g * gamma_data
                    dx = (1.0 / N) * inv_std_c * (
                        N * dxhat
                        - dxhat.sum(axis=0)
                        - x_hat_c * (dxhat * x_hat_c).sum(axis=0)
                    )
                else:
                    dx = g * gamma_data * inv_std_c
                _accum(x, dx)

        return _op(out_data, rg, (x, self.gamma, self.beta), "batchnorm1d", bwd if rg else None)

    def get_config(self):
        return {
            "num_features": self.num_features,
            "eps": self.eps,
            "momentum": self.momentum,
        }

    def __repr__(self):
        return f"BatchNorm1d({self.num_features}, eps={self.eps}, momentum={self.momentum})"


# ---------------------------------------------------------------------------
# BatchNorm2d
# ---------------------------------------------------------------------------

class BatchNorm2d(Module):
    """Batch normalization for 4D inputs (N, C, H, W)."""

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        self.gamma = Tensor(np.ones(num_features), requires_grad=True)
        self.beta = Tensor(np.zeros(num_features), requires_grad=True)
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

    def forward(self, x: Tensor) -> Tensor:
        # x: (N, C, H, W)
        N, C, H, W = x.data.shape
        x_flat = x.data.transpose(1, 0, 2, 3).reshape(C, -1)  # (C, N*H*W)

        if self._training:
            mean = x_flat.mean(axis=1)    # (C,)
            var = x_flat.var(axis=1)      # (C,)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var

        inv_std = 1.0 / np.sqrt(var + self.eps)  # (C,)
        x_hat = (x.data - mean[None, :, None, None]) * inv_std[None, :, None, None]
        out_data = self.gamma.data[None, :, None, None] * x_hat + self.beta.data[None, :, None, None]

        rg = x.requires_grad or self.gamma.requires_grad or self.beta.requires_grad
        n_total = N * H * W
        gamma_data = self.gamma.data.copy()
        inv_std_c = inv_std.copy()
        x_hat_c = x_hat.copy()
        training = self._training

        def bwd(g):
            if self.gamma.requires_grad:
                _accum(self.gamma, (g * x_hat_c).sum(axis=(0, 2, 3)))
            if self.beta.requires_grad:
                _accum(self.beta, g.sum(axis=(0, 2, 3)))
            if x.requires_grad:
                if training:
                    dxhat = g * gamma_data[None, :, None, None]
                    s = inv_std_c[None, :, None, None]
                    dx = (1.0 / n_total) * s * (
                        n_total * dxhat
                        - dxhat.sum(axis=(0, 2, 3), keepdims=True)
                        - x_hat_c * (dxhat * x_hat_c).sum(axis=(0, 2, 3), keepdims=True)
                    )
                else:
                    dx = g * gamma_data[None, :, None, None] * inv_std_c[None, :, None, None]
                _accum(x, dx)

        return _op(out_data, rg, (x, self.gamma, self.beta), "batchnorm2d", bwd if rg else None)

    def get_config(self):
        return {
            "num_features": self.num_features,
            "eps": self.eps,
            "momentum": self.momentum,
        }

    def __repr__(self):
        return f"BatchNorm2d({self.num_features})"


# ---------------------------------------------------------------------------
# Dropout
# ---------------------------------------------------------------------------

class Dropout(Module):
    """
    Inverted dropout: divides by (1-p) during training so no scaling at test time.

    Parameters
    ----------
    p : float, probability of zeroing an element (default 0.5)
    """

    def __init__(self, p: float = 0.5):
        super().__init__()
        assert 0.0 <= p < 1.0, "Dropout probability must be in [0, 1)"
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if not self._training or self.p == 0.0:
            return x

        mask = (np.random.rand(*x.data.shape) > self.p).astype(x.data.dtype)
        mask /= (1.0 - self.p)
        out_data = x.data * mask
        rg = x.requires_grad

        def bwd(g):
            _accum(x, g * mask)

        return _op(out_data, rg, (x,), "dropout", bwd if rg else None)

    def get_config(self):
        return {"p": self.p}

    def __repr__(self):
        return f"Dropout(p={self.p})"


# ---------------------------------------------------------------------------
# Flatten
# ---------------------------------------------------------------------------

class Flatten(Module):
    """Flatten all dimensions starting from start_dim."""

    def __init__(self, start_dim: int = 1):
        super().__init__()
        self.start_dim = start_dim

    def forward(self, x: Tensor) -> Tensor:
        shape = x.data.shape
        new_shape = shape[:self.start_dim] + (-1,)
        return x.reshape(new_shape)

    def __repr__(self):
        return f"Flatten(start_dim={self.start_dim})"


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

class Embedding(Module):
    """
    Simple lookup table (embedding matrix).

    Parameters
    ----------
    num_embeddings : int  — vocabulary size
    embedding_dim  : int  — embedding dimension
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        w_data = np.random.randn(num_embeddings, embedding_dim) * 0.01
        self.weight = Tensor(w_data, requires_grad=True)

    def forward(self, indices) -> Tensor:
        """
        Parameters
        ----------
        indices : array-like of int, shape (*)
        Returns : Tensor of shape (*, embedding_dim)
        """
        if isinstance(indices, Tensor):
            indices = indices.data.astype(int)
        else:
            indices = np.array(indices, dtype=int)

        out_data = self.weight.data[indices]
        rg = self.weight.requires_grad
        idx_copy = indices.copy()
        w_shape = self.weight.data.shape

        def bwd(g):
            if self.weight.grad is None:
                self.weight.grad = np.zeros(w_shape)
            np.add.at(self.weight.grad, idx_copy, g)

        return _op(out_data, rg, (self.weight,), "embedding", bwd if rg else None)

    def get_config(self):
        return {
            "num_embeddings": self.num_embeddings,
            "embedding_dim": self.embedding_dim,
        }

    def __repr__(self):
        return f"Embedding({self.num_embeddings}, {self.embedding_dim})"
