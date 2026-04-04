"""
numl/nn/recurrent.py
--------------------
Recurrent neural network layers:
  - RNNCell, RNN
  - LSTMCell, LSTM
  - GRUCell, GRU

Gradients flow through time via BPTT, which works automatically because each
timestep creates new Tensors in the computation graph.
"""

import numpy as np
from numl.core.tensor import Tensor
from numl.nn.module import Module
from numl.nn.activations import tanh, sigmoid


# ---------------------------------------------------------------------------
# RNN
# ---------------------------------------------------------------------------

class RNNCell(Module):
    """
    Single-step RNN cell: h_t = tanh(x @ W_ih + h_{t-1} @ W_hh + b)

    Parameters
    ----------
    input_size    : int
    hidden_size   : int
    nonlinearity  : 'tanh' or 'relu'
    """

    def __init__(self, input_size: int, hidden_size: int, nonlinearity: str = 'tanh'):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.nonlinearity = nonlinearity

        k = np.sqrt(1.0 / hidden_size)
        self.W_ih = Tensor(np.random.uniform(-k, k, (input_size, hidden_size)),
                           requires_grad=True)
        self.W_hh = Tensor(np.random.uniform(-k, k, (hidden_size, hidden_size)),
                           requires_grad=True)
        self.b = Tensor(np.zeros(hidden_size), requires_grad=True)

    def forward(self, x: Tensor, h: Tensor) -> Tensor:
        pre = x @ self.W_ih + h @ self.W_hh + self.b
        if self.nonlinearity == 'tanh':
            return tanh(pre)
        else:
            from numl.nn.activations import relu
            return relu(pre)


class RNN(Module):
    """
    Multi-layer RNN.

    Parameters
    ----------
    input_size   : int
    hidden_size  : int
    num_layers   : int, default 1
    nonlinearity : 'tanh' or 'relu'
    batch_first  : bool — if True, input is (batch, seq, features)
    dropout      : float — applied between layers (not on last layer)
    bidirectional: bool
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1,
                 nonlinearity: str = 'tanh', batch_first: bool = False,
                 dropout: float = 0.0, bidirectional: bool = False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.nonlinearity = nonlinearity
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        for layer in range(num_layers):
            for direction in range(self.num_directions):
                layer_input = (input_size if layer == 0
                               else hidden_size * self.num_directions)
                cell = RNNCell(layer_input, hidden_size, nonlinearity)
                name = f"cell_{layer}_{direction}"
                object.__setattr__(self, name, cell)
                self._modules[name] = cell

    def forward(self, x: Tensor, h_0: Tensor | None = None) -> tuple:
        """
        Returns (output, h_n)
          output : (seq_len, batch, num_directions * hidden_size)
          h_n    : (num_layers * num_directions, batch, hidden_size)
        """
        if self.batch_first:
            # x: (batch, seq, input) -> (seq, batch, input)
            x = x.transpose((1, 0, 2))

        seq_len, batch, _ = x.data.shape

        if h_0 is None:
            h_0_data = np.zeros((self.num_layers * self.num_directions, batch,
                                 self.hidden_size))
            h_0 = Tensor(h_0_data, requires_grad=False)

        # Split h_0 into per-layer, per-direction tensors
        h = []
        for i in range(self.num_layers * self.num_directions):
            h.append(Tensor(h_0.data[i], requires_grad=h_0.requires_grad))

        h_n = []
        for layer in range(self.num_layers):
            layer_out = []
            for direction in range(self.num_directions):
                cell = self._modules[f"cell_{layer}_{direction}"]
                idx = layer * self.num_directions + direction
                h_t = h[idx]

                seq = range(seq_len) if direction == 0 else range(seq_len - 1, -1, -1)
                outputs = []
                for t in seq:
                    x_t = Tensor(x.data[t], requires_grad=x.requires_grad)
                    h_t = cell(x_t, h_t)
                    outputs.append(h_t)

                if direction == 1:
                    outputs = outputs[::-1]

                layer_out.append(outputs)
                h_n.append(h_t)

            # Combine directions for this layer's output
            if self.num_directions == 2:
                combined = [Tensor(np.concatenate([layer_out[0][t].data,
                                                   layer_out[1][t].data], axis=-1))
                            for t in range(seq_len)]
            else:
                combined = layer_out[0]

            # Apply dropout between layers (not on last)
            if self.dropout > 0.0 and layer < self.num_layers - 1 and self._training:
                from numl.nn.layers import Dropout
                drop = Dropout(self.dropout)
                combined = [drop(c) for c in combined]

            # Stack for next layer input
            x = Tensor(np.stack([c.data for c in combined], axis=0),
                       requires_grad=x.requires_grad)

        # output: stack along seq dim
        from numl.core.ops import stack
        output = stack(combined, axis=0)

        # h_n: stack
        h_n_tensor = Tensor(np.stack([h.data for h in h_n], axis=0))
        return output, h_n_tensor


# ---------------------------------------------------------------------------
# LSTM
# ---------------------------------------------------------------------------

class LSTMCell(Module):
    """
    Single-step LSTM cell.

    Uses combined gate matrix for efficiency:
      [i, f, g, o] = x @ W_ih + h @ W_hh + b
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        k = np.sqrt(1.0 / hidden_size)
        self.W_ih = Tensor(np.random.uniform(-k, k, (input_size, 4 * hidden_size)),
                           requires_grad=True)
        self.W_hh = Tensor(np.random.uniform(-k, k, (hidden_size, 4 * hidden_size)),
                           requires_grad=True)
        self.b = Tensor(np.zeros(4 * hidden_size), requires_grad=True)

    def forward(self, x: Tensor, h: Tensor, c: Tensor) -> tuple:
        """Returns (h_t, c_t)."""
        gates = x @ self.W_ih + h @ self.W_hh + self.b  # (batch, 4*H)
        hs = self.hidden_size

        i_gate = sigmoid(gates[:, :hs] if gates.data.ndim == 2
                         else Tensor(gates.data[:hs]))
        f_gate = sigmoid(gates[:, hs:2*hs] if gates.data.ndim == 2
                         else Tensor(gates.data[hs:2*hs]))
        g_gate = tanh(gates[:, 2*hs:3*hs] if gates.data.ndim == 2
                      else Tensor(gates.data[2*hs:3*hs]))
        o_gate = sigmoid(gates[:, 3*hs:] if gates.data.ndim == 2
                         else Tensor(gates.data[3*hs:]))

        c_t = f_gate * c + i_gate * g_gate
        h_t = o_gate * tanh(c_t)
        return h_t, c_t


class LSTM(Module):
    """
    Multi-layer LSTM.

    Parameters
    ----------
    input_size   : int
    hidden_size  : int
    num_layers   : int, default 1
    batch_first  : bool
    dropout      : float
    bidirectional: bool
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1,
                 batch_first: bool = False, dropout: float = 0.0,
                 bidirectional: bool = False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        for layer in range(num_layers):
            for direction in range(self.num_directions):
                layer_input = (input_size if layer == 0
                               else hidden_size * self.num_directions)
                cell = LSTMCell(layer_input, hidden_size)
                name = f"cell_{layer}_{direction}"
                object.__setattr__(self, name, cell)
                self._modules[name] = cell

    def forward(self, x: Tensor,
                hx: tuple | None = None) -> tuple:
        """
        Returns (output, (h_n, c_n)).
        """
        if self.batch_first:
            x = x.transpose((1, 0, 2))

        seq_len, batch, _ = x.data.shape
        D = self.num_directions

        if hx is None:
            zeros = np.zeros((self.num_layers * D, batch, self.hidden_size))
            h_0 = Tensor(zeros)
            c_0 = Tensor(zeros.copy())
        else:
            h_0, c_0 = hx

        h = [Tensor(h_0.data[i]) for i in range(self.num_layers * D)]
        c = [Tensor(c_0.data[i]) for i in range(self.num_layers * D)]

        h_n, c_n = [], []
        combined = None

        for layer in range(self.num_layers):
            layer_out_fwd = []
            layer_out_bwd = []

            for direction in range(D):
                cell = self._modules[f"cell_{layer}_{direction}"]
                idx = layer * D + direction
                h_t = h[idx]
                c_t = c[idx]

                seq = range(seq_len) if direction == 0 else range(seq_len - 1, -1, -1)
                outputs = []
                for t in seq:
                    x_t = Tensor(x.data[t], requires_grad=x.requires_grad)
                    h_t, c_t = cell(x_t, h_t, c_t)
                    outputs.append(h_t)

                h_n.append(h_t)
                c_n.append(c_t)

                if direction == 0:
                    layer_out_fwd = outputs
                else:
                    layer_out_bwd = outputs[::-1]

            if D == 2:
                combined = [Tensor(np.concatenate([layer_out_fwd[t].data,
                                                   layer_out_bwd[t].data], axis=-1))
                            for t in range(seq_len)]
            else:
                combined = layer_out_fwd

            if self.dropout > 0.0 and layer < self.num_layers - 1 and self._training:
                from numl.nn.layers import Dropout
                drop = Dropout(self.dropout)
                combined = [drop(c_item) for c_item in combined]

            x = Tensor(np.stack([c_item.data for c_item in combined], axis=0))

        from numl.core.ops import stack
        output = stack(combined, axis=0)
        h_n_t = Tensor(np.stack([t.data for t in h_n], axis=0))
        c_n_t = Tensor(np.stack([t.data for t in c_n], axis=0))
        return output, (h_n_t, c_n_t)


# ---------------------------------------------------------------------------
# GRU
# ---------------------------------------------------------------------------

class GRUCell(Module):
    """
    Single-step GRU cell.
      r = sigmoid(x@W_ir + h@W_hr + b_r)
      z = sigmoid(x@W_iz + h@W_hz + b_z)
      n = tanh(x@W_in + r*(h@W_hn + b_hn) + b_in)
      h_t = (1-z)*n + z*h
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        k = np.sqrt(1.0 / hidden_size)
        # Combined input->hidden for r,z,n gates
        self.W_ih = Tensor(np.random.uniform(-k, k, (input_size, 3 * hidden_size)),
                           requires_grad=True)
        self.W_hh = Tensor(np.random.uniform(-k, k, (hidden_size, 3 * hidden_size)),
                           requires_grad=True)
        self.b_ih = Tensor(np.zeros(3 * hidden_size), requires_grad=True)
        self.b_hh = Tensor(np.zeros(3 * hidden_size), requires_grad=True)

    def forward(self, x: Tensor, h: Tensor) -> Tensor:
        hs = self.hidden_size

        gates_x = x @ self.W_ih + self.b_ih
        gates_h = h @ self.W_hh + self.b_hh

        def _col(t, start, end):
            if t.data.ndim == 2:
                return t[:, start:end]
            return Tensor(t.data[start:end])

        r = sigmoid(_col(gates_x, 0, hs) + _col(gates_h, 0, hs))
        z = sigmoid(_col(gates_x, hs, 2*hs) + _col(gates_h, hs, 2*hs))
        n = tanh(_col(gates_x, 2*hs, 3*hs) + r * _col(gates_h, 2*hs, 3*hs))

        h_t = (Tensor(np.ones_like(z.data)) - z) * n + z * h
        return h_t


class GRU(Module):
    """Multi-layer GRU."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1,
                 batch_first: bool = False, dropout: float = 0.0,
                 bidirectional: bool = False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        for layer in range(num_layers):
            for direction in range(self.num_directions):
                layer_input = (input_size if layer == 0
                               else hidden_size * self.num_directions)
                cell = GRUCell(layer_input, hidden_size)
                name = f"cell_{layer}_{direction}"
                object.__setattr__(self, name, cell)
                self._modules[name] = cell

    def forward(self, x: Tensor, h_0: Tensor | None = None) -> tuple:
        if self.batch_first:
            x = x.transpose((1, 0, 2))

        seq_len, batch, _ = x.data.shape
        D = self.num_directions

        if h_0 is None:
            h_0 = Tensor(np.zeros((self.num_layers * D, batch, self.hidden_size)))

        h = [Tensor(h_0.data[i]) for i in range(self.num_layers * D)]
        h_n = []
        combined = None

        for layer in range(self.num_layers):
            layer_out = []
            for direction in range(D):
                cell = self._modules[f"cell_{layer}_{direction}"]
                idx = layer * D + direction
                h_t = h[idx]
                seq = range(seq_len) if direction == 0 else range(seq_len - 1, -1, -1)
                outputs = []
                for t in seq:
                    x_t = Tensor(x.data[t], requires_grad=x.requires_grad)
                    h_t = cell(x_t, h_t)
                    outputs.append(h_t)
                if direction == 1:
                    outputs = outputs[::-1]
                layer_out.append(outputs)
                h_n.append(h_t)

            if D == 2:
                combined = [Tensor(np.concatenate([layer_out[0][t].data,
                                                   layer_out[1][t].data], axis=-1))
                            for t in range(seq_len)]
            else:
                combined = layer_out[0]

            if self.dropout > 0.0 and layer < self.num_layers - 1 and self._training:
                from numl.nn.layers import Dropout
                drop = Dropout(self.dropout)
                combined = [drop(c) for c in combined]

            x = Tensor(np.stack([c.data for c in combined], axis=0))

        from numl.core.ops import stack
        output = stack(combined, axis=0)
        h_n_t = Tensor(np.stack([t.data for t in h_n], axis=0))
        return output, h_n_t
