"""Tests for recurrent layers: RNN, LSTM, GRU."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import unittest
import numpy as np
from numl.core.tensor import Tensor


class TestRNNCell(unittest.TestCase):

    def test_output_shape(self):
        from numl.nn.recurrent import RNNCell
        cell = RNNCell(input_size=4, hidden_size=8)
        x = Tensor(np.random.randn(3, 4))
        h = Tensor(np.zeros((3, 8)))
        h_new = cell(x, h)
        self.assertEqual(h_new.shape, (3, 8))

    def test_tanh_activation(self):
        from numl.nn.recurrent import RNNCell
        cell = RNNCell(input_size=2, hidden_size=4)
        x = Tensor(np.random.randn(5, 2))
        h = Tensor(np.zeros((5, 4)))
        h_new = cell(x, h)
        # tanh output range is (-1, 1)
        self.assertTrue(np.all(h_new.data >= -1))
        self.assertTrue(np.all(h_new.data <= 1))

    def test_grad_flows(self):
        from numl.nn.recurrent import RNNCell
        cell = RNNCell(input_size=3, hidden_size=4)
        x = Tensor(np.random.randn(2, 3), requires_grad=True)
        h = Tensor(np.zeros((2, 4)), requires_grad=True)
        h_new = cell(x, h)
        loss = h_new.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, (2, 3))


class TestRNN(unittest.TestCase):

    def test_output_shape(self):
        from numl.nn.recurrent import RNN
        rnn = RNN(input_size=4, hidden_size=8, num_layers=1)
        # x: (seq_len, batch, input_size)
        x = Tensor(np.random.randn(5, 3, 4))
        output, h_n = rnn(x)
        self.assertEqual(output.shape, (5, 3, 8))
        self.assertEqual(h_n.shape, (1, 3, 8))

    def test_multilayer_shape(self):
        from numl.nn.recurrent import RNN
        rnn = RNN(input_size=4, hidden_size=8, num_layers=2)
        x = Tensor(np.random.randn(6, 2, 4))
        output, h_n = rnn(x)
        self.assertEqual(output.shape, (6, 2, 8))
        self.assertEqual(h_n.shape, (2, 2, 8))

    def test_grad_flows_through_time(self):
        from numl.nn.recurrent import RNN
        rnn = RNN(input_size=3, hidden_size=5, num_layers=1)
        x = Tensor(np.random.randn(4, 2, 3), requires_grad=True)
        output, h_n = rnn(x)
        output.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, (4, 2, 3))


class TestLSTMCell(unittest.TestCase):

    def test_output_shapes(self):
        from numl.nn.recurrent import LSTMCell
        cell = LSTMCell(input_size=4, hidden_size=6)
        x = Tensor(np.random.randn(3, 4))
        h = Tensor(np.zeros((3, 6)))
        c = Tensor(np.zeros((3, 6)))
        h_new, c_new = cell(x, (h, c))
        self.assertEqual(h_new.shape, (3, 6))
        self.assertEqual(c_new.shape, (3, 6))

    def test_gate_ranges(self):
        from numl.nn.recurrent import LSTMCell
        cell = LSTMCell(input_size=2, hidden_size=4)
        x = Tensor(np.random.randn(5, 2))
        h = Tensor(np.zeros((5, 4)))
        c = Tensor(np.zeros((5, 4)))
        h_new, c_new = cell(x, (h, c))
        # h = o * tanh(c), so h in (-1, 1)
        self.assertTrue(np.all(h_new.data >= -1))
        self.assertTrue(np.all(h_new.data <= 1))

    def test_grad_flows(self):
        from numl.nn.recurrent import LSTMCell
        cell = LSTMCell(input_size=3, hidden_size=5)
        x = Tensor(np.random.randn(2, 3), requires_grad=True)
        h = Tensor(np.zeros((2, 5)))
        c = Tensor(np.zeros((2, 5)))
        h_new, c_new = cell(x, (h, c))
        (h_new + c_new).sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, (2, 3))


class TestLSTM(unittest.TestCase):

    def test_output_shape(self):
        from numl.nn.recurrent import LSTM
        lstm = LSTM(input_size=4, hidden_size=8, num_layers=1)
        x = Tensor(np.random.randn(5, 3, 4))
        output, (h_n, c_n) = lstm(x)
        self.assertEqual(output.shape, (5, 3, 8))
        self.assertEqual(h_n.shape, (1, 3, 8))
        self.assertEqual(c_n.shape, (1, 3, 8))

    def test_multilayer_shape(self):
        from numl.nn.recurrent import LSTM
        lstm = LSTM(input_size=4, hidden_size=8, num_layers=2)
        x = Tensor(np.random.randn(6, 2, 4))
        output, (h_n, c_n) = lstm(x)
        self.assertEqual(output.shape, (6, 2, 8))
        self.assertEqual(h_n.shape, (2, 2, 8))
        self.assertEqual(c_n.shape, (2, 2, 8))

    def test_grad_flows(self):
        from numl.nn.recurrent import LSTM
        lstm = LSTM(input_size=3, hidden_size=5, num_layers=1)
        x = Tensor(np.random.randn(4, 2, 3), requires_grad=True)
        output, (h_n, c_n) = lstm(x)
        output.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, (4, 2, 3))

    def test_initial_hidden_provided(self):
        from numl.nn.recurrent import LSTM
        lstm = LSTM(input_size=4, hidden_size=6, num_layers=1)
        x = Tensor(np.random.randn(3, 2, 4))
        h0 = Tensor(np.zeros((1, 2, 6)))
        c0 = Tensor(np.zeros((1, 2, 6)))
        output, (h_n, c_n) = lstm(x, (h0, c0))
        self.assertEqual(output.shape, (3, 2, 6))


class TestGRUCell(unittest.TestCase):

    def test_output_shape(self):
        from numl.nn.recurrent import GRUCell
        cell = GRUCell(input_size=4, hidden_size=6)
        x = Tensor(np.random.randn(3, 4))
        h = Tensor(np.zeros((3, 6)))
        h_new = cell(x, h)
        self.assertEqual(h_new.shape, (3, 6))

    def test_tanh_range(self):
        from numl.nn.recurrent import GRUCell
        cell = GRUCell(input_size=3, hidden_size=5)
        x = Tensor(np.random.randn(4, 3))
        h = Tensor(np.zeros((4, 5)))
        h_new = cell(x, h)
        self.assertTrue(np.all(h_new.data >= -1))
        self.assertTrue(np.all(h_new.data <= 1))

    def test_grad_flows(self):
        from numl.nn.recurrent import GRUCell
        cell = GRUCell(input_size=3, hidden_size=4)
        x = Tensor(np.random.randn(2, 3), requires_grad=True)
        h = Tensor(np.zeros((2, 4)))
        h_new = cell(x, h)
        h_new.sum().backward()
        self.assertIsNotNone(x.grad)


class TestGRU(unittest.TestCase):

    def test_output_shape(self):
        from numl.nn.recurrent import GRU
        gru = GRU(input_size=4, hidden_size=8, num_layers=1)
        x = Tensor(np.random.randn(5, 3, 4))
        output, h_n = gru(x)
        self.assertEqual(output.shape, (5, 3, 8))
        self.assertEqual(h_n.shape, (1, 3, 8))

    def test_multilayer_shape(self):
        from numl.nn.recurrent import GRU
        gru = GRU(input_size=4, hidden_size=8, num_layers=2)
        x = Tensor(np.random.randn(6, 2, 4))
        output, h_n = gru(x)
        self.assertEqual(output.shape, (6, 2, 8))
        self.assertEqual(h_n.shape, (2, 2, 8))

    def test_grad_flows(self):
        from numl.nn.recurrent import GRU
        gru = GRU(input_size=3, hidden_size=5, num_layers=1)
        x = Tensor(np.random.randn(4, 2, 3), requires_grad=True)
        output, h_n = gru(x)
        output.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, (4, 2, 3))

    def test_sequence_classification(self):
        """GRU should learn a simple sequence pattern."""
        from numl.nn.recurrent import GRU
        from numl.nn.layers import Dense
        from numl.nn.loss import BCEWithLogitsLoss
        from numl.optim.adam import Adam

        np.random.seed(42)
        # Task: classify if last token > 0
        def make_data(n=64, seq=5, d=2):
            X = np.random.randn(seq, n, d)
            y = (X[-1, :, 0] > 0).astype(np.float64).reshape(n, 1)
            return X, y

        X_data, y_data = make_data()
        gru = GRU(input_size=2, hidden_size=8)
        dense = Dense(8, 1)
        loss_fn = BCEWithLogitsLoss()
        params = list(gru.parameters()) + list(dense.parameters())
        opt = Adam(params, lr=0.01)

        for _ in range(50):
            opt.zero_grad()
            X = Tensor(X_data)
            out, h_n = gru(X)
            last = Tensor(out.data[-1])  # last timestep
            pred = dense(last)
            loss = loss_fn(pred, Tensor(y_data))
            loss.backward()
            opt.step()

        self.assertLess(loss.item(), 1.0)


if __name__ == '__main__':
    unittest.main()
