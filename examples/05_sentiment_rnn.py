"""
05_sentiment_rnn.py
Sentiment classification with an LSTM on synthetic sequence data.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from numl.core.tensor import Tensor
from numl.nn.recurrent import LSTM
from numl.nn.layers import Dense
from numl.nn.loss import BCEWithLogitsLoss
from numl.optim.adam import Adam
from numl.metrics.classification import accuracy_score

print("=" * 50)
print("Example 5: Sentiment Classification with LSTM")
print("=" * 50)

# Synthetic task: classify if mean of sequence > 0 (positive sentiment)
np.random.seed(42)
N_TRAIN, N_TEST = 400, 100
SEQ_LEN, INPUT_DIM = 10, 8
HIDDEN_DIM = 32

def make_dataset(n, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(SEQ_LEN, n, INPUT_DIM)
    # Label: majority of timesteps have positive mean
    means = X.mean(axis=2)  # (T, N)
    y = (means.mean(axis=0) > 0).astype(np.float64).reshape(n, 1)
    return X, y

X_train, y_train = make_dataset(N_TRAIN, seed=0)
X_test, y_test = make_dataset(N_TEST, seed=1)

# Model: LSTM + linear classifier on last hidden state
lstm = LSTM(input_size=INPUT_DIM, hidden_size=HIDDEN_DIM, num_layers=1)
classifier = Dense(HIDDEN_DIM, 1)
loss_fn = BCEWithLogitsLoss()
params = list(lstm.parameters()) + list(classifier.parameters())
optimizer = Adam(params, lr=1e-3)

print(f"\nTraining LSTM on {N_TRAIN} synthetic sequences (len={SEQ_LEN})...")

for epoch in range(1, 21):
    # Mini-batch training
    idx = np.random.permutation(N_TRAIN)
    batch_size = 64
    total_loss = 0.0
    n_batches = 0
    for start in range(0, N_TRAIN, batch_size):
        b_idx = idx[start:start + batch_size]
        X_b = Tensor(X_train[:, b_idx, :])
        y_b = Tensor(y_train[b_idx])

        optimizer.zero_grad()
        output, (h_n, c_n) = lstm(X_b)
        # Last hidden state: h_n shape (1, B, hidden)
        last_h = Tensor(h_n.data[0])  # (B, hidden)
        logits = classifier(last_h)
        loss = loss_fn(logits, y_b)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1

    if epoch % 5 == 0:
        # Evaluate
        X_t = Tensor(X_test)
        out, (h_n, _) = lstm(X_t)
        last_h = Tensor(h_n.data[0])
        logits = classifier(last_h)
        from numl.core.ops import sigmoid
        probs = sigmoid(logits).data.ravel()
        preds = (probs > 0.5).astype(int)
        acc = accuracy_score(y_test.ravel().astype(int), preds)
        print(f"  Epoch {epoch:3d}  |  Loss: {total_loss/n_batches:.4f}  |  Test acc: {acc:.4f}")

print("\nDone! LSTM learns to classify sequences.")
