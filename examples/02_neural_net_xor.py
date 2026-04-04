"""
02_neural_net_xor.py
Solves the XOR problem using a 2-layer MLP with the numl framework.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from numl.core.tensor import Tensor
from numl.nn.layers import Dense
from numl.nn.activations import ReLU
from numl.nn.sequential import Sequential
from numl.nn.loss import BCEWithLogitsLoss
from numl.optim.adam import Adam

print("=" * 50)
print("Example 2: XOR with a 2-layer MLP")
print("=" * 50)

# XOR dataset
X = Tensor(np.array([[0.0, 0.0],
                     [0.0, 1.0],
                     [1.0, 0.0],
                     [1.0, 1.0]]))
y = Tensor(np.array([[0.0], [1.0], [1.0], [0.0]]))

# Build model
np.random.seed(0)
model = Sequential([
    Dense(2, 16),
    ReLU(),
    Dense(16, 1)
])

loss_fn = BCEWithLogitsLoss()
optimizer = Adam(model.parameters(), lr=0.05)

print("\nTraining...")
losses = []
for epoch in range(1, 501):
    optimizer.zero_grad()
    logits = model(X)
    loss = loss_fn(logits, y)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    if epoch % 100 == 0:
        print(f"  Epoch {epoch:4d}  |  Loss: {loss.item():.6f}")

print(f"\nFinal loss: {losses[-1]:.6f}  (target < 0.05)")

# Evaluate
import numl.core.ops as ops
probs = ops.sigmoid(model(X))
print("\nPredictions (probabilities):")
for i, (xi, pi) in enumerate(zip(X.data, probs.data)):
    label = 1 if pi[0] > 0.5 else 0
    print(f"  XOR({int(xi[0])}, {int(xi[1])}) = {label}  (prob = {pi[0]:.4f})")

assert losses[-1] < 0.05, "XOR did not converge!"
print("\nXOR solved successfully!")
