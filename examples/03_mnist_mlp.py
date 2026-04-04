"""
03_mnist_mlp.py
Trains an MLP on synthetic MNIST-like data (random for portability).
For real MNIST, replace the data loading section.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from numl.core.tensor import Tensor
from numl.nn.layers import Dense, BatchNorm1d, Dropout
from numl.nn.activations import ReLU
from numl.nn.sequential import Sequential
from numl.nn.loss import CrossEntropyLoss
from numl.nn.utils import clip_grad_norm_
from numl.optim.adam import Adam
from numl.optim.schedulers import StepLR
from numl.data.dataset import ArrayDataset
from numl.data.dataloader import DataLoader
from numl.metrics.classification import accuracy_score

print("=" * 50)
print("Example 3: MLP on (Synthetic) MNIST-style Data")
print("=" * 50)

# Generate synthetic 28x28 grayscale data
np.random.seed(42)
N_TRAIN, N_TEST = 1000, 200
N_CLASSES = 10
INPUT_DIM = 784  # 28*28

X_train = np.random.randn(N_TRAIN, INPUT_DIM).astype(np.float64)
y_train = np.random.randint(0, N_CLASSES, N_TRAIN)
X_test = np.random.randn(N_TEST, INPUT_DIM).astype(np.float64)
y_test = np.random.randint(0, N_CLASSES, N_TEST)

# Normalize
X_train = X_train / X_train.std()
X_test = X_test / X_train.std()

# Build model
model = Sequential([
    Dense(INPUT_DIM, 256),
    BatchNorm1d(256),
    ReLU(),
    Dropout(p=0.3),
    Dense(256, 128),
    BatchNorm1d(128),
    ReLU(),
    Dropout(p=0.2),
    Dense(128, N_CLASSES),
])

loss_fn = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=1e-3)
scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

# DataLoader
train_ds = ArrayDataset(X_train, y_train)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

print(f"\nModel parameters: {sum(p.data.size for p in model.parameters()):,}")
print(f"Training on {N_TRAIN} synthetic samples...")

EPOCHS = 5
for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_loss = 0.0
    for X_b, y_b in train_loader:
        optimizer.zero_grad()
        logits = model(Tensor(X_b))
        loss = loss_fn(logits, Tensor(y_b.astype(np.int64)))
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_loss += loss.item()
    scheduler.step()

    # Eval
    model.eval()
    test_logits = model(Tensor(X_test)).data
    pred = np.argmax(test_logits, axis=1)
    acc = accuracy_score(y_test, pred)
    print(f"  Epoch {epoch}  |  Loss: {epoch_loss/len(train_loader):.4f}  |  Test acc: {acc:.4f}")

print("\nNote: Accuracy ~10% is expected on random data (random baseline).")
print("Replace synthetic data with real MNIST for meaningful results.")
