"""
04_mnist_cnn.py
Trains a CNN on synthetic MNIST-like data.
For real MNIST, replace the data loading section.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from numl.core.tensor import Tensor
from numl.nn.layers import Conv2d, Dense, BatchNorm1d, Flatten, MaxPool2d
from numl.nn.activations import ReLU
from numl.nn.sequential import Sequential
from numl.nn.loss import CrossEntropyLoss
from numl.optim.adam import Adam
from numl.data.dataset import ArrayDataset
from numl.data.dataloader import DataLoader
from numl.metrics.classification import accuracy_score

print("=" * 50)
print("Example 4: CNN on (Synthetic) MNIST-style Data")
print("=" * 50)

# Generate synthetic 28x28 grayscale images (N, C, H, W)
np.random.seed(7)
N_TRAIN, N_TEST = 500, 100
N_CLASSES = 10

X_train = np.random.randn(N_TRAIN, 1, 14, 14).astype(np.float64)  # smaller for speed
y_train = np.random.randint(0, N_CLASSES, N_TRAIN)
X_test = np.random.randn(N_TEST, 1, 14, 14).astype(np.float64)
y_test = np.random.randint(0, N_CLASSES, N_TEST)

# Build CNN
model = Sequential([
    Conv2d(1, 8, kernel_size=3, padding=1),    # (N, 8, 14, 14)
    ReLU(),
    MaxPool2d(kernel_size=2, stride=2),         # (N, 8, 7, 7)
    Conv2d(8, 16, kernel_size=3, padding=1),   # (N, 16, 7, 7)
    ReLU(),
    MaxPool2d(kernel_size=2, stride=2),         # (N, 16, 3, 3)
    Flatten(),                                  # (N, 16*3*3=144)
    Dense(144, 64),
    ReLU(),
    Dense(64, N_CLASSES),
])

loss_fn = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=1e-3)

train_ds = ArrayDataset(X_train, y_train)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

print(f"\nModel parameters: {sum(p.data.size for p in model.parameters()):,}")
print(f"Training on {N_TRAIN} synthetic 14×14 images...")

for epoch in range(1, 4):
    model.train()
    epoch_loss = 0.0
    for X_b, y_b in train_loader:
        optimizer.zero_grad()
        logits = model(Tensor(X_b))
        loss = loss_fn(logits, Tensor(y_b.astype(np.int64)))
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    model.eval()
    test_logits = model(Tensor(X_test)).data
    pred = np.argmax(test_logits, axis=1)
    acc = accuracy_score(y_test, pred)
    print(f"  Epoch {epoch}  |  Loss: {epoch_loss/len(train_loader):.4f}  |  Test acc: {acc:.4f}")

print("\nNote: Random data gives ~10% accuracy. Use real MNIST for real results.")
