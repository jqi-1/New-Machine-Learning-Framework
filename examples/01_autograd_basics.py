"""
01_autograd_basics.py
Demonstrates manual gradient computation with the numl autograd engine.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from numl.core.tensor import Tensor
from numl.core import ops

print("=" * 50)
print("Example 1: Autograd Basics")
print("=" * 50)

# --- Scalar computation ---
print("\n--- Scalar: f(x) = x^2 + 2x + 1  at x=3 ---")
x = Tensor(np.array(3.0), requires_grad=True)
f = x ** 2 + Tensor(np.array(2.0)) * x + Tensor(np.array(1.0))
f.backward()
print(f"  f(3)  = {f.item():.4f}  (expected 16.0)")
print(f"  f'(3) = {x.grad:.4f}  (expected 8.0)")

# --- Vector computation ---
print("\n--- Vector: f(w) = sum(w^2) ---")
w = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
loss = (w * w).sum()
loss.backward()
print(f"  loss       = {loss.item():.4f}  (expected 14.0)")
print(f"  d_loss/d_w = {w.grad}  (expected [2, 4, 6])")

# --- Chain rule ---
print("\n--- Chain rule: f(x) = exp(sin(x)) ---")
x = Tensor(np.array([0.0, np.pi / 4]), requires_grad=True)
y = ops.sigmoid(x)
z = y.log()
loss = z.sum()
loss.backward()
print(f"  x     = {x.data}")
print(f"  z     = {z.data}")
print(f"  dz/dx = {x.grad}")

# --- Matrix operations ---
print("\n--- Matrix multiply: C = A @ B, loss = sum(C) ---")
A = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), requires_grad=True)
B = Tensor(np.array([[1.0, 0.0], [0.0, 1.0]]), requires_grad=True)
C = A @ B
loss = C.sum()
loss.backward()
print(f"  C =\n{C.data}")
print(f"  dL/dA =\n{A.grad}  (expected ones)")
print(f"  dL/dB =\n{B.grad}")

# --- Gradient accumulation ---
print("\n--- Gradient accumulation: a used twice ---")
a = Tensor(np.array([2.0, 3.0]), requires_grad=True)
b = a + a  # both paths go through a
c = b.sum()
c.backward()
print(f"  dc/da = {a.grad}  (expected [2, 2])")

# --- Simple gradient descent step ---
print("\n--- Gradient descent: minimize f(w) = ||w||^2 ---")
w = Tensor(np.array([10.0, -5.0, 3.0]), requires_grad=True)
lr = 0.1
for step in range(20):
    loss = (w * w).sum()
    loss.backward()
    w.data -= lr * w.grad
    w.zero_grad()
print(f"  After 20 steps: w = {w.data}")
print(f"  Loss = {(w * w).sum().item():.6f}  (should be ~0)")

print("\nDone!")
