"""
06_linear_regression.py
Demonstrates LinearRegression, Ridge, and Lasso on synthetic data.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from numl.classical.linear import LinearRegression, Ridge, Lasso, ElasticNet
from numl.metrics.regression import mean_squared_error, r2_score, mean_absolute_error

print("=" * 50)
print("Example 6: Linear Regression Variants")
print("=" * 50)

np.random.seed(42)
n = 100
# True: y = 2x1 - 1.5x2 + 0.5x3 + noise
X = np.random.randn(n, 5)
true_coef = np.array([2.0, -1.5, 0.5, 0.0, 0.0])
y = X @ true_coef + np.random.randn(n) * 0.3

# Split
X_tr, X_te = X[:80], X[80:]
y_tr, y_te = y[:80], y[80:]

print("\n--- Ordinary Least Squares ---")
lr = LinearRegression()
lr.fit(X_tr, y_tr)
y_pred = lr.predict(X_te)
print(f"  R²  = {r2_score(y_te, y_pred):.4f}")
print(f"  MSE = {mean_squared_error(y_te, y_pred):.4f}")
print(f"  Coefficients: {lr.coef_}")

print("\n--- Ridge (alpha=0.1) ---")
ridge = Ridge(alpha=0.1)
ridge.fit(X_tr, y_tr)
y_pred = ridge.predict(X_te)
print(f"  R²  = {r2_score(y_te, y_pred):.4f}")
print(f"  Coefficients: {ridge.coef_}")

print("\n--- Lasso (alpha=0.05, sparse) ---")
lasso = Lasso(alpha=0.05, max_iter=2000, tol=1e-4)
lasso.fit(X_tr, y_tr)
y_pred = lasso.predict(X_te)
print(f"  R²  = {r2_score(y_te, y_pred):.4f}")
print(f"  Coefficients: {np.round(lasso.coef_, 4)}")
n_nonzero = np.sum(np.abs(lasso.coef_) > 1e-3)
print(f"  Non-zero coefficients: {n_nonzero} (true: 3)")

print("\n--- ElasticNet (alpha=0.05, l1_ratio=0.5) ---")
en = ElasticNet(alpha=0.05, l1_ratio=0.5, max_iter=2000)
en.fit(X_tr, y_tr)
y_pred = en.predict(X_te)
print(f"  R²  = {r2_score(y_te, y_pred):.4f}")
print(f"  Coefficients: {np.round(en.coef_, 4)}")

print("\n--- Comparison summary ---")
print("  True coefficients:", true_coef)

# Noise-free data should give R² ≥ 0.99
X_clean = np.random.randn(50, 3)
y_clean = X_clean @ np.array([1.0, 2.0, 3.0])
lr2 = LinearRegression().fit(X_clean, y_clean)
r2 = r2_score(y_clean, lr2.predict(X_clean))
print(f"\n  Noise-free R² (should be ~1.0): {r2:.8f}")
