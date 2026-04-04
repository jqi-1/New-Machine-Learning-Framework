"""
10_pca_visualization.py
PCA dimensionality reduction and 2D visualization of high-dimensional data.
Also demonstrates TruncatedSVD.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from numl.decomposition.pca import PCA, IncrementalPCA
from numl.decomposition.svd import TruncatedSVD
from numl.preprocessing.scalers import StandardScaler

print("=" * 50)
print("Example 10: PCA Visualization & Dimensionality Reduction")
print("=" * 50)

# Synthetic high-dimensional data with 3 underlying components
np.random.seed(42)
n_samples = 300
n_features = 20
n_components_true = 3

# Data lives on a 3D subspace in 20D space
A = np.random.randn(n_components_true, n_features)  # basis
Z = np.random.randn(n_samples, n_components_true)   # latent coords
X = Z @ A + np.random.randn(n_samples, n_features) * 0.1
y = np.argmax(Z, axis=1) % 3  # synthetic labels based on dominant component

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Full PCA ---
print(f"\n--- PCA on {n_samples} samples × {n_features} features ---")
pca = PCA(n_components=10)
X_pca = pca.fit_transform(X_scaled)
print(f"  Input shape:  {X_scaled.shape}")
print(f"  Output shape: {X_pca.shape}")
print(f"\n  Explained variance ratio per component:")
cum_var = 0.0
for i, var in enumerate(pca.explained_variance_ratio_[:6]):
    cum_var += var
    bar = '#' * int(var * 200)
    print(f"  PC{i+1}: {var:.4f}  (cumul: {cum_var:.4f})  {bar}")

print(f"\n  Variance explained by first 3 PCs: "
      f"{pca.explained_variance_ratio_[:3].sum():.4f}")
print(f"  (Expected ~high since data has 3 true components)")

# --- 2D projection ---
print("\n--- 2D projection (for visualization) ---")
pca2 = PCA(n_components=2)
X_2d = pca2.fit_transform(X_scaled)
print(f"  Projected to 2D: shape = {X_2d.shape}")
for label in range(3):
    mask = y == label
    center = X_2d[mask].mean(axis=0)
    print(f"  Class {label} center: ({center[0]:.3f}, {center[1]:.3f})")

# --- Reconstruction error ---
print("\n--- Reconstruction error vs n_components ---")
for k in [1, 2, 3, 5, 10, 20]:
    pca_k = PCA(n_components=min(k, n_features))
    X_t = pca_k.fit_transform(X_scaled)
    X_rec = pca_k.inverse_transform(X_t)
    err = np.mean((X_scaled - X_rec) ** 2)
    bar = '#' * max(1, int((1 - err / np.var(X_scaled)) * 30))
    print(f"  k={k:2d}: MSE={err:.4f}  {bar}")

# --- IncrementalPCA ---
print("\n--- IncrementalPCA (streaming) ---")
ipca = IncrementalPCA(n_components=3)
batch_size = 50
for i in range(0, n_samples, batch_size):
    ipca.partial_fit(X_scaled[i:i+batch_size])
X_ipca = ipca.transform(X_scaled)
print(f"  IncrementalPCA output shape: {X_ipca.shape}")

# Compare full PCA vs IncrementalPCA reconstruction
pca3 = PCA(n_components=3)
X_full = pca3.fit_transform(X_scaled)
err_full = np.mean((X_scaled - pca3.inverse_transform(X_full))**2)
err_inc = np.mean((X_scaled - ipca.inverse_transform(X_ipca))**2)
print(f"  Full PCA reconstruction MSE:        {err_full:.6f}")
print(f"  IncrementalPCA reconstruction MSE:  {err_inc:.6f}")

# --- TruncatedSVD ---
print("\n--- TruncatedSVD (randomized) ---")
svd = TruncatedSVD(n_components=5, n_iter=5, random_state=42)
X_svd = svd.fit_transform(X_scaled)
print(f"  TruncatedSVD output shape: {X_svd.shape}")
print(f"  Singular values: {np.round(svd.singular_values_, 3)}")
print(f"  Explained variance ratio: {np.round(svd.explained_variance_ratio_, 4)}")

print("\nDone!")
