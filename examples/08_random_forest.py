"""
08_random_forest.py
Compares single decision tree vs random forest on synthetic breast-cancer-like data.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from numl.classical.tree import DecisionTreeClassifier
from numl.classical.ensemble import RandomForestClassifier
from numl.data.splits import train_test_split, cross_val_score
from numl.metrics.classification import accuracy_score, roc_auc_score, f1_score

print("=" * 50)
print("Example 8: Random Forest vs Decision Tree")
print("=" * 50)

# Synthetic binary classification (breast-cancer-like, 30 features)
np.random.seed(42)
n = 500
n_features = 30
# Informative: first 10 features; rest are noise
X_info = np.random.randn(n, 10)
X_noise = np.random.randn(n, 20)
X = np.hstack([X_info, X_noise])
# Label: sign of linear combination of informative features
coef = np.array([1, -1, 2, -0.5, 1.5, -1, 0.5, -2, 1, -0.5])
logits = X_info @ coef
y = (logits > 0).astype(int)
print(f"  Class balance: {y.mean():.2f} positive")

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)

# Single decision tree
tree = DecisionTreeClassifier(max_depth=5)
tree.fit(X_tr, y_tr)
tree_preds = tree.predict(X_te)

# Random forest
rf = RandomForestClassifier(n_estimators=50, max_depth=5,
                             max_features='sqrt', random_state=0)
rf.fit(X_tr, y_tr)
rf_preds = rf.predict(X_te)

print(f"\n{'Model':<25s} {'Accuracy':>10s} {'F1':>10s}")
print("-" * 47)
for name, preds in [("Decision Tree (depth=5)", tree_preds),
                    ("Random Forest (50 trees)", rf_preds)]:
    acc = accuracy_score(y_te, preds)
    f1 = f1_score(y_te, preds)
    print(f"  {name:<23s} {acc:>10.4f} {f1:>10.4f}")

# Cross-validation comparison
print("\n5-fold cross-validation:")
for name, model in [("Decision Tree", DecisionTreeClassifier(max_depth=5)),
                     ("Random Forest", RandomForestClassifier(n_estimators=20,
                                                               max_depth=5,
                                                               random_state=0))]:
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f"  {name:<15s}: {np.mean(scores):.4f} ± {np.std(scores):.4f}")

# Feature importances
print("\nTop 10 feature importances (Random Forest):")
fi = rf.feature_importances_
top_idx = np.argsort(fi)[::-1][:10]
for rank, i in enumerate(top_idx, 1):
    marker = "* (informative)" if i < 10 else ""
    bar = '#' * int(fi[i] * 100)
    print(f"  {rank:2d}. Feature {i:2d}: {fi[i]:.4f}  {bar}  {marker}")
