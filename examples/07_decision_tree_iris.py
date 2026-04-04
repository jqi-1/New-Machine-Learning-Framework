"""
07_decision_tree_iris.py
Decision tree on synthetic Iris-like data (3-class classification).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from numl.classical.tree import DecisionTreeClassifier
from numl.data.splits import train_test_split
from numl.metrics.classification import (
    accuracy_score, confusion_matrix, classification_report
)

print("=" * 50)
print("Example 7: Decision Tree on Iris-like Data")
print("=" * 50)

# Synthetic 3-class data (Iris-style: 4 features, 3 classes)
np.random.seed(0)
centers = np.array([
    [5.0, 3.5, 1.5, 0.3],
    [6.0, 2.8, 4.5, 1.4],
    [6.5, 3.0, 5.5, 2.0],
])
n_per_class = 50
X = np.vstack([
    np.random.randn(n_per_class, 4) * 0.4 + c
    for c in centers
])
y = np.repeat([0, 1, 2], n_per_class)
class_names = ['setosa', 'versicolor', 'virginica']
feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

# Train/test split
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

# Train decision trees with varying depths
print("\nEffect of max_depth on accuracy:")
for depth in [1, 2, 3, 5, None]:
    tree = DecisionTreeClassifier(max_depth=depth, criterion='gini')
    tree.fit(X_tr, y_tr)
    tr_acc = tree.score(X_tr, y_tr)
    te_acc = tree.score(X_te, y_te)
    depth_str = str(depth) if depth is not None else "inf"
    print(f"  depth={depth_str:3s}  |  train: {tr_acc:.4f}  |  test: {te_acc:.4f}")

# Best model
tree = DecisionTreeClassifier(max_depth=4, criterion='gini')
tree.fit(X_tr, y_tr)
y_pred = tree.predict(X_te)

print(f"\nBest model (depth=4):")
print(f"  Test accuracy: {accuracy_score(y_te, y_pred):.4f}")

print("\nConfusion matrix:")
cm = confusion_matrix(y_te, y_pred)
print(f"  {cm}")

print("\nFeature importances:")
for name, imp in sorted(zip(feature_names, tree.feature_importances_),
                         key=lambda x: -x[1]):
    bar = '#' * int(imp * 40)
    print(f"  {name:<15s} {imp:.4f}  {bar}")

print(f"\nClassification report:")
report = classification_report(y_te, y_pred, target_names=class_names)
print(report)
