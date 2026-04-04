# numl — NumPy Machine Learning Framework

A full machine learning framework built on **pure Python and NumPy** only. No PyTorch, no TensorFlow, no scikit-learn.

## Features

### Neural Networks with Automatic Differentiation
- **Autograd engine**: Reverse-mode autodiff via dynamic computation graph
- **Layers**: Dense, Conv2d (im2col), BatchNorm1d/2d, Dropout
- **Recurrent**: RNN, LSTM, GRU (with BPTT)
- **Activations**: ReLU, LeakyReLU, ELU, Sigmoid, Tanh, Softmax, LogSoftmax, GELU, Swish
- **Losses**: MSE, MAE, CrossEntropy, BCE, BCEWithLogits, Huber, NLL
- **Optimizers**: SGD (momentum, Nesterov), Adam, AdamW, RMSProp, Adagrad
- **LR Schedulers**: StepLR, CosineAnnealingLR, ReduceLROnPlateau, WarmupScheduler

### Classical Machine Learning
- **Linear models**: LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
- **Trees**: DecisionTreeClassifier, DecisionTreeRegressor
- **Ensembles**: RandomForestClassifier/Regressor, GradientBoostingClassifier/Regressor
- **SVM**: SVC, SVR (SMO solver, RBF/linear/poly/sigmoid kernels)
- **Neighbors**: KNNClassifier, KNNRegressor (with KD-tree)
- **Naive Bayes**: GaussianNB, MultinomialNB, BernoulliNB
- **Clustering**: KMeans (kmeans++), DBSCAN

### Data Utilities
- `Dataset`, `ArrayDataset`, `TensorDataset`
- `DataLoader` with shuffling, batching, custom collate
- `train_test_split`, `KFold`, `StratifiedKFold`, `cross_val_score`

### Preprocessing
- **Scalers**: StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, Normalizer
- **Encoders**: LabelEncoder, OneHotEncoder, OrdinalEncoder
- **Imputers**: SimpleImputer, KNNImputer
- **Transforms**: PolynomialFeatures, Binarizer, FunctionTransformer

### Decomposition
- PCA, IncrementalPCA, TruncatedSVD

### Metrics
- **Classification**: accuracy, precision, recall, F1, ROC-AUC, confusion matrix
- **Regression**: MSE, MAE, R², RMSE, MAPE
- **Clustering**: silhouette score, Davies-Bouldin, adjusted Rand index

### Model Serialization
- Neural networks: `.npz` weights + `.json` architecture
- Classical models: pickle

---

## Installation

```bash
pip install -e .
# or just:
pip install numpy
```

## Quick Start

### Neural Network (XOR)

```python
import numpy as np
from numl.core.tensor import Tensor
from numl.nn.layers import Dense
from numl.nn.activations import ReLU, Sigmoid
from numl.nn.sequential import Sequential
from numl.nn.loss import BCEWithLogitsLoss
from numl.optim.adam import Adam

model = Sequential([
    Dense(2, 8),
    ReLU(),
    Dense(8, 1),
])

X = Tensor(np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float64))
y = Tensor(np.array([[0],[1],[1],[0]], dtype=np.float64))

loss_fn = BCEWithLogitsLoss()
opt = Adam(model.parameters(), lr=0.01)

for epoch in range(1000):
    opt.zero_grad()
    pred = model(X)
    loss = loss_fn(pred, y)
    loss.backward()
    opt.step()

print(f"Loss: {loss.item():.4f}")
```

### Classical ML

```python
import numpy as np
from numl.classical.ensemble import RandomForestClassifier
from numl.metrics.classification import accuracy_score

X_train = np.random.randn(100, 4)
y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(int)

clf = RandomForestClassifier(n_estimators=50)
clf.fit(X_train, y_train)
preds = clf.predict(X_train)
print(f"Train accuracy: {accuracy_score(y_train, preds):.3f}")
```

### Save & Load Models

```python
from numl.serialize.io import save_model, load_model

save_model(model, "my_model")          # writes my_model.npz + my_model.json
model2 = load_model("my_model", Sequential)
```

---

## Running Tests

```bash
python -m pytest tests/
# or
python -m unittest discover tests/
```

## Examples

See the `examples/` directory for end-to-end demonstrations:
- `01_autograd_basics.py` — Manual gradient computation
- `02_neural_net_xor.py` — XOR with MLP
- `03_mnist_mlp.py` — MNIST classification
- `06_linear_regression.py` — Ridge/Lasso regression
- `09_kmeans_clustering.py` — KMeans clustering
- ... and more

---

## Architecture

```
numl/
├── core/       # Tensor + autograd engine
├── nn/         # Neural network layers, losses, activations
├── optim/      # Optimizers and LR schedulers
├── classical/  # Classical ML algorithms
├── data/       # Dataset, DataLoader, splits
├── preprocessing/
├── decomposition/
├── metrics/
└── serialize/
```

## License

MIT
