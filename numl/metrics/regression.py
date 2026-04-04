"""
numl/metrics/regression.py
---------------------------
Regression evaluation metrics.
"""

import numpy as np


def mean_squared_error(y_true, y_pred, squared: bool = True) -> float:
    """
    Mean Squared Error (or RMSE if squared=False).

    Parameters
    ----------
    squared : bool — if False, returns RMSE (default True)
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mse = np.mean((y_true - y_pred) ** 2)
    return float(mse if squared else np.sqrt(mse))


def mean_absolute_error(y_true, y_pred) -> float:
    """Mean Absolute Error."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(np.abs(y_true - y_pred)))


def r2_score(y_true, y_pred) -> float:
    """
    R² (coefficient of determination).
    1.0 = perfect, 0.0 = predicting the mean, can be negative.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1.0 - ss_res / (ss_tot + 1e-10))


def mean_absolute_percentage_error(y_true, y_pred) -> float:
    """
    Mean Absolute Percentage Error (MAPE).
    Returns value in [0, inf). Undefined when y_true == 0.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def explained_variance_score(y_true, y_pred) -> float:
    """
    Explained variance: 1 - Var(y - ŷ) / Var(y).
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    var_res = np.var(y_true - y_pred)
    var_y = np.var(y_true)
    return float(1.0 - var_res / (var_y + 1e-10))


def max_error(y_true, y_pred) -> float:
    """Maximum residual error."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.max(np.abs(y_true - y_pred)))


def median_absolute_error(y_true, y_pred) -> float:
    """Median Absolute Error — robust to outliers."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.median(np.abs(y_true - y_pred)))


def mean_squared_log_error(y_true, y_pred) -> float:
    """
    Mean Squared Log Error. Only valid for non-negative predictions.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if np.any(y_pred < 0) or np.any(y_true < 0):
        raise ValueError("MSLE is undefined for negative values.")
    return float(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2))


def huber_loss(y_true, y_pred, delta: float = 1.0) -> float:
    """Huber loss — quadratic for small errors, linear for large."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    err = np.abs(y_true - y_pred)
    return float(np.mean(np.where(err <= delta, 0.5 * err ** 2, delta * (err - 0.5 * delta))))
