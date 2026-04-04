"""
numl/metrics/classification.py
--------------------------------
Classification evaluation metrics.
"""

import numpy as np


def accuracy_score(y_true, y_pred, normalize: bool = True) -> float:
    """
    Fraction (or count) of correctly classified samples.

    Parameters
    ----------
    normalize : if True (default), return fraction; else return count
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    correct = np.sum(y_true == y_pred)
    return correct / len(y_true) if normalize else int(correct)


def confusion_matrix(y_true, y_pred, labels=None) -> np.ndarray:
    """
    Compute confusion matrix C where C[i,j] = number of samples with
    true label i predicted as label j.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    n = len(labels)
    label_to_idx = {lbl: i for i, lbl in enumerate(labels)}
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        if t in label_to_idx and p in label_to_idx:
            cm[label_to_idx[t], label_to_idx[p]] += 1
    return cm


def _precision_recall_fscore(y_true, y_pred, average, zero_division=0):
    """Internal: compute per-class then average P, R, F."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes = np.unique(np.concatenate([y_true, y_pred]))

    precisions, recalls, f1s, supports = [], [], [], []
    for cls in classes:
        tp = np.sum((y_pred == cls) & (y_true == cls))
        fp = np.sum((y_pred == cls) & (y_true != cls))
        fn = np.sum((y_pred != cls) & (y_true == cls))
        support = tp + fn

        p = tp / (tp + fp) if (tp + fp) > 0 else zero_division
        r = tp / (tp + fn) if (tp + fn) > 0 else zero_division
        f = 2 * p * r / (p + r) if (p + r) > 0 else zero_division

        precisions.append(p)
        recalls.append(r)
        f1s.append(f)
        supports.append(support)

    precisions = np.array(precisions)
    recalls = np.array(recalls)
    f1s = np.array(f1s)
    supports = np.array(supports)

    if average == 'binary':
        # Use the last (positive) class
        return float(precisions[-1]), float(recalls[-1]), float(f1s[-1])
    elif average == 'micro':
        tp_sum = np.sum([np.sum((y_pred == c) & (y_true == c)) for c in classes])
        fp_sum = np.sum([np.sum((y_pred == c) & (y_true != c)) for c in classes])
        fn_sum = np.sum([np.sum((y_pred != c) & (y_true == c)) for c in classes])
        p = tp_sum / (tp_sum + fp_sum) if (tp_sum + fp_sum) > 0 else zero_division
        r = tp_sum / (tp_sum + fn_sum) if (tp_sum + fn_sum) > 0 else zero_division
        f = 2 * p * r / (p + r) if (p + r) > 0 else zero_division
        return float(p), float(r), float(f)
    elif average == 'macro':
        return float(precisions.mean()), float(recalls.mean()), float(f1s.mean())
    elif average == 'weighted':
        w = supports / supports.sum()
        return float(np.dot(w, precisions)), float(np.dot(w, recalls)), float(np.dot(w, f1s))
    elif average is None:
        return precisions, recalls, f1s

    raise ValueError(f"Unknown average: {average}")


def precision_score(y_true, y_pred, average: str = 'binary',
                    zero_division: float = 0) -> float:
    return _precision_recall_fscore(y_true, y_pred, average, zero_division)[0]


def recall_score(y_true, y_pred, average: str = 'binary',
                 zero_division: float = 0) -> float:
    return _precision_recall_fscore(y_true, y_pred, average, zero_division)[1]


def f1_score(y_true, y_pred, average: str = 'binary',
             zero_division: float = 0) -> float:
    return _precision_recall_fscore(y_true, y_pred, average, zero_division)[2]


def classification_report(y_true, y_pred, target_names=None,
                          zero_division: float = 0) -> str:
    """Return a text summary of classification metrics per class."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes = np.unique(np.concatenate([y_true, y_pred]))
    if target_names is None:
        target_names = [str(c) for c in classes]

    header = f"{'':>15} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>10}\n\n"
    rows = [header]

    p_all, r_all, f_all = _precision_recall_fscore(y_true, y_pred, average=None, zero_division=zero_division)
    supports = [np.sum(y_true == cls) for cls in classes]

    for name, p, r, f, sup in zip(target_names, p_all, r_all, f_all, supports):
        rows.append(f"{name:>15} {p:>10.2f} {r:>10.2f} {f:>10.2f} {sup:>10}\n")

    rows.append("\n")
    acc = accuracy_score(y_true, y_pred)
    total = len(y_true)
    p_macro, r_macro, f_macro = _precision_recall_fscore(y_true, y_pred, 'macro', zero_division)
    p_w, r_w, f_w = _precision_recall_fscore(y_true, y_pred, 'weighted', zero_division)

    rows.append(f"{'accuracy':>15} {'':>10} {'':>10} {acc:>10.2f} {total:>10}\n")
    rows.append(f"{'macro avg':>15} {p_macro:>10.2f} {r_macro:>10.2f} {f_macro:>10.2f} {total:>10}\n")
    rows.append(f"{'weighted avg':>15} {p_w:>10.2f} {r_w:>10.2f} {f_w:>10.2f} {total:>10}\n")

    return "".join(rows)


def roc_curve(y_true, y_score):
    """
    Compute ROC curve (binary classification).

    Parameters
    ----------
    y_true  : array of 0/1 labels
    y_score : array of scores/probabilities for the positive class

    Returns
    -------
    fpr, tpr, thresholds : np.ndarray
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    sorted_idx = np.argsort(-y_score)
    y_true_sorted = y_true[sorted_idx]
    thresholds = y_score[sorted_idx]

    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos

    tps = np.cumsum(y_true_sorted)
    fps = np.cumsum(1 - y_true_sorted)

    tpr = tps / n_pos
    fpr = fps / n_neg

    # Prepend (0, 0)
    tpr = np.concatenate([[0.0], tpr])
    fpr = np.concatenate([[0.0], fpr])
    thresholds = np.concatenate([[thresholds[0] + 1], thresholds])

    return fpr, tpr, thresholds


def roc_auc_score(y_true, y_score) -> float:
    """Area under the ROC curve (trapezoidal rule)."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.trapz(tpr, fpr))


def average_precision_score(y_true, y_score) -> float:
    """Area under the Precision-Recall curve (interpolated)."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    sorted_idx = np.argsort(-y_score)
    y_sorted = y_true[sorted_idx]

    n_pos = y_true.sum()
    tps = np.cumsum(y_sorted)
    fps = np.cumsum(1 - y_sorted)

    precision = tps / (tps + fps)
    recall = tps / n_pos

    # AP = sum of changes in recall * precision
    recall = np.concatenate([[0.0], recall])
    precision = np.concatenate([[1.0], precision])
    return float(np.trapz(precision, recall))


def log_loss(y_true, y_pred_proba, eps: float = 1e-7) -> float:
    """
    Cross-entropy loss between true labels and predicted probabilities.

    Parameters
    ----------
    y_true       : 1D array of class indices, or 2D one-hot
    y_pred_proba : 2D array (n_samples, n_classes)
    """
    y_pred_proba = np.clip(np.asarray(y_pred_proba, dtype=np.float64), eps, 1 - eps)
    y_true = np.asarray(y_true)

    if y_true.ndim == 1:
        n = len(y_true)
        return -float(np.mean(np.log(y_pred_proba[np.arange(n), y_true])))
    else:
        return -float(np.mean(np.sum(y_true * np.log(y_pred_proba), axis=1)))


def balanced_accuracy_score(y_true, y_pred) -> float:
    """Macro-averaged recall (balanced accuracy for imbalanced classes)."""
    return recall_score(y_true, y_pred, average='macro')


def cohen_kappa_score(y_true, y_pred) -> float:
    """Cohen's kappa statistic."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes = np.unique(np.concatenate([y_true, y_pred]))
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    n = cm.sum()
    p_o = np.diag(cm).sum() / n  # observed agreement
    p_e = np.sum(cm.sum(axis=0) * cm.sum(axis=1)) / n ** 2  # expected agreement

    return float((p_o - p_e) / (1 - p_e + 1e-10))
