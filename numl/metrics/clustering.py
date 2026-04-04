"""
numl/metrics/clustering.py
---------------------------
Clustering evaluation metrics.
"""

import numpy as np


def silhouette_score(X: np.ndarray, labels, metric: str = 'euclidean') -> float:
    """
    Mean Silhouette Coefficient for all samples.

    For sample i:
      a(i) = mean intra-cluster distance
      b(i) = mean nearest-cluster distance
      s(i) = (b(i) - a(i)) / max(a(i), b(i))

    Returns mean s(i) over all samples. Range: [-1, 1], higher is better.
    """
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels)
    unique_labels = np.unique(labels[labels != -1])  # ignore noise (-1)

    if len(unique_labels) < 2:
        raise ValueError("silhouette_score requires at least 2 distinct clusters.")

    n = len(X)
    s = np.zeros(n)

    # Compute pairwise distances
    if metric == 'euclidean':
        sq = np.sum(X ** 2, axis=1)
        D = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * X @ X.T, 0))
    elif metric == 'manhattan':
        D = np.sum(np.abs(X[:, None] - X[None, :]), axis=2)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    for i in range(n):
        if labels[i] == -1:
            s[i] = 0.0
            continue
        own_mask = labels == labels[i]
        own_mask[i] = False

        if own_mask.sum() == 0:
            s[i] = 0.0
            continue

        a_i = D[i, own_mask].mean()

        b_i = np.inf
        for lbl in unique_labels:
            if lbl == labels[i]:
                continue
            other_mask = labels == lbl
            if other_mask.sum() == 0:
                continue
            mean_dist = D[i, other_mask].mean()
            if mean_dist < b_i:
                b_i = mean_dist

        if b_i == np.inf:
            s[i] = 0.0
        else:
            denom = max(a_i, b_i)
            s[i] = (b_i - a_i) / denom if denom > 0 else 0.0

    valid = labels != -1
    return float(s[valid].mean())


def silhouette_samples(X: np.ndarray, labels, metric: str = 'euclidean') -> np.ndarray:
    """Per-sample silhouette scores."""
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels)
    unique_labels = np.unique(labels[labels != -1])
    n = len(X)

    if metric == 'euclidean':
        sq = np.sum(X ** 2, axis=1)
        D = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * X @ X.T, 0))
    else:
        D = np.sum(np.abs(X[:, None] - X[None, :]), axis=2)

    s = np.zeros(n)
    for i in range(n):
        if labels[i] == -1:
            continue
        own_mask = labels == labels[i]
        own_mask[i] = False
        if own_mask.sum() == 0:
            continue
        a_i = D[i, own_mask].mean()
        b_i = min(D[i, labels == lbl].mean()
                  for lbl in unique_labels if lbl != labels[i]
                  and (labels == lbl).sum() > 0)
        denom = max(a_i, b_i)
        s[i] = (b_i - a_i) / denom if denom > 0 else 0.0
    return s


def davies_bouldin_score(X: np.ndarray, labels) -> float:
    """
    Davies-Bouldin Index. Lower is better.

    DB = (1/k) * sum_i max_{j≠i} [ (s_i + s_j) / d(c_i, c_j) ]
    where s_i = mean distance of cluster i to its centroid,
          d(c_i, c_j) = distance between centroids i and j.
    """
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels)
    unique_labels = np.unique(labels[labels != -1])
    k = len(unique_labels)

    centroids = np.array([X[labels == lbl].mean(axis=0) for lbl in unique_labels])
    scatter = np.array([np.mean(np.linalg.norm(X[labels == lbl] - centroids[i], axis=1))
                        for i, lbl in enumerate(unique_labels)])

    db = 0.0
    for i in range(k):
        max_ratio = 0.0
        for j in range(k):
            if i == j:
                continue
            d = np.linalg.norm(centroids[i] - centroids[j])
            if d > 0:
                ratio = (scatter[i] + scatter[j]) / d
                if ratio > max_ratio:
                    max_ratio = ratio
        db += max_ratio

    return float(db / k)


def calinski_harabasz_score(X: np.ndarray, labels) -> float:
    """
    Calinski-Harabasz (Variance Ratio Criterion). Higher is better.

    CH = (BSS / (k-1)) / (WSS / (n-k))
    """
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels)
    unique_labels = np.unique(labels)
    k = len(unique_labels)
    n = len(X)
    global_mean = X.mean(axis=0)

    bss = 0.0  # between-cluster scatter
    wss = 0.0  # within-cluster scatter
    for lbl in unique_labels:
        mask = labels == lbl
        c = X[mask].mean(axis=0)
        bss += mask.sum() * np.sum((c - global_mean) ** 2)
        wss += np.sum((X[mask] - c) ** 2)

    if wss == 0 or k == 1:
        return 0.0
    return float((bss / (k - 1)) / (wss / (n - k)))


def adjusted_rand_score(labels_true, labels_pred) -> float:
    """
    Adjusted Rand Index (ARI). 1.0 = perfect match, ~0 = random.

    Accounts for chance agreement, unlike the raw Rand index.
    """
    labels_true = np.asarray(labels_true)
    labels_pred = np.asarray(labels_pred)
    n = len(labels_true)

    # Contingency table
    classes = np.unique(labels_true)
    clusters = np.unique(labels_pred)
    contingency = np.zeros((len(classes), len(clusters)), dtype=int)
    ci_map = {c: i for i, c in enumerate(classes)}
    cj_map = {c: j for j, c in enumerate(clusters)}
    for t, p in zip(labels_true, labels_pred):
        if t in ci_map and p in cj_map:
            contingency[ci_map[t], cj_map[p]] += 1

    def comb2(x):
        return x * (x - 1) // 2

    sum_comb_c = sum(comb2(contingency[i, j])
                     for i in range(len(classes))
                     for j in range(len(clusters)))

    sum_a = sum(comb2(contingency[i, :].sum()) for i in range(len(classes)))
    sum_b = sum(comb2(contingency[:, j].sum()) for j in range(len(clusters)))

    total = comb2(n)
    expected = sum_a * sum_b / total if total > 0 else 0
    max_index = 0.5 * (sum_a + sum_b)

    denom = max_index - expected
    if denom == 0:
        return 1.0 if sum_comb_c == expected else 0.0
    return float((sum_comb_c - expected) / denom)


def normalized_mutual_info_score(labels_true, labels_pred) -> float:
    """Normalized Mutual Information between two clusterings."""
    labels_true = np.asarray(labels_true)
    labels_pred = np.asarray(labels_pred)
    n = len(labels_true)

    classes = np.unique(labels_true)
    clusters = np.unique(labels_pred)

    # Contingency
    contingency = np.zeros((len(classes), len(clusters)), dtype=float)
    ci_map = {c: i for i, c in enumerate(classes)}
    cj_map = {c: j for j, c in enumerate(clusters)}
    for t, p in zip(labels_true, labels_pred):
        if t in ci_map and p in cj_map:
            contingency[ci_map[t], cj_map[p]] += 1

    contingency /= n
    row_sums = contingency.sum(axis=1)
    col_sums = contingency.sum(axis=0)

    mi = 0.0
    for i in range(len(classes)):
        for j in range(len(clusters)):
            if contingency[i, j] > 0 and row_sums[i] > 0 and col_sums[j] > 0:
                mi += contingency[i, j] * np.log(contingency[i, j] / (row_sums[i] * col_sums[j]))

    h_true = -np.sum(row_sums[row_sums > 0] * np.log(row_sums[row_sums > 0]))
    h_pred = -np.sum(col_sums[col_sums > 0] * np.log(col_sums[col_sums > 0]))

    denom = (h_true + h_pred) / 2
    return float(mi / denom) if denom > 0 else 1.0
