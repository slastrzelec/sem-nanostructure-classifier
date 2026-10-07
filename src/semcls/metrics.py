"""Metrics and bootstrap confidence intervals (numpy only)."""
import numpy as np


def softmax(logits):
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def confusion(y, p, k):
    y = np.asarray(y, dtype=np.int64)
    p = np.asarray(p, dtype=np.int64)
    return np.bincount(y * k + p, minlength=k * k).reshape(k, k)


def f1_per_class(y, p, k):
    """F1 of every class in 0..k-1; a class with no support and no predictions scores 0."""
    cm = confusion(y, p, k).astype(np.float64)
    tp = np.diag(cm)
    denom = cm.sum(axis=1) + cm.sum(axis=0)
    return np.divide(2 * tp, denom, out=np.zeros(k), where=denom > 0)


def macro_f1(y, p, k):
    return float(f1_per_class(y, p, k).mean())


def recall_per_class(y, p, k):
    cm = confusion(y, p, k).astype(np.float64)
    support = cm.sum(axis=1)
    return np.divide(np.diag(cm), support, out=np.zeros(k), where=support > 0)


def accuracy(y, p):
    return float((np.asarray(y) == np.asarray(p)).mean())


def ece(probs, y, n_bins=15):
    """Expected calibration error (top-label confidence, equal-width bins)."""
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == np.asarray(y)).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def bootstrap_dist(metric, y, *arrays, n_boot=1000, seed=0, groups=None):
    """Bootstrap distribution of metric(y, *arrays). Without `groups` samples are resampled one by
    one. With `groups` (e.g. cluster ids) whole groups are resampled, which keeps the correlation
    between near-duplicate images and gives wider, honest intervals. Arrays may be 2-D (rows =
    samples); every array is indexed with the same resample."""
    y = np.asarray(y)
    arrays = [np.asarray(a) for a in arrays]
    rng = np.random.default_rng(seed)
    n = len(y)
    if groups is not None:
        groups = np.asarray(groups)
        order = np.argsort(groups, kind="stable")
        _, starts = np.unique(groups[order], return_index=True)
        members = np.split(order, starts[1:])
    vals = np.empty(n_boot)
    for i in range(n_boot):
        if groups is None:
            idx = rng.integers(0, n, n)
        else:
            idx = np.concatenate([members[j] for j in rng.integers(0, len(members), len(members))])
        vals[i] = metric(y[idx], *[a[idx] for a in arrays])
    return vals


def bootstrap_ci(metric, y, *arrays, n_boot=1000, seed=0, alpha=0.05, groups=None):
    """Percentile bootstrap. metric(y, *arrays) -> float. Returns (point, lo, hi)."""
    vals = bootstrap_dist(metric, y, *arrays, n_boot=n_boot, seed=seed, groups=groups)
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(metric(np.asarray(y), *[np.asarray(a) for a in arrays])), float(lo), float(hi)
