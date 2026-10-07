"""Temperature scaling and selective prediction on stored logits (numpy only). SPEC section 11, 2026-10-07."""
import numpy as np

from . import metrics as M

T_LO, T_HI = 0.05, 10.0
TARGET_ACC = 0.99
MIN_ACCEPTED = 50


def log_softmax(logits):
    z = logits - logits.max(axis=1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


def nll(logits, y, T=1.0):
    ls = log_softmax(np.asarray(logits, dtype=np.float64) / T)
    return float(-ls[np.arange(len(y)), np.asarray(y)].mean())


def fit_temperature(logits, y, lo=T_LO, hi=T_HI, tol=1e-6):
    """argmin_T NLL(logits / T) by golden-section search over log T (NLL is unimodal in log T)."""
    a, b = np.log(lo), np.log(hi)
    g = (np.sqrt(5) - 1) / 2
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = nll(logits, y, np.exp(c)), nll(logits, y, np.exp(d))
    while b - a > tol:
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = nll(logits, y, np.exp(c))
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = nll(logits, y, np.exp(d))
    return float(np.exp((a + b) / 2))


def confidence(logits, T=1.0):
    return M.softmax(np.asarray(logits, dtype=np.float64) / T).max(axis=1)


def choose_threshold(conf, correct, target=TARGET_ACC, min_n=MIN_ACCEPTED):
    """Smallest threshold theta (largest coverage) with accuracy(conf >= theta) >= target and >= min_n
    accepted images, ties included. None if no threshold qualifies."""
    conf, correct = np.asarray(conf, dtype=np.float64), np.asarray(correct, dtype=np.float64)
    order = np.argsort(-conf, kind="stable")
    c, ok = conf[order], correct[order]
    k = np.arange(1, len(c) + 1)
    acc = np.cumsum(ok) / k
    last = np.r_[c[:-1] != c[1:], True]  # last element of every group of tied confidences
    valid = last & (k >= min_n) & (acc >= target)
    if not valid.any():
        return None
    return float(c[np.flatnonzero(valid)[-1]])


def selective(conf, correct, theta):
    """(coverage, accuracy of accepted images; nan if none accepted)."""
    m = np.asarray(conf) >= theta
    return float(m.mean()), (float(np.asarray(correct)[m].mean()) if m.any() else float("nan"))


def aurc(conf, correct):
    """Area under the risk-coverage curve: mean over k of the error rate among the k most confident images."""
    order = np.argsort(-np.asarray(conf), kind="stable")
    ok = np.asarray(correct, dtype=np.float64)[order]
    return float((1.0 - np.cumsum(ok) / np.arange(1, len(ok) + 1)).mean())


def accuracy_at_coverage(conf, correct, coverage):
    """Accuracy of the `coverage` share of most confident images (descriptive)."""
    n = len(conf)
    k = max(1, int(round(coverage * n)))
    order = np.argsort(-np.asarray(conf), kind="stable")[:k]
    return float(np.asarray(correct)[order].mean())


def error_capture(conf, correct, theta):
    """Share of the errors that fall below the threshold (nan if there are no errors)."""
    err = np.asarray(correct) == 0
    return float((np.asarray(conf)[err] < theta).mean()) if err.any() else float("nan")
