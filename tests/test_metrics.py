import numpy as np
import pytest
from sklearn.metrics import f1_score, recall_score

from semcls import metrics as M


def rand_case(n=500, k=5, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, k, n)
    p = np.where(rng.random(n) < 0.7, y, rng.integers(0, k, n))
    return y, p, k


def test_macro_f1_matches_sklearn():
    y, p, k = rand_case()
    assert M.macro_f1(y, p, k) == pytest.approx(f1_score(y, p, average="macro", labels=range(k), zero_division=0))


def test_macro_f1_counts_absent_class_as_zero():
    y, p = np.array([0, 0, 1, 1]), np.array([0, 0, 1, 1])
    assert M.macro_f1(y, p, 3) == pytest.approx(2 / 3)


def test_recall_matches_sklearn():
    y, p, k = rand_case(seed=1)
    assert M.recall_per_class(y, p, k) == pytest.approx(recall_score(y, p, average=None, labels=range(k), zero_division=0))


def test_confusion_counts():
    cm = M.confusion([0, 0, 1, 2], [0, 1, 1, 2], 3)
    assert cm.tolist() == [[1, 1, 0], [0, 1, 0], [0, 0, 1]]


def test_ece_perfectly_calibrated_and_overconfident():
    # always predicts class 0 with confidence 1.0, right half of the time -> ECE 0.5
    probs = np.tile([1.0, 0.0], (100, 1))
    y = np.array([0, 1] * 50)
    assert M.ece(probs, y) == pytest.approx(0.5)
    # confidence 0.5 and accuracy 0.5 -> ECE 0
    probs = np.tile([0.5, 0.5], (100, 1))
    probs[:, 0] += 1e-9
    assert M.ece(probs, np.array([0, 1] * 50)) == pytest.approx(0.0, abs=1e-6)


def test_softmax_rows_sum_to_one():
    s = M.softmax(np.random.default_rng(0).normal(size=(10, 4)) * 50)
    assert s.sum(axis=1) == pytest.approx(np.ones(10))


def test_bootstrap_ci_contains_point_and_is_reproducible():
    y, p, k = rand_case(n=400)
    f = lambda y_, p_: M.macro_f1(y_, p_, k)
    a = M.bootstrap_ci(f, y, p, n_boot=300, seed=3)
    b = M.bootstrap_ci(f, y, p, n_boot=300, seed=3)
    assert a == b
    assert a[1] <= a[0] <= a[2]
    assert a[2] - a[1] < 0.2


def test_cluster_bootstrap_is_wider_for_correlated_samples():
    rng = np.random.default_rng(0)
    k, n_groups, rep = 3, 60, 10
    gy = rng.integers(0, k, n_groups)
    gp = np.where(rng.random(n_groups) < 0.8, gy, rng.integers(0, k, n_groups))
    y, p = np.repeat(gy, rep), np.repeat(gp, rep)  # every sample is duplicated 10 times
    groups = np.repeat(np.arange(n_groups), rep)
    f = lambda y_, p_: M.accuracy(y_, p_)
    _, lo_i, hi_i = M.bootstrap_ci(f, y, p, n_boot=500, seed=1)
    _, lo_g, hi_g = M.bootstrap_ci(f, y, p, n_boot=500, seed=1, groups=groups)
    assert (hi_g - lo_g) > 1.8 * (hi_i - lo_i)


def test_cluster_bootstrap_with_singletons_matches_plain_bootstrap():
    y, p, k = rand_case(n=300, seed=2)
    f = lambda y_, p_: M.macro_f1(y_, p_, k)
    a = M.bootstrap_ci(f, y, p, n_boot=400, seed=5)
    b = M.bootstrap_ci(f, y, p, n_boot=400, seed=5, groups=np.arange(len(y)))
    assert a[0] == b[0]
    assert abs((a[2] - a[1]) - (b[2] - b[1])) < 0.03
