import numpy as np

from semcls import calib as C
from semcls import metrics as M


def _toy(n=4000, k=5, scale=4.0, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, k, n)
    logits = rng.normal(0, 1, (n, k))
    logits[np.arange(n), y] += 1.5          # informative but noisy
    return logits * scale, y                # scale > 1: over-confident


def test_fit_temperature_scales_with_logits_and_keeps_argmax():
    z, y = _toy(scale=1.0)
    T1 = C.fit_temperature(z, y)
    T3 = C.fit_temperature(z * 3.0, y)      # inflating the logits by 3 must inflate T by 3
    assert abs(T3 / T1 - 3.0) < 0.01
    assert np.array_equal((z * 3.0 / T3).argmax(1), (z * 3.0).argmax(1))


def test_fit_temperature_not_worse_than_one_and_inside_bounds():
    z, y = _toy(scale=6.0)
    T = C.fit_temperature(z, y)
    assert C.nll(z, y, T) <= C.nll(z, y, 1.0) + 1e-9
    assert C.T_LO <= T <= C.T_HI
    assert M.ece(M.softmax(z / T), y) < M.ece(M.softmax(z), y)


def test_fit_temperature_matches_grid_search():
    z, y = _toy(scale=5.0, seed=3)
    grid = np.exp(np.linspace(np.log(0.05), np.log(10), 2000))
    best = grid[np.argmin([C.nll(z, y, t) for t in grid])]
    assert abs(np.log(C.fit_temperature(z, y)) - np.log(best)) < 0.01


def test_choose_threshold_basic_and_ties():
    conf = np.array([0.99, 0.98, 0.97, 0.9, 0.8, 0.7])
    ok = np.array([1, 1, 1, 0, 1, 0])
    assert C.choose_threshold(conf, ok, target=1.0, min_n=1) == 0.97
    assert C.choose_threshold(conf, ok, target=1.0, min_n=4) is None
    conf2 = np.array([0.9, 0.9, 0.5, 0.5])          # ties: cannot split the 0.9 pair
    ok2 = np.array([1, 0, 1, 1])
    assert C.choose_threshold(conf2, ok2, target=0.99, min_n=1) is None
    assert C.choose_threshold(conf2, ok2, target=0.75, min_n=1) == 0.5


def test_selective_aurc_capture_and_coverage_accuracy():
    conf = np.array([0.9, 0.8, 0.7, 0.6])
    ok = np.array([1, 1, 0, 1])
    cov, acc = C.selective(conf, ok, 0.75)
    assert (cov, acc) == (0.5, 1.0)
    assert np.isnan(C.selective(conf, ok, 0.95)[1])
    assert C.error_capture(conf, ok, 0.75) == 1.0
    assert C.accuracy_at_coverage(conf, ok, 0.75) == 2 / 3
    # perfect ranking (all errors least confident) has a lower AURC than the reverse
    good = C.aurc(np.array([0.9, 0.8, 0.7, 0.6]), np.array([1, 1, 0, 0]))
    bad = C.aurc(np.array([0.9, 0.8, 0.7, 0.6]), np.array([0, 0, 1, 1]))
    assert good < bad


def test_e4_end_to_end_on_synthetic_runs():
    from semcls.e4 import build_e4
    rng = np.random.default_rng(1)
    k, n = 3, 600
    sp = dict(files=[f"f{i}.jpg" for i in range(2 * n)], group={"0.94": {"cluster": list(range(2 * n))}})

    def mk(part, seed, off):
        y = rng.integers(0, k, n)
        z = rng.normal(0, 1, (n, k)); z[np.arange(n), y] += 2.0
        return dict(name=f"r{seed}", seed=seed, classes=["a", "b", "c"], names=[f"f{i + off}.jpg" for i in range(n)], y=y, logits=z * 3)
    val = {"convnext_tiny": [mk("val", s, 0) for s in range(3)]}
    test = {"convnext_tiny": [mk("test", s, n) for s in range(3)]}
    # same images for every seed: reuse the first run's y/names
    for g in (val, test):
        for r in g["convnext_tiny"][1:]:
            r["y"], r["names"] = g["convnext_tiny"][0]["y"], g["convnext_tiny"][0]["names"]
    md, res = build_e4(val, test, sp, n_boot=20, seed=0)
    r = res["convnext_tiny"]
    assert len(r["runs"]) == 3 and all(x["T"] > 1 for x in r["runs"])       # logits were inflated
    assert r["d_ece"]["point"] < 0
    assert "E4b" in md and "ConvNeXt-Tiny" in md
