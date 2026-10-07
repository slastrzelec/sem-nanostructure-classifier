"""Leakage and consistency checks for splits/splits.json (SPEC section 7)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
SPLITS = ROOT / "splits" / "splits.json"
MANIFEST = ROOT / "reports" / "cache384_manifest.csv.gz"
NAMES = {"train", "val", "test"}
MIN_PER_CLASS_EVAL = 15  # smallest acceptable class count in val and in test
RATIO_TOL = 0.005  # clusters are indivisible, but the assignment is tuned until shares are within half a point
BIG_CLUSTER = 10  # images in clusters of at least this size must be spread evenly over the splits
BIG_SHARE_TOL = 0.05  # absolute, vs the share in the whole dataset


@pytest.fixture(scope="module")
def sp():
    return json.loads(SPLITS.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def man():
    m = pd.read_csv(MANIFEST)
    return m[m["note"].str.startswith("kept")].set_index("cache_name")


def variants(sp):
    yield "random", sp["random"], None
    for tau, v in sp["group"].items():
        yield f"group@{tau}", v["split"], v["cluster"]


def test_files_match_manifest(sp, man):
    assert len(set(sp["files"])) == len(sp["files"]) == sp["meta"]["n"]
    assert set(sp["files"]) == set(man.index)
    assert [man.loc[f, "class"] for f in sp["files"]] == sp["class"]


def test_no_exact_duplicate_content(sp, man):
    sha = man.loc[sp["files"], "orig_sha256"]
    assert sha.is_unique


def test_valid_labels_and_lengths(sp):
    n = sp["meta"]["n"]
    for name, split, cluster in variants(sp):
        assert len(split) == n, name
        assert set(split) == NAMES, name
        if cluster is not None:
            assert len(cluster) == n, name


def test_no_cluster_in_two_splits(sp):
    for tau, v in sp["group"].items():
        d = pd.DataFrame({"c": v["cluster"], "s": v["split"]})
        assert (d.groupby("c")["s"].nunique() == 1).all(), tau


def test_class_coverage(sp):
    y = np.array(sp["class"])
    for name, split, _ in variants(sp):
        t = pd.crosstab(y, np.array(split))
        assert t[["val", "test"]].min().min() >= MIN_PER_CLASS_EVAL, (name, t)
        assert (t["train"] > t[["val", "test"]].max(axis=1)).all(), name


def test_split_ratios(sp):
    n = sp["meta"]["n"]
    for name, split, _ in variants(sp):
        share = pd.Series(split).value_counts() / n
        for s, r in sp["meta"]["ratios"].items():
            assert abs(share[s] - r) < RATIO_TOL, (name, s, share[s])


def test_group_splits_differ_from_random(sp):
    # sanity: the group split is not accidentally identical to the random one
    for tau, v in sp["group"].items():
        assert np.mean(np.array(v["split"]) == np.array(sp["random"])) < 0.95


def test_large_clusters_spread_over_splits(sp):
    # Version 1 of the group split sent every large cluster to val/test (0% of train images were in
    # clusters >= 10, 28-31% of val/test), which made val/test unrepresentative. The share of
    # images that belong to large clusters must be similar in all three splits.
    for tau, v in sp["group"].items():
        cl, s = np.array(v["cluster"]), np.array(v["split"])
        big = np.bincount(cl)[cl] >= BIG_CLUSTER
        for name in NAMES:
            assert abs(big[s == name].mean() - big.mean()) < BIG_SHARE_TOL, (tau, name, big[s == name].mean(), big.mean())
