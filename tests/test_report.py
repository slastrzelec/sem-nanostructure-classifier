"""Tests of the result aggregation (src/semcls/report.py) on synthetic runs."""
import json

import numpy as np
import pytest

from semcls import report as R

K = 3
CLASSES = ["a", "b", "c"]


def make_splits(n=120, seed=0):
    rng = np.random.default_rng(seed)
    files = [f"f{i}.jpg" for i in range(n)]
    cluster = np.arange(n)
    cluster[10:20] = 10          # a cluster of 10 images
    cluster[30:34] = 30
    rand = np.array(["train"] * 80 + ["val"] * 20 + ["test"] * 20)
    rng.shuffle(rand)
    rand[10:20] = "train"        # whole cluster 10 in train
    rand[30] = "train"           # cluster 30: one member in train ...
    rand[31] = "test"            # ... and one in test
    y = [CLASSES[i % K] for i in range(n)]
    return dict(files=files, **{"class": y}, random=rand.tolist(),
                group={"0.94": dict(cluster=cluster.tolist(), split=rand.tolist())})


def write_run(root, name, variant, backbone, seed, names, y, logits, sha="S1", part="val", done=True):
    d = root / name
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps(dict(variant=variant, backbone=backbone, seed=seed,
                                                   classes=CLASSES, splits_sha256=sha)))
    if done:
        (d / "done.flag").write_text("ok")
    np.savez(d / f"{part}_logits.npz", names=np.array(names), y=np.array(y), logits=logits)
    return d


def logits_for(y, correct, rng):
    lg = rng.normal(size=(len(y), K)) * 0.1
    pred = np.where(correct, y, (np.asarray(y) + 1) % K)
    lg[np.arange(len(y)), pred] += 5
    return lg


@pytest.fixture
def runs_dir(tmp_path):
    sp = make_splits()
    idx = np.where(np.asarray(sp["random"]) == "test")[0]
    names = [sp["files"][i] for i in idx]
    y = [i % K for i in idx]
    rng = np.random.default_rng(1)
    for seed, acc in enumerate([0.9, 0.8, 0.7]):
        write_run(tmp_path, f"random__resnet50__s{seed}", "random", "resnet50", seed, names, y,
                  logits_for(y, rng.random(len(y)) < acc, rng), part="test")
    return tmp_path, sp, names, y


def test_load_runs_filters_by_sha_and_files(tmp_path):
    names, y = ["f0.jpg", "f1.jpg"], [0, 1]
    lg = np.eye(K)[y]
    write_run(tmp_path, "r_ok", "random", "resnet50", 0, names, y, lg, sha="NEW")
    write_run(tmp_path, "r_old", "random", "resnet50", 1, names, y, lg, sha="OLD")
    write_run(tmp_path, "g_old", "group:0.94", "resnet50", 0, names, y, lg, sha="OLD")
    write_run(tmp_path, "g_unfinished", "group:0.94", "resnet50", 1, names, y, lg, sha="NEW", done=False)
    write_run(tmp_path, "g_noval", "group:0.94", "resnet50", 2, names, y, lg, sha="NEW", part="test")
    runs, skipped = R.load_runs([tmp_path], "val", "NEW", ["OLD"])
    got = sorted((r["variant"], r["seed"]) for r in runs)
    assert got == [("random", 0), ("random", 1)]            # legacy sha accepted for random only
    reasons = dict(skipped)
    assert "g_old" in reasons and "g_noval" in reasons and "g_unfinished" not in reasons


def test_load_runs_rejects_duplicates(tmp_path):
    names, y = ["f0.jpg"], [0]
    lg = np.eye(K)[y]
    write_run(tmp_path / "a", "r0", "random", "resnet50", 0, names, y, lg)
    write_run(tmp_path / "b", "r0", "random", "resnet50", 0, names, y, lg)
    with pytest.raises(ValueError, match="duplicate"):
        R.load_runs([tmp_path / "a", tmp_path / "b"], "val", "S1")


def test_group_runs_requires_same_images(tmp_path):
    lg = np.eye(K)[[0, 1]]
    write_run(tmp_path, "r0", "random", "resnet50", 0, ["f0.jpg", "f1.jpg"], [0, 1], lg)
    write_run(tmp_path, "r1", "random", "resnet50", 1, ["f0.jpg", "f2.jpg"], [0, 1], lg)
    runs, _ = R.load_runs([tmp_path], "val", "S1")
    with pytest.raises(ValueError, match="different images"):
        R.group_runs(runs)


def test_group_stats_mean_sd_and_ci(runs_dir):
    root, sp, names, y = runs_dir
    runs, _ = R.load_runs([root], "test", "S1")
    s = R.group_stats(R.group_runs(runs)[("random", "resnet50")], sp, n_boot=200)
    acc = np.array(s["acc"]["per_seed"])
    assert s["acc"]["mean"] == pytest.approx(acc.mean())
    assert s["acc"]["sd"] == pytest.approx(acc.std(ddof=1))
    assert s["seeds"] == [0, 1, 2] and s["n"] == len(names)
    for kind in ("acc", "macro_f1"):
        m = s[kind]
        assert m["ci_img"][0] <= m["mean"] <= m["ci_img"][1]
        assert m["ci_clu"][0] <= m["mean"] <= m["ci_clu"][1]
    assert np.asarray(s["confusion"]).sum() == 3 * len(names)


def test_gap_ci_point_and_interval(runs_dir):
    root, sp, names, y = runs_dir
    runs, _ = R.load_runs([root], "test", "S1")
    a = R.group_runs(runs)[("random", "resnet50")]
    best, worst = [a[0]], [a[2]]
    g = R.gap_ci(best, worst, sp, "acc", n_boot=300)
    assert g["point"] > 0
    assert g["img"][0] < g["point"] < g["img"][1]
    assert g["img"][0] > -0.2 and g["clu"][1] < 0.8
    zero = R.gap_ci(best, best, sp, "acc", n_boot=300)
    assert zero["point"] == 0 and zero["img"][0] < 0 < zero["img"][1]


def test_neighbour_in_train_flag():
    sp = make_splits()
    names = ["f31.jpg", "f12.jpg", "f50.jpg"]
    flag = R.neighbour_in_train(sp, names)
    assert flag[0]                                   # f31's cluster has f30 in train
    assert flag[1]                                   # cluster 10 is entirely in train
    train_clusters = {c for c, s in zip(sp["group"]["0.94"]["cluster"], sp["random"]) if s == "train"}
    assert flag[2] == (sp["group"]["0.94"]["cluster"][50] in train_clusters)


def test_macro_f1_present_ignores_absent_classes():
    y, p = np.array([0, 0, 1, 1]), np.array([0, 0, 1, 1])
    assert R.macro_f1_present(y, p, 3) == 1.0
    assert R.M.macro_f1(y, p, 3) == pytest.approx(2 / 3)


def test_check_random_identical():
    a, b = make_splits(), make_splits()
    R.check_random_identical(a, b)
    b["random"][0] = "val" if b["random"][0] != "val" else "test"
    with pytest.raises(ValueError):
        R.check_random_identical(a, b)


def test_build_report_end_to_end(runs_dir):
    root, sp, names, y = runs_dir
    runs, skipped = R.load_runs([root], "test", "S1")
    md, res = R.build_report(runs, skipped, sp, "test", n_boot=50)
    assert "Main table" in md and "random" in md and "near-duplicate" in md
    assert "random|resnet50" in res["groups"]
    json.dumps(res, default=float)


def test_load_runs_skips_region_runs(tmp_path):
    names, y = ["f0.jpg"], [0]
    lg = np.eye(K)[y]
    write_run(tmp_path, "r_full", "random", "resnet50", 0, names, y, lg)
    d2 = write_run(tmp_path, "r_nobottom", "random", "resnet50", 1, names, y, lg)
    cfg = json.loads((d2 / "config.json").read_text())
    cfg["cfg"] = {"region": "no_bottom"}
    (d2 / "config.json").write_text(json.dumps(cfg))
    runs, skipped = R.load_runs([tmp_path], "val", "S1")
    assert [r["name"] for r in runs] == ["r_full"]
    assert dict(skipped)["r_nobottom"].startswith("region run")
