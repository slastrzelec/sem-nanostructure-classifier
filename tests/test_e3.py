import json

import numpy as np
import pytest

from semcls import e3 as E

K = 3
CLASSES = ["a", "b", "c"]


def sp_for(n):
    cl = np.arange(n)
    cl[:6] = 0
    return dict(files=[f"f{i}.jpg" for i in range(n)], group={"0.94": dict(cluster=cl.tolist())})


def write(root, region, seed, y, logits, sha="S", part="val", done=True, variant="group:0.94", backbone="convnext_tiny"):
    d = root / f"{variant}__{backbone}__{region}__s{seed}".replace(":", "")
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps(dict(variant=variant, backbone=backbone, seed=seed, classes=CLASSES,
                                                   splits_sha256=sha, cfg=dict(region=region))))
    if done:
        (d / "done.flag").write_text("ok")
    n = len(y)
    np.savez(d / f"{part}_logits.npz", names=np.array([f"f{i}.jpg" for i in range(n)]), y=np.array(y), logits=logits)


def logits(y, acc, rng):
    lg = rng.normal(size=(len(y), K)) * 0.1
    pred = np.where(rng.random(len(y)) < acc, y, (np.asarray(y) + 1) % K)
    lg[np.arange(len(y)), pred] += 5
    return lg


@pytest.fixture
def setup(tmp_path):
    n = 150
    y = [i % K for i in range(n)]
    rng = np.random.default_rng(0)
    for region, acc in [("full", 0.95), ("no_bottom", 0.94), ("no_top", 0.90), ("bottom_strip", 0.6), ("top_strip", 0.6)]:
        for s in range(3):
            write(tmp_path, region, s, y, logits(y, acc, rng))
    return tmp_path, sp_for(n), y


def test_load_filters(setup):
    root, sp, y = setup
    rng = np.random.default_rng(1)
    write(root, "no_bottom", 7, y, logits(y, 0.9, rng), sha="OTHER")           # other splits file
    write(root, "no_top", 8, y, logits(y, 0.9, rng), variant="random")           # other variant
    write(root, "top_strip", 9, y, logits(y, 0.9, rng), backbone="resnet50")     # other backbone
    write(root, "full", 5, y, logits(y, 0.9, rng), done=False)                  # unfinished
    g = E.load_e3_runs([root], "val", "S")
    assert sorted(g) == sorted(E.ORDER) and all(len(v) == 3 for v in g.values())
    assert E.load_e3_runs([root], "test", "S") == {}


def test_duplicates_rejected(setup):
    root, sp, y = setup
    other = root.parent / (root.name + "_b")
    write(other, "full", 0, y, np.zeros((len(y), K)))
    with pytest.raises(ValueError, match="duplicate"):
        E.load_e3_runs([root, other], "val", "S")


def test_different_images_rejected(setup):
    root, sp, y = setup
    g = E.load_e3_runs([root], "val", "S")
    g["no_top"][0]["names"] = list(reversed(g["no_top"][0]["names"]))
    with pytest.raises(ValueError, match="different images"):
        E.check_same(g)


def test_report_contrasts_signs_and_intervals(setup):
    root, sp, y = setup
    g = E.load_e3_runs([root], "val", "S")
    md, res = E.build_e3(g, sp, "val", n_boot=200)
    c = {(x["a"], x["b"]): x for x in res["contrasts"]}
    d = c[("full", "no_top")]["acc"]
    assert d["point"] > 0 and d["img"][0] < d["point"] < d["img"][1]
    assert c[("bottom_strip", "top_strip")]["acc"]["img"][0] < 0 < c[("bottom_strip", "top_strip")]["acc"]["img"][1]
    assert "Paired differences" in md and "E3a bottom strip only" in md
    assert res["regions"]["full"]["macro_f1"] and len(res["regions"]["full"]["macro_f1"]) == 3
    json.dumps(res)
