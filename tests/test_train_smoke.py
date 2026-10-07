import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from PIL import Image  # noqa: E402

from semcls.evaltest import evaluate_test  # noqa: E402
from semcls.train import run  # noqa: E402


@pytest.fixture
def tiny(tmp_path):
    rng = np.random.default_rng(0)
    imgs = tmp_path / "img"
    imgs.mkdir()
    files, classes = [], []
    for c, cname in enumerate(["a", "b", "c"]):
        for i in range(10):
            f = f"{cname}/{cname}{i}__{i:08d}.jpg"
            (imgs / f).parent.mkdir(exist_ok=True)
            base = np.full((96, 128, 3), 60 * (c + 1), np.uint8)
            Image.fromarray(np.clip(base + rng.integers(0, 30, base.shape), 0, 255).astype(np.uint8)).save(imgs / f)
            files.append(f)
            classes.append(cname)
    split = ["train"] * 6 + ["val"] * 2 + ["test"] * 2
    sp = {"files": files, "class": classes, "random": split * 3, "group": {},
          "meta": {"n": 30}}
    p = tmp_path / "splits.json"
    p.write_text(json.dumps(sp))
    return p, imgs, tmp_path / "runs"


def test_train_then_test_once(tiny):
    splits, imgs, runs = tiny
    ov = dict(epochs=2, batch=6, accum=1, workers=0, hw=(96, 128), pretrained=False, patience=0)
    out = run("random", "resnet50", 0, splits, imgs, runs, overrides=ov, device="cpu")
    assert (out / "done.flag").exists() and (out / "best.pt").exists()
    v = np.load(out / "val_logits.npz")
    assert v["logits"].shape == (6, 3) and len(v["names"]) == 6
    assert not (out / "test_logits.npz").exists()  # training never produced test outputs
    path = evaluate_test(out, splits, imgs, device="cpu", workers=0)
    t = np.load(path)
    assert t["logits"].shape == (6, 3)
    with pytest.raises(RuntimeError):
        evaluate_test(out, splits, imgs, device="cpu", workers=0)
    # a finished run is skipped, not retrained
    assert run("random", "resnet50", 0, splits, imgs, runs, overrides=ov, device="cpu") == out


def test_convnext_forward_backward(tiny):
    splits, imgs, runs = tiny
    ov = dict(epochs=1, batch=4, accum=2, workers=0, hw=(96, 128), pretrained=False, patience=0)
    out = run("random", "convnext_tiny", 1, splits, imgs, runs, overrides=ov, device="cpu")
    assert json.loads((out / "config.json").read_text())["epochs_run"] == 1


def test_region_run_name_config_and_test_eval(tiny):
    splits, imgs, runs = tiny
    ov = dict(epochs=1, batch=4, accum=1, workers=0, hw=(96, 128), pretrained=False, patience=0, region="no_bottom")
    out = run("random", "resnet50", 0, splits, imgs, runs, overrides=ov, device="cpu")
    assert out.name == "random__resnet50__no_bottom__s0"
    assert json.loads((out / "config.json").read_text())["cfg"]["region"] == "no_bottom"
    t = np.load(evaluate_test(out, splits, imgs, device="cpu", workers=0))
    assert t["logits"].shape == (6, 3)
