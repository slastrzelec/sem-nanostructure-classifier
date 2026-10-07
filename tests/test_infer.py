import io
import json
import os
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from PIL import Image  # noqa: E402

from semcls import infer as I  # noqa: E402
from semcls.data import SemDataset  # noqa: E402
from semcls.models import build  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "data" / "demo_samples"
CKPT = ROOT / "data" / "models" / "convnext_tiny_group094_s2_best.pt"


def _png(w=64, h=48, fmt="PNG", seed=0):
    rng = np.random.default_rng(seed)
    buf = io.BytesIO()
    Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8)).save(buf, format=fmt)
    return buf.getvalue()


@pytest.fixture(scope="module")
def clf():
    torch.manual_seed(0)
    model, _ = build(I.BACKBONE, len(I.CLASSES), pretrained=False)  # random weights: no checkpoint needed
    return I.Classifier(model)


def test_tensor_from_bytes_equals_semdataset_from_path():
    files = sorted((ROOT / "app" / "samples").rglob("*.jpg"))
    if not files:
        pytest.skip("no demo samples")
    f = files[0]
    a = I.Classifier.to_tensor(f.read_bytes())
    b, _ = SemDataset([f], [0], train=False)[0]
    assert a.dtype == torch.uint8 and tuple(a.shape) == (3, 384, 512)
    assert torch.equal(a, b)


def test_wrong_hash_refused(tmp_path):
    p = tmp_path / "fake.pt"
    p.write_bytes(b"not a checkpoint")
    with pytest.raises(I.ChecksumError):
        I.Classifier.load(p)


@pytest.mark.parametrize("data,msg", [
    (b"", "empty"),
    (b"x" * (I.MAX_BYTES + 1), "larger"),
    (b"this is not an image", "cannot read"),
    (_png(fmt="GIF"), "unsupported"),
    (_png(fmt="BMP"), "unsupported"),
    (_png(w=I.MAX_SIDE + 1, h=40), "outside"),
    (_png(w=8, h=8), "outside"),
    (_png()[:-40], "cannot read"),   # truncated PNG
])
def test_upload_rejected(data, msg):
    with pytest.raises(I.UploadError, match=msg):
        I.validate_upload(data)


def test_upload_accepted_png_and_jpeg():
    I.validate_upload(_png())
    I.validate_upload(_png(fmt="JPEG"))


def _logit_for_confidence(conf, k=10, T=I.TEMPERATURE):
    """Logits whose calibrated top probability equals conf (one class high, the rest 0)."""
    a = T * np.log(conf * (k - 1) / (1 - conf))
    z = np.zeros(k)
    z[3] = a
    return z


def test_uncertain_flag_around_theta():
    for conf, flag in ((I.THETA - 0.002, True), (I.THETA + 0.002, False)):
        p = I.calibrated_probs(_logit_for_confidence(conf))
        assert abs(p.max() - conf) < 1e-9
        assert I.is_uncertain(p.max()) is flag
    assert I.is_uncertain(I.THETA) is False  # c >= theta is accepted


def test_calibration_keeps_argmax_and_sums_to_one():
    z = np.random.default_rng(1).normal(0, 5, 10)
    p = I.calibrated_probs(z)
    assert p.argmax() == z.argmax() and abs(p.sum() - 1) < 1e-12
    assert p.max() < np.exp(z - z.max()).max() / np.exp(z - z.max()).sum() + 1e-12  # softer than T = 1


def test_no_file_written_during_request(clf, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
    before = sorted(os.listdir(tmp_path))
    r = clf.predict(_png(w=300, h=200, fmt="JPEG"), with_cam=True)
    assert sorted(os.listdir(tmp_path)) == before == []
    assert len(r.top) == 3 and abs(r.probs.sum() - 1) < 1e-9


def test_grad_cam_shape_range_and_finite(clf):
    r = clf.predict(_png(w=512, h=384), with_cam=True)
    assert r.cam.shape == (384, 512) and np.isfinite(r.cam).all()
    assert r.cam.min() >= 0 and r.cam.max() <= 1.0 + 1e-9
    assert r.image.shape == (384, 512, 3) and r.image.dtype == np.uint8


def test_cam_does_not_change_the_logits(clf):
    data = _png(w=512, h=384)
    a = clf.predict(data, with_cam=False).probs
    b = clf.predict(data, with_cam=True).probs
    assert np.allclose(a, b, atol=1e-6)


@pytest.mark.skipif(not CKPT.exists() or not (SAMPLES / "samples_val.json").exists(), reason="checkpoint or samples missing")
def test_checkpoint_reproduces_stored_val_logits():
    clf = I.Classifier.load(CKPT)
    S = json.loads((SAMPLES / "samples_val.json").read_text())
    out = np.stack([clf.logits((SAMPLES / s["name"]).read_bytes()) for s in S])
    ref = np.array([s["logits"] for s in S])
    assert np.array_equal(out.argmax(1), ref.argmax(1))
    assert np.abs(out - ref).max() < 0.05   # fp32 CPU vs fp16 GPU (measured 0.0105)
