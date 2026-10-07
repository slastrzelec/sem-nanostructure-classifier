"""Inference for the demo: checked upload, calibrated confidence, "uncertain" flag, optional Grad-CAM.

Constants come from SPEC section 11 (checkpoint 2026-10-07, T and theta from E4, 2026-10-07)."""
import hashlib
import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

from .data import SemDataset
from .gradcam import grad_cam
from .models import build
from .train import prep

CLASSES = ["Biological", "Fibres", "Films_Coated_Surface", "MEMS_devices_and_electrodes", "Nanowires",
           "Particles", "Patterned_surface", "Porous_Sponge", "Powder", "Tips"]
BACKBONE = "convnext_tiny"
CHECKPOINT_SHA256 = "7e856feee47b86c3fb97d9a581bd7a8c9f1bc24454654e82330003fd0f3a194f"
TEMPERATURE = 3.283   # E4a, fitted on val for this checkpoint
THETA = 0.9150        # E4b, threshold on the calibrated top confidence (accuracy of accepted val images >= 0.99)
MAX_BYTES = 10 * 1024 * 1024
MAX_SIDE = 4096
ALLOWED_FORMATS = ("JPEG", "PNG")


class UploadError(ValueError):
    """The uploaded file is not accepted (size, type, dimensions or not decodable)."""


class ChecksumError(RuntimeError):
    """The checkpoint file does not match the recorded sha256."""


def sha256_of(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def validate_upload(data):
    """Raises UploadError unless `data` is a PNG/JPEG of at most MAX_BYTES and MAX_SIDE x MAX_SIDE that decodes."""
    if not isinstance(data, (bytes, bytearray)) or len(data) == 0:
        raise UploadError("empty upload")
    if len(data) > MAX_BYTES:
        raise UploadError(f"file larger than {MAX_BYTES // (1024 * 1024)} MB")
    try:
        im = Image.open(io.BytesIO(data))
        fmt, (w, h) = im.format, im.size  # header only, nothing decoded yet
        if fmt not in ALLOWED_FORMATS:
            raise UploadError(f"unsupported format {fmt}; use PNG or JPEG")
        if w > MAX_SIDE or h > MAX_SIDE or w < 32 or h < 32:
            raise UploadError(f"image size {w}x{h} outside 32..{MAX_SIDE} pixels per side")
        im.load()
    except UploadError:
        raise
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError, Image.DecompressionBombError) as e:
        raise UploadError(f"cannot read the image ({type(e).__name__})") from e


def calibrated_probs(logits, T=TEMPERATURE):
    z = np.asarray(logits, dtype=np.float64) / T
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def is_uncertain(confidence, theta=THETA):
    return bool(confidence < theta)


@dataclass
class Prediction:
    probs: np.ndarray          # calibrated, one per class
    top: list                  # [(class name, probability)] best first
    confidence: float
    uncertain: bool
    image: np.ndarray          # the 384 x 512 RGB uint8 image the model saw
    cam: np.ndarray = None     # 384 x 512 in [0, 1] (Grad-CAM of the predicted class) or None


class Classifier:
    def __init__(self, model, device="cpu"):
        self.model = model.to(device).eval()
        self.device = device

    @classmethod
    def load(cls, path, expected_sha256=CHECKPOINT_SHA256, device="cpu"):
        path = Path(path)
        if sha256_of(path) != expected_sha256:
            raise ChecksumError(f"{path.name}: sha256 differs from the checkpoint recorded in SPEC; refusing to load")
        model, _ = build(BACKBONE, len(CLASSES), pretrained=False)
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        return cls(model, device)

    @staticmethod
    def to_tensor(data):
        """Same preprocessing as training/evaluation (SemDataset, validation branch), from bytes, in memory."""
        validate_upload(data)
        x, _ = SemDataset([io.BytesIO(bytes(data))], [0], train=False)[0]
        return x

    def logits(self, data):
        x = self.to_tensor(data)[None]
        with torch.no_grad():
            return self.model(prep(x, self.device)).float().cpu().numpy()[0]

    def predict(self, data, top_k=3, with_cam=False):
        x = self.to_tensor(data)[None]
        xn = prep(x, self.device)
        cam = None
        if with_cam:
            z, cam = grad_cam(self.model, xn)
            z = z.numpy()
        else:
            with torch.no_grad():
                z = self.model(xn).float().cpu().numpy()[0]
        p = calibrated_probs(z)
        order = np.argsort(-p)[:top_k]
        conf = float(p.max())
        return Prediction(probs=p, top=[(CLASSES[i], float(p[i])) for i in order], confidence=conf,
                          uncertain=is_uncertain(conf), image=x[0].permute(1, 2, 0).numpy().copy(), cam=cam)
