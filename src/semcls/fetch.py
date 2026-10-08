"""Download of the demo checkpoint with an integrity check (SPEC section 11, deployment entry 2026-10-08).

Torch-free on purpose. The file is written to a temporary name, size-capped, hashed, and renamed into place only
when the sha256 matches; otherwise nothing is left behind."""
import hashlib
import os
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

CHECKPOINT_URL = ("https://huggingface.co/slastrzelec/sem-nanostructure-classifier-convnext-tiny/"
                  "resolve/main/convnext_tiny_group094_s2_best.pt")
CHECKPOINT_BYTES = 111_365_791


class DownloadError(RuntimeError):
    pass


def fetch_verified(url, dest, sha256, max_bytes, timeout=60, allowed_schemes=("https",), chunk=1 << 20):
    scheme = urlparse(url).scheme
    if scheme not in allowed_schemes:
        raise DownloadError(f"scheme '{scheme}' is not allowed")
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dest.parent, suffix=".part")
    os.close(fd)
    h, n = hashlib.sha256(), 0
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r, open(tmp, "wb") as f:
            while True:
                b = r.read(chunk)
                if not b:
                    break
                n += len(b)
                if n > max_bytes:
                    raise DownloadError("download is larger than the expected checkpoint")
                h.update(b)
                f.write(b)
        if h.hexdigest() != sha256:
            raise DownloadError("sha256 of the downloaded file differs from the recorded one; discarded")
        os.replace(tmp, dest)
    except urllib.error.URLError as e:
        raise DownloadError(f"download failed: {e.reason}") from e
    finally:
        Path(tmp).unlink(missing_ok=True)
    return dest


def ensure_checkpoint(path, sha256, url=CHECKPOINT_URL, max_bytes=CHECKPOINT_BYTES, fallback_dir=None,
                      allowed_schemes=("https",)):
    """Return the path of the checkpoint, downloading it when absent. An existing file is returned unchanged;
    Classifier.load checks its hash."""
    path = Path(path)
    if path.exists():
        return path
    try:
        return fetch_verified(url, path, sha256, max_bytes, allowed_schemes=allowed_schemes)
    except OSError as e:
        if isinstance(e, urllib.error.URLError):
            raise
        fb = Path(fallback_dir) if fallback_dir else Path(tempfile.gettempdir()) / "semcls"
        alt = fb / path.name
        if alt.exists():
            return alt
        return fetch_verified(url, alt, sha256, max_bytes, allowed_schemes=allowed_schemes)
