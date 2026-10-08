import hashlib

import pytest

from semcls import fetch

PAYLOAD = b"x" * 5000
SHA = hashlib.sha256(PAYLOAD).hexdigest()


@pytest.fixture
def src(tmp_path):
    p = tmp_path / "remote.bin"
    p.write_bytes(PAYLOAD)
    return p.as_uri()


FILE_OK = ("file",)


def test_download_and_verify(tmp_path, src):
    dest = tmp_path / "m" / "ckpt.pt"
    out = fetch.fetch_verified(src, dest, SHA, len(PAYLOAD), allowed_schemes=FILE_OK)
    assert out == dest and dest.read_bytes() == PAYLOAD
    assert [p.name for p in dest.parent.iterdir()] == ["ckpt.pt"]


def test_hash_mismatch_leaves_nothing(tmp_path, src):
    dest = tmp_path / "m" / "ckpt.pt"
    with pytest.raises(fetch.DownloadError, match="sha256"):
        fetch.fetch_verified(src, dest, "0" * 64, len(PAYLOAD), allowed_schemes=FILE_OK)
    assert list(dest.parent.iterdir()) == []


def test_size_cap(tmp_path, src):
    dest = tmp_path / "m" / "ckpt.pt"
    with pytest.raises(fetch.DownloadError, match="larger"):
        fetch.fetch_verified(src, dest, SHA, 100, allowed_schemes=FILE_OK)
    assert list(dest.parent.iterdir()) == []


def test_only_https_by_default(tmp_path, src):
    with pytest.raises(fetch.DownloadError, match="scheme"):
        fetch.fetch_verified(src, tmp_path / "c.pt", SHA, len(PAYLOAD))
    with pytest.raises(fetch.DownloadError, match="scheme"):
        fetch.fetch_verified("http://example.org/c.pt", tmp_path / "c.pt", SHA, len(PAYLOAD))


def test_existing_file_is_not_downloaded(tmp_path):
    dest = tmp_path / "ckpt.pt"
    dest.write_bytes(b"local")
    assert fetch.ensure_checkpoint(dest, SHA, url="https://invalid.invalid/x") == dest
    assert dest.read_bytes() == b"local"


def test_fallback_when_target_dir_unusable(tmp_path, src):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a directory")
    out = fetch.ensure_checkpoint(blocker / "ckpt.pt", SHA, url=src, max_bytes=len(PAYLOAD),
                                  fallback_dir=tmp_path / "fb", allowed_schemes=FILE_OK)
    assert out == tmp_path / "fb" / "ckpt.pt" and out.read_bytes() == PAYLOAD


def test_constants_match_the_recorded_checkpoint():
    assert fetch.CHECKPOINT_URL.startswith("https://huggingface.co/")
    assert fetch.CHECKPOINT_BYTES == 111_365_791
