#!/usr/bin/env python3
"""E0 step 1 - stream images straight from the downloaded zip and extract per-image features.

Raw images are never extracted to disk. For every archive (.tar inside the zip) we record:
  * archive-level MD5 + SHA-256 (integrity)
  * per image: sha256 of the file bytes, size, mode, dHash, pHash, per-row mean/std (info-bar detection),
    a 64x48 grayscale thumbnail (for later duplicate analysis).
Outputs go to data/interim/e0/ (git-ignored).
"""
import argparse, hashlib, io, json, tarfile, time, zipfile
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.fft import dct

H_ROWS = 768
THUMB = (64, 48)


class HashingReader(io.RawIOBase):
    def __init__(self, raw):
        self.raw, self.n = raw, 0
        self.md5, self.sha = hashlib.md5(), hashlib.sha256()

    def readable(self):
        return True

    def readinto(self, b):
        data = self.raw.read(len(b))
        n = len(data)
        b[:n] = data
        self.md5.update(data); self.sha.update(data); self.n += n
        return n

    def drain(self):
        while True:
            d = self.raw.read(1 << 20)
            if not d:
                break
            self.md5.update(d); self.sha.update(d); self.n += len(d)


def pack(bits):
    b = np.zeros(64, dtype=np.uint8)
    b[:len(bits)] = bits
    return int.from_bytes(np.packbits(b).tobytes(), "big")


def features(data):
    im = Image.open(io.BytesIO(data))
    w, h = im.size
    mode = im.mode
    g = im.convert("L")
    a = np.asarray(g, dtype=np.float32)
    d = np.asarray(g.resize((9, 8), Image.BILINEAR), dtype=np.float32)
    dh = pack((d[:, 1:] > d[:, :-1]).flatten())
    p = np.asarray(g.resize((32, 32), Image.BILINEAR), dtype=np.float32)
    c = dct(dct(p, axis=0, norm="ortho"), axis=1, norm="ortho")[:8, :8].flatten()[1:]
    ph = pack(c > np.median(c))
    rm = np.full(H_ROWS, np.nan, np.float16); rs = np.full(H_ROWS, np.nan, np.float16)
    k = min(h, H_ROWS)
    rm[:k] = a.mean(1)[:k]; rs[:k] = a.std(1)[:k]
    thumb = np.asarray(g.resize(THUMB, Image.BILINEAR), dtype=np.uint8)
    gray_rgb = False
    if mode == "RGB":
        s = np.asarray(im.resize((64, 48)), dtype=np.int16)
        gray_rgb = bool(np.abs(s[..., 0] - s[..., 1]).max() <= 2 and np.abs(s[..., 1] - s[..., 2]).max() <= 2)
    return w, h, mode, gray_rgb, dh, ph, rm, rs, thumb


def process(zpath, member, out):
    stem = member[:-4]
    dst = out / f"e0_{stem}.npz"
    if dst.exists():
        print("skip", member, flush=True); return
    t0 = time.time()
    names, sha, nb, W, Hh, modes, grays, dhs, phs, rms, rss, thumbs = ([] for _ in range(12))
    with zipfile.ZipFile(zpath) as z, z.open(member) as raw:
        hr = HashingReader(raw)
        with tarfile.open(fileobj=io.BufferedReader(hr, 1 << 20), mode="r|") as t:
            for m in t:
                if not m.isfile():
                    continue
                data = t.extractfile(m).read()
                try:
                    w, h, mode, gr, dh, ph, rm, rs, th = features(data)
                except Exception as e:
                    print("ERR", m.name, e, flush=True); continue
                names.append(m.name); sha.append(hashlib.sha256(data).hexdigest()); nb.append(len(data))
                W.append(w); Hh.append(h); modes.append(mode); grays.append(gr)
                dhs.append(dh); phs.append(ph); rms.append(rm); rss.append(rs); thumbs.append(th)
                if len(names) % 500 == 0:
                    print(f"{member}: {len(names)} imgs, {time.time()-t0:.0f}s", flush=True)
        hr.drain()
    np.savez_compressed(
        dst, names=np.array(names), sha256=np.array(sha), nbytes=np.array(nb), w=np.array(W), h=np.array(Hh),
        mode=np.array(modes), gray_rgb=np.array(grays), dhash=np.array(dhs, dtype=np.uint64),
        phash=np.array(phs, dtype=np.uint64), rowmean=np.stack(rms), rowstd=np.stack(rss), thumb=np.stack(thumbs))
    info = dict(archive=member, images=len(names), bytes=hr.n, md5=hr.md5.hexdigest(), sha256=hr.sha.hexdigest(),
                seconds=round(time.time() - t0, 1))
    (out / f"e0_{stem}.archive.json").write_text(json.dumps(info, indent=1))
    print("done", info, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("members", nargs="+")
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for mem in a.members:
        process(a.zip, mem, Path(a.out))
