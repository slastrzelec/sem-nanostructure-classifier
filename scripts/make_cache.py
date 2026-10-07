#!/usr/bin/env python3
"""Build the 512x384 JPEG cache (short side 384) straight from the downloaded zip.

* exact duplicates (same SHA-256, found in E0) are dropped - the first file (sorted by name) is kept;
* every cache file is named <Class>/<stem>__<sha8>.jpg and lists its original SHA-256 in the manifest;
* output: data/cache384/<Class>_384[.partIofN].tar + manifest CSV (one row per kept or dropped image).
Raw images are never extracted to disk. Usage: make_cache.py --zip Z Class.tar [--part I --of N]
"""
import argparse, csv, hashlib, io, tarfile, time, zipfile
from pathlib import Path
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "cache384"
SIZE = (512, 384)
QUALITY = 95


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("member")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cls = a.member[:-4]
    tag = "" if a.of == 1 else f".part{a.part}of{a.of}"
    dst = OUT / f"{cls}_384{tag}.tar"
    man = OUT / f"{cls}_384{tag}.manifest.csv"
    if dst.exists():
        print("skip", dst.name); return
    e0 = pd.read_csv(ROOT / "reports" / "e0" / "e0_per_image_features.csv.gz", usecols=["names", "sha256", "cls", "w", "h"])
    e0 = e0[e0.cls == cls].sort_values("names")
    first = e0.drop_duplicates("sha256", keep="first")
    keep = set(first.names)
    dup_of = dict(zip(e0.sha256, first.names))
    sha_e0 = dict(zip(e0.names, e0.sha256))
    t0 = time.time(); rows = []; k = 0; kept = 0
    tmp = dst.with_suffix(".tar.tmp")
    with zipfile.ZipFile(a.zip) as z, z.open(a.member) as raw, \
            tarfile.open(fileobj=io.BufferedReader(raw, 1 << 20), mode="r|") as t, tarfile.open(tmp, "w") as out:
        for m in t:
            if not m.isfile():
                continue
            k += 1
            if (k - 1) % a.of != a.part:
                continue
            data = t.extractfile(m).read()
            sha = hashlib.sha256(data).hexdigest()
            assert sha_e0[m.name] == sha, f"SHA mismatch vs E0 for {m.name}"
            if m.name not in keep:
                rows.append([cls, m.name, sha, "", "dropped_exact_duplicate_of:" + dup_of[sha]]); continue
            im = Image.open(io.BytesIO(data)).convert("RGB")
            w, h = im.size
            im = im.resize(SIZE, Image.LANCZOS)
            buf = io.BytesIO(); im.save(buf, "JPEG", quality=QUALITY, subsampling=0)
            name = f"{cls}/{Path(m.name).stem}__{sha[:8]}.jpg"
            ti = tarfile.TarInfo(name); ti.size = buf.tell(); ti.mtime = 0
            buf.seek(0); out.addfile(ti, buf)
            rows.append([cls, m.name, sha, name, f"kept;orig_size={w}x{h}"]); kept += 1
            if kept % 500 == 0:
                print(f"{a.member} part{a.part}: {kept} kept, {time.time()-t0:.0f}s", flush=True)
    with open(man, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f); wr.writerow(["class", "orig_name", "orig_sha256", "cache_name", "note"]); wr.writerows(rows)
    tmp.rename(dst)
    print("done", dst.name, "kept", kept, "rows", len(rows), "MB", round(dst.stat().st_size / 1e6, 1), f"{time.time()-t0:.0f}s", flush=True)


main()
