"""Access to splits/splits.json. Test files are unreachable unless explicitly requested."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def load_splits(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def class_names(sp):
    return sorted(set(sp["class"]))


def canon_variant(variant):
    """'group:0.90' and 'group:0.9' are the same variant; returns the canonical spelling."""
    if variant == "random":
        return variant
    kind, _, tau = variant.partition(":")
    if kind != "group":
        raise KeyError(f"unknown split variant {variant!r}")
    return f"group:{float(tau):g}"


def split_vector(sp, variant):
    """variant: 'random' or 'group:<tau>' (e.g. 'group:0.94')."""
    variant = canon_variant(variant)
    if variant == "random":
        return sp["random"]
    tau = variant.partition(":")[2]
    if tau not in sp["group"]:
        raise KeyError(f"unknown split variant {variant!r}; have random and "
                       + ", ".join("group:" + t for t in sp["group"]))
    return sp["group"][tau]["split"]


def get_split(sp, variant, parts, allow_test=False):
    """Return {part: (file names, integer labels)} for the requested parts."""
    if "test" in parts and not allow_test:
        raise PermissionError("the test split is only available to the final test evaluation")
    vec = np.array(split_vector(sp, variant))
    classes = class_names(sp)
    cid = {c: i for i, c in enumerate(classes)}
    y = np.array([cid[c] for c in sp["class"]], dtype=np.int64)
    files = np.array(sp["files"])
    out = {}
    for p in parts:
        if p not in ("train", "val", "test"):
            raise ValueError(p)
        m = vec == p
        out[p] = (files[m].tolist(), y[m])
    return out


def index_images(root):
    """Map image basename -> path for all .jpg files under root."""
    by_base = {}
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if fn.lower().endswith(".jpg"):
                if fn in by_base:
                    raise RuntimeError(f"duplicate basename {fn}")
                by_base[fn] = os.path.join(dp, fn)
    return by_base


def resolve_paths(files, by_base):
    paths = [by_base.get(os.path.basename(f)) for f in files]
    missing = [f for f, p in zip(files, paths) if p is None]
    if missing:
        raise FileNotFoundError(f"{len(missing)} images not found, e.g. {missing[:3]}")
    return paths
