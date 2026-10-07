"""Leakage-free splits (SPEC sections 3 and 4).

  grid  : cluster statistics over a grid of cosine thresholds; writes nothing but reports.
  split : random stratified split (E1) and group-aware splits (E2) for the given thresholds,
          verified exactly against all image pairs above the lowest threshold.

Usage:
  python scripts/make_splits.py grid  --emb data/interim/emb_resnet50.npz
  python scripts/make_splits.py split --emb data/interim/emb_resnet50.npz --taus 0.80,0.86,0.92

Embeddings are centred (dataset mean) and L2-normalised; similarity = cosine. No labels are used
for clustering. Clusters = connected components of the graph "cosine >= tau" (single linkage).
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "reports" / "cache384_manifest.csv.gz"
OUT_DIR = ROOT / "reports" / "splits"
SPLITS_JSON = ROOT / "splits" / "splits.json"
NAMES = ("train", "val", "test")
RATIOS = (0.70, 0.15, 0.15)
BIN = 0.002  # histogram / floor step
MAX_PAIRS = 30_000_000


def load(emb_path):
    man = pd.read_csv(MANIFEST)
    man = man[man["note"].str.startswith("kept")].set_index("cache_name")
    z = np.load(emb_path, allow_pickle=False)
    names = [str(x) for x in z["names"]]
    assert len(set(names)) == len(names), "duplicate names in embeddings"
    assert set(names) == set(man.index), "embeddings and manifest differ"
    man = man.loc[names]
    e = z["emb"].astype(np.float32)
    e -= e.mean(axis=0, keepdims=True)
    e /= np.linalg.norm(e, axis=1, keepdims=True) + 1e-12
    return names, man["class"].to_numpy(), man["orig_sha256"].to_numpy(), e


def collect(emb, floor=0.5, cap=MAX_PAIRS, block=1000):
    """All unordered pairs with cosine >= floor (floor is raised if more than `cap` pairs),
    histogram of all pair similarities (bin = BIN) and nearest-neighbour similarity per image."""
    n = len(emb)
    nb = int(round(2 / BIN))
    hist = np.zeros(nb + 1, np.int64)
    nn = np.full(n, -1.0, np.float32)
    A, B, V = [], [], []
    total = 0
    for i in range(0, n, block):
        S = emb[i:i + block] @ emb[i:].T
        r = S.shape[0]
        S[:, :r][np.tril_indices(r)] = -1.0  # keep only column > row
        nn[i:i + r] = np.maximum(nn[i:i + r], S.max(axis=1))
        nn[i:] = np.maximum(nn[i:], S.max(axis=0))
        q = np.clip(((S + 1.0) / BIN).astype(np.int32), 0, nb)
        hist += np.bincount(q.ravel(), minlength=nb + 1)
        rr, cc = np.nonzero(S >= floor)
        A.append((rr + i).astype(np.int32))
        B.append((cc + i).astype(np.int32))
        V.append(S[rr, cc].astype(np.float32))
        total += len(rr)
        while total > cap:
            floor += BIN
            a, b, v = (np.concatenate(x) for x in (A, B, V))
            m = v >= floor
            A, B, V = [a[m]], [b[m]], [v[m]]
            total = int(m.sum())
        print(f"  block {i // block + 1}/{-(-n // block)}  pairs>={floor:.3f}: {total}", flush=True)
    a, b, v = (np.concatenate(x) for x in (A, B, V))
    m = v >= floor
    return a[m], b[m], v[m], floor, hist, nn


def components(n, a, b, v, tau):
    m = v >= tau
    g = coo_matrix((np.ones(int(m.sum()), np.int8), (a[m], b[m])), shape=(n, n))
    return connected_components(g, directed=False)[1]


def cluster_stats(lab, y, tau):
    sizes = np.bincount(lab)
    df = pd.DataFrame({"c": lab, "y": y})
    ncls = df.groupby("c")["y"].nunique()
    mixed = ncls[ncls > 1].index
    return {
        "tau": tau,
        "clusters": int(len(sizes)),
        "singletons": int((sizes == 1).sum()),
        "images_in_non_singleton": int(sizes[sizes > 1].sum()),
        "largest": int(sizes.max()),
        "p99_size": float(np.percentile(sizes, 99)),
        "mixed_class_clusters": int(len(mixed)),
        "images_in_mixed": int(sizes[mixed].sum()),
    }


def group_split(lab, y, seed):
    """Greedy, class-stratified assignment of whole clusters to train/val/test.

    Clusters are taken largest first (random ties). A cluster goes to the split that leaves the
    running allocation closest to the target ratios *of everything assigned so far*, per class and
    relative to the class size. Big clusters are therefore spread over the splits in proportion
    (version 1 compared against the final per-split targets, which sent every large cluster to
    val/test; see SPEC section 11, 2026-10-06), and the many small clusters at the end correct
    the class balance."""
    classes = np.unique(y)
    ci = {c: k for k, c in enumerate(classes)}
    yk = np.array([ci[c] for c in y])
    nc, ng = len(classes), int(lab.max()) + 1
    G = np.zeros((ng, nc), np.int64)
    np.add.at(G, (lab, yk), 1)
    Nc = G.sum(axis=0).astype(np.float64)
    w = 1.0 / np.maximum(Nc, 1.0) ** 2
    r = np.asarray(RATIOS, np.float64)[:, None]
    cnt = np.zeros((3, nc))
    cum = np.zeros(nc)
    rng = np.random.default_rng(seed)
    order = np.lexsort((rng.random(ng), -G.sum(axis=1)))  # largest first, random ties
    assign = np.zeros(ng, np.int8)
    for g in order:
        gc = G[g].astype(np.float64)
        cum += gc
        dev = cnt - r * cum  # deviation of every split from its share of the running total
        best, best_c = 0, None
        for s in range(3):
            d = dev.copy()
            d[s] += gc
            c = float((d * d * w).sum())
            if best_c is None or c < best_c:
                best, best_c = s, c
        assign[g] = best
        cnt[best] += gc
    return np.array(NAMES)[assign[lab]]


def random_split(y, seed):
    idx = np.arange(len(y))
    tr, rest = train_test_split(idx, test_size=1 - RATIOS[0], stratify=y, random_state=seed)
    va, te = train_test_split(rest, test_size=0.5, stratify=y[rest], random_state=seed)
    out = np.empty(len(y), dtype="<U5")
    out[tr], out[va], out[te] = "train", "val", "test"
    return out


def table(df):
    cols = [str(c) for c in df.columns]
    rows = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        rows.append("| " + " | ".join(str(x) for x in r.tolist()) + " |")
    return "\n".join(rows)


def counts(split, y):
    return pd.crosstab(pd.Series(y, name="class"), pd.Series(split, name="split")).reindex(columns=list(NAMES), fill_value=0)


def stage_grid(args):
    names, y, sha, e = load(args.emb)
    a, b, v, floor, hist, nn = collect(e, floor=args.floor)
    print(f"stored pairs >= {floor:.3f}: {len(v)}")
    q = np.percentile(nn, [1, 5, 25, 50, 75, 95, 99])
    taus = [round(t, 3) for t in np.arange(max(floor, 0.5), 0.99, 0.02) if t >= floor - 1e-9]
    rows = [cluster_stats(components(len(y), a, b, v, t), y, t) for t in taus]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "grid_stats.csv", index=False)
    pairs_above = np.cumsum(hist[::-1])[::-1] // 1
    with open(OUT_DIR / "grid_report.md", "w", encoding="utf-8") as f:
        f.write("# Threshold grid (cosine on centred ResNet-50 features)\n\n")
        f.write(f"Images: {len(y)}. Pairs stored down to cosine {floor:.3f}: {len(v)}.\n\n")
        f.write("Nearest-neighbour cosine per image, percentiles 1/5/25/50/75/95/99: "
                + ", ".join(f"{x:.3f}" for x in q) + "\n\n")
        f.write(table(df.assign(p99_size=df["p99_size"].round(1))) + "\n")
    print(df.assign(p99_size=df["p99_size"].round(1)).to_string(index=False))
    print("NN cosine percentiles 1/5/25/50/75/95/99:", np.round(q, 3))


def stage_split(args):
    taus = sorted(float(t) for t in args.taus.split(","))
    names, y, sha, e = load(args.emb)
    n = len(y)
    a, b, v, floor, hist, nn = collect(e, floor=min(taus))
    assert floor <= min(taus) + 1e-9, (
        f"pair cap reached: lowest usable threshold is {floor:.3f}, requested {min(taus)}")
    emb_hash = hashlib.sha256(Path(args.emb).read_bytes()).hexdigest()
    res = {"random": random_split(y, args.seed)}
    clusters = {}
    for t in taus:
        lab = components(n, a, b, v, t)
        clusters[t] = lab
        res[t] = group_split(lab, y, args.seed)

    lines = ["# Splits report\n", f"Seed {args.seed}, ratios {RATIOS}, images {n}, embeddings sha256 `{emb_hash[:16]}`.\n"]
    ok = True
    for key, sp in res.items():
        title = "E1 random stratified" if key == "random" else f"E2 group-aware, tau={key}"
        lines.append(f"\n## {title}\n")
        c = counts(sp, y)
        c["val+test min"] = c[["val", "test"]].min(axis=1)
        tot = c[list(NAMES)].sum()
        lines.append("Share of images: " + ", ".join(f"{s} {tot[s] / n:.3f}" for s in NAMES) + "\n")
        lines.append(table(c.reset_index()) + "\n")
        # exact leakage check on all stored pairs
        for t in taus:
            cross = int((sp[a[v >= t]] != sp[b[v >= t]]).sum())
            if key == "random":
                lines.append(f"- pairs with cosine >= {t} that cross splits: **{cross}** (of {int((v >= t).sum())})")
            elif key == t:
                lines.append(f"- pairs with cosine >= {t} that cross splits: **{cross}**")
                ok &= cross == 0
                lab = clusters[t]
                cs = pd.DataFrame({"c": lab, "s": sp}).groupby("c")["s"].nunique()
                bad = int((cs > 1).sum())
                lines.append(f"- clusters spanning more than one split: **{bad}**")
                ok &= bad == 0
                sz = np.bincount(lab)[lab]
                lines.append("- images in clusters of size >= 10, by split: " + ", ".join(
                    f"{s_} {100 * float((sz[sp == s_] >= 10).mean()):.1f}%" for s_ in NAMES)
                    + f"; largest cluster by split: " + ", ".join(f"{s_} {int(sz[sp == s_].max())}" for s_ in NAMES))
                st = cluster_stats(lab, y, t)
                st = cluster_stats(lab, y, t)
                lines.append("- clusters: " + ", ".join(f"{k}={v_}" for k, v_ in st.items() if k != "tau"))
    assert ok, "leakage check failed"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "splits_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    SPLITS_JSON.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "meta": {
            "seed": args.seed, "ratios": dict(zip(NAMES, RATIOS)), "n": n,
            "similarity": "cosine on dataset-centred, L2-normalised ResNet-50 (ImageNet) features",
            "clustering": "connected components of the graph cosine >= tau",
            "taus": taus, "embeddings_sha256": emb_hash,
        },
        "files": names,
        "class": [str(x) for x in y],
        "random": res["random"].tolist(),
        "group": {f"{t:g}": {"cluster": clusters[t].tolist(), "split": res[t].tolist()} for t in taus},
    }
    SPLITS_JSON.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    print("wrote", SPLITS_JSON, "and", OUT_DIR / "splits_report.md")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["grid", "split"])
    ap.add_argument("--emb", required=True)
    ap.add_argument("--taus", default="")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--floor", type=float, default=0.5)
    args = ap.parse_args()
    stage_grid(args) if args.stage == "grid" else stage_split(args)
