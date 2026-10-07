#!/usr/bin/env python3
"""E0 step 2 - turn the extracted features into the E0 data-audit report (numbers only come from this run)."""
import glob, hashlib, json, platform, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / "data" / "interim" / "e0"
OUT = ROOT / "reports" / "e0"
OUT.mkdir(parents=True, exist_ok=True)
BRIGHT, DARK, MIN_RUN = 225.0, 30.0, 20

# ---------------------------------------------------------------- load
parts, thumbs, rowm, rows_ = [], [], [], []
archives = []
for f in sorted(glob.glob(str(IN / "e0_*.npz"))):
    z = np.load(f)
    parts.append(pd.DataFrame({k: z[k] for k in ["names", "sha256", "nbytes", "w", "h", "mode", "gray_rgb", "dhash", "phash"]}))
    thumbs.append(z["thumb"]); rowm.append(z["rowmean"]); rows_.append(z["rowstd"])
for f in sorted(glob.glob(str(IN / "e0_*.archive.json"))):
    archives.append(json.loads(Path(f).read_text()))
df = pd.concat(parts, ignore_index=True)
thumb = np.concatenate(thumbs).reshape(len(df), -1).astype(np.float32)
rm = np.concatenate(rowm).astype(np.float32); rs = np.concatenate(rows_).astype(np.float32)
df["cls"] = df.names.str.split("/").str[0]
df["code"] = df.names.str.split("/").str[1].str.split("_").str[0]
df["stem"] = df.names.str.split("/").str[1].str.rsplit(".", n=1).str[0].str.split("_", n=1).str[1]
N = len(df)
res = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "n_images": int(N),
       "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
       "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}

# ---------------------------------------------------------------- classes
cc = df.cls.value_counts().rename_axis("class").reset_index(name="n")
cc["share_%"] = (100 * cc.n / N).round(2)
res["class_counts"] = dict(zip(cc["class"], cc.n.astype(int)))
res["imbalance_max_over_min"] = round(float(cc.n.max() / cc.n.min()), 1)
res["code_per_class"] = {c: g.code.value_counts().to_dict() for c, g in df.groupby("cls")}
cc.to_csv(OUT / "e0_class_counts.csv", index=False)

# ---------------------------------------------------------------- sizes / formats
res["sizes_wxh"] = {f"{k[0]}x{k[1]}": int(v) for k, v in df.groupby(["w", "h"]).size().items()}
res["modes"] = df["mode"].value_counts().to_dict()
res["rgb_that_is_really_gray_%"] = round(100 * float(df.loc[df["mode"] == "RGB", "gray_rgb"].mean()), 1)
res["file_kb_by_class"] = (df.groupby("cls").nbytes.median() / 1024).round(0).astype(int).to_dict()

# ---------------------------------------------------------------- exact duplicates
g = df.groupby("sha256")
dup = g.filter(lambda x: len(x) > 1)
grp = dup.groupby("sha256").agg(n=("cls", "size"), n_classes=("cls", "nunique"), classes=("cls", lambda s: ",".join(sorted(set(s)))))
res["exact_dup_groups"] = int(len(grp))
res["exact_dup_images_in_groups"] = int(len(dup))
res["exact_dup_extra_copies"] = int(len(dup) - len(grp))
res["exact_dup_groups_across_classes"] = int((grp.n_classes > 1).sum())
res["filename_stem_collisions"] = int(df.stem.duplicated(keep=False).sum())
grp.reset_index().to_csv(OUT / "e0_exact_duplicates.csv", index=False)

# ---------------------------------------------------------------- near duplicates (pHash prefilter + thumbnail correlation)
ph = df.phash.values.astype(np.uint64); dh = df.dhash.values.astype(np.uint64)
hist = np.zeros(65, dtype=np.int64)
CH = 256
for i0 in range(0, N, CH):  # pass 1: distance histogram only (memory-safe)
    d = np.bitwise_count(ph[i0:i0 + CH, None] ^ ph[None, :]).astype(np.int16)
    m = np.arange(N)[None, :] > np.arange(i0, min(i0 + CH, N))[:, None]
    hist += np.bincount(d[m], minlength=65)
MAX_PAIRS = 3_000_000
T_STORE = max([t for t in range(0, 9) if hist[:t + 1].sum() <= MAX_PAIRS] or [0])
pi, pj, pd_ = [], [], []
for i0 in range(0, N, CH):  # pass 2: store pairs with distance <= T_STORE
    d = np.bitwise_count(ph[i0:i0 + CH, None] ^ ph[None, :]).astype(np.int16)
    m = np.arange(N)[None, :] > np.arange(i0, min(i0 + CH, N))[:, None]
    a, b = np.nonzero(m & (d <= T_STORE))
    pi.append(a + i0); pj.append(b); pd_.append(d[a, b])
pi, pj, pdist = np.concatenate(pi), np.concatenate(pj), np.concatenate(pd_)
res["phash_pair_distance_hist_0_to_16"] = {int(k): int(hist[k]) for k in range(17)}
res["phash_pairs_stored_le_T"] = {"T": int(T_STORE), "pairs": int(len(pi))}
t = thumb - thumb.mean(1, keepdims=True); t /= (np.linalg.norm(t, axis=1, keepdims=True) + 1e-9)
corr = np.empty(len(pi), dtype=np.float32)
for s in range(0, len(pi), 5000):
    corr[s:s + 5000] = np.einsum("ij,ij->i", t[pi[s:s + 5000]], t[pj[s:s + 5000]])
cls_arr = df.cls.values
same_cls = cls_arr[pi] == cls_arr[pj]
sha_arr = df.sha256.values
res["near_dup"] = {}
labs = {}
for T in [x for x in (2, 4, 8) if x <= T_STORE]:
    ok = (pdist <= T) & (corr >= 0.95) & (sha_arr[pi] != sha_arr[pj])
    n_imgs = len(np.unique(np.concatenate([pi[ok], pj[ok]])))
    gr = coo_matrix((np.ones(ok.sum()), (pi[ok], pj[ok])), shape=(N, N))
    ncomp, lab = connected_components(gr, directed=False)
    sizes = np.bincount(lab); multi = np.where(sizes > 1)[0]
    labs[T] = (lab, sizes, multi)
    cross = sum(1 for c in multi if df.cls.values[lab == c].shape[0] and len(set(cls_arr[lab == c])) > 1)
    res["near_dup"][f"phash<={T}&thumb_corr>=0.95"] = {
        "pairs": int(ok.sum()), "pairs_same_class": int((ok & same_cls).sum()), "pairs_cross_class": int((ok & ~same_cls).sum()),
        "images_with_near_dup_%": round(100 * n_imgs / N, 2), "clusters": int(len(multi)),
        "largest_cluster": int(sizes[multi].max()) if len(multi) else 0, "clusters_spanning_classes": int(cross)}
keep = (pdist <= T_STORE) & (corr >= 0.95)
pd.DataFrame({"a": df.names.values[pi[keep]], "b": df.names.values[pj[keep]], "phash_dist": pdist[keep], "thumb_corr": corr[keep].round(3)}) \
    .sort_values(["phash_dist", "thumb_corr"], ascending=[True, False]).head(2000).to_csv(OUT / "e0_near_dup_pairs_top2000.csv", index=False)

# ---------------------------------------------------------------- info bar (bright / dark full-width row bands)
def longest_run(mask):
    if not mask.any():
        return -1, 0
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    s, e = np.where(d == 1)[0], np.where(d == -1)[0]
    k = np.argmax(e - s)
    return int(s[k]), int(e[k] - s[k])

bs = np.array([longest_run(np.nan_to_num(r, nan=0) > BRIGHT) for r in rm])
ds = np.array([longest_run(np.nan_to_num(r, nan=255) < DARK) for r in rm])
df["bright_start"], df["bright_len"] = bs[:, 0], bs[:, 1]
df["dark_start"], df["dark_len"] = ds[:, 0], ds[:, 1]
df["has_bright_band"] = df.bright_len >= MIN_RUN
df["has_dark_band"] = df.dark_len >= MIN_RUN
df["bright_in_bottom_20%"] = df.has_bright_band & (df.bright_start >= 0.8 * 768)
res["info_bar"] = {
    "definition": f"longest run of >= {MIN_RUN} consecutive rows with mean gray > {BRIGHT} (bright) or < {DARK} (dark)",
    "bright_band_%": round(100 * float(df.has_bright_band.mean()), 1), "dark_band_%": round(100 * float(df.has_dark_band.mean()), 1),
    "bright_band_starts_in_bottom_20%_of_image_%": round(100 * float(df["bright_in_bottom_20%"].mean()), 1),
    "bright_len_percentiles_p5_p50_p95": [int(x) for x in np.percentile(df.bright_len[df.has_bright_band], [5, 50, 95])] if df.has_bright_band.any() else [],
    "bright_start_percentiles_p5_p50_p95": [int(x) for x in np.percentile(df.bright_start[df.has_bright_band], [5, 50, 95])] if df.has_bright_band.any() else [],
    "bright_band_%_by_class": (100 * df.groupby("cls").has_bright_band.mean()).round(1).to_dict(),
    "bright_start_hist_64row_bins": {int(k): int(v) for k, v in zip(*np.unique(df.bright_start[df.has_bright_band] // 64 * 64, return_counts=True))}}


# ---------------------------------------------------------------- follow-ups: what the big clusters / resolutions / colour really are
df["thumb_std"] = thumb.std(1); df["thumb_mean"] = thumb.mean(1)
Tc = 4 if 4 in labs else max(labs)
lab, sizes, multi = labs[Tc]
top = []
for c in multi[np.argsort(-sizes[multi])][:8]:
    idx = np.where(lab == c)[0]
    top.append({"size": int(len(idx)), "classes": df.cls.iloc[idx].value_counts().to_dict(),
                "mean_thumb_std": round(float(df.thumb_std.iloc[idx].mean()), 1), "mean_brightness": round(float(df.thumb_mean.iloc[idx].mean()), 1)})
res["top_clusters_at_T%d" % Tc] = top
res["median_thumb_std_all_images"] = round(float(df.thumb_std.median()), 1)
inclust = np.isin(lab, multi)
df["in_near_dup_cluster"] = inclust
res[f"near_dup_cluster_membership_pct_by_class_T{Tc}"] = (100 * df.groupby("cls").in_near_dup_cluster.mean()).round(1).to_dict()
res["resolution_by_class"] = {c: {f"{a}x{b}": int(n) for (a, b), n in g.groupby(["w", "h"]).size().items()} for c, g in df.groupby("cls")}
res["rgb_really_gray_%_by_class"] = (100 * df[df["mode"] == "RGB"].groupby("cls").gray_rgb.mean()).round(1).to_dict()
res["dark_band_%_by_class"] = (100 * df.groupby("cls").has_dark_band.mean()).round(1).to_dict()
res["no_bright_no_dark_band_%"] = round(100 * float((~df.has_bright_band & ~df.has_dark_band).mean()), 1)

df.drop(columns=["dhash", "phash"]).to_csv(OUT / "e0_per_image_features.csv.gz", index=False)
res["archives"] = archives
(OUT / "e0_summary.json").write_text(json.dumps(res, indent=1, default=str))
print(json.dumps({k: v for k, v in res.items() if k not in ("archives", "code_per_class")}, indent=1, default=str))
