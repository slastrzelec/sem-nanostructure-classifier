"""Aggregation of finished runs into result tables (numpy only, no torch).

Used by scripts/report_runs.py. Works on val_logits.npz or test_logits.npz of finished runs."""
import json
from pathlib import Path

import numpy as np

from . import metrics as M
from .splits import canon_variant

BOOT_TAU = "0.94"  # clusters used for the cluster bootstrap (SPEC section 11, 2026-10-06)
VARIANT_ORDER = ["random", "group:0.94", "group:0.9", "group:0.98"]


def check_random_identical(sp_a, sp_b):
    """E1 runs were trained with an older splits file; they stay valid only if the file list,
    labels and random split are identical."""
    for key in ("files", "class", "random"):
        if sp_a[key] != sp_b[key]:
            raise ValueError(f"splits files differ in '{key}'; E1 runs cannot be matched to the current file")


def load_runs(run_dirs, part, splits_sha, legacy_sha=()):
    """Finished runs with a <part>_logits.npz. Group runs must come from the current splits file;
    random runs also from `legacy_sha` files. Returns (runs, skipped[(name, reason)])."""
    runs, skipped, seen = [], [], {}
    for root in run_dirs:
        for d in sorted(Path(root).iterdir()):
            if not (d / "config.json").exists() or not (d / "done.flag").exists():
                continue
            info = json.loads((d / "config.json").read_text())
            variant = canon_variant(info["variant"])
            region = info.get("cfg", {}).get("region", "full")
            if region != "full":
                skipped.append((d.name, f"region run ({region}); E3 runs are not part of the main tables"))
                continue
            ok = {splits_sha} | (set(legacy_sha) if variant == "random" else set())
            if info["splits_sha256"] not in ok:
                skipped.append((d.name, "trained on another version of splits.json"))
                continue
            f = d / f"{part}_logits.npz"
            if not f.exists():
                skipped.append((d.name, f"no {part}_logits.npz"))
                continue
            key = (variant, info["backbone"], int(info["seed"]))
            if key in seen:
                raise ValueError(f"duplicate run {key}: {seen[key]} and {d}")
            seen[key] = d
            z = np.load(f, allow_pickle=False)
            runs.append(dict(name=d.name, variant=variant, backbone=info["backbone"], seed=int(info["seed"]),
                             classes=list(info["classes"]), names=[str(x) for x in z["names"]],
                             y=z["y"].astype(np.int64), logits=z["logits"].astype(np.float64)))
    return runs, skipped


def group_runs(runs):
    """{(variant, backbone): runs sorted by seed}; all seeds of a group must share the same images."""
    out = {}
    for r in runs:
        out.setdefault((r["variant"], r["backbone"]), []).append(r)
    for key, rs in out.items():
        rs.sort(key=lambda r: r["seed"])
        for r in rs[1:]:
            if r["names"] != rs[0]["names"] or not np.array_equal(r["y"], rs[0]["y"]):
                raise ValueError(f"{key}: runs evaluated on different images")
    return out


def cluster_ids(sp, names, tau=BOOT_TAU):
    idx = {n: i for i, n in enumerate(sp["files"])}
    cl = np.asarray(sp["group"][tau]["cluster"])
    return cl[[idx[n] for n in names]]


def neighbour_in_train(sp, names, tau=BOOT_TAU):
    """True for images whose tau-cluster has a member in the *random* split's training set."""
    cl = np.asarray(sp["group"][tau]["cluster"])
    train_clusters = set(cl[np.asarray(sp["random"]) == "train"].tolist())
    idx = {n: i for i, n in enumerate(sp["files"])}
    return np.array([cl[idx[n]] in train_clusters for n in names])


def macro_f1_present(y, p, k):
    """Macro-F1 over the classes that occur in y (for subsets where some class is absent)."""
    present = np.bincount(np.asarray(y), minlength=k) > 0
    return float(M.f1_per_class(y, p, k)[present].mean())


def _preds(runs):
    return np.stack([r["logits"].argmax(1) for r in runs], axis=1)


def metric_fn(kind, k):
    """Mean over seeds (columns of P) of a per-run metric; signature f(y, P)."""
    one = {"macro_f1": lambda y, p: M.macro_f1(y, p, k),
           "acc": lambda y, p: M.accuracy(y, p),
           "macro_f1_present": lambda y, p: macro_f1_present(y, p, k)}[kind]
    return lambda y, P: float(np.mean([one(y, P[:, s]) for s in range(P.shape[1])]))


def group_stats(runs, sp, n_boot=1000, seed=0):
    """Metrics of one (variant, backbone): per-seed values, mean +- sd over seeds, CIs of the seed
    mean by image bootstrap and by cluster bootstrap, ECE, per-class recall, summed confusion."""
    k = len(runs[0]["classes"])
    y, P = runs[0]["y"], _preds(runs)
    cl = cluster_ids(sp, runs[0]["names"])
    out = dict(seeds=[r["seed"] for r in runs], n=int(len(y)), k=k)
    for kind in ("macro_f1", "acc"):
        f = metric_fn(kind, k)
        per = [f(y, P[:, [s]]) for s in range(P.shape[1])]
        out[kind] = dict(per_seed=per, mean=float(np.mean(per)),
                         sd=float(np.std(per, ddof=1)) if len(per) > 1 else float("nan"),
                         ci_img=M.bootstrap_ci(f, y, P, n_boot=n_boot, seed=seed)[1:],
                         ci_clu=M.bootstrap_ci(f, y, P, n_boot=n_boot, seed=seed, groups=cl)[1:])
    out["ece"] = float(np.mean([M.ece(M.softmax(r["logits"]), y) for r in runs]))
    out["recall"] = np.mean([M.recall_per_class(y, P[:, s], k) for s in range(P.shape[1])], axis=0).tolist()
    out["confusion"] = sum(M.confusion(y, P[:, s], k) for s in range(P.shape[1])).tolist()
    return out


def gap_ci(runs_a, runs_b, sp, kind="macro_f1", n_boot=1000, seed=0, alpha=0.05):
    """metric(A) - metric(B) where A and B are evaluated on different images (independent
    bootstraps). Returns {'point', 'img': (lo, hi), 'clu': (lo, hi)}."""
    k = len(runs_a[0]["classes"])
    f = metric_fn(kind, k)
    ya, Pa, ca = runs_a[0]["y"], _preds(runs_a), cluster_ids(sp, runs_a[0]["names"])
    yb, Pb, cb = runs_b[0]["y"], _preds(runs_b), cluster_ids(sp, runs_b[0]["names"])
    out = dict(point=f(ya, Pa) - f(yb, Pb))
    for tag, ga, gb in (("img", None, None), ("clu", ca, cb)):
        da = M.bootstrap_dist(f, ya, Pa, n_boot=n_boot, seed=seed, groups=ga)
        db = M.bootstrap_dist(f, yb, Pb, n_boot=n_boot, seed=seed + 1, groups=gb)
        out[tag] = tuple(float(v) for v in np.percentile(da - db, [100 * alpha / 2, 100 * (1 - alpha / 2)]))
    return out


def stratified(runs, sp, n_boot=1000, seed=0):
    """E1 runs only: the same models and the same test images, split by whether the image's tau=0.94
    cluster has a member in the training set. Descriptive; the strata differ in class mix."""
    k = len(runs[0]["classes"])
    y, P = runs[0]["y"], _preds(runs)
    flag = neighbour_in_train(sp, runs[0]["names"])
    cl = cluster_ids(sp, runs[0]["names"])
    out = {}
    for tag, m in (("with_neighbour_in_train", flag), ("no_neighbour_in_train", ~flag)):
        res = dict(n=int(m.sum()))
        for kind in ("acc", "macro_f1_present"):
            f = metric_fn(kind, k)
            res[kind] = dict(mean=f(y[m], P[m]), ci_clu=M.bootstrap_ci(f, y[m], P[m], n_boot=n_boot, seed=seed, groups=cl[m])[1:])
        out[tag] = res
    return out


# ---- rendering ----------------------------------------------------------------------------

def _t(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines) + "\n"


def _pm(d, nd=4):
    return f"{d['mean']:.{nd}f} ± {d['sd']:.{nd}f}" if d["sd"] == d["sd"] else f"{d['mean']:.{nd}f}"


def _ci(ci, nd=4):
    return f"[{ci[0]:.{nd}f}, {ci[1]:.{nd}f}]"


def _order(keys):
    def k(key):
        v, b = key
        return (VARIANT_ORDER.index(v) if v in VARIANT_ORDER else 99, b)
    return sorted(keys, key=k)


def build_report(runs, skipped, sp, part, n_boot=1000, seed=0):
    """Returns (markdown, results dict)."""
    groups = group_runs(runs)
    classes = runs[0]["classes"]
    stats = {key: group_stats(rs, sp, n_boot, seed) for key, rs in groups.items()}
    L = [f"# Results on the {part} split\n",
         f"Runs: {len(runs)}; bootstrap: {n_boot} resamples; 'image' = images resampled one by one, 'cluster' = whole "
         f"tau={BOOT_TAU} clusters resampled; CIs are for the mean over seeds (seeds resampled jointly); ± is the sample "
         "standard deviation over seeds.\n"]
    if part == "val":
        L.append("Note: validation sets of different variants contain different images; differences between variants are "
                 "not test results.\n")
    if skipped:
        L.append("Skipped runs: " + "; ".join(f"{n} ({why})" for n, why in skipped) + "\n")

    L.append("\n## Main table\n")
    rows = []
    for key in _order(stats):
        s = stats[key]
        rows.append([key[0], key[1], len(s["seeds"]), s["n"], _pm(s["macro_f1"]), _ci(s["macro_f1"]["ci_img"]),
                     _ci(s["macro_f1"]["ci_clu"]), _pm(s["acc"]), f"{s['ece']:.4f}"])
    L.append(_t(["variant", "backbone", "seeds", "images", "macro-F1", "95% CI image", "95% CI cluster", "accuracy", "ECE"], rows))

    gap_rows, strat = [], {}
    for b in sorted({b for _, b in groups}):
        if ("random", b) in groups and ("group:0.94", b) in groups:
            for kind in ("macro_f1", "acc"):
                g = gap_ci(groups[("random", b)], groups[("group:0.94", b)], sp, kind, n_boot, seed)
                gap_rows.append([b, kind, f"{g['point']:+.4f}", _ci(g["img"]), _ci(g["clu"])])
    if gap_rows:
        L.append("\n## E1 (random) minus E2 (group, tau=0.94)\n")
        L.append("The two splits contain different images, so this gap mixes leakage with differences between the samples.\n\n")
        L.append(_t(["backbone", "metric", "E1 - E2", "95% CI image", "95% CI cluster"], gap_rows))

    srows = []
    for b in sorted({b for _, b in groups}):
        if ("random", b) in groups:
            st = stratified(groups[("random", b)], sp, n_boot, seed)
            strat[b] = st
            for tag, res in st.items():
                srows.append([b, tag.replace("_", " "), res["n"], f"{res['acc']['mean']:.4f}", _ci(res["acc"]["ci_clu"]),
                              f"{res['macro_f1_present']['mean']:.4f}", _ci(res["macro_f1_present"]["ci_clu"])])
    if srows:
        L.append(f"\n## E1 {part} images with and without a near-duplicate cluster-mate in the training set\n")
        L.append("Same models, same images, split by whether the image's tau=0.94 cluster has a member in the E1 training set. "
                 "Descriptive only: the two subsets differ in class mix and difficulty. Macro-F1 is over the classes present in the subset.\n\n")
        L.append(_t(["backbone", "subset", "images", "accuracy", "95% CI cluster", "macro-F1", "95% CI cluster"], srows))

    sens = [(v, b) for (v, b) in _order(groups) if b == "resnet50" and v in VARIANT_ORDER]
    if len(sens) > 1:
        L.append("\n## Sensitivity to the clustering threshold (ResNet-50, seed 0)\n")
        rows = []
        for v, b in sens:
            rs = [r for r in groups[(v, b)] if r["seed"] == 0]
            if rs:
                s = group_stats(rs, sp, n_boot, seed)
                rows.append([v, s["n"], f"{s['macro_f1']['mean']:.4f}", _ci(s["macro_f1"]["ci_clu"]), f"{s['acc']['mean']:.4f}"])
        L.append(_t(["variant", "images", "macro-F1", "95% CI cluster", "accuracy"], rows))

    L.append("\n## Per-class recall (mean over seeds)\n")
    rows = [[f"{v} / {b}"] + [f"{x:.3f}" for x in stats[(v, b)]["recall"]] for v, b in _order(stats)]
    L.append(_t(["run"] + classes, rows))

    L.append("\n## Most frequent confusions (true -> predicted, share of the true class, summed over seeds)\n")
    for v, b in _order(stats):
        cm = np.asarray(stats[(v, b)]["confusion"], dtype=float)
        support = cm.sum(axis=1, keepdims=True)
        rate = np.divide(cm, support, out=np.zeros_like(cm), where=support > 0)
        np.fill_diagonal(rate, 0)
        top = np.dstack(np.unravel_index(np.argsort(-rate, axis=None)[:5], rate.shape))[0]
        items = [f"{classes[i]} -> {classes[j]} {100 * rate[i, j]:.1f}% ({int(cm[i, j])})" for i, j in top if cm[i, j] > 0]
        L.append(f"- {v} / {b}: " + "; ".join(items))
    results = dict(part=part, n_boot=n_boot, classes=classes,
                   groups={f"{v}|{b}": s for (v, b), s in stats.items()},
                   gaps=gap_rows, stratified=strat, skipped=skipped)
    return "\n".join(L) + "\n", results
