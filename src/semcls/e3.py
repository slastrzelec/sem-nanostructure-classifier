"""E3 tables: with-bar reference vs bar removed / strips (numpy only). All E3 runs and the reference are
ConvNeXt-Tiny on group tau = 0.94 and are evaluated on the same images, so differences are paired."""
import json
from pathlib import Path

import numpy as np

from . import metrics as M
from .report import cluster_ids, metric_fn
from .splits import canon_variant

ORDER = ["full", "no_bottom", "no_top", "bottom_strip", "top_strip"]
LABEL = {"full": "with bar (reference)", "no_bottom": "E3b bar removed (no bottom)",
         "no_top": "E3b-ctrl top removed (no top)", "bottom_strip": "E3a bottom strip only",
         "top_strip": "E3a-ctrl top strip only"}
# (A, B, question) reported as metric(A) - metric(B)
CONTRASTS = [("full", "no_bottom", "drop caused by removing the bar zone"),
             ("full", "no_top", "drop caused by removing the same amount from the top (control)"),
             ("no_top", "no_bottom", "control minus bar removed"),
             ("bottom_strip", "top_strip", "bar-zone strip minus top strip")]


def load_e3_runs(run_dirs, part, splits_sha, variant="group:0.94", backbone="convnext_tiny"):
    """{region: [run dicts sorted by seed]}; only finished runs of the given variant/backbone trained on
    the current splits file. Raises on duplicates, on different images and on seed sets that differ."""
    out, seen = {}, {}
    for root in run_dirs:
        for d in sorted(Path(root).iterdir()):
            if not (d / "config.json").exists() or not (d / "done.flag").exists():
                continue
            info = json.loads((d / "config.json").read_text())
            if canon_variant(info["variant"]) != canon_variant(variant) or info["backbone"] != backbone:
                continue
            if info["splits_sha256"] != splits_sha:
                continue
            f = d / f"{part}_logits.npz"
            if not f.exists():
                continue
            region = info.get("cfg", {}).get("region", "full")
            key = (region, int(info["seed"]))
            if key in seen:
                raise ValueError(f"duplicate run {key}: {seen[key]} and {d}")
            seen[key] = d
            z = np.load(f, allow_pickle=False)
            out.setdefault(region, []).append(
                dict(name=d.name, seed=int(info["seed"]), classes=list(info["classes"]),
                     names=[str(x) for x in z["names"]], y=z["y"].astype(np.int64), logits=z["logits"].astype(np.float64)))
    for region, rs in out.items():
        rs.sort(key=lambda r: r["seed"])
    return out


def check_same(groups):
    ref = next(iter(groups.values()))[0]
    seeds = None
    for region, rs in groups.items():
        s = [r["seed"] for r in rs]
        seeds = s if seeds is None else seeds
        if s != seeds:
            raise ValueError(f"{region}: seeds {s} differ from {seeds}")
        for r in rs:
            if r["names"] != ref["names"] or not np.array_equal(r["y"], ref["y"]):
                raise ValueError(f"{r['name']}: evaluated on different images")
    return ref


def _preds(rs):
    return np.stack([r["logits"].argmax(1) for r in rs], axis=1)


def paired(P, y, a, b, kind, k, cl, n_boot, seed):
    """metric(a) - metric(b) of the seed means on the same images: point, image CI, cluster CI."""
    f = metric_fn(kind, k)
    m = lambda yy, A, B: f(yy, A) - f(yy, B)
    out = dict(point=float(m(y, P[a], P[b])))
    for tag, g in (("img", None), ("clu", cl)):
        d = M.bootstrap_dist(m, y, P[a], P[b], n_boot=n_boot, seed=seed, groups=g)
        out[tag] = [float(v) for v in np.percentile(d, [2.5, 97.5])]
    return out


def build_e3(groups, sp, part, n_boot=1000, seed=0):
    ref = check_same(groups)
    y, k, cl = ref["y"], len(ref["classes"]), cluster_ids(sp, ref["names"])
    regions = [r for r in ORDER if r in groups]
    P = {r: _preds(groups[r]) for r in regions}
    res = dict(part=part, n=int(len(y)), seeds=[r["seed"] for r in groups[regions[0]]], regions={}, contrasts=[])
    fm, fa = metric_fn("macro_f1", k), metric_fn("acc", k)
    for r in regions:
        f1 = [fm(y, P[r][:, [s]]) for s in range(P[r].shape[1])]
        ac = [fa(y, P[r][:, [s]]) for s in range(P[r].shape[1])]
        rec = np.mean([M.recall_per_class(y, P[r][:, s], k) for s in range(P[r].shape[1])], axis=0)
        res["regions"][r] = dict(macro_f1=f1, acc=ac, recall=rec.tolist(), runs=[x["name"] for x in groups[r]])
    for a, b, q in CONTRASTS:
        if a in P and b in P:
            res["contrasts"].append(dict(a=a, b=b, question=q,
                                         macro_f1=paired(P, y, a, b, "macro_f1", k, cl, n_boot, seed),
                                         acc=paired(P, y, a, b, "acc", k, cl, n_boot, seed)))
    cls = list(ref["classes"])
    md = [f"# E3 info-bar test, {part} split", "",
          f"ConvNeXt-Tiny, group tau = 0.94. {len(y)} {part} images, seeds {res['seeds']}; every variant is evaluated on the "
          "same images, so differences are paired. Bootstrap: 95% percentile intervals of the difference of seed means "
          f"({n_boot} resamples, by images and by tau = 0.94 clusters). The intervals cover the evaluation sample only, not "
          "the variation between trainings (see sd over seeds).", "",
          "## Per variant (seed mean, sd over seeds)", "",
          "| variant | macro-F1 | sd | accuracy | sd | per-seed macro-F1 |", "|---|---|---|---|---|---|"]
    for r in regions:
        d = res["regions"][r]
        sd = lambda v: float(np.std(v, ddof=1)) if len(v) > 1 else float("nan")
        md.append(f"| {LABEL[r]} | {np.mean(d['macro_f1']):.4f} | {sd(d['macro_f1']):.4f} | {np.mean(d['acc']):.4f} | "
                  f"{sd(d['acc']):.4f} | {', '.join(f'{v:.4f}' for v in d['macro_f1'])} |")
    md += ["", "## Paired differences (A - B, percentage points)", "",
           "| A - B | question | macro-F1 | 95% CI images | 95% CI clusters | accuracy | 95% CI images | 95% CI clusters |",
           "|---|---|---|---|---|---|---|---|"]
    pp = lambda v: f"{100 * v:+.2f}"
    for c in res["contrasts"]:
        row = [f"{c['a']} - {c['b']}", c["question"]]
        for kind in ("macro_f1", "acc"):
            t = c[kind]
            row += [pp(t["point"]), f"[{pp(t['img'][0])}, {pp(t['img'][1])}]", f"[{pp(t['clu'][0])}, {pp(t['clu'][1])}]"]
        md.append("| " + " | ".join(row) + " |")
    md += ["", "## Per-class recall (seed mean)", "", "| class | n | " + " | ".join(regions) + " |",
           "|---|---|" + "---|" * len(regions)]
    for i, c in enumerate(cls):
        md.append(f"| {c} | {int((y == i).sum())} | " + " | ".join(f"{res['regions'][r]['recall'][i]:.3f}" for r in regions) + " |")
    md += ["", "Chance level of macro-F1 on 10 classes is about 0.10. Classes with few images (Porous_Sponge, Fibres, "
           "Films_Coated_Surface): one image changes recall by several points."]
    return "\n".join(md) + "\n", res
