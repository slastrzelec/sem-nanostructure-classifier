"""E4a/E4b tables: temperature scaling and selective prediction on stored logits (numpy only).
SPEC section 11, 2026-10-07. Everything is fitted on val; test only receives the fixed T and theta."""
import numpy as np

from . import calib as C
from . import metrics as M
from .report import cluster_ids

NAMES = {"convnext_tiny": "ConvNeXt-Tiny", "resnet50": "ResNet-50"}
DEMO = ("convnext_tiny", 2)


def _ci(metric, y, L, cl, n_boot, seed):
    out = {}
    for tag, g in (("img", None), ("clu", cl)):
        d = M.bootstrap_dist(metric, y, L, n_boot=n_boot, seed=seed, groups=g)
        out[tag] = [float(v) for v in np.nanpercentile(d, [2.5, 97.5])]
    return out


def _check(val, test):
    if [r["seed"] for r in val] != [r["seed"] for r in test]:
        raise ValueError("val and test runs have different seeds")
    for v, t in zip(val, test):
        if v["name"] != t["name"] or v["classes"] != t["classes"]:
            raise ValueError(f"val/test run mismatch: {v['name']} vs {t['name']}")


def analyse_backbone(val, test, cl, n_boot, seed):
    _check(val, test)
    k = len(val[0]["classes"])
    yv, yt = val[0]["y"], test[0]["y"]
    runs = []
    for v, t in zip(val, test):
        T = C.fit_temperature(v["logits"], yv)
        cv, ct = C.confidence(v["logits"], T), C.confidence(t["logits"], T)
        okv = (v["logits"].argmax(1) == yv).astype(float)
        okt = (t["logits"].argmax(1) == yt).astype(float)
        assert np.array_equal((t["logits"] / T).argmax(1), t["logits"].argmax(1))  # accuracy / F1 unchanged
        theta = C.choose_threshold(cv, okv)
        r = dict(name=v["name"], seed=v["seed"], T=T,
                 nll=dict(val_before=C.nll(v["logits"], yv), val_after=C.nll(v["logits"], yv, T),
                          test_before=C.nll(t["logits"], yt), test_after=C.nll(t["logits"], yt, T)),
                 ece=dict(val_before=M.ece(M.softmax(v["logits"]), yv), val_after=M.ece(M.softmax(v["logits"] / T), yv),
                          test_before=M.ece(M.softmax(t["logits"]), yt), test_after=M.ece(M.softmax(t["logits"] / T), yt)),
                 acc_val=float(okv.mean()), acc_test=float(okt.mean()), theta=theta,
                 aurc_test=C.aurc(ct, okt), acc_at_cov={str(c): C.accuracy_at_coverage(ct, okt, c) for c in (0.90, 0.95)})
        if theta is not None:
            r["val"] = dict(zip(("coverage", "sel_acc"), C.selective(cv, okv, theta)))
            r["test"] = dict(zip(("coverage", "sel_acc"), C.selective(ct, okt, theta)),
                             error_capture=C.error_capture(ct, okt, theta))
            r["reject_rate_class"] = [float((ct[yt == c] < theta).mean()) if (yt == c).any() else float("nan") for c in range(k)]
        runs.append(r)
    S = len(runs)
    L = np.stack([t["logits"] for t in test], axis=1)  # (n, seeds, classes)
    Ts = np.array([r["T"] for r in runs])
    res = dict(runs=runs, n_test=int(len(yt)), n_val=int(len(yv)))
    ece_d = lambda y, LL: float(np.mean([M.ece(M.softmax(LL[:, s] / Ts[s]), y) - M.ece(M.softmax(LL[:, s]), y) for s in range(S)]))
    nll_d = lambda y, LL: float(np.mean([C.nll(LL[:, s], y, Ts[s]) - C.nll(LL[:, s], y) for s in range(S)]))
    res["d_ece"] = dict(point=ece_d(yt, L), **_ci(ece_d, yt, L, cl, n_boot, seed))
    res["d_nll"] = dict(point=nll_d(yt, L), **_ci(nll_d, yt, L, cl, n_boot, seed))
    if all(r["theta"] is not None for r in runs):
        th = np.array([r["theta"] for r in runs])
        conf = lambda LL: [C.confidence(LL[:, s], Ts[s]) for s in range(S)]
        cov_f = lambda y, LL: float(np.mean([(c >= th[s]).mean() for s, c in enumerate(conf(LL))]))
        acc_f = lambda y, LL: float(np.nanmean([((LL[:, s].argmax(1) == y)[c >= th[s]]).mean() if (c >= th[s]).any() else np.nan
                                                for s, c in enumerate(conf(LL))]))
        res["coverage"] = dict(point=cov_f(yt, L), **_ci(cov_f, yt, L, cl, n_boot, seed))
        res["sel_acc"] = dict(point=acc_f(yt, L), **_ci(acc_f, yt, L, cl, n_boot, seed))
    else:
        res["coverage"] = res["sel_acc"] = None
    return res


def build_e4(val_groups, test_groups, sp, n_boot=1000, seed=0):
    out = {}
    for b in NAMES:
        if b in val_groups and b in test_groups:
            names = test_groups[b][0]["names"]
            out[b] = analyse_backbone(val_groups[b], test_groups[b], cluster_ids(sp, names), n_boot, seed)
            out[b]["classes"] = list(test_groups[b][0]["classes"])
            out[b]["test_y"] = [int(v) for v in np.bincount(test_groups[b][0]["y"], minlength=len(out[b]["classes"]))]
    return render(out, n_boot), out


def render(res, n_boot):
    f4 = lambda v: "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.4f}"
    f3 = lambda v: "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.3f}"
    mean = lambda rs, g: float(np.mean([g(r) for r in rs]))
    iv = lambda d, p=3, sg=False: f"[{d['img'][0]:{'+' if sg else ''}.{p}f}, {d['img'][1]:{'+' if sg else ''}.{p}f}] / [{d['clu'][0]:{'+' if sg else ''}.{p}f}, {d['clu'][1]:{'+' if sg else ''}.{p}f}]"
    first = next(iter(res.values()))
    md = ["# E4a/E4b: temperature scaling and selective prediction", "",
          f"Group tau = 0.94, seeds 0-2, stored logits ({first['n_val']} val and {first['n_test']} test images). Post-hoc: test logits were seen "
          "before this analysis; T and the threshold theta are fitted on val only and applied unchanged to test (SPEC 2026-10-07). "
          f"Intervals: 95% percentile bootstrap, image / cluster (tau = 0.94), {n_boot} resamples, seed means, T and theta fixed; they cover the "
          "evaluation sample only, not the variation between trainings.", ""]
    md += ["## E4a Temperature scaling (seed mean; per-seed T in the last column)", "",
           "| model | T | NLL val before > after* | NLL test before > after | ECE val before > after* | ECE test before > after | d ECE test (after - before), img / clu | d NLL test, img / clu | per-seed T |",
           "|---|---|---|---|---|---|---|---|---|"]
    for b, r in res.items():
        rs = r["runs"]
        pair = lambda key, a, c: f"{f4(mean(rs, lambda x: x[key][a]))} > {f4(mean(rs, lambda x: x[key][c]))}"
        md.append(f"| {NAMES[b]} | {mean(rs, lambda x: x['T']):.3f} | {pair('nll', 'val_before', 'val_after')} | {pair('nll', 'test_before', 'test_after')} | "
                  f"{pair('ece', 'val_before', 'val_after')} | {pair('ece', 'test_before', 'test_after')} | "
                  f"{r['d_ece']['point']:+.4f} {iv(r['d_ece'], 4, True)} | {r['d_nll']['point']:+.4f} {iv(r['d_nll'], 4, True)} | "
                  + ", ".join("%.2f" % x["T"] for x in rs) + " |")
    md += ["", "*in-sample for T. Accuracy and macro-F1 are unchanged by temperature scaling (argmax invariant, asserted).", "",
           f"## E4b Selective prediction (rule: largest coverage with val accuracy of accepted >= {C.TARGET_ACC}, >= {C.MIN_ACCEPTED} images)", "",
           "| model | theta (mean) | val coverage | val sel. acc | test coverage, img / clu | test sel. acc, img / clu | error capture (test) | overall test acc (random rejection) | AURC test | sel. acc @ cov 0.90 / 0.95 |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for b, r in res.items():
        rs = r["runs"]
        if r["coverage"] is None:
            md.append(f"| {NAMES[b]} | not attainable for at least one seed ({[x['seed'] for x in rs if x['theta'] is None]}) | | | | | | | | |")
            continue
        md.append(f"| {NAMES[b]} | {mean(rs, lambda x: x['theta']):.4f} | {f3(mean(rs, lambda x: x['val']['coverage']))} | {f4(mean(rs, lambda x: x['val']['sel_acc']))} | "
                  f"{r['coverage']['point']:.3f} {iv(r['coverage'])} | {r['sel_acc']['point']:.4f} {iv(r['sel_acc'], 4)} | "
                  f"{f3(mean(rs, lambda x: x['test']['error_capture']))} | {mean(rs, lambda x: x['acc_test']):.4f} | {mean(rs, lambda x: x['aurc_test']):.4f} | "
                  f"{mean(rs, lambda x: x['acc_at_cov']['0.9']):.4f} / {mean(rs, lambda x: x['acc_at_cov']['0.95']):.4f} |")
    md += ["", "### Per seed", "", "| run | T | theta | val cov | val sel. acc | test cov | test sel. acc | test acc | ECE test before > after |", "|---|---|---|---|---|---|---|---|---|"]
    for b, r in res.items():
        for x in r["runs"]:
            tag = " (demo)" if (b, x["seed"]) == DEMO else ""
            th = f"{x['theta']:.4f}" if x["theta"] is not None else "not attainable"
            v = x.get("val", {}); t = x.get("test", {})
            md.append(f"| {NAMES[b]} s{x['seed']}{tag} | {x['T']:.3f} | {th} | {f3(v.get('coverage'))} | {f4(v.get('sel_acc'))} | {f3(t.get('coverage'))} | "
                      f"{f4(t.get('sel_acc'))} | {x['acc_test']:.4f} | {x['ece']['test_before']:.4f} > {x['ece']['test_after']:.4f} |")
    md += ["", "### Rejection rate per class on test (share of the class below theta, seed mean)", ""]
    bs = [b for b, r in res.items() if all("reject_rate_class" in x for x in r["runs"])]
    if bs:
        md += ["| class | n test | " + " | ".join(NAMES[b] for b in bs) + " |", "|---|---|" + "---|" * len(bs)]
        for i, c in enumerate(res[bs[0]]["classes"]):
            md.append(f"| {c} | {res[bs[0]]['test_y'][i]} | " + " | ".join(f3(float(np.mean([x['reject_rate_class'][i] for x in res[b]['runs']]))) for b in bs) + " |")
    md += ["", "Reading notes: theta is selected on val, so the val selective accuracy is optimistic and the val-to-test gap is part of the result. "
           "Classes with few images (Porous_Sponge, Fibres, Films_Coated_Surface): one image changes a rate by several points. "
           "ECE of about 3100 images is biased upwards; interpret the interval of the difference, not small ECE values."]
    return "\n".join(md) + "\n"
