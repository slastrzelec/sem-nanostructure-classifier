"""E4a/E4b: temperature scaling and selective prediction from stored logits (SPEC section 11, 2026-10-07).

  python scripts/e4_report.py --runs <runs dir> [<runs dir> ...] [--boot 1000 --seed 0 --out reports/e4/calib_selective.md]

Needs val_logits.npz and test_logits.npz of the group tau = 0.94 runs (ConvNeXt-Tiny, ResNet-50, seeds 0-2)."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for cand in (ROOT / "src", ROOT.parent / "src"):
    if (cand / "semcls").is_dir():
        sys.path.insert(0, str(cand))
        break

from semcls.e4 import build_e4  # noqa: E402
from semcls.report import group_runs, load_runs  # noqa: E402
from semcls.splits import load_splits, sha256_file  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--runs", nargs="+", required=True)
ap.add_argument("--splits", default=str(ROOT / "splits" / "splits.json"))
ap.add_argument("--boot", type=int, default=1000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default=str(ROOT / "reports" / "e4" / "calib_selective.md"))
args = ap.parse_args()

sp = load_splits(args.splits)
sha = sha256_file(args.splits)
groups = {}
for part in ("val", "test"):
    runs, _ = load_runs(args.runs, part, sha)
    g = group_runs([r for r in runs if r["variant"] == "group:0.94"])
    groups[part] = {b: rs for (v, b), rs in g.items()}
md, res = build_e4(groups["val"], groups["test"], sp, n_boot=args.boot, seed=args.seed)
out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(md, encoding="utf-8")
out.with_suffix(".json").write_text(json.dumps(res, indent=1), encoding="utf-8")
print(md)
print("wrote", out)
