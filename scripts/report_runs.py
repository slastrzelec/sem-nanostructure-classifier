"""Aggregate finished runs into result tables (validation or test split).

  python scripts/report_runs.py --runs <runs dir> [<runs dir> ...] --part val
  python scripts/report_runs.py --runs <runs dir> ... --part test      # only after the freeze in SPEC section 11

Runs trained on splits_v1.json are accepted for the random split only (its random section is
identical to splits.json; checked here)."""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for cand in (HERE, ROOT / "src"):  # next to the script (Kaggle code dataset) or in the repository
    if (cand / "semcls").is_dir():
        sys.path.insert(0, str(cand))
        break
SPLITS_DIR = HERE if (HERE / "splits.json").exists() else ROOT / "splits"

from semcls.report import build_report, check_random_identical, load_runs  # noqa: E402
from semcls.splits import load_splits, sha256_file  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--runs", nargs="+", required=True, help="directories that contain run directories")
ap.add_argument("--part", choices=["val", "test"], default="val")
ap.add_argument("--splits", default=str(SPLITS_DIR / "splits.json"))
ap.add_argument("--legacy-splits", nargs="*", default=[str(SPLITS_DIR / "splits_v1.json")])
ap.add_argument("--boot", type=int, default=1000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default=None, help="markdown output (default reports/results/<part>.md)")
args = ap.parse_args()

sp = load_splits(args.splits)
legacy = [p for p in args.legacy_splits if Path(p).exists()]
for p in legacy:
    check_random_identical(sp, load_splits(p))
runs, skipped = load_runs(args.runs, args.part, sha256_file(args.splits), [sha256_file(p) for p in legacy])
if not runs:
    sys.exit(f"no finished runs with {args.part}_logits.npz found; skipped: {skipped}")
md, res = build_report(runs, skipped, sp, args.part, n_boot=args.boot, seed=args.seed)
out = Path(args.out) if args.out else ROOT / "reports" / "results" / f"{args.part}.md"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(md, encoding="utf-8")
out.with_suffix(".json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
print(md)
print("wrote", out)
