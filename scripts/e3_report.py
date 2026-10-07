"""Aggregate the E3 runs (bar removal / strips) and the with-bar reference into tables.

  python scripts/e3_report.py --runs <dir with E3 runs> <dir with reference runs> --part val
  python scripts/e3_report.py ... --part test      # only after the freeze entry in SPEC section 11"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for cand in (ROOT / "src", ROOT.parent / "src"):
    if (cand / "semcls").is_dir():
        sys.path.insert(0, str(cand))
        break

from semcls.e3 import build_e3, load_e3_runs  # noqa: E402
from semcls.splits import load_splits, sha256_file  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--runs", nargs="+", required=True, help="directories that contain run directories")
ap.add_argument("--part", choices=["val", "test"], default="val")
ap.add_argument("--splits", default=str(ROOT / "splits" / "splits.json"))
ap.add_argument("--boot", type=int, default=1000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default=None, help="markdown output (default reports/e3/<part>.md)")
args = ap.parse_args()

sp = load_splits(args.splits)
groups = load_e3_runs(args.runs, args.part, sha256_file(args.splits))
if "full" not in groups or len(groups) < 2:
    sys.exit(f"need the with-bar reference and at least one E3 variant with {args.part}_logits.npz; found {sorted(groups)}")
md, res = build_e3(groups, sp, args.part, n_boot=args.boot, seed=args.seed)
out = Path(args.out) if args.out else ROOT / "reports" / "e3" / f"{args.part}.md"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(md, encoding="utf-8")
out.with_suffix(".json").write_text(json.dumps(res, indent=1), encoding="utf-8")
print(md)
print("wrote", out)
