"""Final test evaluation of finished runs (each checkpoint at most once)."""
import argparse
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
for cand in (here, here.parent / "src", Path("/kaggle/input/nano-ss-code")):
    if (cand / "semcls").is_dir():
        sys.path.insert(0, str(cand))
        break

ap = argparse.ArgumentParser()
ap.add_argument("--runs", default="/kaggle/working/runs")
ap.add_argument("--splits", default="/kaggle/input/nano-ss-code/splits.json")
ap.add_argument("--images", default="/kaggle/input")
ap.add_argument("--only", nargs="*", default=None, help="run directory names; default: all finished runs")
args = ap.parse_args()

from semcls.evaltest import evaluate_test  # noqa: E402

for d in sorted(Path(args.runs).iterdir()):
    if not (d / "done.flag").exists() or (args.only and d.name not in args.only):
        continue
    if (d / "test.lock").exists():
        print("already evaluated, skipping:", d.name)
        continue
    print("test:", d.name, evaluate_test(d, args.splits, args.images))
