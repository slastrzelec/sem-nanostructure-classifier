"""Run training jobs (cross product of variants x backbones x seeds) on Kaggle or locally.

Example (Kaggle, code dataset attached as /kaggle/input/nano-ss-code):
  python kaggle_run.py --variants random group:0.94 --backbones resnet50 --seeds 0 1 2
Use --shard i/n to split the job list over several processes (one per GPU).
"""
import argparse
import itertools
import json
import os
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
for cand in (here, here.parent / "src", Path("/kaggle/input/nano-ss-code")):
    if (cand / "semcls").is_dir():
        sys.path.insert(0, str(cand))
        break

ap = argparse.ArgumentParser()
ap.add_argument("--variants", nargs="+", required=True)
ap.add_argument("--backbones", nargs="+", required=True)
ap.add_argument("--seeds", nargs="+", type=int, required=True)
ap.add_argument("--regions", nargs="+", default=["full"], help="E3: full bottom_strip top_strip no_bottom no_top")
ap.add_argument("--splits", default=None)
ap.add_argument("--images", default="/kaggle/input")
ap.add_argument("--out", default="/kaggle/working/runs")
ap.add_argument("--shard", default="0/1")
ap.add_argument("--set", nargs="*", default=[], help="config overrides key=json_value, e.g. epochs=2")
args = ap.parse_args()

from semcls.train import run  # noqa: E402

splits = args.splits
if splits is None:
    for c in (here / "splits.json", here.parent / "splits" / "splits.json",
              Path("/kaggle/input/nano-ss-code/splits.json")):
        if c.exists():
            splits = str(c)
            break
assert splits, "splits.json not found"
overrides = {kv.split("=", 1)[0]: json.loads(kv.split("=", 1)[1]) for kv in args.set}
i, n = (int(x) for x in args.shard.split("/"))
jobs = list(itertools.product(args.variants, args.backbones, args.regions, args.seeds))[i::n]
print("jobs:", jobs, flush=True)
for variant, backbone, region, seed in jobs:
    run(variant, backbone, seed, splits, args.images, args.out, overrides={**overrides, "region": region})
