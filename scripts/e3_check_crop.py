"""E3 pre-check (label-free, read-only): does the bar-zone crop of SPEC section 11 remove the info bar?

Runs the E0 band detector (same thresholds) on row means of the 512x384 cache, before and after cropping
the bottom BAR_FRAC. Uses only train+val images of the group split tau = 0.94; test images are not opened.

  python scripts/e3_check_crop.py --cache data/cache384 --splits splits/splits.json"""
import argparse
import io
import json
import sys
import tarfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from semcls import bar as B  # noqa: E402
from semcls.splits import get_split, load_splits  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--cache", default=str(ROOT / "data" / "cache384"))
ap.add_argument("--splits", default=str(ROOT / "splits" / "splits.json"))
ap.add_argument("--variant", default="group:0.94")
ap.add_argument("--fracs", type=float, nargs="*", default=[0.15, 0.1875, 0.20, 0.22, 0.25, 0.30])
ap.add_argument("--out", default=str(ROOT / "reports" / "e3" / "crop_check"))
args = ap.parse_args()

FRACS = args.fracs
sp = load_splits(args.splits)
data = get_split(sp, args.variant, ("train", "val"))        # never the test part
want = {}
for part in ("train", "val"):
    for f, y in zip(*data[part]):
        want[f] = y
cls_of = dict(zip(sp["files"], sp["class"]))
stats = defaultdict(lambda: defaultdict(int))
seen = 0
for tar in sorted(Path(args.cache).glob("*.tar")):
    with tarfile.open(tar) as t:
        for m in t:
            if not m.isfile():
                continue
            key = m.name
            if key not in want:
                continue
            im = Image.open(io.BytesIO(t.extractfile(m).read())).convert("L")
            if im.size != (512, 384):
                im = im.resize((512, 384), Image.BILINEAR)
            rm = np.asarray(im, np.float64).mean(axis=1)
            h = len(rm)
            y0, y1 = B.region_rows("no_bottom", h)
            s0, s1 = B.region_rows("bottom_strip", h)
            c = cls_of[key]
            for grp in ("all", c):
                st = stats[grp]
                st["n"] += 1
                for kind in ("bright", "dark"):
                    st[f"full_{kind}_band"] += B.band(rm, kind)[1] >= B.min_run(h)
                    st[f"full_{kind}_touch_bottom"] += B.band_touches_bottom(rm, kind, h)
                    st[f"crop_{kind}_band"] += B.band(rm[y0:y1], kind)[1] >= B.min_run(h)
                    st[f"crop_{kind}_touch_bottom"] += B.band_touches_bottom(rm[y0:y1], kind, h)
                    st[f"strip_{kind}_band"] += B.band(rm[s0:s1], kind)[1] >= B.min_run(h)
                for kind in ("bright", "dark"):
                    st[f"full_{kind}_barlike"] += B.has_bar_like_band(rm, kind, h)
                    st[f"crop_{kind}_barlike"] += B.has_bar_like_band(rm[y0:y1], kind, h)
                    st[f"strip_{kind}_barlike"] += B.has_bar_like_band(rm[s0:s1], kind, h)
                st["crop_any_barlike"] += (B.has_bar_like_band(rm[y0:y1], "bright", h)
                                           or B.has_bar_like_band(rm[y0:y1], "dark", h))
                st["full_any_barlike"] += (B.has_bar_like_band(rm, "bright", h) or B.has_bar_like_band(rm, "dark", h))
                for fr in FRACS:                       # sensitivity: bright bar-like band left after a larger/smaller crop
                    yy = h - int(round(h * fr))
                    st[f"sens_{fr:.4f}_bright_barlike"] += B.has_bar_like_band(rm[:yy], "bright", h)
                    st[f"sens_{fr:.4f}_bright_barlike_nearline"] += B.has_bar_like_band(rm[max(0, yy - int(round(0.12 * h))):yy], "bright", h)
                st["crop_any_touch_bottom"] += (B.band_touches_bottom(rm[y0:y1], "bright", h)
                                                or B.band_touches_bottom(rm[y0:y1], "dark", h))
            seen += 1
missing = len(want) - seen
if missing:
    sys.exit(f"{missing} images of train/val were not found in the cache tars under {args.cache}")

keys = ["n", "full_bright_band", "full_bright_touch_bottom", "crop_bright_band", "crop_bright_touch_bottom",
        "strip_bright_band", "full_dark_band", "full_dark_touch_bottom", "crop_dark_band",
        "crop_dark_touch_bottom", "strip_dark_band", "crop_any_touch_bottom",
        "full_bright_barlike", "crop_bright_barlike", "strip_bright_barlike", "full_dark_barlike",
        "crop_dark_barlike", "strip_dark_barlike", "full_any_barlike", "crop_any_barlike"]
sens = sorted(k for k in stats["all"] if k.startswith("sens_"))
res = {g: {k: int(v[k]) for k in keys + (sens if g == "all" else [])} for g, v in stats.items()}
a = res["all"]
pct = lambda g, k: 100.0 * res[g][k] / res[g]["n"]
threshold = 1.0
verdict = "PASS" if pct("all", "crop_any_touch_bottom") <= threshold else "FAIL"
verdict2 = "PASS" if pct("all", "crop_any_barlike") <= threshold else "FAIL: enlarge the crop and record it in SPEC section 11"
lines = [f"# E3 pre-check: bar-zone crop (bottom {B.BAR_FRAC:.2%} = {B.bar_rows(384)} of 384 cache rows)", "",
         f"Images: {a['n']} (train+val of {args.variant}; test not opened). E0 detector, run length {B.min_run(384)} rows at 384, "
         f"thresholds bright > {B.BRIGHT}, dark < {B.DARK}. 'touch bottom' = longest band ends within 2 rows of the last row.", "",
         "| quantity | % of images |", "|---|---|"]
for k in keys[1:]:
    lines.append(f"| {k} | {pct('all', k):.2f} |")
lines += ["", f"Criterion as first written in SPEC (band touching the new bottom edge <= {threshold}%): **{verdict}** "
          f"(measured {pct('all', 'crop_any_touch_bottom'):.2f}%).",
          f"Amended criterion (SPEC; band of bar-like length, {B.min_run(384)}-{round(B.MAX_RUN_768 * 384 / 768)} rows at 384, "
          f"either colour, anywhere in the cropped image, <= {threshold}%): **{verdict2}** "
          f"(measured {pct('all', 'crop_any_barlike'):.2f}%; before the crop {pct('all', 'full_any_barlike'):.2f}%).", "",
          "By class, % of images (full band bright / bright touching bottom before crop / any band touching bottom after crop):", "",
          "| class | n | full bright | full bright touch | after crop touch | after crop bar-like |", "|---|---|---|---|---|---|"]
for g in sorted(k for k in res if k != "all"):
    lines.append(f"| {g} | {res[g]['n']} | {pct(g, 'full_bright_band'):.1f} | {pct(g, 'full_bright_touch_bottom'):.1f} | {pct(g, 'crop_any_touch_bottom'):.1f} | {pct(g, 'crop_any_barlike'):.1f} |")
lines += ["", "Sensitivity (bright bar-like band left in the cropped image; 'near line' = within the last 12% of rows above the crop line):", "",
          "| crop fraction | rows removed at 384 | bright bar-like left % | near the crop line % |", "|---|---|---|---|"]
for fr in FRACS:
    lines.append(f"| {fr:.4f} | {int(round(384 * fr))} | {pct('all', f'sens_{fr:.4f}_bright_barlike'):.2f} | {pct('all', f'sens_{fr:.4f}_bright_barlike_nearline'):.2f} |")
out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
out.with_suffix(".json").write_text(json.dumps(res, indent=1), encoding="utf-8")
print("\n".join(lines))
