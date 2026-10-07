"""Regions of the image used in E3 (info-bar shortcut test) and the band detector of E0 (numpy only).

The bar is not at a fixed row (E0: start row median 663 of 768, 5th percentile 629), so one rule is
applied to every image: BAR_FRAC of the height is the 'bar zone' (bottom 18.75%: rows 624-767 of 768,
rows 312-383 of the 384-row cache, k = 72 rows)."""
import numpy as np

BAR_FRAC = 0.1875
REGIONS = ("full", "bottom_strip", "top_strip", "no_bottom", "no_top")
# thresholds of the E0 detector (mean gray of a full-width row); the run length of E0 (20 rows at 768)
# is scaled with the height
BRIGHT, DARK, MIN_RUN_768 = 225.0, 30.0, 20


def bar_rows(h):
    """Height of the bar zone for an image with h rows."""
    return int(round(h * BAR_FRAC))


def region_rows(region, h):
    """(y0, y1) of the kept rows."""
    k = bar_rows(h)
    if region == "full":
        return 0, h
    if region == "bottom_strip":
        return h - k, h
    if region == "top_strip":
        return 0, k
    if region == "no_bottom":
        return 0, h - k
    if region == "no_top":
        return k, h
    raise ValueError(f"unknown region {region!r}; expected one of {REGIONS}")


def longest_run(mask):
    """(start, length) of the longest run of True; (0, 0) if none."""
    best, best_s, cur, cur_s = 0, 0, 0, 0
    for i, v in enumerate(np.asarray(mask, bool)):
        if v:
            if cur == 0:
                cur_s = i
            cur += 1
            if cur > best:
                best, best_s = cur, cur_s
        else:
            cur = 0
    return best_s, best


def min_run(h):
    return max(1, int(round(MIN_RUN_768 * h / 768)))


def band(rowmean, kind):
    """(start, length) of the longest bright (> BRIGHT) or dark (< DARK) run of row means."""
    r = np.nan_to_num(np.asarray(rowmean, np.float64), nan=0.0 if kind == "bright" else 255.0)
    return longest_run(r > BRIGHT if kind == "bright" else r < DARK)


def band_touches_bottom(rowmean, kind, h_full, tol_768=4):
    """True if the longest band has the minimal E0 length (scaled to h_full, the height of the
    uncropped image) and ends within tol rows of the last row of `rowmean`."""
    s, ln = band(rowmean, kind)
    tol = max(1, int(round(tol_768 * h_full / 768)))
    return ln >= min_run(h_full) and s + ln >= len(rowmean) - tol


MAX_RUN_768 = 60   # E0: 99th percentile of bright bar length is 53 rows at 768; longer bands are image content


def runs(mask):
    """All runs of True as (start, length)."""
    out, cur_s, cur = [], 0, 0
    for i, v in enumerate(np.asarray(mask, bool)):
        if v:
            if cur == 0:
                cur_s = i
            cur += 1
        elif cur:
            out.append((cur_s, cur))
            cur = 0
    if cur:
        out.append((cur_s, cur))
    return out


def has_bar_like_band(rowmean, kind, h_full):
    """True if some bright/dark run has a bar-like length: >= the E0 minimum and <= MAX_RUN_768 (scaled).
    Used to look for a bar (fragment) left in a cropped image; long bands are image content."""
    r = np.nan_to_num(np.asarray(rowmean, np.float64), nan=0.0 if kind == "bright" else 255.0)
    lo, hi = min_run(h_full), max(min_run(h_full), int(round(MAX_RUN_768 * h_full / 768)))
    return any(lo <= ln <= hi for _, ln in runs(r > BRIGHT if kind == "bright" else r < DARK))
