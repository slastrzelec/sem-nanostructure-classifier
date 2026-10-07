"""Tests of the E3 regions and the band detector (numpy only); dataset test needs torch."""
import numpy as np
import pytest

from semcls import bar as B


def test_region_rows_cache_and_original_height():
    assert B.bar_rows(384) == 72 and B.bar_rows(768) == 144
    assert B.region_rows("full", 384) == (0, 384)
    assert B.region_rows("bottom_strip", 384) == (312, 384)
    assert B.region_rows("top_strip", 384) == (0, 72)
    assert B.region_rows("no_bottom", 384) == (0, 312)
    assert B.region_rows("no_top", 384) == (72, 384)
    assert B.region_rows("no_bottom", 768) == (0, 624)       # rows 624-767 are the bar zone
    with pytest.raises(ValueError):
        B.region_rows("middle", 384)


def test_no_region_shares_rows_across_bar_and_image_wrongly():
    h = 384
    a, b = B.region_rows("no_bottom", h), B.region_rows("bottom_strip", h)
    assert a[1] == b[0] and (a[1] - a[0]) + (b[1] - b[0]) == h   # complementary
    a, b = B.region_rows("top_strip", h), B.region_rows("no_top", h)
    assert a[1] == b[0] and (a[1] - a[0]) + (b[1] - b[0]) == h
    assert (B.region_rows("no_bottom", h)[1] - B.region_rows("no_bottom", h)[0]
            == B.region_rows("no_top", h)[1] - B.region_rows("no_top", h)[0])  # same amount of image kept


def test_longest_run():
    assert B.longest_run([0, 1, 1, 0, 1, 1, 1, 0]) == (4, 3)
    assert B.longest_run([0, 0]) == (0, 0)
    assert B.longest_run([1, 1, 1]) == (0, 3)


def rowmeans(h, bar_start, bar_len=34, level=250.0, base=100.0):
    r = np.full(h, base)
    if bar_start is not None:
        r[bar_start:bar_start + bar_len] = level
    return r


def test_bar_removed_by_crop_for_the_e0_start_range():
    h = 384
    for start768 in (629, 663, 677, 700):                 # 5th percentile .. beyond the 95th
        r = rowmeans(h, start768 // 2, bar_len=17)       # 34 rows at 768 -> 17 rows at 384
        y0, y1 = B.region_rows("no_bottom", h)
        assert B.band(r[y0:y1], "bright")[1] < B.min_run(h)   # nothing left of the bar
        y0, y1 = B.region_rows("bottom_strip", h)
        assert B.band(r[y0:y1], "bright")[1] >= B.min_run(h) - 1  # the strip contains it


def test_band_touches_bottom_detects_leftover():
    h = 384
    r = rowmeans(h, 300, bar_len=12)                       # a bar that starts above the crop line
    cropped = r[:312]
    assert B.band_touches_bottom(cropped, "bright", h)    # still a band at the new bottom edge
    assert not B.band_touches_bottom(rowmeans(h, None)[:312], "bright", h)
    assert not B.band_touches_bottom(rowmeans(h, 100, bar_len=12)[:312], "bright", h)  # band in the middle


def test_dataset_regions_shapes_and_content():
    pytest.importorskip("torch")
    from PIL import Image
    from semcls.data import SemDataset
    import tempfile, os
    h, w = 96, 128
    a = np.zeros((h, w, 3), np.uint8)
    a[:, :, 0] = np.arange(h)[:, None] * 2               # row index encoded in the red channel
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "x.png")
        Image.fromarray(a).save(p)
        k = B.bar_rows(h)
        for region, (y0, y1) in [("full", (0, h)), ("bottom_strip", (h - k, h)), ("top_strip", (0, k)),
                                 ("no_bottom", (0, h - k)), ("no_top", (k, h))]:
            x, _ = SemDataset([p], [0], train=False, hw=(h, w), region=region)[0]
            assert tuple(x.shape) == (3, y1 - y0, w)
            assert np.array_equal(x[0].numpy(), a[y0:y1, :, 0])
            xt, _ = SemDataset([p], [0], train=True, hw=(h, w), region=region)[0]
            assert tuple(xt.shape) == (3, y1 - y0, w)    # augmentation keeps the cropped size


def test_bar_like_band_ignores_long_content_bands():
    h = 384
    assert B.has_bar_like_band(rowmeans(h, 100, bar_len=17), "bright", h)
    assert not B.has_bar_like_band(rowmeans(h, 100, bar_len=60), "bright", h)   # long band = content
    assert not B.has_bar_like_band(rowmeans(h, 100, bar_len=5), "bright", h)    # too short
    assert not B.has_bar_like_band(rowmeans(h, None), "bright", h)
    dark = np.full(h, 100.0)
    dark[200:215] = 5.0
    assert B.has_bar_like_band(dark, "dark", h) and not B.has_bar_like_band(dark, "bright", h)
    assert B.runs([0, 1, 1, 0, 1]) == [(1, 2), (4, 1)]
