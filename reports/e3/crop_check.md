# E3 pre-check: bar-zone crop (bottom 18.75% = 72 of 384 cache rows)

Images: 17714 (train+val of group:0.94; test not opened). E0 detector, run length 10 rows at 384, thresholds bright > 225.0, dark < 30.0. 'touch bottom' = longest band ends within 2 rows of the last row.

| quantity | % of images |
|---|---|
| full_bright_band | 57.69 |
| full_bright_touch_bottom | 0.29 |
| crop_bright_band | 2.97 |
| crop_bright_touch_bottom | 0.77 |
| strip_bright_band | 57.28 |
| full_dark_band | 14.23 |
| full_dark_touch_bottom | 1.79 |
| crop_dark_band | 12.84 |
| crop_dark_touch_bottom | 4.04 |
| strip_dark_band | 9.99 |
| crop_any_touch_bottom | 4.82 |
| full_bright_barlike | 57.04 |
| crop_bright_barlike | 1.07 |
| strip_bright_barlike | 56.68 |
| full_dark_barlike | 6.11 |
| crop_dark_barlike | 5.14 |
| strip_dark_barlike | 2.22 |
| full_any_barlike | 60.84 |
| crop_any_barlike | 6.19 |

Criterion as first written in SPEC (band touching the new bottom edge <= 1.0%): **FAIL** (measured 4.82%).
Amended criterion (SPEC; band of bar-like length, 10-30 rows at 384, either colour, anywhere in the cropped image, <= 1.0%): **FAIL: enlarge the crop and record it in SPEC section 11** (measured 6.19%; before the crop 60.84%).

By class, % of images (full band bright / bright touching bottom before crop / any band touching bottom after crop):

| class | n | full bright | full bright touch | after crop touch | after crop bar-like |
|---|---|---|---|---|---|
| Biological | 815 | 70.1 | 0.2 | 0.4 | 0.1 |
| Fibres | 128 | 82.8 | 0.0 | 0.0 | 0.0 |
| Films_Coated_Surface | 259 | 20.5 | 0.0 | 0.4 | 0.4 |
| MEMS_devices_and_electrodes | 3894 | 37.1 | 0.3 | 2.6 | 11.6 |
| Nanowires | 3184 | 68.2 | 0.1 | 4.9 | 4.6 |
| Particles | 3250 | 53.6 | 0.0 | 6.5 | 6.7 |
| Patterned_surface | 3957 | 62.6 | 0.4 | 1.2 | 5.2 |
| Porous_Sponge | 146 | 30.1 | 0.0 | 0.0 | 0.0 |
| Powder | 741 | 75.8 | 0.1 | 0.0 | 0.0 |
| Tips | 1340 | 78.2 | 1.3 | 24.9 | 5.3 |

Sensitivity (bright bar-like band left in the cropped image; 'near line' = within the last 12% of rows above the crop line):

| crop fraction | rows removed at 384 | bright bar-like left % | near the crop line % |
|---|---|---|---|
| 0.1500 | 58 | 8.65 | 8.07 |
| 0.1875 | 72 | 1.07 | 0.38 |
| 0.2000 | 77 | 1.09 | 0.34 |
| 0.2200 | 84 | 1.09 | 0.38 |
| 0.2500 | 96 | 1.08 | 0.37 |
| 0.3000 | 115 | 1.04 | 0.39 |
