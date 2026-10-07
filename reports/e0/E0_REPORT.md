# E0 - Data audit (NFFA-EUROPE 100% SEM Dataset)

Generated 2026-10-05 09:35:38 by `scripts/e0_extract.py` + `scripts/e0_report.py` (script sha256 `81301efe128eca64...`). All numbers below come from `reports/e0/e0_summary.json`.

## 1. Integrity

All 10 archives were streamed from the downloaded zip; MD5 and SHA-256 of each are in `checksums/`. Total images: **21,169** (matches the 21,169 stated by the publisher).

## 2. Classes

| Class | Images | Share % |
|---|---:|---:|
| Patterned_surface | 4,752 | 22.4 |
| MEMS_devices_and_electrodes | 4,583 | 21.6 |
| Particles | 3,905 | 18.4 |
| Nanowires | 3,815 | 18.0 |
| Tips | 1,621 | 7.7 |
| Biological | 962 | 4.5 |
| Powder | 898 | 4.2 |
| Films_Coated_Surface | 309 | 1.5 |
| Porous_Sponge | 174 | 0.8 |
| Fibres | 150 | 0.7 |

Imbalance (largest / smallest): **31.7x**. With a 15% test split the smallest classes would have only 22 (Fibres), 26 (Porous_Sponge) and 46 (Films_Coated_Surface) test images, so per-class metrics will have wide confidence intervals.

## 3. Formats

1022x766: 5, 1023x767: 55, 1024x768: 20,893, 2048x1536: 38, 3072x2304: 178. All files are RGB JPEG; 13.5% of them are visually grayscale (the rest carry a colour tint). Non-standard sizes (2048x1536, 3072x2304) occur only in some classes (`resolution_by_class` in the JSON), so image size is a potential shortcut and will be neutralised by resizing.

## 4. Duplicates

- Exact duplicates (same SHA-256): **318 groups, 332 extra copies (1.6%)**, none across classes. Collapse before splitting.

- Near duplicates `phash<=2&thumb_corr>=0.95`: 7.2% of images in a cluster; 446 clusters, largest 303; pairs same class 2,218, cross class 1,009.
- Near duplicates `phash<=4&thumb_corr>=0.95`: 11.95% of images in a cluster; 591 clusters, largest 657; pairs same class 8,648, cross class 6,392.
- Near duplicates `phash<=8&thumb_corr>=0.95`: 21.53% of images in a cluster; 737 clusters, largest 2012; pairs same class 43,007, cross class 50,669.

Caveat: clusters are single-linkage, so chains of low-detail images inflate the largest cluster (at T<=4 it mixes 9 classes, mean contrast below the dataset median). Single-class clusters (e.g. 152 MEMS images) look like real same-sample series. The pHash+thumbnail test is only a pre-filter; the group definition for E2 will use embedding similarity and be checked on a visual sample.

## 5. Information bar

Definition: longest run of >= 20 consecutive rows with mean gray > 225.0 (bright) or < 30.0 (dark).

- Bright band present in **60.3%** of images (start row median 663 of 768, length median 34 rows); dark band in 14.1%; neither detected in 33.1%.
- Bright band by class (%): Biological 72.2, Fibres 80.7, Films_Coated_Surface 20.1, MEMS_devices_and_electrodes 46.1, Nanowires 68.0, Particles 53.6, Patterned_surface 64.5, Porous_Sponge 36.2, Powder 72.6, Tips 80.8.
- Dark band by class (%): Biological 0.1, Fibres 0.0, Films_Coated_Surface 1.0, MEMS_devices_and_electrodes 21.0, Nanowires 13.0, Particles 18.2, Patterned_surface 7.1, Porous_Sponge 0.6, Powder 0.2, Tips 29.3.

Bar presence, colour and band type differ strongly between classes, so a shortcut is plausible. E0 only describes this; E3 measures how much a model uses it.

## 6. Consequences for the next steps

1. Collapse exact duplicates before any split.
2. Group-aware split by embedding-similarity clusters (E2), stratified, with a minimum per-class count in val/test.
3. Report macro-F1 with bootstrap CIs; per-class results for the three small classes are indicative only.
4. Neutralise size/colour cues (resize, optional grayscale) and run E3 (bar crop vs bar only).
