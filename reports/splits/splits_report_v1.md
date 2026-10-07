# Splits report

Seed 42, ratios (0.7, 0.15, 0.15), images 20837, embeddings sha256 `8b28e63456116f58`.


## E1 random stratified

Share of images: train 0.700, val 0.150, test 0.150

| class | train | val | test | val+test min |
|---|---|---|---|---|
| Biological | 671 | 144 | 144 | 144 |
| Fibres | 105 | 23 | 22 | 22 |
| Films_Coated_Surface | 213 | 46 | 45 | 45 |
| MEMS_devices_and_electrodes | 3207 | 687 | 687 | 687 |
| Nanowires | 2622 | 562 | 562 | 562 |
| Particles | 2676 | 573 | 574 | 573 |
| Patterned_surface | 3258 | 698 | 699 | 698 |
| Porous_Sponge | 120 | 26 | 26 | 26 |
| Powder | 610 | 130 | 131 | 130 |
| Tips | 1103 | 237 | 236 | 236 |

- pairs with cosine >= 0.9 that cross splits: **21943** (of 47265)
- pairs with cosine >= 0.94 that cross splits: **5982** (of 13103)
- pairs with cosine >= 0.98 that cross splits: **534** (of 1368)

## E2 group-aware, tau=0.9

Share of images: train 0.684, val 0.167, test 0.149

| class | train | val | test | val+test min |
|---|---|---|---|---|
| Biological | 671 | 144 | 144 | 144 |
| Fibres | 105 | 23 | 22 | 22 |
| Films_Coated_Surface | 212 | 46 | 46 | 46 |
| MEMS_devices_and_electrodes | 3207 | 687 | 687 | 687 |
| Nanowires | 2622 | 562 | 562 | 562 |
| Particles | 2532 | 724 | 567 | 567 |
| Patterned_surface | 3259 | 698 | 698 | 698 |
| Porous_Sponge | 120 | 26 | 26 | 26 |
| Powder | 609 | 131 | 131 | 131 |
| Tips | 914 | 434 | 228 | 228 |

- pairs with cosine >= 0.9 that cross splits: **0**
- clusters spanning more than one split: **0**
- clusters: clusters=13307, singletons=11444, images_in_non_singleton=9393, largest=1205, p99_size=6.0, mixed_class_clusters=12, images_in_mixed=2341

## E2 group-aware, tau=0.94

Share of images: train 0.700, val 0.150, test 0.150

| class | train | val | test | val+test min |
|---|---|---|---|---|
| Biological | 671 | 144 | 144 | 144 |
| Fibres | 105 | 23 | 22 | 22 |
| Films_Coated_Surface | 212 | 46 | 46 | 46 |
| MEMS_devices_and_electrodes | 3207 | 687 | 687 | 687 |
| Nanowires | 2622 | 562 | 562 | 562 |
| Particles | 2677 | 573 | 573 | 573 |
| Patterned_surface | 3259 | 698 | 698 | 698 |
| Porous_Sponge | 120 | 26 | 26 | 26 |
| Powder | 609 | 131 | 131 | 131 |
| Tips | 1104 | 236 | 236 | 236 |

- pairs with cosine >= 0.94 that cross splits: **0**
- clusters spanning more than one split: **0**
- clusters: clusters=16811, singletons=15318, images_in_non_singleton=5519, largest=295, p99_size=4.0, mixed_class_clusters=2, images_in_mixed=7

## E2 group-aware, tau=0.98

Share of images: train 0.700, val 0.150, test 0.150

| class | train | val | test | val+test min |
|---|---|---|---|---|
| Biological | 671 | 144 | 144 | 144 |
| Fibres | 105 | 23 | 22 | 22 |
| Films_Coated_Surface | 212 | 46 | 46 | 46 |
| MEMS_devices_and_electrodes | 3207 | 687 | 687 | 687 |
| Nanowires | 2622 | 562 | 562 | 562 |
| Particles | 2677 | 573 | 573 | 573 |
| Patterned_surface | 3259 | 698 | 698 | 698 |
| Porous_Sponge | 120 | 26 | 26 | 26 |
| Powder | 609 | 131 | 131 | 131 |
| Tips | 1104 | 236 | 236 | 236 |

- pairs with cosine >= 0.98 that cross splits: **0**
- clusters spanning more than one split: **0**
- clusters: clusters=20099, singletons=19639, images_in_non_singleton=1198, largest=31, p99_size=2.0, mixed_class_clusters=0, images_in_mixed=0
