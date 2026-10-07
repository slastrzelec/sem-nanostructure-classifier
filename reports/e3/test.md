# E3 info-bar test, test split

ConvNeXt-Tiny, group tau = 0.94. 3123 test images, seeds [0, 1, 2]; every variant is evaluated on the same images, so differences are paired. Bootstrap: 95% percentile intervals of the difference of seed means (1000 resamples, by images and by tau = 0.94 clusters). The intervals cover the evaluation sample only, not the variation between trainings (see sd over seeds).

## Per variant (seed mean, sd over seeds)

| variant | macro-F1 | sd | accuracy | sd | per-seed macro-F1 |
|---|---|---|---|---|---|
| with bar (reference) | 0.9558 | 0.0077 | 0.9719 | 0.0011 | 0.9472, 0.9622, 0.9579 |
| E3b bar removed (no bottom) | 0.9421 | 0.0128 | 0.9674 | 0.0047 | 0.9457, 0.9279, 0.9527 |
| E3b-ctrl top removed (no top) | 0.9560 | 0.0025 | 0.9704 | 0.0008 | 0.9585, 0.9535, 0.9561 |
| E3a bottom strip only | 0.8826 | 0.0098 | 0.9028 | 0.0061 | 0.8719, 0.8911, 0.8850 |
| E3a-ctrl top strip only | 0.8946 | 0.0056 | 0.9125 | 0.0013 | 0.8926, 0.9009, 0.8902 |

## Paired differences (A - B, percentage points)

| A - B | question | macro-F1 | 95% CI images | 95% CI clusters | accuracy | 95% CI images | 95% CI clusters |
|---|---|---|---|---|---|---|---|
| full - no_bottom | drop caused by removing the bar zone | +1.37 | [+0.41, +2.33] | [+0.29, +2.40] | +0.45 | [+0.17, +0.76] | [+0.12, +0.78] |
| full - no_top | drop caused by removing the same amount from the top (control) | -0.03 | [-0.75, +0.70] | [-0.75, +0.76] | +0.15 | [-0.14, +0.43] | [-0.15, +0.44] |
| no_top - no_bottom | control minus bar removed | +1.39 | [+0.33, +2.45] | [+0.19, +2.49] | +0.30 | [-0.02, +0.67] | [-0.09, +0.67] |
| bottom_strip - top_strip | bar-zone strip minus top strip | -1.19 | [-3.18, +0.82] | [-3.32, +1.17] | -0.97 | [-1.88, -0.03] | [-2.08, +0.13] |

## Per-class recall (seed mean)

| class | n | full | no_bottom | no_top | bottom_strip | top_strip |
|---|---|---|---|---|---|---|
| Biological | 144 | 0.991 | 0.988 | 0.991 | 0.958 | 0.970 |
| Fibres | 22 | 0.970 | 0.985 | 0.909 | 0.864 | 0.879 |
| Films_Coated_Surface | 45 | 0.844 | 0.733 | 0.867 | 0.733 | 0.711 |
| MEMS_devices_and_electrodes | 687 | 0.979 | 0.977 | 0.972 | 0.886 | 0.905 |
| Nanowires | 562 | 0.979 | 0.971 | 0.979 | 0.942 | 0.942 |
| Particles | 573 | 0.983 | 0.986 | 0.985 | 0.930 | 0.938 |
| Patterned_surface | 698 | 0.961 | 0.958 | 0.960 | 0.882 | 0.878 |
| Porous_Sponge | 26 | 0.885 | 0.885 | 0.923 | 0.718 | 0.846 |
| Powder | 130 | 0.954 | 0.936 | 0.956 | 0.933 | 0.928 |
| Tips | 236 | 0.973 | 0.972 | 0.969 | 0.857 | 0.908 |

Chance level of macro-F1 on 10 classes is about 0.10. Classes with few images (Porous_Sponge, Fibres, Films_Coated_Surface): one image changes recall by several points.
