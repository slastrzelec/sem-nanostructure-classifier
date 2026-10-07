# E3 info-bar test, val split

ConvNeXt-Tiny, group tau = 0.94. 3128 val images, seeds [0, 1, 2]; every variant is evaluated on the same images, so differences are paired. Bootstrap: 95% percentile intervals of the difference of seed means (1000 resamples, by images and by tau = 0.94 clusters). The intervals cover the evaluation sample only, not the variation between trainings (see sd over seeds).

## Per variant (seed mean, sd over seeds)

| variant | macro-F1 | sd | accuracy | sd | per-seed macro-F1 |
|---|---|---|---|---|---|
| with bar (reference) | 0.9588 | 0.0035 | 0.9718 | 0.0030 | 0.9551, 0.9592, 0.9619 |
| E3b bar removed (no bottom) | 0.9570 | 0.0070 | 0.9676 | 0.0013 | 0.9548, 0.9514, 0.9648 |
| E3b-ctrl top removed (no top) | 0.9678 | 0.0027 | 0.9725 | 0.0016 | 0.9709, 0.9669, 0.9657 |
| E3a bottom strip only | 0.8887 | 0.0030 | 0.9012 | 0.0039 | 0.8856, 0.8889, 0.8916 |
| E3a-ctrl top strip only | 0.8958 | 0.0045 | 0.8982 | 0.0007 | 0.9009, 0.8923, 0.8943 |

## Paired differences (A - B, percentage points)

| A - B | question | macro-F1 | 95% CI images | 95% CI clusters | accuracy | 95% CI images | 95% CI clusters |
|---|---|---|---|---|---|---|---|
| full - no_bottom | drop caused by removing the bar zone | +0.18 | [-0.62, +0.96] | [-0.64, +0.96] | +0.42 | [+0.11, +0.76] | [+0.11, +0.73] |
| full - no_top | drop caused by removing the same amount from the top (control) | -0.90 | [-1.67, -0.13] | [-1.80, -0.21] | -0.07 | [-0.40, +0.27] | [-0.44, +0.27] |
| no_top - no_bottom | control minus bar removed | +1.08 | [+0.26, +1.97] | [+0.33, +2.02] | +0.49 | [+0.14, +0.85] | [+0.13, +0.88] |
| bottom_strip - top_strip | bar-zone strip minus top strip | -0.71 | [-2.36, +0.91] | [-2.88, +1.31] | +0.30 | [-0.74, +1.32] | [-0.97, +1.84] |

## Per-class recall (seed mean)

| class | n | full | no_bottom | no_top | bottom_strip | top_strip |
|---|---|---|---|---|---|---|
| Biological | 144 | 0.972 | 0.975 | 0.981 | 0.912 | 0.933 |
| Fibres | 23 | 0.957 | 0.971 | 0.986 | 0.942 | 0.913 |
| Films_Coated_Surface | 46 | 0.819 | 0.812 | 0.862 | 0.746 | 0.710 |
| MEMS_devices_and_electrodes | 687 | 0.984 | 0.985 | 0.979 | 0.906 | 0.896 |
| Nanowires | 562 | 0.977 | 0.968 | 0.970 | 0.953 | 0.907 |
| Particles | 574 | 0.985 | 0.983 | 0.985 | 0.907 | 0.925 |
| Patterned_surface | 698 | 0.961 | 0.954 | 0.962 | 0.868 | 0.870 |
| Porous_Sponge | 26 | 0.949 | 0.949 | 0.987 | 0.756 | 0.885 |
| Powder | 131 | 0.964 | 0.952 | 0.975 | 0.944 | 0.957 |
| Tips | 237 | 0.956 | 0.954 | 0.972 | 0.861 | 0.885 |

Chance level of macro-F1 on 10 classes is about 0.10. Classes with few images (Porous_Sponge, Fibres, Films_Coated_Surface): one image changes recall by several points.
