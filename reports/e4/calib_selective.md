# E4a/E4b: temperature scaling and selective prediction

Group tau = 0.94, seeds 0-2, stored logits (3128 val and 3123 test images). Post-hoc: test logits were seen before this analysis; T and the threshold theta are fitted on val only and applied unchanged to test (SPEC 2026-10-07). Intervals: 95% percentile bootstrap, image / cluster (tau = 0.94), 1000 resamples, seed means, T and theta fixed; they cover the evaluation sample only, not the variation between trainings.

## E4a Temperature scaling (seed mean; per-seed T in the last column)

| model | T | NLL val before > after* | NLL test before > after | ECE val before > after* | ECE test before > after | d ECE test (after - before), img / clu | d NLL test, img / clu | per-seed T |
|---|---|---|---|---|---|---|---|---|
| ConvNeXt-Tiny | 3.263 | 0.2462 > 0.1050 | 0.2417 > 0.1030 | 0.0241 > 0.0072 | 0.0235 > 0.0064 | -0.0172 [-0.0184, -0.0115] / [-0.0187, -0.0112] | -0.1387 [-0.1775, -0.1019] / [-0.1786, -0.0992] | 3.21, 3.29, 3.28 |
| ResNet-50 | 2.233 | 0.1957 > 0.1287 | 0.1874 > 0.1242 | 0.0238 > 0.0086 | 0.0238 > 0.0085 | -0.0153 [-0.0177, -0.0085] / [-0.0183, -0.0076] | -0.0632 [-0.0832, -0.0447] / [-0.0878, -0.0413] | 1.89, 2.50, 2.31 |

*in-sample for T. Accuracy and macro-F1 are unchanged by temperature scaling (argmax invariant, asserted).

## E4b Selective prediction (rule: largest coverage with val accuracy of accepted >= 0.99, >= 50 images)

| model | theta (mean) | val coverage | val sel. acc | test coverage, img / clu | test sel. acc, img / clu | error capture (test) | overall test acc (random rejection) | AURC test | sel. acc @ cov 0.90 / 0.95 |
|---|---|---|---|---|---|---|---|---|---|
| ConvNeXt-Tiny | 0.9054 | 0.937 | 0.9901 | 0.940 [0.933, 0.947] / [0.930, 0.949] | 0.9915 [0.9886, 0.9942] / [0.9884, 0.9943] | 0.714 | 0.9719 | 0.0025 | 0.9938 / 0.9903 |
| ResNet-50 | 0.8797 | 0.903 | 0.9902 | 0.908 [0.900, 0.916] / [0.895, 0.920] | 0.9921 [0.9897, 0.9945] / [0.9891, 0.9946] | 0.799 | 0.9647 | 0.0030 | 0.9930 / 0.9840 |

### Per seed

| run | T | theta | val cov | val sel. acc | test cov | test sel. acc | test acc | ECE test before > after |
|---|---|---|---|---|---|---|---|---|
| ConvNeXt-Tiny s0 | 3.214 | 0.8928 | 0.939 | 0.9901 | 0.944 | 0.9925 | 0.9709 | 0.0241 > 0.0084 |
| ConvNeXt-Tiny s1 | 3.292 | 0.9083 | 0.943 | 0.9902 | 0.939 | 0.9911 | 0.9731 | 0.0225 > 0.0055 |
| ConvNeXt-Tiny s2 (demo) | 3.283 | 0.9150 | 0.929 | 0.9900 | 0.938 | 0.9908 | 0.9718 | 0.0240 > 0.0051 |
| ResNet-50 s0 | 1.886 | 0.8856 | 0.892 | 0.9903 | 0.896 | 0.9904 | 0.9622 | 0.0220 > 0.0113 |
| ResNet-50 s1 | 2.500 | 0.9025 | 0.898 | 0.9900 | 0.900 | 0.9943 | 0.9667 | 0.0244 > 0.0082 |
| ResNet-50 s2 | 2.312 | 0.8510 | 0.918 | 0.9903 | 0.928 | 0.9917 | 0.9651 | 0.0250 > 0.0059 |

### Rejection rate per class on test (share of the class below theta, seed mean)

| class | n test | ConvNeXt-Tiny | ResNet-50 |
|---|---|---|---|
| Biological | 144 | 0.016 | 0.042 |
| Fibres | 22 | 0.242 | 0.136 |
| Films_Coated_Surface | 45 | 0.370 | 0.459 |
| MEMS_devices_and_electrodes | 687 | 0.038 | 0.070 |
| Nanowires | 562 | 0.055 | 0.072 |
| Particles | 573 | 0.044 | 0.084 |
| Patterned_surface | 698 | 0.057 | 0.088 |
| Porous_Sponge | 26 | 0.436 | 0.231 |
| Powder | 130 | 0.128 | 0.215 |
| Tips | 236 | 0.055 | 0.106 |

Reading notes: theta is selected on val, so the val selective accuracy is optimistic and the val-to-test gap is part of the result. Classes with few images (Porous_Sponge, Fibres, Films_Coated_Surface): one image changes a rate by several points. ECE of about 3100 images is biased upwards; interpret the interval of the difference, not small ECE values.
