# Results on the val split

Runs: 14; bootstrap: 1000 resamples; 'image' = images resampled one by one, 'cluster' = whole tau=0.94 clusters resampled; CIs are for the mean over seeds (seeds resampled jointly); ± is the sample standard deviation over seeds.

Note: validation sets of different variants contain different images; differences between variants are not test results.


## Main table

| variant | backbone | seeds | images | macro-F1 | 95% CI image | 95% CI cluster | accuracy | ECE |
|---|---|---|---|---|---|---|---|---|
| random | convnext_tiny | 3 | 3126 | 0.9696 ± 0.0024 | [0.9585, 0.9784] | [0.9571, 0.9786] | 0.9734 ± 0.0027 | 0.0234 |
| random | resnet50 | 3 | 3126 | 0.9611 ± 0.0065 | [0.9511, 0.9702] | [0.9494, 0.9699] | 0.9664 ± 0.0022 | 0.0220 |
| group:0.94 | convnext_tiny | 3 | 3128 | 0.9588 ± 0.0035 | [0.9462, 0.9693] | [0.9432, 0.9693] | 0.9718 ± 0.0030 | 0.0241 |
| group:0.94 | resnet50 | 3 | 3128 | 0.9455 ± 0.0026 | [0.9315, 0.9569] | [0.9258, 0.9602] | 0.9634 ± 0.0020 | 0.0238 |
| group:0.9 | resnet50 | 1 | 3127 | 0.9385 | [0.9201, 0.9540] | [0.9126, 0.9529] | 0.9613 | 0.0266 |
| group:0.98 | resnet50 | 1 | 3125 | 0.9549 | [0.9400, 0.9680] | [0.9381, 0.9678] | 0.9667 | 0.0227 |


## E1 (random) minus E2 (group, tau=0.94)

The two splits contain different images, so this gap mixes leakage with differences between the samples.


| backbone | metric | E1 - E2 | 95% CI image | 95% CI cluster |
|---|---|---|---|---|
| convnext_tiny | macro_f1 | +0.0108 | [-0.0045, 0.0263] | [-0.0058, 0.0294] |
| convnext_tiny | acc | +0.0017 | [-0.0048, 0.0088] | [-0.0062, 0.0095] |
| resnet50 | macro_f1 | +0.0156 | [0.0004, 0.0311] | [-0.0019, 0.0371] |
| resnet50 | acc | +0.0030 | [-0.0045, 0.0106] | [-0.0062, 0.0129] |


## E1 val images with and without a near-duplicate cluster-mate in the training set

Same models, same images, split by whether the image's tau=0.94 cluster has a member in the E1 training set. Descriptive only: the two subsets differ in class mix and difficulty. Macro-F1 is over the classes present in the subset.


| backbone | subset | images | accuracy | 95% CI cluster | macro-F1 | 95% CI cluster |
|---|---|---|---|---|---|---|
| convnext_tiny | with neighbour in train | 705 | 0.9972 | [0.9944, 0.9995] | 0.9980 | [0.9957, 0.9996] |
| convnext_tiny | no neighbour in train | 2421 | 0.9665 | [0.9598, 0.9734] | 0.9606 | [0.9447, 0.9722] |
| resnet50 | with neighbour in train | 705 | 0.9957 | [0.9915, 0.9990] | 0.9957 | [0.9890, 0.9991] |
| resnet50 | no neighbour in train | 2421 | 0.9579 | [0.9506, 0.9647] | 0.9508 | [0.9377, 0.9631] |


## Sensitivity to the clustering threshold (ResNet-50, seed 0)

| variant | images | macro-F1 | 95% CI cluster | accuracy |
|---|---|---|---|---|
| random | 3126 | 0.9603 | [0.9466, 0.9709] | 0.9655 |
| group:0.94 | 3128 | 0.9466 | [0.9242, 0.9642] | 0.9623 |
| group:0.9 | 3127 | 0.9385 | [0.9126, 0.9529] | 0.9613 |
| group:0.98 | 3125 | 0.9549 | [0.9381, 0.9678] | 0.9667 |


## Per-class recall (mean over seeds)

| run | Biological | Fibres | Films_Coated_Surface | MEMS_devices_and_electrodes | Nanowires | Particles | Patterned_surface | Porous_Sponge | Powder | Tips |
|---|---|---|---|---|---|---|---|---|---|---|
| random / convnext_tiny | 0.991 | 0.942 | 0.920 | 0.967 | 0.974 | 0.990 | 0.962 | 1.000 | 0.987 | 0.975 |
| random / resnet50 | 0.981 | 0.928 | 0.848 | 0.966 | 0.965 | 0.981 | 0.959 | 1.000 | 0.964 | 0.972 |
| group:0.94 / convnext_tiny | 0.972 | 0.957 | 0.819 | 0.984 | 0.977 | 0.985 | 0.961 | 0.949 | 0.964 | 0.956 |
| group:0.94 / resnet50 | 0.963 | 0.971 | 0.746 | 0.972 | 0.970 | 0.980 | 0.951 | 0.987 | 0.964 | 0.955 |
| group:0.9 / resnet50 | 0.931 | 1.000 | 0.761 | 0.959 | 0.982 | 0.976 | 0.961 | 0.846 | 0.923 | 0.970 |
| group:0.98 / resnet50 | 0.993 | 1.000 | 0.848 | 0.983 | 0.956 | 0.976 | 0.960 | 1.000 | 0.946 | 0.958 |


## Most frequent confusions (true -> predicted, share of the true class, summed over seeds)

- random / convnext_tiny: Films_Coated_Surface -> Powder 3.6% (5); Fibres -> Powder 2.9% (2); Patterned_surface -> MEMS_devices_and_electrodes 2.6% (55); MEMS_devices_and_electrodes -> Patterned_surface 2.1% (44); Fibres -> Porous_Sponge 1.4% (1)
- random / resnet50: Fibres -> MEMS_devices_and_electrodes 4.3% (3); Films_Coated_Surface -> Powder 3.6% (5); Films_Coated_Surface -> Patterned_surface 3.6% (5); Patterned_surface -> MEMS_devices_and_electrodes 3.2% (66); Films_Coated_Surface -> Nanowires 2.9% (4)
- group:0.94 / convnext_tiny: Films_Coated_Surface -> Nanowires 12.3% (17); Films_Coated_Surface -> Particles 4.3% (6); Powder -> Particles 3.1% (12); Fibres -> Tips 2.9% (2); Porous_Sponge -> Tips 2.6% (2)
- group:0.94 / resnet50: Films_Coated_Surface -> Nanowires 18.8% (26); Films_Coated_Surface -> Particles 4.3% (6); Patterned_surface -> MEMS_devices_and_electrodes 3.3% (70); Powder -> Particles 2.5% (10); MEMS_devices_and_electrodes -> Patterned_surface 1.8% (38)
- group:0.9 / resnet50: Films_Coated_Surface -> Nanowires 13.0% (6); Porous_Sponge -> Nanowires 11.5% (3); Powder -> Particles 4.6% (6); Films_Coated_Surface -> Patterned_surface 4.3% (2); Films_Coated_Surface -> Powder 4.3% (2)
- group:0.98 / resnet50: Films_Coated_Surface -> Nanowires 6.5% (3); Films_Coated_Surface -> Particles 4.3% (2); Powder -> Particles 3.1% (4); Patterned_surface -> MEMS_devices_and_electrodes 3.0% (21); Tips -> Patterned_surface 3.0% (7)
