# Results on the test split

Runs: 14; bootstrap: 1000 resamples; 'image' = images resampled one by one, 'cluster' = whole tau=0.94 clusters resampled; CIs are for the mean over seeds (seeds resampled jointly); ± is the sample standard deviation over seeds.


## Main table

| variant | backbone | seeds | images | macro-F1 | 95% CI image | 95% CI cluster | accuracy | ECE |
|---|---|---|---|---|---|---|---|---|
| random | convnext_tiny | 3 | 3126 | 0.9684 ± 0.0038 | [0.9578, 0.9771] | [0.9579, 0.9771] | 0.9781 ± 0.0005 | 0.0187 |
| random | resnet50 | 3 | 3126 | 0.9570 ± 0.0073 | [0.9446, 0.9669] | [0.9445, 0.9670] | 0.9715 ± 0.0023 | 0.0186 |
| group:0.94 | convnext_tiny | 3 | 3123 | 0.9558 ± 0.0077 | [0.9411, 0.9673] | [0.9393, 0.9664] | 0.9719 ± 0.0011 | 0.0235 |
| group:0.94 | resnet50 | 3 | 3123 | 0.9442 ± 0.0056 | [0.9286, 0.9557] | [0.9237, 0.9567] | 0.9647 ± 0.0023 | 0.0238 |
| group:0.9 | resnet50 | 1 | 3124 | 0.9474 | [0.9305, 0.9597] | [0.9301, 0.9612] | 0.9619 | 0.0259 |
| group:0.98 | resnet50 | 1 | 3126 | 0.9607 | [0.9467, 0.9727] | [0.9465, 0.9728] | 0.9677 | 0.0214 |


## E1 (random) minus E2 (group, tau=0.94)

The two splits contain different images, so this gap mixes leakage with differences between the samples.


| backbone | metric | E1 - E2 | 95% CI image | 95% CI cluster |
|---|---|---|---|---|
| convnext_tiny | macro_f1 | +0.0126 | [-0.0039, 0.0293] | [-0.0030, 0.0318] |
| convnext_tiny | acc | +0.0062 | [-0.0008, 0.0125] | [-0.0008, 0.0132] |
| resnet50 | macro_f1 | +0.0128 | [-0.0048, 0.0316] | [-0.0061, 0.0348] |
| resnet50 | acc | +0.0069 | [-0.0005, 0.0141] | [-0.0014, 0.0150] |


## E1 test images with and without a near-duplicate cluster-mate in the training set

Same models, same images, split by whether the image's tau=0.94 cluster has a member in the E1 training set. Descriptive only: the two subsets differ in class mix and difficulty. Macro-F1 is over the classes present in the subset.


| backbone | subset | images | accuracy | 95% CI cluster | macro-F1 | 95% CI cluster |
|---|---|---|---|---|---|---|
| convnext_tiny | with neighbour in train | 732 | 0.9986 | [0.9963, 1.0000] | 0.9967 | [0.9862, 1.0000] |
| convnext_tiny | no neighbour in train | 2394 | 0.9719 | [0.9659, 0.9770] | 0.9574 | [0.9412, 0.9692] |
| resnet50 | with neighbour in train | 732 | 0.9959 | [0.9913, 0.9991] | 0.9949 | [0.9840, 0.9995] |
| resnet50 | no neighbour in train | 2394 | 0.9641 | [0.9572, 0.9699] | 0.9432 | [0.9256, 0.9574] |


## Sensitivity to the clustering threshold (ResNet-50, seed 0)

| variant | images | macro-F1 | 95% CI cluster | accuracy |
|---|---|---|---|---|
| random | 3126 | 0.9506 | [0.9318, 0.9640] | 0.9690 |
| group:0.94 | 3123 | 0.9405 | [0.9166, 0.9555] | 0.9622 |
| group:0.9 | 3124 | 0.9474 | [0.9301, 0.9612] | 0.9619 |
| group:0.98 | 3126 | 0.9607 | [0.9465, 0.9728] | 0.9677 |


## Per-class recall (mean over seeds)

| run | Biological | Fibres | Films_Coated_Surface | MEMS_devices_and_electrodes | Nanowires | Particles | Patterned_surface | Porous_Sponge | Powder | Tips |
|---|---|---|---|---|---|---|---|---|---|---|
| random / convnext_tiny | 0.991 | 0.970 | 0.867 | 0.975 | 0.985 | 0.995 | 0.971 | 0.987 | 0.964 | 0.973 |
| random / resnet50 | 0.981 | 0.985 | 0.844 | 0.966 | 0.982 | 0.986 | 0.967 | 0.962 | 0.954 | 0.968 |
| group:0.94 / convnext_tiny | 0.991 | 0.970 | 0.844 | 0.979 | 0.979 | 0.983 | 0.961 | 0.885 | 0.954 | 0.973 |
| group:0.94 / resnet50 | 0.981 | 0.985 | 0.778 | 0.967 | 0.976 | 0.974 | 0.957 | 0.872 | 0.951 | 0.970 |
| group:0.9 / resnet50 | 0.979 | 0.955 | 0.822 | 0.964 | 0.982 | 0.965 | 0.966 | 0.846 | 0.893 | 0.958 |
| group:0.98 / resnet50 | 0.993 | 1.000 | 0.911 | 0.971 | 0.973 | 0.979 | 0.948 | 0.962 | 0.977 | 0.962 |


## Most frequent confusions (true -> predicted, share of the true class, summed over seeds)

- random / convnext_tiny: Films_Coated_Surface -> Particles 5.9% (8); Films_Coated_Surface -> Nanowires 5.2% (7); Fibres -> Patterned_surface 3.0% (2); Films_Coated_Surface -> Powder 2.2% (3); Powder -> Particles 2.0% (8)
- random / resnet50: Films_Coated_Surface -> Nanowires 8.9% (12); Films_Coated_Surface -> Particles 4.4% (6); Porous_Sponge -> Nanowires 2.6% (2); Films_Coated_Surface -> Powder 2.2% (3); MEMS_devices_and_electrodes -> Patterned_surface 2.1% (43)
- group:0.94 / convnext_tiny: Films_Coated_Surface -> Particles 5.9% (8); Films_Coated_Surface -> Nanowires 4.4% (6); Porous_Sponge -> Particles 3.8% (3); Fibres -> Tips 3.0% (2); Porous_Sponge -> Nanowires 2.6% (2)
- group:0.94 / resnet50: Films_Coated_Surface -> Nanowires 8.9% (12); Films_Coated_Surface -> Particles 8.9% (12); Porous_Sponge -> Powder 6.4% (5); Porous_Sponge -> Patterned_surface 3.8% (3); Films_Coated_Surface -> Powder 3.0% (4)
- group:0.9 / resnet50: Films_Coated_Surface -> Nanowires 11.1% (5); Porous_Sponge -> Powder 7.7% (2); Powder -> Particles 5.3% (7); Fibres -> Patterned_surface 4.5% (1); Films_Coated_Surface -> Particles 4.4% (2)
- group:0.98 / resnet50: Films_Coated_Surface -> Powder 4.4% (2); Patterned_surface -> MEMS_devices_and_electrodes 4.3% (30); Porous_Sponge -> Powder 3.8% (1); Films_Coated_Surface -> Porous_Sponge 2.2% (1); Films_Coated_Surface -> Particles 2.2% (1)
