---
license: mit
library_name: pytorch
pipeline_tag: image-classification
tags:
  - scanning-electron-microscopy
  - nanostructures
  - convnext
  - calibration
  - leakage-free-evaluation
---

# SEM nanostructure classifier (ConvNeXt-Tiny, group split, seed 2)

ConvNeXt-Tiny fine-tuned to classify scanning-electron-microscopy (SEM) images into 10 nanostructure classes. It is the checkpoint behind the demo of the project [`sem-nanostructure-classifier`](https://github.com/slastrzelec/sem-nanostructure-classifier), whose goal is an honest, leakage-free evaluation rather than a leaderboard score.

## Files

| file | sha256 |
|---|---|
| `convnext_tiny_group094_s2_best.pt` (PyTorch `state_dict`, 111,365,791 bytes) | `7e856feee47b86c3fb97d9a581bd7a8c9f1bc24454654e82330003fd0f3a194f` |

Load with `torch.load(path, map_location="cpu", weights_only=True)` into `torchvision.models.convnext_tiny` whose last layer is `Linear(768, 10)`; verify the hash first. The project repository contains the exact inference code (`src/semcls/infer.py`).

## Intended use and not intended use

For research, teaching and as a demonstration of leakage-aware evaluation on SEM images. **Not** for industrial, quality-control or scientific conclusions: it knows only 10 classes from one dataset and returns one of them for any input, including images that are not SEM images. There is no out-of-distribution detector.

Classes: Biological, Fibres, Films_Coated_Surface, MEMS_devices_and_electrodes, Nanowires, Particles, Patterned_surface, Porous_Sponge, Powder, Tips.

## Training data

NFFA-EUROPE "100% SEM Dataset" (Aversa, Modarres, Cozzini, Ciancio; *Scientific Data* 5, 180172, 2018; DOI 10.23728/b2share.80df8606fcdb4b2bae1656f0dc6db8ba; CC BY). 21,169 images, 332 exact duplicate copies removed, 20,837 used, resized to 512 x 384. Split 70/15/15 with **whole near-duplicate clusters** assigned to one split (clusters = connected components of cosine similarity >= 0.94 on frozen ResNet-50 features), so no near-duplicate crosses train, validation and test. Training images: 14,586.

## Training procedure

ImageNet-pretrained ConvNeXt-Tiny, full fine-tuning; AdamW (learning rate 1e-4 backbone, 1e-3 head, weight decay 0.01), 1 warm-up epoch then cosine decay, 12 epochs, effective batch 32, mixed precision, horizontal flip and a light random crop; no class weights; epoch chosen by validation macro-F1 (seed 2: 0.962). One fixed configuration, no hyper-parameter search. Trained on a free Kaggle T4 GPU.

## Evaluation (test split, 3,123 images, evaluated once)

| | macro-F1 | accuracy |
|---|---|---|
| this checkpoint (seed 2) | 0.958 | 0.972 |
| mean of 3 seeds, ConvNeXt-Tiny | 0.956 (95% CI over clusters 0.939-0.966) | 0.972 |

Seeds differ by about 1 pp macro-F1; the confidence intervals describe the test sample only, not the variation between trainings. Per-class recall of the 3-seed mean: Films_Coated_Surface 0.84 and Porous_Sponge 0.89 are the weakest, both with fewer than 50 test images; the others are 0.95-0.99.

**Calibration.** The raw model is over-confident. Temperature T = 3.283 (fitted on the validation split) reduces the test expected calibration error of this checkpoint from 0.0240 to 0.0051. The demo reports the temperature-scaled probability.

**Selective prediction.** Predictions with calibrated confidence below 0.915 are flagged "uncertain" (threshold chosen on validation so that accepted validation images are at least 99% correct). On the test split this checkpoint accepts 93.8% of the images with 99.1% accuracy (overall 97.2%). Rejections concentrate in rare classes (about a quarter to a half of Fibres, Films_Coated_Surface and Porous_Sponge images, 2-6% of the large classes). The calibration and threshold analysis was done after the test logits were known; all parameters were fitted on validation only.

## Limitations and risks

- Single lab, single dataset; no external test set. Behaviour on images from other instruments, magnifications or preparations is unknown.
- The group definition (embedding similarity) is a proxy for "same sample".
- Rare classes have 22-46 test images; their metrics are uncertain.
- Many SEM images carry an information bar at the bottom. Removing the bottom 18.75% of the image did not break the model (macro-F1 -1.4 pp on test, -0.2 pp on validation for the 3-seed mean), but the test only shows that the bar is not necessary, not that the model ignores it.
- Uploads of the demo are resized from their original resolution, while training images were resized from 1024 x 768 originals by a separate caching step; results on raw uploads can differ slightly from the reported ones.
- Grad-CAM maps in the demo are illustrations, not explanations of the decision.

## Licence and citation

Weights and code: MIT. Please cite the dataset (Aversa et al., 2018) and the B2SHARE record above.
