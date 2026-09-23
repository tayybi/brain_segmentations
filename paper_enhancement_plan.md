# Paper Enhancement Plan

## Working title
Transformer-Based Mouse Brain Extraction with a ViT Decoder Segmentation Pipeline

## What the repository currently supports

- Binary mouse brain segmentation with paired image-mask supervision.
- A transformer-style backbone with a residual decoder in [improved_segmentation_pipeline.py](/home/tali1/brain_segmentations/improved_segmentation_pipeline.py).
- Two predefined dataset splits:
  - `70/15/15`: train 152 slices from 14 animals, validation 33 slices from 3 animals, test 34 slices from 3 animals plus one hidden `.DS_Store` artifact.
  - `70/10/20`: train 152 slices from 14 animals, validation 22 slices from 2 animals, test 44 slices from 4 animals.
- Masked-image sweep assets under `data/masked_sweeps/`, suggesting a self-supervised or pretext-learning angle, but no paper-ready results table yet.

## Claims you can make now

- The study investigates transformer-based mouse brain extraction using a ViT-style encoder and convolutional decoder.
- The dataset split is animal-level rather than random slice-level, which reduces leakage risk across train, validation, and test sets.
- The optimization objective is a BCE-plus-Dice hybrid loss, and model selection should be based on validation Dice.

## Claims you should not make yet

- Any final Dice or IoU improvement over baseline.
- Any superiority over U-Net, Swin, ConvNeXt, or prior literature.
- Any benefit from masked pretraining unless the ablation is run and reported.

## Strong paper structure

1. Problem
Mouse brain extraction is a binary segmentation task where accurate boundaries matter for downstream morphometric or registration workflows.

2. Gap
Classical CNN pipelines work reasonably well, but they may underuse global context in low-contrast anatomical slices.

3. Proposed method
Use a ViT-style token encoder with a residual convolutional decoder, trained with BCE plus Dice loss on animal-separated splits.

4. Evaluation
Report Dice as the primary metric. Add IoU, precision, and recall if available. Keep validation and test results separated.

5. Contribution language
- Adapts a transformer-based segmentation architecture to mouse brain extraction.
- Uses animal-level data splitting to reduce leakage.
- Establishes a reproducible training pipeline for future ablations on augmentation, decoder design, and pretraining.

## Minimum experiments needed for a credible submission

1. Baseline
Plain U-Net or your original notebook model.

2. Proposed model
Current ViT plus residual decoder pipeline.

3. Ablation A
`BCE` vs `BCE + Dice`.

4. Ablation B
Weak augmentation vs current augmentation.

5. Optional ablation
With and without masked-image pretraining.

## Results table template

| Model | Split | Loss | Augmentation | Dice | IoU | Precision | Recall |
|---|---|---|---|---:|---:|---:|---:|
| Baseline U-Net | 70/15/15 | BCE | light | TBA | TBA | TBA | TBA |
| ViT + residual decoder | 70/15/15 | BCE + Dice | light | TBA | TBA | TBA | TBA |
| ViT + residual decoder | 70/15/15 | BCE + Dice | stronger | TBA | TBA | TBA | TBA |

## Figures you should include

1. Pipeline diagram
Image -> patch/token encoder -> residual decoder -> binary mask.

2. Qualitative predictions
At least 3 rows: input, ground truth, prediction, error map.

3. Training curves
Training loss, validation loss, validation Dice.

## Discussion points worth writing explicitly

- Why animal-level splitting matters more than slice-level randomization.
- Why Dice is the main metric for a foreground-sparse segmentation problem.
- Whether the transformer encoder helps global anatomical consistency or only changes optimization behavior.
- Limitations from sample size and 2D slice training.

## Immediate cleanup before writing results

- Remove or ignore hidden files such as `.DS_Store`.
- Record exact train/validation/test animal IDs in the manuscript.
- Save per-run metrics in `results/` instead of leaving example numbers in tracker notes.
