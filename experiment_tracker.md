# Brain Segmentation Experiment Tracker

This file tracks the experiments and model variants developed for the paper work in this folder.

## Main project goal
Enhance mouse brain extraction performance using transformer-based segmentation models and compare configurations for paper-quality results.

## Experiment structure

- `experiments/baseline_vit_unet/` : baseline ViT + UNet-like decoder
- `experiments/upgrade_vit_decoder/` : improved decoder, augmentation, loss tuning
- `experiments/compare_backbones/` : compare ViT, Swin, or ConvNeXt backbones
- `results/` : validation metrics, plots, saved summaries
- `scripts/` : reusable training and evaluation scripts

## Current status

- Baseline notebook copied and organized under this workspace
- Improved modular pipeline scaffold created
- All future experiments should be saved under this project folder

## Suggested experiment log format

For each run, record:
- date
- dataset split
- backbone
- loss
- augmentation
- batch size
- learning rate
- dice
- iou
- notes

Example:

- Exp 01: ViT + BCE + Dice + flip/rotation, dice=0.91
- Exp 02: ViT + residual decoder + stronger aug, dice=0.93
- Exp 03: Swin + BCE + Dice, dice=0.94
