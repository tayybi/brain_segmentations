# Mouse Brain Extraction v2 - Enhancement Roadmap

This folder already contains a strong baseline built around a ViT + decoder segmentation pipeline. The next step is to turn it into a more reliable, reproducible, and production-ready pipeline.

## Current strengths

- ViT backbone with masked image modeling (MIM) pretraining
- Decoder with UNet-like upsampling
- Augmentation for training stability
- Dice and BCE-based loss options
- Early stopping and checkpoint saving

## High-impact improvements to implement next

### 1. Move from notebook code to a modular training script

The current model is embedded in a notebook, which makes iteration slow and hard to reproduce. Split the pipeline into:

- data loading
- model definition
- training loop
- validation metrics
- inference utilities

This makes it easier to compare experiments and keep checkpoints organized.

### 2. Improve loss design for brain masks

Brain extraction is a class-imbalanced binary segmentation problem. Use:

- BCE + Dice loss as the default baseline
- Focal + Dice loss for harder boundary cases
- optional Lovasz loss or Tversky loss if false positives remain high

A good starting point is to keep BCE + Dice as the main loss while monitoring Dice, IoU, and precision/recall.

### 3. Add stronger augmentation and preprocessing

The current pipeline already includes flips and rotation. Expand it with:

- elastic deformation
- small intensity jitter
- CLAHE or histogram normalization
- random brightness/contrast
- center-crop + resize consistency for anatomy alignment

This is especially helpful for mouse brain images where shape and contrast can vary substantially.

### 4. Use a more robust decoder

The current decoder is acceptable, but there are a few improvements worth exploring:

- skip connections from early ViT feature maps
- ConvNeXt or Swin backbone instead of plain ViT
- U-Net style feature fusion from multiple resolutions
- residual blocks in decoder

The best upgrade path is to compare:

1. ViT + simple UNet decoder
2. ViT + residual decoder
3. Swin Transformer + U-Net decoder
4. ConvNeXt + segmentation head

### 5. Add better validation and model selection

Use validation metrics beyond loss:

- Dice score
- IoU
- precision
- recall
- Hausdorff distance on boundary regions

Keep the checkpoint with the best validation Dice, not only the lowest loss. This is important for small anatomical structures and boundary precision.

### 6. Add inference-time improvements

The notebook already includes a simple horizontal flip TTA. Extend that to:

- flip in both axes
- multi-scale inference
- threshold tuning on validation set
- morphological cleanup after prediction

For brain extraction, post-processing with connected-component filtering is often useful.

### 7. Add dataset quality checks

Before training, verify:

- all images and masks align
- mask values are binary
- image dimensions match expected size
- annotations are not empty or corrupted
- no train/val leakage across slices or animals

These checks prevent biased results and clean training instability.

### 8. Add experiment tracking

Track:

- hyperparameters
- train loss and validation loss curves
- dice per epoch
- best checkpoint path
- GPU/CPU runtime

A simple CSV or TensorBoard logging setup is enough for a strong improvement.

## Recommended enhancement order

1. Refactor the notebook into a Python training script
2. Add BCE + Dice + improved validation metrics
3. Add stronger augmentation and preprocessing
4. Add TTA and post-processing for inference
5. Compare ViT vs Swin/ConvNeXt backbone
6. Add ablation studies to justify the final model

## Suggested next target

The most realistic next upgrade is:

- keep the current ViT backbone
- strengthen the decoder with residual blocks and skip connections
- use BCE + Dice loss
- improve augmentation
- evaluate with Dice + IoU + boundary metrics
- add TTA and connected-component filtering at inference

This gives a meaningful improvement without rewriting the entire project.

## Files of interest

- [model_vit_unet_v2.ipynb](model_vit_unet_v2.ipynb)
- [data](data)
- [checkpoints](checkpoints)

## Hybrid SSL Follow-on Study

`run_hybrid_ssl_study.py` implements the new paper's 2.5D hybrid CNN-Transformer experiment. It reconstructs masked training slices during SSL pretraining, then predicts each center-slice mask from its preceding, center, and following MRI slices. The test and validation animals are excluded from pretraining.

The experiment design and commands are in [experiment_protocol.md](experiment_protocol.md). Start the SSL-pretrained model with:

```bash
python run_hybrid_ssl_study.py full --output-dir experiments/hybrid_mae_seed1 --seed 1
```

See [model_architecture.md](model_architecture.md) for diagrams of the from-scratch and SSL-pretrained experiments.
