# Paper Draft Skeleton

## Abstract
Accurate brain extraction is a prerequisite for many downstream neuroimaging analyses, yet performance can degrade when boundaries are weak or image contrast varies across slices. This work studies a transformer-based segmentation pipeline for mouse brain extraction using a ViT-style encoder and a residual convolutional decoder. The model is trained for binary segmentation with a hybrid BCE-plus-Dice objective and evaluated on animal-separated train, validation, and test splits to reduce information leakage across slices from the same specimen. The repository currently supports reproducible training on a `70/15/15` split with 14 training animals, 3 validation animals, and 3 test animals. Final quantitative comparisons remain to be filled in after controlled baseline and ablation runs. The intended contribution is a reproducible transformer-based benchmark for mouse brain extraction and a framework for studying the effects of decoder design, augmentation, and pretraining.

## 1. Introduction
Mouse brain extraction is a core preprocessing step for registration, atlas mapping, and volumetric analysis. Errors at this stage propagate to later processing, so robust foreground-background separation is important even when anatomy is small and contrast is inconsistent. Convolutional encoder-decoder models such as U-Net have become standard for biomedical segmentation, but transformer-based encoders offer stronger global receptive fields that may better capture whole-structure context. This paper evaluates whether a ViT-style segmentation pipeline is a useful direction for mouse brain extraction.

## 2. Related Work
Traditional brain extraction methods rely on thresholding, morphology, deformable models, or atlas priors. More recent biomedical segmentation systems are dominated by U-Net variants and hybrid CNN-transformer models. For this paper, keep the related work focused on three threads: classical skull stripping or brain masking, U-Net style biomedical segmentation, and transformer-based dense prediction.

## 3. Methods

### 3.1 Task Definition
The task is binary segmentation of mouse brain tissue from 2D grayscale image slices with paired manual masks.

### 3.2 Dataset and Split Strategy
Two dataset split layouts exist in the repository. The recommended primary split is `70/15/15`, containing 152 training slices from animals `1c, 2c, 3c, 4c, 5c, 6c, 7c, 8c, 9c, 10c, 21c, 22c, 23c, 24c`, 33 validation slices from animals `25c, 26c, 27c`, and 34 test slices from animals `28c, 29c, 30c`. This split is animal-separated, which is preferable to random slice splitting because adjacent slices from the same animal are highly correlated.

### 3.3 Model
The implemented model contains:
- A ViT-style tokenization stage using a strided convolutional patch projection from one grayscale channel to a 768-dimensional embedding space.
- A transformer encoder created with `timm`.
- A residual-style convolutional decoder that upsamples token features back to full-resolution binary masks.

### 3.4 Optimization
Training uses a hybrid BCE-plus-Dice loss. BCE stabilizes pixelwise optimization, while Dice emphasizes overlap on the sparse foreground region. The current training script uses separate learning rates for the backbone and decoder and saves the best checkpoint according to validation Dice.

### 3.5 Augmentation and Preprocessing
Current augmentation includes random horizontal flips, vertical flips, and small-angle rotation. Images are resized to `256 x 256`, converted to single-channel tensors, and normalized per image.

## 4. Experimental Design

### 4.1 Baselines
You need at least one strong baseline for the paper:
- Original notebook model or plain U-Net.
- Proposed ViT plus residual decoder model.

### 4.2 Evaluation Metrics
Dice should be the primary metric. IoU, precision, and recall should be added if possible. If boundary quality is central to the paper, add a boundary-sensitive metric such as Hausdorff distance.

### 4.3 Ablations
Recommended ablations:
- `BCE` vs `BCE + Dice`
- baseline decoder vs residual decoder
- light augmentation vs stronger augmentation
- optional: with vs without masked-image pretraining

## 5. Results
This section should report only measured results. Do not reuse example numbers from planning notes. Present a compact quantitative table and a figure with representative successful and failure cases.

## 6. Discussion
Discuss whether the transformer encoder improves global shape consistency, whether improvements are mostly due to the decoder or loss, and how sensitive performance is to the small dataset size. Be explicit that the present pipeline is 2D and may not capture full 3D anatomical continuity.

## 7. Limitations
- Small number of animals.
- 2D slice-wise training.
- No final cross-model comparison recorded yet in `results/`.
- Hidden file artifacts such as `.DS_Store` indicate the dataset packaging still needs cleanup.

## 8. Conclusion
This study frames mouse brain extraction as a reproducible transformer-based segmentation problem and provides a practical training pipeline for evaluating ViT-style encoders with convolutional decoders. The final paper should emphasize measured gains, animal-level evaluation, and controlled ablations rather than broad architectural claims.
