# Hybrid CNN-Transformer Model Architecture

Both experiments use the same final segmentation model. The only difference is how the encoder is initialized before segmentation training.

```mermaid
flowchart LR
    A["Three MRI slices\nprevious, center, next\n3 x 256 x 256"]
    B["CNN stem\nlocal edges and textures\n64 x 64 x 64"]
    C["CNN downsampling\n256 x 16 x 16"]
    D["Transformer\n4 layers, 8 heads\n256 spatial tokens"]
    E["CNN decoder\nupsampling"]
    F["Brain-mask prediction\n1 x 256 x 256"]

    A --> B --> C --> D --> E --> F
    B -. fine-detail skip connection .-> E
```

## Experiment 1: Hybrid model from scratch

```mermaid
flowchart TD
    A["Randomly initialized CNN-Transformer encoder"]
    B["Segmentation training\n152 training slice triplets + 152 masks"]
    C["Validate using 33 slices from 3 mice"]
    D["Test once using 34 slices from 3 unseen mice"]
    A --> B --> C --> D
```

The model starts with no MRI knowledge. It learns both image features and brain-mask prediction only from the labelled training data.

## Experiment 2: SSL-pretrained hybrid model

```mermaid
flowchart TD
    A["152 training slice triplets\nNo masks needed"]
    B["Hide random patches\nDefault: 16 x 16 patches, 50% hidden"]
    C["CNN-Transformer encoder"]
    D["Reconstruction decoder\nRebuild the center MRI slice"]
    E["Keep pretrained encoder weights\nDiscard reconstruction decoder"]
    F["Segmentation decoder"]
    G["Fine-tune with 152 training masks"]
    H["Validate, then test on unseen mice"]

    A --> B --> C --> D --> E --> F --> G --> H
```

Here the encoder first learns general MRI anatomy by rebuilding masked images. It is then fine-tuned to segment the brain mask.

## Fair comparison

| Item | From scratch | SSL-pretrained |
|---|---|---|
| Final segmentation architecture | Same | Same |
| Training mice | Same 14 mice | Same 14 mice |
| MRI input | Same three adjacent slices | Same three adjacent slices |
| Segmentation masks | Same 152 masks | Same 152 masks |
| Initial encoder weights | Random | Learned from masked reconstruction |
| Validation and test mice | Same | Same |

Run each configuration with seeds `1`, `2`, and `3`. If the SSL-pretrained version consistently has higher test Dice than the from-scratch version, the improvement is attributable to masked-image pretraining rather than a different segmentation architecture.
