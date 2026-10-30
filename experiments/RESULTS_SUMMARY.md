# Experiment summary (29 Oct 2026)

All results: 20 mice, five-fold animal-level CV stratified by cohort, three seeds,
per-mouse means, paired Wilcoxon tests. Paper: `paper_isbi2027/main.tex` (ISBI 2027).

## Label propagation (paper)
| Setting | Dice | HD95 px | Folder |
|---|---|---|---|
| All slices annotated | 0.946 | 9.0 | `label_prop` (full) |
| 3 slices, anchors only | 0.924 | 13.2 | `label_prop` |
| 3 slices + registration, anchor loss, agreement | 0.933 | 11.0 | `label_prop` |
| 3 slices + SAM2 | 0.950 | 8.9 | `label_prop_sam2_quartile` |
| 2 slices + SAM2 | 0.949 | 8.9 | `label_prop_sam2_two_third` |
| 1 slice, anchors only | 0.891 | 20.4 | `label_prop_sam2_one_mid` |
| 1 slice + SAM2 | 0.950 | 9.0 | `label_prop_sam2_one_mid` |
| SAM2 applied directly to each mouse (1 annotated slice) | 0.956 | 8.1 | `results/sam2_pseudo_one_mid.npz` |

Spread anchor layout: `label_prop_spread`, `label_prop_sam2_spread`.

## Self-supervised pretraining (negative result on this data)
`cross_slice_ssl` (Transformer 2.5D) and `arch_{cnn_25d,cnn_2d,transformer_2d}`.
Cross-slice SSL (objectives A-D) helped only Transformer 2.5D at 1 slice (HD95 20.6 -> 18.3 px);
it was equal or worse in the other seven settings, as was masked image modelling.
With one annotated slice, 2D input beat 2.5D (CNN 0.905 vs 0.891).
Next step: repeat at scale on the unlabelled Bruker archive (coronal T1/T2 RARE, mice only,
labelled mice removed); sample in `data/Mice_study_sample`.

## nnU-Net baseline (`nnunet_baseline`, 2D, 250 epochs, one run per fold)
torch.compile must stay off (nnUNet_compile=f): with it, torch 2.10 gave striped predictions (Dice 0.03); broken runs in `nnunet/broken_compile`.
| Annotation | Dice | HD95 px |
|---|---|---|
| All slices | 0.960 | 6.9 |
| 1 slice + SAM2 | 0.958 (non-inferior, p<0.001) | 7.6 |
| 1 slice, others ignored | 0.868 | 23.1 |

## Cross-cohort (`cross_cohort`, our network, 3 seeds)
256->200: slice 0.891, +SAM2 0.929 (non-inferior), full 0.931. 200->256: 0.777, 0.790, 0.809 (not non-inferior).

## Longer pretraining (`ssl_long`, 16000 steps, seed 1, one slice)
Cross-slice ABCD 0.896 / 18.0 px (Dice p=0.50, HD95 p=0.009 vs no pretraining); masked modelling 0.887.
