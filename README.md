# One Annotated Slice per Mouse Suffices

Code, per-mouse results and paper source for *One Annotated Slice per Mouse Suffices:
Label Propagation for Brain Extraction in Preclinical MRI* (ISBI 2027, in preparation).

We ask how few manually outlined slices per mouse are needed to train a brain extraction
network when the remaining slices receive automatically propagated labels. With one
annotated slice per training mouse (9% of slices) and labels propagated by a frozen SAM2,
the trained network is non-inferior to full annotation at a margin of 0.01 Dice
(0.950 vs 0.946; nnU-Net 0.958 vs 0.960).

## Data

`data/dataset_split_70_15_15/` holds 20 mice (218 slices, 9.4 T Bruker, T1/T2-weighted)
with expert brain masks, in two cohorts: 1c–10c at 256×256 and 21c–30c at 200×200.
`data/dataset_split_70_10_20/` holds the same 218 slices in an earlier 70/10/20 split.
The code reads only `dataset_split_70_15_15`, pools all its mice and builds its own
five-fold animal-level split, stratified by cohort (`make_split` in `run_meta_ssl_study.py`).

## Code

| File | Purpose |
|---|---|
| `run_label_propagation_study.py` | Main study: anchors, propagators (copy, interpolation, self-training, registration, SAM2), agreement filtering, training and scoring |
| `run_meta_ssl_study.py` | Shared data loading, splits, 2.5D CNN–Transformer network and metrics |
| `run_cross_slice_ssl_study.py` | Self-supervised pretraining (masked modelling and cross-slice objectives) |
| `scripts/sam2_propagate.py` | SAM2.1 propagation of the annotated slices through each mouse (separate environment) |
| `scripts/nnunet_baseline.py`, `scripts/launch_nnunet.sh` | nnU-Net v2 baseline (separate environment, `nnUNet_compile=f`) |
| `scripts/launch_*_sweep.sh` | Run every fold × seed in parallel on the GPUs |
| `scripts/summarize_label_prop.py` | Per-mouse means, label quality and paired tests |
| `scripts/make_*_figure*.py` | Paper figures |

## Reproducing

```bash
# SAM2 labels for an anchor layout (quartile, one_mid, two_third, spread)
.venv_sam2/bin/python scripts/sam2_propagate.py --anchor-layout one_mid

# One fold and seed of the main study
python run_label_propagation_study.py --fold 0 --seed 1 --anchor-layout one_mid --methods sparse sam2 full

# All folds and seeds; cross-cohort runs use --split A_to_B or B_to_A
OUT=experiments/label_prop_sam2_one_mid EXTRA="--anchor-layout one_mid --methods sparse sam2" \
    scripts/launch_label_prop_sweep.sh 8 2

# nnU-Net baseline
.venv_nnunet/bin/python scripts/nnunet_baseline.py prepare
scripts/launch_nnunet.sh 8
.venv_nnunet/bin/python scripts/nnunet_baseline.py score
```

## Results

`experiments/<study>/fold<k>_seed<s>/metrics.csv` holds per-mouse test scores for every run,
and `experiments/RESULTS_SUMMARY.md` maps each number in the paper to its folder.
The paper source is in `paper_isbi2027/` (`main.tex`, figures in `figures/`).

Not in the repository (large or regenerable): SAM2 label files (`results/*.npz`),
nnU-Net models and model checkpoints.
