# Hybrid SSL Study Protocol

## Primary comparison

Evaluate the following methods using identical animal-level splits:

1. Previous supervised 2D U-Net.
2. Previous masked-reconstruction SSL 2D U-Net.
3. `run_hybrid_ssl_study.py finetune` without a pretrained checkpoint.
4. `run_hybrid_ssl_study.py full`, which uses masked reconstruction pretraining on training animals only.

The primary endpoint is mean per-animal test Dice. Report mean and standard deviation across animals and keep the selected checkpoint based only on validation-animal Dice.

## Label-efficiency experiment

Run methods 3 and 4 with `--label-fraction 0.25`, `0.5`, and `1.0`. The script selects whole training animals, never individual slices, so no animal is partially labelled in an experiment.

## Required controls

- Use `dataset_split_70_15_15` consistently for all four methods.
- Do not use validation or test images in MAE pretraining.
- Repeat each final configuration with at least three seeds.
- Evaluate and test only after selecting hyperparameters using validation animals.
- Do not reuse the previous paper's text or figures; cite it as the CNN SSL baseline.

## Commands

```bash
# Transformer trained from scratch
python run_hybrid_ssl_study.py finetune --output-dir experiments/hybrid_scratch_seed1 --seed 1
python run_hybrid_ssl_study.py evaluate --output-dir experiments/hybrid_scratch_seed1 --checkpoint experiments/hybrid_scratch_seed1/best_segmentation.pt

# MAE-pretrained hybrid model
python run_hybrid_ssl_study.py full --output-dir experiments/hybrid_mae_seed1 --seed 1

# Label-efficiency condition
python run_hybrid_ssl_study.py full --output-dir experiments/hybrid_mae_25pct_seed1 --label-fraction 0.25 --seed 1
```
