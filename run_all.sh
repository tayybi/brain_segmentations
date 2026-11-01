#!/usr/bin/env bash
set -e
cd ~/brain_segmentation
PY=.venv_sam2/bin/python
if [ ! -x "$PY" ]; then
  echo "== installing SAM2 environment"
  python3 -m venv .venv_sam2
  .venv_sam2/bin/pip install -q torch torchvision numpy scipy pillow
  .venv_sam2/bin/pip install -q "git+https://github.com/facebookresearch/sam2.git"
fi
$PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
mkdir -p checkpoints/sam2 results
if [ ! -s checkpoints/sam2/sam2.1_hiera_large.pt ]; then
  echo "== downloading SAM2 checkpoint"
  wget -q -O checkpoints/sam2/sam2.1_hiera_large.pt https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
fi
if [ ! -s results/sam2_pseudo_one_mid.npz ]; then
  echo "== SAM2 labels"
  CUDA_VISIBLE_DEVICES=1 $PY scripts/sam2_propagate.py --anchor-layout one_mid
fi
echo "== adaptation runs (seed 1)"
mkdir -p experiments/cohort_adaptation
CUDA_VISIBLE_DEVICES=0 $PY run_cohort_adaptation.py --split B_to_A --seed 1 > experiments/cohort_adaptation/B_to_A_seed1.out 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY run_cohort_adaptation.py --split A_to_B --seed 1 > experiments/cohort_adaptation/A_to_B_seed1.out 2>&1 &
wait
echo "== summary"
$PY scripts/summarize_cohort_adaptation.py experiments/cohort_adaptation
