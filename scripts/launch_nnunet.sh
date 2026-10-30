#!/usr/bin/env bash
# Plans, trains (2d, fold "all", 250 epochs) and predicts every nnU-Net baseline dataset, two per GPU.
# Usage: scripts/launch_nnunet.sh [n_gpus]
set -u
cd "$(dirname "$0")/.."
export nnUNet_raw=$PWD/nnunet/raw nnUNet_preprocessed=$PWD/nnunet/preprocessed nnUNet_results=$PWD/nnunet/results
# torch.compile on torch 2.10 produced striped, collapsing predictions (Dice 0.03); keep it off.
export nnUNet_compile=f
BIN=$PWD/.venv_nnunet/bin
N_GPUS=${1:-8}
mkdir -p nnunet/logs

run_one() {
  local index=$1 name=$2
  local id=${name:7:3} gpu=$((index % N_GPUS))
  local out=nnunet/predictions/$name
  [ -f "$out/.done" ] && { echo "skip $name"; return; }
  {
    "$BIN/nnUNetv2_plan_and_preprocess" -d "$id" -c 2d -np 4 --verify_dataset_integrity &&
    CUDA_VISIBLE_DEVICES=$gpu "$BIN/nnUNetv2_train" "$id" 2d all -tr nnUNetTrainer_250epochs &&
    CUDA_VISIBLE_DEVICES=$gpu "$BIN/nnUNetv2_predict" -i "nnunet/raw/$name/imagesTs" -o "$out" -d "$id" -c 2d -f all -tr nnUNetTrainer_250epochs &&
    touch "$out/.done"
  } > "nnunet/logs/$name.log" 2>&1
  echo "finished $name (exit $?)"
}
export -f run_one; export N_GPUS BIN

ls nnunet/raw | awk '{print NR-1, $0}' | xargs -P $((N_GPUS * 2)) -n 2 bash -c 'run_one "$0" "$1"'
