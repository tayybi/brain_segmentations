#!/usr/bin/env bash
# Runs run_cross_slice_ssl_study.py for every fold x seed, two jobs per GPU.
# Usage: [OUT=experiments/label_prop_x] [EXTRA="--anchor-layout spread"] scripts/launch_label_prop_sweep.sh [n_gpus] [jobs_per_gpu]
set -u
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-/home/tali1/miniconda3/bin/python}
N_GPUS=${1:-8}
PER_GPU=${2:-2}
OUT=${OUT:-experiments/cross_slice_ssl}
EXTRA=${EXTRA:-}
mkdir -p "$OUT/logs"

jobs=()
for regime in sparse3; do
  for seed in 1 2 3; do
    for fold in 0 1 2 3 4; do
      jobs+=("$regime $fold $seed")
    done
  done
done

run_job() {
  local index=$1 regime=$2 fold=$3 seed=$4
  local gpu=$((index % N_GPUS))
  local done_file="$OUT/fold${fold}_seed${seed}/val_scores.json"
  [ -f "$done_file" ] && { echo "skip $regime fold$fold seed$seed"; return; }
  CUDA_VISIBLE_DEVICES=$gpu "$PYTHON" run_cross_slice_ssl_study.py --fold "$fold" --seed "$seed" \
    --output-dir "$OUT/fold${fold}_seed${seed}" $EXTRA > "$OUT/logs/${regime}_fold${fold}_seed${seed}.log" 2>&1
  echo "finished $regime fold$fold seed$seed (exit $?)"
}
export -f run_job
export PYTHON N_GPUS OUT EXTRA

for i in "${!jobs[@]}"; do echo "$i ${jobs[$i]}"; done | xargs -P $((N_GPUS * PER_GPU)) -L 1 bash -c 'run_job "$@"' _
echo "all jobs finished"
