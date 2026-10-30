#!/usr/bin/env bash
# Runs run_meta_ssl_study.py for every regime x fold x seed, two jobs per GPU.
# Usage: scripts/launch_meta_ssl_sweep.sh [n_gpus] [jobs_per_gpu]
set -u
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-/home/tali1/miniconda3/bin/python}
N_GPUS=${1:-8}
PER_GPU=${2:-2}
mkdir -p experiments/meta_ssl/logs

jobs=()
for regime in full sparse; do
  for seed in 1 2 3; do
    for fold in 0 1 2 3 4; do
      jobs+=("$regime $fold $seed")
    done
  done
done

run_job() {
  local index=$1 regime=$2 fold=$3 seed=$4
  local gpu=$((index % N_GPUS))
  local done_file="experiments/meta_ssl/$regime/fold${fold}_seed${seed}/val_scores.json"
  [ -f "$done_file" ] && { echo "skip $regime fold$fold seed$seed"; return; }
  CUDA_VISIBLE_DEVICES=$gpu "$PYTHON" run_meta_ssl_study.py --regime "$regime" --fold "$fold" --seed "$seed" \
    > "experiments/meta_ssl/logs/${regime}_fold${fold}_seed${seed}.log" 2>&1
  echo "finished $regime fold$fold seed$seed (exit $?)"
}
export -f run_job
export PYTHON N_GPUS

for i in "${!jobs[@]}"; do echo "$i ${jobs[$i]}"; done | xargs -P $((N_GPUS * PER_GPU)) -L 1 bash -c 'run_job "$@"' _
echo "all jobs finished"
