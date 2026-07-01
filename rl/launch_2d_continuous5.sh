#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/furuta_tilt_2d"
PY="$HOME/furuta_rl/.venv/bin/python"

for seed in 0 1 2 3 4; do
  tag="tilt2d_cont_s${seed}"
  setsid -f env CUDA_VISIBLE_DEVICES="$seed" "$PY" rl/train_tqc_2d.py \
    --warmstart rl/models/clean20_master_2d_warmstart.zip \
    --teacher-data rl/teacher_2d_retention_100k.npz \
    --tag "$tag" \
    --seed "$seed" \
    --steps 1750000 \
    --nenv 8 \
    --eval-freq 25000 \
    --n-target 30 \
    --n-guard 20 \
    --actor-start 25000 \
    > "train_${tag}.log" 2>&1 < /dev/null
done

echo "launched five continuous 2D seeds"
