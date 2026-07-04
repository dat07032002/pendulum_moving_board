#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/furuta_tilt_2d/rl"
PY="$HOME/furuta_rl/.venv/bin/python"

for seed in 0 1 2; do
  tag="c360_s0ft_s${seed}"
  setsid -f env \
    CUDA_VISIBLE_DEVICES="$seed" \
    FURUTA_VMAX=10 \
    FURUTA_UP_THRESH=0.984807753 \
    FURUTA_CABLE_LIMIT_DEG=360 \
    FURUTA_SUCCESS_ARM_LIMIT_DEG=330 \
    FURUTA_ARM_CENTER_W=0.02 \
    FURUTA_TIGHT_UPRIGHT_W=0.25 \
    FURUTA_TIGHT_UPRIGHT_SCALE_DEG=10 \
    FURUTA_CABLE_WARNING_W=0.20 \
    FURUTA_CABLE_WARNING_START_DEG=270 \
    "$PY" train_c360_finetune_2d.py \
      --warmstart models/pm10_60_up10_cable_s0/best_stage_5.zip \
      --tag "$tag" \
      --seed "$seed" \
      --steps 800000 \
      --nenv 8 \
      --warmup-steps 25000 \
      --actor-lr 1e-5 \
      --critic-lr 1e-4 \
      --eval-freq 50000 \
      --eval-target-episodes 60 \
      --eval-guard-episodes 20 \
      --consecutive-passes 2 \
      > "train_${tag}.log" 2>&1 < /dev/null
done

echo "launched 3 cable-360 S0 fine-tuning seeds"
