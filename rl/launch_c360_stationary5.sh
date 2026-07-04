#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/furuta_tilt_2d/rl"
PY="$HOME/furuta_rl/.venv/bin/python"

for seed in 0 1 2 3 4; do
  actor_lr="5e-6"
  rehearsal_args=(--teacher-data teacher_s1_safe_nominal_100k.npz)
  variant="primary"
  if (( seed == 3 )); then
    actor_lr="3e-6"
    variant="conservative"
  elif (( seed == 4 )); then
    variant="no_rehearsal"
    rehearsal_args=(--no-rehearsal)
  fi
  tag="stationary_c360_nominal_${variant}_s${seed}"
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
    "$PY" train_c360_stationary_2d.py \
      --warmstart models/c360_s0ft_s1/best_stage_3.zip \
      --tag "$tag" \
      --seed "$seed" \
      --steps 400000 \
      --nenv 8 \
      --warmup-steps 25000 \
      --actor-lr "$actor_lr" \
      --critic-lr 1e-4 \
      --rehearsal-fraction 0.15 \
      --eval-freq 50000 \
      "${rehearsal_args[@]}" \
      > "train_${tag}.log" 2>&1 < /dev/null
done

echo "launched 5 stationary nominal physical-limit fine-tuning seeds"
