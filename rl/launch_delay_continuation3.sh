#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/furuta_tilt_2d/rl"
PY="$HOME/furuta_rl/.venv/bin/python"
COMMON=(
  FURUTA_VMAX=10
  FURUTA_UP_THRESH=0.984807753
  FURUTA_CABLE_LIMIT_DEG=360
  FURUTA_SUCCESS_ARM_LIMIT_DEG=330
  FURUTA_ARM_CENTER_W=0.02
  FURUTA_TIGHT_UPRIGHT_W=0.25
  FURUTA_TIGHT_UPRIGHT_SCALE_DEG=10
  FURUTA_CABLE_WARNING_W=0.20
  FURUTA_CABLE_WARNING_START_DEG=270
)

for seed in 0 1 2; do
  tag="delay12_from_dr_s1_s${seed}"
  setsid -f env \
    CUDA_VISIBLE_DEVICES="$seed" \
    "${COMMON[@]}" \
    "$PY" train_c360_delay_continuation_2d.py \
      --warmstart models/progressive_dr_s1/best_safe.zip \
      --teacher-data teacher_s1_safe_nominal_100k.npz \
      --tag "$tag" --seed "$seed" --steps 200000 --nenv 8 \
      --warmup-steps 50000 --actor-lr 5e-7 --critic-lr 3e-5 \
      > "train_${tag}.log" 2>&1 < /dev/null
done

echo "launched 3 action-delay continuation seeds"
