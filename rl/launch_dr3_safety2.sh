#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/furuta_tilt_2d/rl"
PY="$HOME/furuta_rl/.venv/bin/python"
COMMON=(
  FURUTA_VMAX=10
  FURUTA_UP_THRESH=0.984807753
  FURUTA_SUCCESS_ARM_LIMIT_DEG=330
  FURUTA_ARM_CENTER_W=0.02
  FURUTA_TIGHT_UPRIGHT_W=0.25
  FURUTA_TIGHT_UPRIGHT_SCALE_DEG=10
  FURUTA_CABLE_WARNING_W=0.20
  FURUTA_CABLE_WARNING_START_DEG=270
)

for seed in 0 1 2; do
  tag="progressive_dr_s${seed}"
  setsid -f env \
    CUDA_VISIBLE_DEVICES="$seed" \
    FURUTA_CABLE_LIMIT_DEG=360 \
    "${COMMON[@]}" \
    "$PY" train_c360_progressive_dr_2d.py \
      --warmstart models/stationary_c360_nominal_primary_s1/best_safe.zip \
      --teacher-data teacher_s1_safe_nominal_100k.npz \
      --tag "$tag" --seed "$seed" --steps 500000 --nenv 8 \
      --warmup-steps 25000 --actor-lr 3e-6 --critic-lr 1e-4 \
      > "train_${tag}.log" 2>&1 < /dev/null
done

for seed in 3 4; do
  tag="safety_c345_s${seed}"
  setsid -f env \
    CUDA_VISIBLE_DEVICES="$seed" \
    FURUTA_CABLE_LIMIT_DEG=345 \
    "${COMMON[@]}" \
    "$PY" train_c360_stationary_2d.py \
      --warmstart models/stationary_c360_nominal_primary_s1/best_safe.zip \
      --teacher-data teacher_s1_safe_nominal_100k.npz \
      --tag "$tag" --seed "$seed" --steps 300000 --nenv 8 \
      --warmup-steps 25000 --actor-lr 3e-6 --critic-lr 1e-4 \
      --rehearsal-fraction 0.15 --eval-freq 50000 \
      > "train_${tag}.log" 2>&1 < /dev/null
done

echo "launched 3 progressive-DR and 2 nominal safety-reserve seeds"
