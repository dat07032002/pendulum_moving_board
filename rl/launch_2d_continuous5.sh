#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/furuta_tilt_2d/rl"
PY="$HOME/furuta_rl/.venv/bin/python"

for seed in 0 1 2 3 4; do
  if (( seed < 3 )); then
    variant="cable"
    cable_limit="360"
    success_limit="330"
  else
    variant="free"
    cable_limit="none"
    success_limit="none"
  fi
  tag="pm10_60_up10_${variant}_s${seed}"
  setsid -f env \
    CUDA_VISIBLE_DEVICES="$seed" \
    FURUTA_VMAX=10 \
    FURUTA_UP_THRESH=0.984807753 \
    FURUTA_CABLE_LIMIT_DEG="$cable_limit" \
    FURUTA_SUCCESS_ARM_LIMIT_DEG="$success_limit" \
    FURUTA_ARM_CENTER_W=0.02 \
    "$PY" train_phaseb_2d.py \
      --warmstart models/up15_best.zip \
      --tag "$tag" \
      --seed "$seed" \
      --steps 900000 \
      --nenv 8 \
      --gamma 0.99 \
      --warmup-steps 50000 \
      --actor-lr 3e-5 \
      --ladder pm10_60 \
      --no-teacher \
      > "train_${tag}.log" 2>&1 < /dev/null
done

echo "launched 3 cable-aware and 2 free-arm +/-10 deg / 60 deg/s seeds"
