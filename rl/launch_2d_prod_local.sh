#!/usr/bin/env bash
# Local sequential A/B: 3 cable-aware and 2 free-arm seeds at +/-10 deg / 60 deg/s.
set -uo pipefail
cd "$(dirname "$0")"

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
  echo "=== $(date '+%H:%M:%S') launching ${tag} ==="
  FURUTA_VMAX=10 \
  FURUTA_UP_THRESH=0.984807753 \
  FURUTA_CABLE_LIMIT_DEG="$cable_limit" \
  FURUTA_SUCCESS_ARM_LIMIT_DEG="$success_limit" \
  FURUTA_ARM_CENTER_W=0.02 \
  python train_phaseb_2d.py \
    --warmstart models/up15_best.zip \
    --tag "${tag}" \
    --seed "${seed}" \
    --steps 900000 \
    --gamma 0.99 \
    --warmup-steps 50000 \
    --actor-lr 3e-5 \
    --ladder pm10_60 \
    --no-teacher \
    > "prod_${tag}.log" 2>&1
  echo "=== $(date '+%H:%M:%S') ${tag} finished (exit $?) ==="
done
echo "=== all production seeds done ==="
