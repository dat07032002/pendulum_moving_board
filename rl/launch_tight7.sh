#!/usr/bin/env bash
# Tight-hold Stage A: 7-deg gate fine-tune from the deployed best_safe.
# Robustness-first: eval gate stays at the canonical +/-10 deg (in-script).
set -euo pipefail

cd "$HOME/furuta_tilt_2d/rl"
PY="$HOME/furuta_rl/.venv/bin/python"
COMMON=(
  FURUTA_VMAX=10
  FURUTA_CABLE_LIMIT_DEG=360
  FURUTA_SUCCESS_ARM_LIMIT_DEG=330
  FURUTA_ARM_CENTER_W=0.02
  FURUTA_CABLE_WARNING_W=0.20
  FURUTA_CABLE_WARNING_START_DEG=270
)
# NOTE: FURUTA_UP_THRESH / TIGHT_UPRIGHT_* / ACTION_RATE_W are set by the script
# itself from --up-thresh-deg etc. Do not also set them here.

for seed in 0 1 2; do
  tag="tight7_s${seed}"
  setsid -f env \
    CUDA_VISIBLE_DEVICES="$seed" \
    "${COMMON[@]}" \
    "$PY" train_tight_hold_2d.py \
      --warmstart models/progressive_dr_s1/best_safe.zip \
      --teacher-data teacher_s1_safe_nominal_100k.npz \
      --tag "$tag" --seed "$seed" --steps 400000 --nenv 8 \
      --warmup-steps 25000 --actor-lr 5e-6 --critic-lr 1e-4 \
      --rehearsal-fraction 0.15 \
      --up-thresh-deg 7 --tight-scale-deg 7 --tight-w 0.35 --action-rate-w 0.06 \
      > "train_${tag}.log" 2>&1 < /dev/null
done

echo "launched 3 tight7 seeds"
