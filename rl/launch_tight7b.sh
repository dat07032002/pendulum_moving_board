#!/usr/bin/env bash
# Tight-hold Stage A': bold pressure from the critic-recalibrated tight7_s2 checkpoint.
# The first tight7 campaign proved the BC anchor (teacher_coef=1.0) + 5e-6 actor LR
# freezes the actor's style; the critic is now calibrated to the tight reward, so
# this run weakens the anchor and lets the actor move. Eval gate stays +/-10 deg.
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

for seed in 0 1 2; do
  tag="tight7b_s${seed}"
  setsid -f env \
    CUDA_VISIBLE_DEVICES="$seed" \
    "${COMMON[@]}" \
    "$PY" train_tight_hold_2d.py \
      --warmstart models/tight7_s2/best_safe.zip \
      --teacher-data teacher_s1_safe_nominal_100k.npz \
      --tag "$tag" --seed "$((seed + 10))" --steps 400000 --nenv 8 \
      --warmup-steps 10000 --actor-lr 1e-5 --critic-lr 1e-4 \
      --rehearsal-fraction 0.15 --teacher-coef 0.2 --teacher-ratio 0.05 \
      --up-thresh-deg 7 --tight-scale-deg 6 --tight-w 0.8 --action-rate-w 0.3 \
      > "train_${tag}.log" 2>&1 < /dev/null
done

echo "launched 3 tight7b seeds"
