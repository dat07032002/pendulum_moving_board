#!/usr/bin/env bash
# tight7e: slew (3 V/tick) + first-order actuator-lag DR (2-8 ms, sim-only).
# Runs ALONGSIDE tight7d on the two idle GPUs to give a three-way comparison
# (s1 baseline / slew-only / slew+lag). The lag models the real motor-electrical
# chain behind the 26 Hz hardware limit cycle; the real rig has it physically,
# so no firmware change is associated with this variant.
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

for i in 0 1; do
  tag="tight7e_s${i}"
  setsid -f env \
    CUDA_VISIBLE_DEVICES="$((i + 3))" \
    "${COMMON[@]}" \
    "$PY" train_tight_hold_2d.py \
      --warmstart models/tight7c_s1/best_safe.zip \
      --teacher-data teacher_s1_safe_nominal_100k.npz \
      --tag "$tag" --seed "$((i + 40))" --steps 400000 --nenv 8 \
      --warmup-steps 25000 --actor-lr 1e-5 --critic-lr 1e-4 \
      --rehearsal-fraction 0.15 --teacher-coef 0.2 --teacher-ratio none \
      --up-thresh-deg 7 --tight-scale-deg 6 --tight-w 0.8 --action-rate-w 0.1 \
      --slew-v 3.0 --act-lag-ms 2,8 \
      > "train_${tag}.log" 2>&1 < /dev/null
done

echo "launched 2 tight7e seeds (slew 3 V/tick + lag DR 2-8 ms)"
