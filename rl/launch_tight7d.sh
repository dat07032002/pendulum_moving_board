#!/usr/bin/env bash
# tight7d: adapt the smooth/robust tight7c_s1 policy to a 3 V/tick actuator slew
# limit (FURUTA_SLEW_V_PER_TICK), which will be deployed in firmware as
# RL_SLEW_V_PER_TICK. Motivation: hardware showed a +/-10 V 200 Hz bang-bang limit
# cycle (dV RMS ~9 V/tick) that reward shaping alone did not fix; the slew limiter
# changes the actuator itself, matched between sim and firmware.
# Plant change -> keep the 25k critic-only warmup and the 0.2 BC anchor.
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
  tag="tight7d_s${seed}"
  setsid -f env \
    CUDA_VISIBLE_DEVICES="$seed" \
    "${COMMON[@]}" \
    "$PY" train_tight_hold_2d.py \
      --warmstart models/tight7c_s1/best_safe.zip \
      --teacher-data teacher_s1_safe_nominal_100k.npz \
      --tag "$tag" --seed "$((seed + 30))" --steps 400000 --nenv 8 \
      --warmup-steps 25000 --actor-lr 1e-5 --critic-lr 1e-4 \
      --rehearsal-fraction 0.15 --teacher-coef 0.2 --teacher-ratio none \
      --up-thresh-deg 7 --tight-scale-deg 6 --tight-w 0.8 --action-rate-w 0.1 \
      --slew-v 3.0 \
      > "train_${tag}.log" 2>&1 < /dev/null
done

echo "launched 3 tight7d seeds (slew 3 V/tick)"
