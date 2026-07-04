#!/usr/bin/env bash
# tight7f: actuator-lag-only retrain (NO slew limiter).
# Evidence 2026-07-04 night: (a) 6 ms lag in sim reproduces the hardware 26 Hz
# limit cycle exactly (m|da| 0.2 -> 0.4-1.1, success collapse, critic blind);
# (b) the 3 V/tick hard slew limiter collapsed all 5 tight7d/e seeds (hidden
# actuator state = partially observed MDP). So: model the lag, let the
# action-rate penalty teach damping, keep the actuator fully observable.
# No firmware pairing needed - the real rig has the lag physically.
# GPUs 0-2: action_rate_w 0.3 (strong). GPUs 3-4: 0.15 (mild) as a hedge.
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

launch() {  # seed gpu arw tag
  setsid -f env \
    CUDA_VISIBLE_DEVICES="$2" \
    "${COMMON[@]}" \
    "$PY" train_tight_hold_2d.py \
      --warmstart models/tight7c_s1/best_safe.zip \
      --teacher-data teacher_s1_safe_nominal_100k.npz \
      --tag "$4" --seed "$1" --steps 400000 --nenv 8 \
      --warmup-steps 25000 --actor-lr 1e-5 --critic-lr 1e-4 \
      --rehearsal-fraction 0.15 --teacher-coef 0.2 --teacher-ratio none \
      --up-thresh-deg 7 --tight-scale-deg 6 --tight-w 0.8 --action-rate-w "$3" \
      --slew-v 0 --act-lag-ms 3,9 \
      > "train_${4}.log" 2>&1 < /dev/null
}

launch 50 0 0.3  tight7f_s0
launch 51 1 0.3  tight7f_s1
launch 52 2 0.3  tight7f_s2
launch 53 3 0.15 tight7f_m0
launch 54 4 0.15 tight7f_m1

echo "launched 5 tight7f seeds (lag DR 3-9 ms, no slew; arw 0.3 x3 + 0.15 x2)"
