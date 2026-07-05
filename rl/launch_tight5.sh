#!/usr/bin/env bash
# Stage B: tighten the upright gate toward +/-5 deg from the tight7f_s2 winner.
# Delay stays FIXED at 1 (delay-{1,2} training is falsified: three recipes
# collapsed; delay-2 needs action-history obs, out of scope). Lag DR retained
# (the plant model tight7f_s2 mastered). GPUs 0-2: 5-deg gate; GPUs 3-4: 6-deg
# intermediate hedge (staged-tightening lesson from TIGHT_UPRIGHT_RESULTS.md).
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

launch() {  # seed gpu gate scale tag
  setsid -f env \
    CUDA_VISIBLE_DEVICES="$2" \
    "${COMMON[@]}" \
    "$PY" train_tight_hold_2d.py \
      --warmstart models/tight7f_s2/best_safe.zip \
      --teacher-data teacher_s1_safe_nominal_100k.npz \
      --tag "$5" --seed "$1" --steps 400000 --nenv 8 \
      --warmup-steps 25000 --actor-lr 1e-5 --critic-lr 1e-4 \
      --rehearsal-fraction 0.15 --teacher-coef 0.2 --teacher-ratio none \
      --up-thresh-deg "$3" --tight-scale-deg "$4" --tight-w 1.0 --action-rate-w 0.3 \
      --slew-v 0 --act-lag-ms 3,9 --delay-steps 1 \
      > "train_${5}.log" 2>&1 < /dev/null
}

launch 70 0 5 5 tight5_s0
launch 71 1 5 5 tight5_s1
launch 72 2 5 5 tight5_s2
launch 73 3 6 5 tight6_h0
launch 74 4 6 5 tight6_h1

echo "launched 5 Stage B seeds (5-deg gate x3, 6-deg hedge x2)"
