#!/usr/bin/env bash
# tight7g: delay-robustness consolidation from tight7f_s2.
# Lag sweep 2026-07-04 evening: tight7f_s2 is immune to first-order lag up to
# 20 ms at delay=1 (m|da| ~0.01) but limit-cycles at delay=2 (m|da| 0.19-0.34).
# The remaining hardware 27-31 Hz cycle is therefore a fractional-tick transport
# delay effect. Fix: per-episode delay in {1,2} + lag DR, from the s2f warm start,
# fixed soft BC anchor. NOT the failed 2026-07-03 recipe (that switched to
# delay-2-only mid-run with the 1e6 BC clamp active).
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

for seed in 0 1 2 3 4; do
  tag="tight7g_s${seed}"
  setsid -f env \
    CUDA_VISIBLE_DEVICES="$seed" \
    "${COMMON[@]}" \
    "$PY" train_tight_hold_2d.py \
      --warmstart models/tight7f_s2/best_safe.zip \
      --teacher-data teacher_s1_safe_nominal_100k.npz \
      --tag "$tag" --seed "$((seed + 60))" --steps 400000 --nenv 8 \
      --warmup-steps 25000 --actor-lr 1e-5 --critic-lr 1e-4 \
      --rehearsal-fraction 0.15 --teacher-coef 0.2 --teacher-ratio none \
      --up-thresh-deg 7 --tight-scale-deg 6 --tight-w 0.8 --action-rate-w 0.3 \
      --slew-v 0 --act-lag-ms 3,9 --delay-steps 1,2 \
      > "train_${tag}.log" 2>&1 < /dev/null
done

echo "launched 5 tight7g seeds (delay {1,2} + lag DR 3-9 ms)"
