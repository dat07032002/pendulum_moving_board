#!/usr/bin/env bash
# tight8: the action-history campaign. Obs 10->12 (a_{t-2}, a_{t-3} appended)
# restores observability for delay <= 3, which three campaigns proved is the
# blocker for delay-2 robustness (the hardware limit cycle's root cause).
# Warm start = tight7f_s2 surgically expanded (expand_obs_warmstart.py,
# zero-init new input columns -> exact behavioral equivalence at start).
# Everything else identical to the successful tight7f recipe.
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
  tag="tight8_s${seed}"
  setsid -f env \
    CUDA_VISIBLE_DEVICES="$seed" \
    "${COMMON[@]}" \
    "$PY" train_tight_hold_2d.py \
      --warmstart models/tight7f_s2_h2.zip \
      --teacher-data teacher_s1_safe_nominal_100k.npz \
      --tag "$tag" --seed "$((seed + 80))" --steps 400000 --nenv 8 \
      --warmup-steps 25000 --actor-lr 1e-5 --critic-lr 1e-4 \
      --rehearsal-fraction 0.15 --teacher-coef 0.2 --teacher-ratio none \
      --up-thresh-deg 7 --tight-scale-deg 6 --tight-w 0.8 --action-rate-w 0.3 \
      --slew-v 0 --act-lag-ms 3,9 --delay-steps 1,2 --act-history 2 \
      > "train_${tag}.log" 2>&1 < /dev/null
done

echo "launched 5 tight8 seeds (act-history 2, delay {1,2}, lag DR)"
