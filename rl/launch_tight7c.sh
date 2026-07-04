#!/usr/bin/env bash
# Tight-hold Stage A'': same pressure as tight7b, but with the BC anchor actually
# weakened: --teacher-ratio none disables the adaptive rescaling in retention_tqc
# that drove effective_teacher_coef to its 1e6 cap and froze the actor's behavior
# in tight7 and tight7b. Anchor is now a fixed coef 0.2 that fades as the student
# stays close to the teacher. Eval gate unchanged (+/-10 deg, zero-hit).
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
  tag="tight7c_s${seed}"
  setsid -f env \
    CUDA_VISIBLE_DEVICES="$seed" \
    "${COMMON[@]}" \
    "$PY" train_tight_hold_2d.py \
      --warmstart models/tight7_s2/best_safe.zip \
      --teacher-data teacher_s1_safe_nominal_100k.npz \
      --tag "$tag" --seed "$((seed + 20))" --steps 400000 --nenv 8 \
      --warmup-steps 10000 --actor-lr 1e-5 --critic-lr 1e-4 \
      --rehearsal-fraction 0.15 --teacher-coef 0.2 --teacher-ratio none \
      --up-thresh-deg 7 --tight-scale-deg 6 --tight-w 0.8 --action-rate-w 0.3 \
      > "train_${tag}.log" 2>&1 < /dev/null
done

echo "launched 3 tight7c seeds"
