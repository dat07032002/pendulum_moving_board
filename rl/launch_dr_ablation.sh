#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/furuta_tilt"
PY="$HOME/furuta_rl/.venv/bin/python"
MODEL="${1:-rl/models/clean20_master_verified91p5.zip}"
N="${2:-500}"
SEED0="${3:-30000}"
OUT="${4:-dr_ablation_clean20}"

components=(
  motor_gear
  arm_damping
  pole_damping
  arm_friction
  pole_friction
  pole_inertia
  obs_noise
  action_delay
  imu_noise
)

mkdir -p "$OUT"

launch_eval() {
  local name="$1"
  local enabled="$2"
  setsid -f env \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    CUDA_VISIBLE_DEVICES="" \
    "$PY" rl/eval_policy.py "$MODEL" \
      --tilt_deg 20 \
      --dr \
      --dr_components "$enabled" \
      --p_corner 0.10 \
      -n "$N" \
      --seed0 "$SEED0" \
      --arm free \
      --save_npz "$OUT/$name.npz" \
      > "$OUT/$name.log" 2>&1 < /dev/null
}

# Matched anchors: nominal parameters with DR RNG active, and the complete DR stack.
launch_eval "only_none" "none"
launch_eval "full_all" "all"

for component in "${components[@]}"; do
  launch_eval "only_${component}" "$component"

  enabled=()
  for candidate in "${components[@]}"; do
    if [[ "$candidate" != "$component" ]]; then
      enabled+=("$candidate")
    fi
  done
  enabled_csv="$(IFS=,; echo "${enabled[*]}")"
  launch_eval "without_${component}" "$enabled_csv"
done

echo "launched 20 matched evaluations in $OUT (N=$N, seed0=$SEED0)"
