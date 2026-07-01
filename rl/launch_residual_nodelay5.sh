#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/furuta_tilt"
PY="$HOME/furuta_rl/.venv/bin/python"
DR_COMPONENTS="motor_gear,arm_damping,pole_damping,arm_friction,pole_friction,pole_inertia,obs_noise,imu_noise"

for seed in 0 1 2 3 4; do
  tag="residual_nodelay_s${seed}"
  setsid -f env CUDA_VISIBLE_DEVICES="$seed" "$PY" rl/train_tqc.py \
    --residual_base rl/models/clean20_master_verified91p5.zip \
    --residual_scale 0.05 \
    --free_arm \
    --arm_center_w 0 \
    --start_stage 5 \
    --max_stage 9 \
    --eval_tilt_deg 20 \
    --steps 2000000 \
    --nenv 8 \
    --learning_starts 50000 \
    --lr 1e-4 \
    --ent_coef 0.01 \
    --tqd 2 \
    --clean_floor 0.70 \
    --p_corner 0.10 \
    --dr_components "$DR_COMPONENTS" \
    --tilt_amp_min_fraction 0.70 \
    --tilt_rate_min 1.20 \
    --stop_success 0.80 \
    --n_eval 50 \
    --eval_freq_steps 100000 \
    --seed "$seed" \
    --tag "$tag" \
    > "train_${tag}.log" 2>&1 < /dev/null
done
