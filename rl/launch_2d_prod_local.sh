#!/usr/bin/env bash
# Local production run: validated Phase-B recipe (re-init critic, gamma=0.99, gentle +/-10 deg
# ladder, soft advancing gate, retention floors), 3 seeds run SEQUENTIALLY on the single local
# GPU. Target envelope: +/-10 deg up to 120 deg/s, both axes.
set -uo pipefail
cd "$(dirname "$0")"

for seed in 0 1 2; do
  tag="prod2d_v1_s${seed}"
  echo "=== $(date '+%H:%M:%S') launching ${tag} ==="
  python train_phaseb_2d.py \
    --tag "${tag}" \
    --seed "${seed}" \
    --steps 900000 \
    --gamma 0.99 \
    --warmup-steps 50000 \
    --actor-lr 3e-5 \
    > "prod_${tag}.log" 2>&1
  echo "=== $(date '+%H:%M:%S') ${tag} finished (exit $?) ==="
done
echo "=== all production seeds done ==="
