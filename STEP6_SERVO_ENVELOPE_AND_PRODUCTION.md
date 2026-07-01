# Step 6: Phase B validation, servo/feasibility check, envelope decision, production launch

Date: 2026-06-30

## Phase B result (method validation)

The Phase-B recipe — keep warm-start actor, **re-initialize the critic**, **gamma=0.99**, short
frozen-actor critic warmup, gentle pitch-speed ladder, soft *advancing* curriculum gate — was
tested on one local seed (`phaseb_test_s0`, 350k steps). Outcome:

- **Critic fix fully validated.** The critic stayed calibrated throughout (level Q ~281 vs true
  return-to-go 283) and positive on every regime. The negative divergence that killed the prior
  run did not recur, even after the actor unfroze. Retention stayed ~100% the whole run.
- **Real improvement where feasible.** Cleared stages up to pitch +/-15 deg 70-85 deg/s (0.87)
  and improved pitch +/-10 deg 120 deg/s from 0.55 (warm start) to 0.73. The old run *died* at
  pitch +/-10 deg 60-90 deg/s; this one sails past it.
- **A real wall at pitch +/-15 deg, high speed.** Stage 4 (+/-15 deg, 85-100 deg/s) plateaued at
  0.30-0.37 across 175k steps; stage 5 (+/-15 deg, 100-120 deg/s) started at 0.17. Flat, with a
  healthy critic — i.e. not a training/critic problem.

## Servo / feasibility check

Question: is the +/-15 deg 120 deg/s wall an artifact of board-servo overshoot (the Step-4
reports noted ~140 deg/s realized for a 120 deg/s reference)?

- `rl/servo_id_2d.py`: the bare position servo (kp=80) **tracks the reference accurately**
  (overshoot ~1.00, realized 113 vs reference 113 deg/s, <1 deg RMS). The reference itself is
  not overshot.
- `rl/probe_servo_stiffness_2d.py`: the ~140 deg/s realized rate is **reaction-driven** — the
  Furuta motor torque shoves the compliant mount. A stiff mount (kp=800) cuts abs-max board rate
  137 -> 108 deg/s, **but raises sustained success only 0.15 -> 0.20** at +/-15 deg 120 deg/s.

Conclusion: the wall is **not** a servo artifact. +/-15 deg amplitude at ~120 deg/s continuous
pitch is at the +/-6 V authority/bandwidth edge for sustaining true-vertical balance. Reducing
realized board motion does not unlock it.

## Envelope decision

**Target envelope for v1: +/-10 deg, up to 120 deg/s, both axes.** (+/-10 deg 120 deg/s is
feasible — 0.73 and climbing in Phase B.) **+/-15 deg fast is out of scope** as a training
target; +/-15 deg robustness margin is kept only through the retention guards (slow +/-15 deg
and static +/-15 deg corners). The real board's measured motion (hardware, BNO086) may not even
demand +/-15 deg 120 deg/s continuous; revisit with logged trajectories.

## Production run (local)

`rl/launch_2d_prod_local.sh` -> `rl/train_phaseb_2d.py` with the +/-10 deg ladder:

| Stage | Axis | Angle | Speed deg/s |
|---:|---|---:|---|
| 0 | pitch | 10 | 40-60 |
| 1 | pitch | 10 | 60-80 |
| 2 | pitch | 10 | 80-100 |
| 3 | pitch | 10 | 100-120 |
| 4 | both | 10 | 60-90 |
| 5 | both | 10 | 90-120 |
| 6 | both | 10 | 30-120 |

- Recipe: re-init critic, gamma=0.99, 50k frozen-actor warmup, actor_lr 3e-5, critic_lr 1e-4,
  teacher_fraction 0.25, soft advancing gate (advance on >=0.78 or stage timeout -> advance),
  retention floors (level/roll >=0.90, slow >=0.85, corners >=0.85) logged each eval.
- 3 seeds (`prod2d_v1_s0/1/2`), run **sequentially** on the single local RTX 5070, 900k steps
  each (~overnight). Tags/logs: `models/prod2d_v1_s{0,1,2}/`, `rl/prod_prod2d_v1_s{0,1,2}.log`.

## Selection (next)

Select the best seed by independent >=500-episode evaluation across all conditions plus the
critic-calibration gate (`rl/probe_capability_2d.py`). The eval documents the real envelope:
where it is >=95% vs where it degrades. Do not select on a training-buffer peak.
