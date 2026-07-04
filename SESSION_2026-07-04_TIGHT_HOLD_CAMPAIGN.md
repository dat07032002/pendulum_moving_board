# Tight-hold campaign: fewer falls, tighter upright, smoother motor (2026-07-04)

Goals (user): (1) reduce fall-and-recover under board tilt, (2) tighten upright to +/-5 deg,
(3) reduce motor vibration. Robustness-first: never regress the +/-10 deg success/cable gates.

## Infrastructure added

- `FURUTA_ACTION_RATE_W` (env reward knob, default 0.02 = legacy) and angle-space `up_bonus`
  (old `min(0.99, up_thresh+0.02)` saturated below ~8 deg gates) in `rl/furuta_env_2d.py`.
- Selection metrics `occ7`/`occ5` (fraction of upright time within 7/5 deg) and `mean_abs_da`
  (vibration proxy) in `rl/verify_2d.py` and `rl/probe_capability_2d.py`.
- `rl/train_tight_hold_2d.py`: stationary-recipe fine-tune with tight-gate args; training envs
  get the tight gate, eval envs pinned to the canonical +/-10 deg gate.
- `FURUTA_SLEW_V_PER_TICK` actuator slew limit in `rl/furuta_env_2d.py`, matched
  `RL_SLEW_V_PER_TICK` in firmware `rlStep` (state `rl_applied_v`, reset at engage), and
  `export_policy.py --slew-v` embeds the define so policy and actuator always travel together.

## Critical discovery: the BC anchor clamp

`retention_tqc.py` `teacher_target_ratio` rescales the behavior-clone coefficient to
`min(1e6, ratio*|rl_actor_loss|/teacher_loss)`. A warm-started student has teacher_loss ~1e-7,
so the coef pins at the 1e6 cap = an infinitely stiff behavior clamp. Campaigns tight7 (400k x 3)
and tight7b produced ZERO behavior change (all metrics identical to warm start to 3 decimals)
while weights drifted in the null space and the critic recalibrated (Q 262->307 vs RTG 307).
Fix: `--teacher-ratio none` -> fixed soft coef (0.2). This clamp also explains why earlier
"retention" fine-tunes never regressed OR improved much.

## tight7c (first run with a free actor): 500-ep finals

Warm start tight7_s2/best_safe (recalibrated critic), gate 7 deg, tight_w 0.8 @ 6 deg,
action_rate_w 0.3, anchor 0.2 fixed, actor lr 1e-5. Seeds diverged, dip-and-recover, s0 never
passed safety.

- **tight7c_s1** (deployed 2026-07-04): nominal level 95.0 / roll 97.2 / both45 94.8 / both60
  94.2 (baseline: 99.0/92.8/90.6/88.0); hits 35/4000; m|da| 0.14-0.25 (baseline 0.28-0.54);
  **DR success 79-85% vs baseline 48-62%** — the robustness-margin fix.
- tight7c_s2: safest cable (11/4000) but lower success everywhere and weaker DR (63-73%).
- occ5 unchanged (~0.62) for both — tightness goal still open (Stage B pending).

## Hardware A/B (s1 flashed, sha 664e378b...25ae64)

30 s hand-tilt runs vs the 2026-07-03 baseline log (comparable ~3.3-3.5 deg tilt):

- in-band 62% vs 51%; drop->recover 5 vs 9; longest hold 10.1 s vs 4.2 s. Goal (1) delivered.
- Vibration NOT fixed on hardware: dV RMS 9.06 vs 9.71 V/tick. Diagnosis (analyze_hold_state):
  theta mean -0.06 deg (cal fine) but std 5.5 deg, theta_dot std 6.6 rad/s, 61% of ticks >9 V,
  sign flip every ~3 ticks: a **self-sustained +/-10 V 200 Hz limit cycle** from plant mismatch
  (stiction / motor dynamics the sim lacks). The sim-smooth policy is only conditionally smooth.

## In flight overnight: tight7d (slew) + tight7e (slew + lag DR)

- **tight7d** (GPUs 0-2, 3 seeds from tight7c_s1): `--slew-v 3.0` (3 V/tick actuator, matched
  sim+firmware), gate 7 deg, action_rate_w 0.1, anchor 0.2, warmup 25k (plant change ->
  critic-first). Auto-chain screens the candidates AND tight7c_s1-on-slewed-plant at 300 eps
  (seed block 520000) when training ends.
- **tight7e** (GPUs 3-4, 2 seeds, same recipe): additionally `--act-lag-ms 2,8` — a sim-only
  first-order actuator lag (per-episode uniform 2-8 ms via `FURUTA_ACT_LAG_TAU_MS`), modelling
  the motor-electrical chain behind the 26 Hz limit cycle. The real rig has this lag physically,
  so no firmware pairing is required (unlike the slew limit). Its auto-chain screens on the
  slew-only plant with the SAME seed block, so tomorrow gives a directly comparable three-way:
  s1 baseline / slew-only / slew+lag.

## Evening measurements: friction ruled out, limit cycle fingerprinted

Arm friction ID (`arm_friction_id.py`, motor-driven, guard-protected):

- Breakaway voltage 0.20-0.50 V (mean 0.37 V) across 8 positions/directions — the sim's
  `frictionloss=6e-3` predicts 0.47 V, so stiction is slightly LESS than modeled. Stiction is
  NOT the vibration cause. Steady-speed sweep aborted safely by the position guard (constant-V
  bursts cover 280 deg too fast); coast-down variant possible later, low priority.
- Raw data: `eval/hw_bringup/arm_friction_raw.txt`, `arm_friction_results.csv`.

Limit-cycle spectroscopy (`analyze_limit_cycle.py` on the s1 hold data): coherent **26 Hz**
oscillation in theta, theta_dot, AND V, voltage sign-flip at ~25 Hz, theta amplitude ~5 deg.
Pendulum natural frequency is 2.3 Hz -> this is a control-loop limit cycle at the frequency
where loop phase margin vanishes, i.e. a few ms of unmodeled high-frequency lag (motor
electrical/FOC/filter chain). Consistency check for the fix in flight: a 26 Hz +/-10 V cycle
requires ~8.2 V/tick of slew; the 3 V/tick limiter caps it at ~3.7 V amplitude, and the
tight7d policies are being trained WITH that actuator so they will not fight it.

## Next steps

1. Review tight7d screens; winner -> 500-ep finals (slewed plant, nominal + DR).
2. Export winner with `--slew-v 3` (embeds RL_SLEW_V_PER_TICK), flash, hardware A/B
   (dV RMS target: well under 3 V/tick by construction; listen for the difference).
3. Do NOT flash slew firmware with a non-slew policy or vice versa — the exporter pairing
   prevents this if you always regenerate `policy_weights.h`.
4. Stage B (+/-5 deg tightness) from the slew winner; occ5 is the remaining unmet goal.
5. Longer term: stiction/Stribeck sysid to close the plant gap at its root.
