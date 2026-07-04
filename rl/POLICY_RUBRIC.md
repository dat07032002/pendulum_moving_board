# Policy acceptance rubric — Furuta swing-up + balance (TQC)

## Current corrected-bearing experiment (2026-07-01)

- Command envelope: board tilt within +/-10 deg and reference speed no higher than 60 deg/s.
- Upright requirement: pole remains within +/-10 deg of true gravity vertical.
- Cable-aware seeds: hard termination at arm +/-360 deg; success is eligible only inside
  +/-330 deg.
- Free-arm seeds: no hard or success arm bound during training, but retain the same 0.02
  arm-centering reward.
- Screen each candidate over 300 fresh deterministic episodes on the `pm10_60` grid, then verify
  the selected deployment winner over 500 fresh episodes per condition.
- Re-evaluate free-arm candidates with the +/-360 deg hard and +/-330 deg success limits imposed.
  A checkpoint that crosses the physical cable boundary is not deployable.
- Report sustained success, catch success, arm p95/max, fraction outside +/-330 deg, cable hits,
  action saturation, and action smoothness. Do not select on success rate alone.

How we decide a trained policy is good enough to deploy. Sim passes are **necessary but not
sufficient** — the real judge is the hardware. Use this for Step 6 (validate) before Step 7
(deploy). Evaluate the **deterministic** policy (mean action, no exploration noise).

## Pass 1 — deterministic eval (run ACROSS the domain-randomization range, not just nominal)
- **Success rate ≥ 80%** across randomized friction / KM / latency (not just nominal sim).
- **Hold time ≈ full episode** (sustained balance, not just clearing the 0.5 s success bar).
- **Action not always saturated** (headroom; not bang-bang ±6 V).
- **Action smooth** — low high-frequency content / small `mean|Δa|` (CAPS metric). A jerky-but-
  unsaturated policy still excites the open-loop-motor vibration / limit-cycle and won't transfer.

## Pass 2 — multiple seeds
- Consistent across **3–5 seeds** (and/or parallel configs). Not one lucky run. RL is high-variance.

## Pass 3 — region-of-attraction (RoA) handoff
- The **states swing-up delivers** the pole into must lie **inside the balance controller's
  catchable region** (otherwise the two work alone but fail combined).
- **Arm angle and action stay realistic/physical** throughout (no exploiting sim quirks).

## Pass 4 — sim-to-real readiness (the gap the sim passes don't cover)
- **DR robustness** = Pass 1 re-stated: success must hold across the measured ~2× motor-param
  spread, not at nominal only. This is the #1 predictor of real-world success.
- **Latency robustness** — survives the real 1-step + EMA-filter lag (modeled in the env; verify
  it isn't relying on instantaneous reaction).
- **Sign alignment check (before trusting hardware)** — cos(θ) is sign-immune, but `sinθ`, `θ̇`,
  and the action sign must match the firmware (analog of the LQR sign check). Verify at PC-in-loop.

## Pass 5 — the final judge: real hardware
- **PC-in-loop** on the rig: swings up / holds (latency-limited, for validation).
- **On-chip MLP** (`MODE_RL`): standalone swing-up + balance, arm soft-limit intact; compare vs
  the LQR baseline (`bal`).

## Quick reference
sim-good = Pass 1–3 (under DR) + smoothness. deploy-ready = + Pass 4. done = Pass 5 on hardware.

---

# Tilt-project additions (randomly-tilting base, ±30°)

This project extends the above to a base that tilts ±30° (LX-16A servo), with `β`/`β̇` from a
BNO086 IMU. The policy must balance to **true (gravity) vertical**, not base-frame. Extra gates:

## Pass 1-T — success under random tilt (the headline gate)
- **Success ≥ 80% under random ±30° tilt**, across tilt **amplitude (0–30°) AND rate (0–2 rad/s)**,
  *on top of* the existing plant DR (KM/friction/latency). Evaluate deterministically.
- **True-vertical hold:** `true_up` (geometric, from `_true_up()`) stays high through the whole tilt
  trajectory — i.e. it tracks **gravity** vertical as the board moves, not base-frame "up".
- **Arm stays within ±360°** the whole time (no cable-limit hits); auto-recovery is a backstop, not
  the plan.

## Pass 3-T — worst-orientation robustness
- Survives **transits through arm φ ≈ 90°** (swing plane aligned with the tilt → max disturbance,
  per Phase 0). Don't only test the benign φ≈0 orientation.

## Pass 4-T — IMU / β sim-to-real readiness
- **IMU robustness:** holds with the characterized BNO086 noise and independent 100 Hz
  orientation/gyro reports sampled by the 200 Hz controller.
- **β sign/scale alignment** at PC-in-loop: `+β` in firmware (IMU tilt direction) must match the sim
  convention, alongside the existing `sinθ`/`θ̇`/action sign checks. `β` is normalized by 0.6 rad.
- **Use the IMU's mag-free fusion** (Game Rotation Vector / Gravity) — the motors disturb the
  magnetometer.

## Pass 5-T — hardware (final)
- PC-in-loop then on-chip `MODE_RL` (**8-D obs**), balancing while the LX-16A drives random ±30°
  tilt; quantify success rate + arm/cable margin under tilt.

**Tilt quick ref:** tilt-sim-good = Pass 1-T + 3-T (under tilt+plant DR) + smoothness.
deploy-ready = + Pass 4-T. done = Pass 5-T on the tilting rig.

## Canonical independent evaluation

Do not select a model from a training callback or a single 100-episode peak. Use deterministic
inference over at least 500 fresh episodes, preserve the per-episode evidence, and record the model
SHA-256 printed by the evaluator. Compare candidates on identical seeds:

```bash
python rl/eval_policy.py FIRST.zip --compare SECOND.zip \
  --tilt_deg 20 --dr -n 500 --seed0 9000 --arm free \
  --save_npz paired_eval_500.npz
```

For the final deployment gate, change to `--tilt_deg 30 --arm cable`. Review all reported metrics,
not only sustained success:

- sustained and catch success with 95% confidence intervals;
- final true-vertical quality and post-catch balance occupancy;
- action saturation and `mean|delta action|`;
- maximum arm excursion and cable margin;
- exposure and balance quality near `phi=+-90 deg`;
- realized tilt amplitude/rate and DR draws.

The paired confidence interval must support any claim that one checkpoint is better. A passing
simulation result still requires the sign/scale and hardware gates above.
