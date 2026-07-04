# First hardware deployment of the 10-D policy (2026-07-03)

Continuation of `SESSION_2026-07-03_DELAY_TRAINING_AND_DEPLOYMENT_HANDOFF.md`. Outcome: the
10-D policy `rl/models/progressive_dr_s1/best_safe.zip` is flashed and balancing on hardware,
including autonomous drop-and-recover under hand-applied board tilt.

## Decision that unblocked deployment: hardware latency is one step

Measured on the rig (motor-off log capture + 0.8 V torque step at `vlim 2`):

- control loop 5.008 ms mean / 6 ms max (clean 200 Hz);
- BNO086 reports mean 2.12 ticks (~100 Hz), worst observed gap 30 ms (guard is 100 ms);
- commanded voltage appears in the same tick the command is processed; encoder velocity responds
  within 1-2 ticks (the second tick is acceleration + quantization, not latency).

End-to-end action latency is therefore ~1 control step, matching `FurutaEnv` `delay=1` semantics.
The two-step-delay robustness requirement was unnecessary; no delay=2 retraining is needed.
Confirmed the entire policy family collapses at forced delay=2 (best_safe: 0-2.5% at 40 eps/cell),
and that the delay-continuation "eligible" model retains full delay=1 competence
(87-99%/cell, 22 cable hits/2400) despite failing delay=2 — see
`eval/delay_s0_eligible_nominal_d1_300.json`, `eval/best_safe_nominal_d2_40.json`.

## Final simulation gate (fresh seeds 220000+, 500 eps/cell)

Nominal delay=1 (`eval/final_best_safe_d1_nominal_500.json`):

- level 99.0, roll60 92.8, pitch30 97.8, pitch45 97.8, pitch60 98.0,
  both30 95.2, both45 90.6, both60 88.0 percent;
- 28 cable hits / 4,000 (0.70%, matching the recorded 0.67%);
- critic calibration sane (Q=263 vs RTG=307 at level).

Full DR delay=1 (`eval/final_best_safe_d1_dr_500.json`): 47.6-61.6% success, catch 60-68%,
35 hits/4,000. Thin robustness margin, known caveat; nominal is representative because the rig is
sysid-calibrated (`sysid.json` is the post-new-bearing fit).

## Export and flash

- Fixed a real exporter bug: `--vmax 10` emitted the invalid C literal `#define RL_VMAX 10f`;
  `rl/export_policy.py` now emits `10.0f` (repr formatting).
- Exported `firmware/furuta_foc/policy_weights.h`: OBS=10, forward-pass error 3.28e-6,
  sha256 `c950d804...cd7986`, `RL_VMAX 10.0f`.
- Compiled `esp32:esp32:esp32` (30% flash / 8% RAM) and flashed to COM5.

## Motor-off preflight (all PASS)

- `health`: AS5600 and AS5048A healthy; `imu`: valid, level tare good.
- `rlcheck` at level: board obs ~0; +roll/+pitch holds read the correct channels with the
  0.667-at-10-deg scale, symmetric both directions, cross-axis leakage <=0.04;
  gyro channels sign-consistent with angle rates.
- Stale-IMU fault test: with BNO086 unplugged, `rlcheck` FAILED and `rl` REFUSED (0 V).
  Recovery verified after replug.

## Powered bring-up

| Stage | Result |
|---|---|
| vlim 1, 4 s | Direction correct (+V -> +phi accel, corr +0.42). Arm wound to 333.7°: cable guard fired, homing recovery engaged. Expected below 10 V. |
| vlim 2, 4 s | corr +0.53, peak arm 192°, clean. IMU healthy through the whole vigorous run. |
| vlim 3, 4 s | corr +0.66, peak arm 83°, clean. |
| vlim 6, 6 s | corr +0.62, arm hit 360.4° (30° overshoot past the 330° guard at ~36 rad/s). Full auto-recovery cycled twice: guard -> homing -> recenter -> re-engage. |
| vlim 10, 8 s | First run failed to catch - led to discovery of the pole-encoder fault below. After fix: catch at 0.9 s, holds 0.5-0.7 s. |
| vlim 10, 30 s | Catch at 0.6 s; 51% of run inside ±10°; holds up to 4.2 s; 9 autonomous drop->recover cycles under hand-applied ±3.5° board tilt; arm brushed 356° once, recovered. |

Logs: `eval/hw_bringup/stage*_vlim*.txt` (per-tick CSV, 200 Hz).

## Incidents and fixes

1. **Stale pole calibration.** Flash held `UPRIGHT_RAW=708` from before the bearing swap; hanging
   read 155° (and later -31°) instead of 180°. Any balance attempt was hopeless. `calhang` fixed.
   Lesson: re-run `calhang` after any pole-assembly work; verify hanging=±180 / upright=0 before
   powered runs.
2. **AS5600 field collapse (root cause: twisted encoder cable).** After the vigorous runs, a
   physical 360° pendulum rotation swept the raw angle only ~25° with wobble (AGC pegged 115/128).
   Looked exactly like a slipped/axial magnet; actual cause was the twisted cable bundle dragging
   the sensor/magnet geometry off-axis. Untwisting restored full 0-4095 sweep. `raw`-based check
   (`raw` command bypasses the software glitch filter) is the discriminating test.
3. **BNO086 warm-boot wedge.** The BNO086 is not reset by ESP32 DTR/warm reboots and repeatedly
   came up "missing" after reboots (roughly every other one). Neither an init retry loop nor an
   I2C 9-pulse bus-clear (both now in firmware; harmless) recovers a hard wedge - only a full
   power cycle does. Guards fail safe every time (`rl` refused, 0 V). Real fix: wire BNO086 NRST
   to a spare GPIO (or MOSFET its 3V3) so firmware can hard-reset it at boot.
4. **Workflow: serial opens no longer reboot the board.** All host-side helper scripts now open
   COM5 with DTR/RTS deasserted, so connecting does not reset the ESP32 (and cannot wedge the
   BNO086 or invalidate the tare mid-session).

## Firmware changes this session (flashed)

- 10-D policy header (`policy_weights.h`) with `RL_VMAX 10.0f`.
- `initBNO()`: 5-attempt retry with 200 ms spacing + `clearI2CBus()` 9-pulse SCL bus-clear before
  `Wire1.begin()`.

## Open items

1. **Wire BNO086 NRST to a GPIO** - the warm-boot wedge will keep costing manual power cycles.
2. **Glue the AS5600 magnet / add cable strain relief** so the twist cannot recur; consider a
   `health`-style AGC threshold warning in firmware (AGC > ~100 means degraded field).
3. **Glitch-filter escape**: `AS5600_MAX_RAW_STEP` rejection has no escape path; after N
   consecutive rejections it should accept the new reading (currently it can latch stale forever).
4. **Hold-fraction gap vs sim** (51% in-band on hardware vs ~98% nominal sim at level). Note the
   comparison is not apples-to-apples: the 30 s run had hand-applied ±3.5° board tilt, where sim
   success is 88-95% per episode. `sysid.json` is already the post-new-bearing fit, so the model
   is current. Next diagnostics, in order: (a) a level, stationary 30 s run for a fair baseline;
   (b) quantify supply-voltage sag at 10 V commands; (c) verify theta calibration precision
   (calhang repeatability) and IMU tare offset after warm-up.
5. **Guard overshoot**: momentum carries ~30° past the 330° software guard at high arm speed;
   fine at the current 360° cable budget but worth a speed-dependent guard if the budget shrinks.
