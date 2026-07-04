# Handoff: cable safety, delay training, verification, and deployment

Date: 2026-07-03 (America/Chicago)

This is the current continuation point. Also read `SESSION_2026-07-01_RIG_IMU_SYSID.md` for the
detailed rig, sensor, pole-system-ID, and simulation-calibration work.

## Target

- Two-axis board envelope: +/-10 degrees, maximum 60 degrees/s.
- Upright success: +/-10 degrees.
- Production training voltage: 10 V.
- Physical cable travel: +/-360 degrees from center.
- Success-safe arm margin: +/-330 degrees.
- ESP32 control: 200 Hz.
- BNO086 observations: 100 Hz.
- Allowed action-delay model: 1-2 control steps (5-10 ms). Hardware experience says 3 steps fails.

The current blocker is **two-step delay robustness**. The latest actor remains useful at one-step
timing but fails catastrophically when the verifier forces two steps.

## Hardware and firmware

- Firmware: `firmware/furuta_foc/furuta_foc.ino`.
- Power-on/reset leaves the motor idle. Only `rl` enables policy control.
- BNO086 is on dedicated `Wire1`: SDA GPIO32, SCL GPIO33, 400 kHz, address 0x4B with 0x4A fallback.
- Game Rotation Vector and calibrated gyro run at 100 Hz.
- Boot performs a 500 ms software level tare.
- Physical +roll -> IMU +roll / gyro-X.
- Physical +pitch -> IMU +pitch / gyro-Y.
- Source cable guard is +/-330 degrees relative to the arm position recorded at `rl` engagement.
- `firmware/furuta_foc/policy_weights.h` is still the old 6-D policy. No 10-D policy was flashed.
- The ESP32 was disconnected during the latest work (COM5 absent).

Prepared firmware changes, all compiled but **not flashed**:

- supports legacy 6-D and new 10-D actors;
- 10-D order matches `Furuta2DEnv._obs()` exactly:
  `[cos(theta), sin(theta), theta_dot/15, clip(phi/pi,+/-2), phi_dot/25, prev_action,
  clip(roll/rad(15),+/-2), clip(pitch/rad(15),+/-2), gyro_x/rad(80), gyro_y/rad(80)]`;
- `rl` refuses a 10-D policy if BNO086 is invalid, untared, or over 100 ms stale;
- stale IMU data while running stops the policy and motor;
- `rlcheck` prints the normalized observation and deterministic action while motor remains off;
- generated policy headers embed `RL_VMAX`; firmware no longer assumes the actor was trained at 6 V.

Both the legacy build and a temporary 10-D build compile for `esp32:esp32:esp32`. The temporary
10-D header was only a compatibility test, not an approved deployment policy.

## Measured rig model

Latest pole fit in `sysid.json`:

- period 440 ms, alpha 204;
- rho approximately 0.928-0.930;
- damping `b_theta` approximately 3.4e-5 N*m*s/rad;
- Coulomb friction `Tf` approximately 0.15-0.16 mN*m;
- deadband approximately 0.9-1.0 degrees.

BNO086 model:

- requesting both reports at 200 Hz was not sustainable;
- stable reports at 100 Hz with a 200 Hz control loop;
- gyro latency 3.7 ms;
- orientation approximately 2-3 ms behind gyro;
- orientation noise approximately 0.012 degrees;
- gyro noise approximately 0.5 degrees/s;
- modeled gyro bias range approximately 0.06 degrees/s.

## Canonical training environment

```text
FURUTA_VMAX=10
FURUTA_UP_THRESH=0.984807753
FURUTA_CABLE_LIMIT_DEG=360
FURUTA_SUCCESS_ARM_LIMIT_DEG=330
FURUTA_ARM_CENTER_W=0.02
FURUTA_TIGHT_UPRIGHT_W=0.25
FURUTA_TIGHT_UPRIGHT_SCALE_DEG=10
FURUTA_CABLE_WARNING_W=0.20
FURUTA_CABLE_WARNING_START_DEG=270
```

Full DR:

- motor gear 0.010-0.016, nominal 0.0127;
- arm damping 3e-4 to 1e-3, nominal 9.4e-4;
- pole damping 1.4e-5 to 5.4e-5;
- arm friction 4e-3 to 8e-3, nominal 6e-3;
- pole friction 6.5e-5 to 2.65e-4;
- pole inertia scale 0.92-1.08;
- observation angle noise 0-0.01 rad;
- action FIFO length 1 or 2 only;
- corner probability 0.1.

Important delay semantics: `FurutaEnv` initializes `act_buf=[0]*delay`. Thus `delay=1` applies the
previous action and `delay=2` applies the action from two control periods earlier.

## Safe rehearsal

Server dataset: `rl/teacher_s1_safe_nominal_100k.npz`.

- 100,000 successful cable-safe transitions;
- balanced across level, roll60, pitch60, both30, both45, both60;
- nominal +/-360/330 environment;
- only accepted trajectories with maximum arm <=270 degrees;
- training uses 15% teacher and 85% new transitions.

No-rehearsal controls repeatedly collapsed. Keep rehearsal unless performing a deliberate ablation.

## Model lineage and verified results

### Original and corrected-plant models

`up15_best.zip` is hash-identical to `models/up15_s1/best_stage_6.zip`: old +/-15 degree,
120 degrees/s, 10 V, originally unrestricted arm.

`models/c360_s0ft_s1/best_stage_3.zip`, 300 episodes x 8:

- success: level 98.3, roll60 91.0, pitch30 99.0, pitch45 97.3, pitch60 98.0,
  both30 95.7, both45 89.7, both60 87.7 percent;
- 33 cable hits / 2,400.

`models/stationary_c360_nominal_primary_s1/best_safe.zip` became the main warm start:

- success: level 98.3, roll60 92.0, pitch30 99.3, pitch45 97.7, pitch60 97.7,
  both30 95.3, both45 90.0, both60 88.0 percent;
- 21 cable hits / 2,400.

### Nominal cable-safety fine-tunes

- `safety_c345_s3/tqc_final.zip`: 17 hits / 2,400; roll60 88.3, both45 91.7,
  both60 87.3 percent.
- `safety_c345_s4/tqc_final.zip`: 17 hits / 2,400; roll60 88.0, both45 91.7,
  both60 87.0 percent.

They reduced hits by 19% versus the 21-hit warm start, but lost roll performance and failed the
zero-hit gate.

### Progressive DR

Runs: `progressive_dr_s0`, `progressive_dr_s1`, `progressive_dr_s2`, 500k each.

- 0-100k: 25% mechanical DR;
- 100-200k: 50% mechanical DR;
- 200-300k: full mechanical DR;
- 300-400k: full mechanical plus observation noise;
- 400-500k: full DR including randomized 1-2 step delay.

Final critics collapsed toward zero/negative Q while real discounted returns remained around 270.
Final checkpoints were rejected.

Best candidate: `models/progressive_dr_s1/best_safe.zip`, saved near 350k before action delay.
Nominal 300 x 8:

- success: level 97.3, roll60 88.7, pitch30 96.7, pitch45 97.3, pitch60 97.3,
  both30 95.7, both45 91.7, both60 87.0 percent;
- 16 cable hits / 2,400.

This is the lowest nominal cable-hit result before delay continuation, but is neither zero-hit nor
delay-robust.

## Delay-continuation experiment

Files: `rl/train_c360_delay_continuation_2d.py` and `rl/launch_delay_continuation3.sh`.

Warm start: `models/progressive_dr_s1/best_safe.zip`.
Runs: `delay12_from_dr_s1_s0`, `s1`, `s2`.

- 200k, 8 environments;
- critic reset; actor frozen for 50k;
- actor LR 5e-7, critic LR 3e-5;
- 15% rehearsal;
- evaluation every 25k;
- 0-50k: full mechanical/observation DR, fixed delay=1;
- 50-125k: randomized delay=1/2;
- 125-200k: fixed delay=2.

All completed. At 175k every seed had a zero-hit internal evaluation. S0 repeated it at 200k and
created `eligible_model.zip`. This label is misleading: `StationarySafetyEval` always built a
nominal delay=1 evaluation environment and did not evaluate the active training delay.

Critics collapsed again at 200k:

- S0 Q/RTG -32/263;
- S1 -32/263;
- S2 -55/264.

## Explicit-delay verification

`rl/verify_2d.py` now supports `--delay-steps 1|2` and `--randomize`, records plant/delay metadata,
and suppresses mismatched nominal Q calibration during DR or delay=2 tests.

Candidate: `models/delay12_from_dr_s1_s0/eligible_model.zip`.

Nominal plant, forced delay=2, 300 x 8:

- success: level 2.7, roll60 1.3, pitch30 1.7, pitch45 2.0, pitch60 1.0,
  both30 0.0, both45 0.0, both60 0.0 percent;
- 27 cable hits / 2,400.

Full DR, forced delay=2, 300 x 8:

- success: level 3.3, roll60 0.7, pitch30 4.0, pitch45 3.3, pitch60 0.7,
  both30 0.7, both45 0.0, both60 0.3 percent;
- 17 cable hits / 2,400.

Conclusion: the S0 eligible actor is not deployable at two-step delay. Training really did use
delay=2 after 125k; the likely cause is failed adaptation associated with critic collapse, not a
stage-setting bug.

Nominal delay=1 300 x 8 was still running at the last successful status check. Its longer runtime
suggests longer surviving episodes, but retrieve actual metrics:

- log: `~/furuta_tilt_2d/rl/verify_delay_s0_nominal_d1_300.log`;
- JSON: `~/furuta_tilt_2d/rl/eval/delay_s0_eligible_nominal_d1_300.json`.

The final status query failed because the local machine temporarily could not resolve the server.

## Server

```powershell
ssh -i $HOME/.ssh/aere_codex_ed25519 tn22833@aere-a83514.ae.utexas.edu
```

- Project: `~/furuta_tilt_2d/rl`
- Python: `~/furuta_rl/.venv/bin/python`
- Five RTX 6000 Ada GPUs.
- Files are synchronized manually with `scp`; remote is not a Git repository.

## Deployment preparation

Files: `rl/export_policy.py`, `firmware/furuta_foc/furuta_foc.ino`, `FIRST_DEPLOYMENT.md`.

Exporter supports 6/8/10-D actors, requires explicit `--vmax`, embeds checkpoint SHA-256 and
`RL_VMAX`, and verifies its NumPy/C-equivalent forward pass against SB3. A test export matched
within `3.28e-6`, below the `1e-4` gate.

Re-export any eventual winner because `--vmax` was added after the temporary header was generated:

```powershell
python rl/export_policy.py --model <approved.zip> --vmax 10 `
  --out firmware/furuta_foc/policy_weights.h
```

Do not flash a current candidate. None has passed cable and delay gates.

## Unresolved safety issues

1. No model has achieved zero cable hits over the full 2,400-episode nominal grid.
2. The latest delay-trained model fails catastrophically at delay=2.
3. Firmware arm zero is set at `rl` engagement. Physically center the cable first.
4. Automatic cable recovery is unvalidated on hardware. Do not rely on it initially.
5. Simulation verification does not replace motor-off `health`, `imu`, and `rlcheck`.

## Recommended next steps

1. Retrieve the completed nominal delay=1 result.
2. Screen `ckpt_125000`, `ckpt_150000`, `best_safe`/`ckpt_175000`, and `ckpt_200000` with only
   30-50 episodes per condition at explicit delay=2 before spending more full-grid time.
3. If every checkpoint fails delay=2:
   - measure whether hardware end-to-end latency truly requires two FIFO steps;
   - add a second previous action/action-history observation if delay=2 is real;
   - train delay=2 from the start of critic reconstruction instead of changing the MDP at 125k;
   - evaluate and select on delay=2 during training;
   - consider recurrent/history-based policies if the delayed system is partially observable.
4. If hardware latency is one step, train and verify the measured one-step model rather than impose
   an unsupported robustness requirement.
5. Only after nominal, exact-delay, and DR gates pass: run final 500-episode verification, export
   with `--vmax 10`, compile, flash with motor power disabled, and follow `FIRST_DEPLOYMENT.md`.

## Workspace caution

The local worktree is intentionally dirty with user changes and generated data. Do not reset or
discard unrelated work. No commit was requested. Use `apply_patch` for source edits.
