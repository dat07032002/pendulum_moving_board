# Project lineage: methods, challenges, and lessons

A consolidated retrospective across the three linked projects, documenting what worked, what was
hard and how it was solved, and the useful facts that carried forward. Written 2026-06-30.

Source projects (all on this machine):
- **P1 — level-ground Furuta:** `C:\Users\thanh\Desktop\LQR_pendulum` (branch `foc`; GitHub
  `dat07032002/lqr_pendulum`). See its `HANDOFF.md`.
- **P2 — one-axis tilt (1D):** `C:\Users\thanh\Desktop\tilt_pendulum`. See `HANDOFF.md`,
  `SESSION_2026-06-26.md`, `SESSION_2026-06-27_TO_29.md`, `SESSION_2026-06-29_CLEAN91_RESIDUAL_DR.md`.
- **P3 — two-axis tilt (2D):** `C:\Users\thanh\Desktop\tilt_pendulum_2d` (this folder). See
  `STEP3–STEP6`, `SESSION_2026-06-30_2D_BOARD_AND_TRAINING.md`, `STEP5_CRITIC_DIAGNOSIS.md`.

---

## 0. The three projects at a glance

| | P1 level-ground | P2 one-axis tilt | P3 two-axis tilt |
|---|---|---|---|
| Task | swing-up + balance, flat | balance through ±20–30° 1-axis tilt | balance through ±15° roll+pitch |
| Obs | 6-D | 8-D (+β, β̇) | 10-D (+roll, pitch, gyroX, gyroY) |
| Balance target | base-frame up | **true gravity vertical** | true gravity vertical |
| Sensing added | AS5600/AS5048A encoders | + BNO086 IMU (1 axis) | + BNO086 (roll/pitch/gyro) |
| Status | **deployed on ESP32, works** | verified master 91.5% (free-arm) | best model verified; 11 V result pending |
| Anchor model | `fix_sde/best_model.zip` | `clean20_master_verified91p5.zip` | `prod2d` seeds + `v11_pm15_nt` |

The through-line: **train a friction-robust nonlinear balancer once, then extend it to
progressively harder disturbances (1-axis, then 2-axis board motion) by transfer + fine-tuning,
never from scratch.**

---

## 1. Successful methods (what worked, and carried forward)

### 1.1 The core RL recipe (established P1)
- **TQC** (Truncated Quantile Critics, sb3-contrib) as the off-policy actor-critic. Small actor
  `[64,64]` (ESP32-portable), larger critic. Only the actor is deployed → the critic is "free."
- **Observation:** `[cosθ, sinθ, θ̇/15, φ/π, φ̇/25, prev_action]` (pre-normalized, no VecNormalize
  → trivial deployment). `prev_action` lets a memoryless actor cope with the modeled action delay.
- **Reward:** `cos(θ)` backbone (drives swing-up *and* balance); velocity penalty **gated to the
  upper half** so pumping isn't punished; **CAPS action-smoothness** `−0.02(Δa)²` (ICRA 2021 —
  smooth control transfers to real motors); arm-centering; `+2` bonus **and success both gated on
  arm < 90°** (closes the "balance at the cable edge" loophole).
- **P3 retargeted the reward to true gravity-vertical** (`_true_up()` from the pole body's world
  orientation), because once the base tilts, base-frame "up" ≠ gravity "up".

### 1.2 Transfer / warm-start instead of from-scratch (P2→P3)
- The verified lower-dimensional master is the warm start for the next project. P3's 10-D warm
  start was a **training-free weight transfer** of the 8-D 1D master: old inputs copied, new
  roll/gyro-X columns zeroed, tilt columns rescaled for the changed normalizers. Verified
  numerically equivalent (max action diff `2.1e-6`).
- **Why:** re-deriving swing-up + balance from scratch is the fragile, expensive part and invites
  TQC seed-variance failures. Transfer preserves it. (P3 confirmed: from-scratch was never worth it.)

### 1.3 Critic reset + frozen-actor warmup — the P3 breakthrough
- **The single most important 2D fix.** The transferred critic was out-of-distribution for the new
  fast-motion regime; naive fine-tuning drove it to diverge **negative** (predicting −1800 for
  states worth +1000), starving the actor of a usable gradient → the run stalled.
- **Fix:** *discard* the transferred critic, **re-initialize** it, and run a **frozen-actor
  warmup** (pure policy evaluation) until the critic is positive and calibrated, before letting it
  drive the actor. Plus **γ lowered to 0.99** (shorter, lower-variance targets → ~3× faster
  calibration, no divergence). Details in `STEP5_CRITIC_DIAGNOSIS.md`.
- Diagnostic that made this findable: a probe comparing critic `Q(s,a)` to the empirical discounted
  return-to-go on balanced states (`rl/probe_capability_2d.py`).

### 1.4 Retention (anti-forgetting) — evolved across P2→P3
- Framing (P2 Phase C): fine-tuning under distribution shift is a **forgetting-mitigation problem**
  (ICML 2024), not just TQC overestimation. Methods drawn from CLEAR (NeurIPS 2019), offline-to-
  online balanced replay (CoRL 2022), RLPD (ICML 2023).
- **`RetentionTQC`** (`rl/retention_tqc.py`): mixes a fixed **teacher dataset** of *successful*
  frozen-policy transitions into every replay batch, adds a **behavior-cloning MSE** pulling the
  actor toward teacher actions, supports **separate actor/critic LRs**, and **freezes the actor
  during critic adaptation**.
- **Make it optional:** the teacher must match the training dynamics. For the 11 V experiment the
  6-V teacher actions over-actuate at 11 V and *fought* adaptation → a `--no-teacher` mode was added.

### 1.5 Curriculum with a SOFT, ADVANCING gate
- Stage the disturbance (tilt angle/speed) from easy to hard. Advance when rolling success clears a
  **soft** threshold (~0.6–0.78) **or** a per-stage timeout that **advances** (never kills) the seed.
- This is a hard-won correction (see 2.1): the old hard-0.9 gate that *killed* stalled seeds is the
  documented failure mode. Small speed increments keep success high so the critic always has a
  mostly-succeeding regime to anchor on.

### 1.6 Verification & selection discipline (P2 Phase C, used everywhere since)
- **Two success metrics:** `is_catch_success` (reached balance at all) vs **`is_success`**
  (finished normally AND balanced ≥80% of the final 2 s). **Select on sustained success**, never on
  catch.
- **Never select from a small training-eval peak.** 100-episode peaks were *repeatedly* sampling
  noise. Verify any claimed winner with **≥500 fresh episodes**, fixed `--seed0`, **Wilson 95% CIs**.
- **Save `best_success_model` / `best_stage_N` separately from the final checkpoint** — final
  policies are often worse than their peak.
- **Paired evaluation:** always consume every DR/tilt RNG draw even when a component is disabled, so
  conditions stay seed-paired across ablations.

### 1.7 On-chip deployment pipeline (P1)
- `export_policy.py` dumps the actor MLP to a C header, **verifies NumPy inference vs SB3 to <1e-6
  on CPU** (the ESP32's float32 path), and embeds model SHA-256 provenance.
- Firmware `MODE_RL` rebuilds the identical observation from the same sensors at 200 Hz.

---

## 2. Challenges and how they were solved

### 2.1 TQC run-to-run variance + brittle curriculum gate (P1/P2)
Hard 0.7–0.9 success gates with no seed trapped stalled runs. **Fix:** set a seed, **soften the
gate + add a per-stage step timeout that advances**, run multiple seeds, keep the best.

### 2.2 Entropy collapse (P2, 2026-06-26)
Stage-0 learned then oscillated/dropped; `ent_coef` ran away to ~0.77 under **gSDE + auto
target-entropy** → policy too stochastic to balance. **Fix:** constrain entropy — `--no_sde`
(chosen default) or `--target_entropy -2`. (Not a critic-divergence issue; `critic_loss` was stable.)

### 2.3 nenv sample-efficiency trap (P2)
`gradient_steps = nenv//2`, so nenv=16 does bigger, staler update blocks per env-step → worse
*sample* efficiency than nenv=8 at the same step budget. **Use nenv=8.** (PPO's "more envs better"
intuition does NOT apply to this off-policy setup.)

### 2.4 Catastrophic forgetting during fine-tuning (P2 Phase C)
Naive off-policy fine-tuning under DR/tilt shift destabilized the pretrained behavior. **Fix:**
retention-aware fine-tuning (teacher replay + behavior-cloning + frozen-actor critic adaptation).
It solved the *forgetting symptom* (clean success stayed high) but by itself didn't *improve*
robustness — which pointed to the real bottleneck (2.6).

### 2.5 The inert retention objective (found P2, re-found P3)
The retention loss contributed **<0.1% of the actor objective** (RL actor loss ~1300 vs
`teacher_coef×teacher_loss` ~1). An adaptive-coefficient "fix" in P3 was **still** defeated by a
1e6 clamp (teacher ≈0.01% of loss). **Lesson:** verify the retention term's *magnitude*, not just
its presence. In practice a calibrated critic + tiny actor LR mattered more than the teacher term.

### 2.6 Action delay is the DR bottleneck (P2 ablation, 2026-06-29)
Component-wise DR ablation on the 91.5% master: disabling any mechanical/sensor DR barely moved
full-DR success, but **action delay alone dropped it to 48.8%**; without delay it recovered to 93%.

| Delay | success |
|---|---|
| 1 step | ~93–96% |
| 2 steps | ~37–52% |
| 3 steps | ~0–2% |

**A bounded residual/amplitude correction cannot repair an unobserved multi-step phase lag — it's
non-Markov.** Fixes: verify the *real* delay (env said 1–2, DR sampled 1–3); drop unsupported
endpoints; or expose action history / use a recurrent policy so the problem is Markov again.

### 2.7 Fast continuous pitch wall (P3)
Warm start was ~99% on level/roll/slow/static but ~10% on fast continuous pitch. Diagnosed (2.3-
style ruled out) as **critic divergence, not capability** (the policy had torque headroom at
failure). Solved by the critic reset + γ fix (1.3). The remaining ±15°-fast wall at 6 V was shown
(servo experiment, `STEP6`) to be **real physics, not a sim artifact** → scoped the target to ±10°.

### 2.8 NaN divergence at scale (P3, 2026-06-30)
4 of 5 parallel server seeds crashed with the actor Gaussian going NaN (exploding gradients, likely
from the reinit critic's large early gradients; stochastic). **Fix:** **gradient-norm clipping**
(`grad_clip=10`). Stock SB3 TQC has none; always keep it on for reinit-critic runs.

### 2.9 Motor authority may erase the wall (P3, 2026-06-30 — open)
The sim capped the motor at **6 V**, but the hardware (TMC6300) runs **~11 V** (≈1.83× torque).
Retraining at 11 V (teacher-free) **cleared the entire ±15° ladder at 0.93–0.97** where 6 V topped
out ~0.10–0.33. Strong evidence that **authority, not physics, was the limit** — and that the real
robot is stronger than the 6 V sim implied. Gated on 500-ep verification + a **hardware thermal
check** (can open-loop FOC sustain 11 V under near-continuous torque?). This echoes P1's ablation:
"motor authority matters" (KM×0.75 → 17% success).

---

## 3. Useful info / key findings

### 3.1 Plant (sysid, `sysid.json`) — shared across all projects
| Quantity | Value |
|---|---|
| Pendulum `alpha` (mlg/J) | 214 (period 0.43 s), validated 3–4× |
| Pendulum friction | Coulomb ~0.35 mN·m, viscous ζ≈0.034 |
| Arm friction | Coulomb ~6.5 mN·m, viscous damping 9.4e-4 |
| Motor `KM` | **0.0127 N·m/V**, J_arm 6.84e-5 |
| Coupling sign | `+V → +θ̇ and +φ̇` (confirmed) |
| Latency | ~1 control step (5 ms), zero jitter; step-test: dead 15 ms / τ 10 ms |
| Motor params 2× inconsistent | open-loop FOC nonlinearity → **domain-randomize widely** |

### 3.2 Hardware gotchas (P1)
- Open-loop FOC, **no current sensing** → torque ≈ voltage only approximately; **FOC offset must be
  symmetric** (bidirectional cal; a 30° offset gave 2× direction-asymmetric torque → balance
  impossible).
- **PWM ~1 kHz default**; 20 kHz cut torque ~4× with the high-side-PWM scheme.
- **AS5600 cable limits the arm to ±180°** — the original classical-control failure mode (LQR
  balances the pole but the arm winds to the limit; no integral action vs the friction offset).
- Export gotcha: gSDE actors clip mean to ±2 (`Hardtanh`) before `tanh`; replicate it or the export
  diverges only at saturation. Verify export on **CPU**, not CUDA.

### 3.3 Actuator-realism ablation (P1) — what actually breaks transfer
Ranked by damage: **ideal/instant actuator model** (adding 20 ms lag → 33%, 50 ms → 0%) and **weak
arm-envelope shaping** (arm-as-flywheel) were the top realism suspects; the step-test later showed
the *real* lag is only ~15 ms (already modeled), so those lag fears were pessimistic and swing-up
**did transfer**. Motor-gain sensitivity was the real one (see 2.9).

### 3.4 Training server (UT AERE) — shared infra
- `ssh -i ~/.ssh/aere_codex_ed25519 tn22833@aere-a83514.ae.utexas.edu` (needs UT VPN).
- **5× RTX 6000 Ada (48 GB), 256 cores, no SLURM** — used since P1. CPU-bound task (MuJoCo physics)
  → GPU util ~15%; run one seed per GPU, 5 seeds parallel.
- venv `~/furuta_rl/.venv` (sb3-contrib 2.9, torch cu124). Driver needs **cu124** torch build.
- **Access failure mode:** SSH can accept the key then drop the session (server-side account/PAM/
  quota); recovered on its own 2026-06-30. **Launch gotcha (P3):** `cd` into `rl/` before launching
  or the CWD-relative `--warmstart` path fails.
- **pkill/pgrep self-match:** patterns containing the tag match the SSH shell's own argv — kill by
  PID excluding `$$`, not `pkill -f <tag>`.

### 3.5 Selection numbers worth remembering
- P1 deployed `fix_sde/best_model.zip`: ~88% randomized-DR, deployed and working on hardware.
- P2 master `clean20_master_verified91p5.zip`: **91.5% sustained** (free-arm, clean plant); full-DR
  only ~45% (action-delay-limited).
- P3 local seed 0 (verified 500 ep): ~100% level/slow, ~34% at both ±10° 120°/s, **calibrated critic**.
- P3 11 V run (30-ep): 0.93–0.97 across the full ±15° envelope (verification pending).

---

## 4. Meta-lessons (the principles that repeatedly paid off)

1. **Transfer, don't restart.** Every project warm-started from the previous verified model.
2. **Diagnose before you scale.** The critic-divergence and action-delay findings came from cheap,
   targeted probes — not from throwing compute at the problem.
3. **Distinguish capability limits from training limits.** "Not saturated at failure" → training
   problem; "authority ablation kills it" → physics/hardware. This fork decided every next step.
4. **Trust verified numbers, not peaks.** 500 fresh episodes + Wilson CIs; select on sustained
   success and on `best_*` checkpoints, never the final one.
5. **Instrument the mechanism.** Log Q-vs-return-to-go, retention-loss magnitude, log-probs — the
   bugs (negative critic, inert teacher, entropy runaway) were invisible in success rate alone.
6. **Match the teacher/curriculum to the actual dynamics** (voltage, delay). A mismatched teacher or
   an over-aggressive stage silently sabotages the run.
7. **Preserve the working artifact.** Never overwrite a verified master; new experiments write new
   files with provenance (SHA-256).

---

## 5. Open questions / unresolved

- **11 V thermal reality:** can the open-loop-FOC gimbal motor *sustain* 11 V under near-continuous
  fast-motion torque without overheating? Sim says 11 V erases the ±15° wall; hardware must confirm.
- **Action delay on the real 2-axis rig:** the P2 bottleneck. Needs measured delay + possibly action
  history / recurrence before cable-constrained deployment.
- **Real board-motion envelope:** the 2D motion generator and BNO086 timing/noise are *modeled*.
  Replace with logged hardware BNO086 trajectories.
- **Firmware still 6-D:** deployment needs the firmware extended to the 10-D + IMU observation, with
  the sim↔hardware sign/scale checks and a fresh `<1e-6` export.
- **Cable limit for 2D:** all 2D work so far is free-arm; the ±180° cable constraint is not yet
  reintroduced.
