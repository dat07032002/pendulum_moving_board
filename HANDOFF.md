# HANDOFF — Furuta pendulum on a randomly-TILTING base (±30°)

Last updated 2026-06-26. This is the full state + context so another agent can continue.
The step-plan lives at `~/.claude/plans/async-beaming-teapot.md` (the "ACTIVE PLAN" section at top).

This is **project #2**, built on top of a completed project #1. Read both sections below.

---

## 0. TL;DR / current state

**Goal:** mount the working GM3506 Furuta rig on a board that **tilts ±30° about one axis** (driven
by a Hiwonder **LX-16A** servo making *random* tilts), and have the pendulum **keep balancing
upright (true gravity-vertical)** through the motion. Board tilt `β`/`β̇` are measured by a **BNO086
IMU** on the board. One ESP32 runs everything. Deploy on-chip like project #1.

**Where we are:**
- ✅ **Phase 0** (sim feasibility) done — ±30° is physically feasible; orientation-dependent.
- ✅ **Phase 1 CODE** done — 8-D tilt env, true-vertical reward, tilt curriculum, obs-agnostic
  exporter. All validated locally.
- ✅ **Phase 1 TRAIN config — FIXED & VALIDATED (2026-06-26).** The stage-0 stall was an **entropy
  collapse** (`ent_coef` ran away to ~0.77 under gSDE + auto target-entropy). The diag sweep (300k,
  nenv=8, seed 0) resolved it: **`--no_sde` crosses 0.6 @130k and holds 1.00 (240k→300k)**; the
  `--target_entropy -2` variant (gSDE on) also holds 1.00 (crosses 0.6 @100k). **Winner locked as the
  `train_tqc.py` default (gSDE now OFF by default; `--sde` re-enables for the contrast seed).** Ready
  to launch on the UT server. Logs: `rl/A_nosde.log`, `rl/B_targent.log`.
- ⬜ **Phases 2–5** (hardware: LX-16A + BNO086 wiring/firmware, deploy, iterate) — not started.

**▶ IMMEDIATE NEXT ACTION (START HERE tomorrow):**
1. **Finish the entropy validation sweep** (was interrupted). Run the stage-0 diag to ~300k on the
   two candidates and pick the one that crosses 0.6 and HOLDS (vs the baseline that collapses):
   - `cd ~/.../tilt_pendulum/rl`
   - `python diag_stage0.py --no_sde --steps 300000 --nenv 8 --seed 0 > A.log 2>&1`  (gSDE off)
   - `python diag_stage0.py --target_entropy -2 --steps 300000 --nenv 8 --seed 0 > B.log 2>&1`
   - (NOTE: don't pipe through `tail`/`grep` to a file — it block-buffers; redirect raw so you can
     read progress. Each ~17 min on a 5070.)
2. **Lock the winning config** as the `train_tqc.py` default (it already has `--no_sde / --ent_coef /
   --target_entropy` knobs and the eval-at-±30°-tilt fix).
3. **Launch on the UT server** (`~/furuta_tilt/`, reuse `~/furuta_rl/.venv`): **3 seeds with the
   winner + 1 seed with the gSDE-variant contrast** (user asked for an sde A/B), `--nenv 8
   --steps 8000000 --eval_tilt_deg 30 --seed {0,1,2}`, GPUs 0–3. Example:
   `CUDA_VISIBLE_DEVICES=0 nohup ~/furuta_rl/.venv/bin/python rl/train_tqc.py --no_sde --nenv 8
   --steps 8000000 --seed 0 --tag tilt_s0 > train_tilt_s0.log 2>&1 &`  (drop `--no_sde` on the
   contrast seed). nohup = survives logout.
4. Then monitor → **keep best `best_model.zip` across seeds** (now eval'd under ±30° tilt) → judge
   vs `rl/POLICY_RUBRIC.md` (Tilt additions) → Phase 2 hardware.

Project #1 peaked at only **~0.7 M steps** (5 stages); expect this to peak ~1–3 M (8 stages). 8 M is
a ceiling, not a target — `best_model` captures the peak; stop early once seeds plateau.

---

## 1. Project #1 (DONE, deployed) — context you inherit

A self-balancing Furuta pendulum on a **GM3506 gimbal motor**, level ground. Classical LQR failed
on hardware (pendulum-pivot friction → arm winds to the ±180° cable limit). We trained a **single
TQC policy** (MuJoCo + domain randomization + curriculum) for swing-up + balance and **deployed it
on-chip** (ESP32 `MODE_RL`, boot auto-start + auto-recovery). It works on hardware.

- **That project lives in a SEPARATE folder** `c:/Users/thanh/Desktop/LQR_pendulum` (branch `foc`,
  pushed to `github.com/dat07032002/lqr_pendulum`). **Do not modify it.** This `tilt_pendulum`
  folder is an independent copy (fresh git repo) seeded from it.
- Hardware (shared with this project): ESP32 @ COM5/921600, 200 Hz loop; **TMC6300** open-loop FOC
  (no current sense, ~11 V); **AS5048A** arm encoder (SPI, 14-bit); **AS5600** pendulum encoder
  (I²C 0x36, 12-bit); arm limited to **±180°** by the AS5600 cable.
- Measured plant params in `sysid.json`: `alpha=214` (pole), `KM≈0.0127`, `J_arm≈6.84e-5`,
  arm damping `9.4e-4`/friction `~6e-3`, pole damping `5.06e-5`/friction `0.35e-3`; `+V→+θ̇,+φ̇`.
- Firmware (`firmware/furuta_foc/furuta_foc.ino`) already has: `MODE_RL` on-chip 2-layer MLP, the
  `rl` command, boot auto-start (~4 s), auto-recovery (unwind+retry on cable hit), NVS-persisted
  FOC/AS5600 cal, `log`/`nolog` 200 Hz stream. **This is the level-ground (6-D obs) firmware** — it
  must be extended to 8-D + IMU for this project (Phase 3).
- **Key lesson (post-mortem) baked into this project:** a 2nd training attempt (v2) failed not from
  the reward shaping but from **TQC run-to-run variance + a brittle hard-0.7 curriculum gate that
  trapped a stalled run** (no seed was set). Fixes carried here: **set a seed, soften the gate to
  0.6 + a per-stage step-timeout, run multiple seeds and keep the best.**

---

## 2. Project #2 (THIS one) — the tilting base

### 2.1 Why the IMU is needed
The AS5600 measures the pole **relative to the (now-tilting) base**, so it can't see gravity. If the
controller balances base-frame "up", under tilt that's *not* true vertical → it holds the pole off
the zero-torque point → needs continuous torque → **winds the arm** (the original failure mode). So
the controller must know the board tilt `β` to locate true vertical. → **BNO086 IMU on the board.**

### 2.2 Hardware decisions (confirmed with user)
- **Tilt actuator:** Hiwonder **LX-16A** serial bus servo (single-wire half-duplex UART @ 115200,
  ~17 kg·cm, 6–8.4 V; lib `madhephaestus/lx16a-servo`). **Actuator only** — drives the random ±30°
  tilt; its pot readback is NOT used (so its noise/backlash don't matter for sensing).
- **Tilt sensor:** **SparkFun BNO086** IMU (BNO08x family, on-chip fusion). Mounted on the board,
  one axis = `β`. **Read over I²C @ 200 Hz** (Game Rotation Vector / Gravity — **mag-free**, because
  the motors disturb the magnetometer). `β̇` from the gyro. Default I²C addr 0x4A/0x4B (coexists with
  the AS5600 @ 0x36; use the 2nd I²C controller if the bus gets tight). Zero `β` at level on boot.
  Adafruit **BNO085** is the drop-in alternative.
- **One ESP32** runs GM3506 FOC + RL + servo command + IMU read.
- **Keep the ±180° arm cable limit** (deployed-v1 behavior: arm-centering + soft limit + guard).

### 2.3 Observation (8-D) — must match sim ↔ firmware exactly
```
[cosθ, sinθ, θ̇/15, clip(φ/π,±2), φ̇/25, prev_action, β/0.6, β̇/3]
```
- `θ` = pole from AS5600 (base/board frame — sensor-faithful), `φ` = arm (AS5048A).
- `β`,`β̇` = board tilt vs gravity (BNO086). The policy infers true-vertical from `θ,φ,β`.
- Action ∈[-1,1] → ±6 V on the GM3506 (unchanged). Normalizers in `furuta_env.py` (TH_SCALE=15,
  PHI_SCALE=25, BETA_SCALE=0.6, BETADOT_SCALE=3).

### 2.4 Reward (per 200 Hz step) — proven v1 reward, retargeted to TRUE vertical
```
up = _true_up()                         # cos of pole angle from GRAVITY vertical (geometric)
r  = up                                 # main: be upright vs gravity (swing-up + balance)
   - 0.20*(φ/π)²                        # arm-centering (avoid cable wind)
   - 0.005*a² - 0.002*φ̇²               # control effort / arm speed (small)
   - 0.02*(a-prev_a)²                   # CAPS action smoothness (transfers to real motor)
if up>0.5:  r -= 0.01*θ̇²               # settle: damp pole ONLY near the top (pumping stays free)
if up>0.92 and |θ̇|<3 and |φ|<90°: r += 2.0   # bonus: genuinely balanced AND arm bounded
# terminate: |φ|>180° → r-=10 (cable); once up, if up<0 → terminate (anti reward-farm); 10 s limit
# success (curriculum/eval): up>0.9 & |θ̇|<4 & |φ|<90° held >0.5 s
```
`arm_envelope_w=0` (the v2 arm-envelope was exonerated + unnecessary — left as a knob, off). The
**key change vs project #1**: `up` is geometric **true-vertical** (`_true_up()` from the pole body's
world orientation), not base-frame `cos(θ)`.

### 2.5 Domain randomization (per episode; ON from curriculum stage 1)
Plant: KM[0.008–0.020], arm damping[3e-4–10e-4], pole damping[2e-5–1e-4], arm friction[4e-3–8e-3],
pole friction[0.2e-3–0.6e-3], pole inertia ±8%, obs noise[0–0.01], action delay 1–3 steps.
**Corner-weighted:** `p_corner=0.3` → 30% of draws pushed to a min/max extreme (worst-case coverage).
Tilt/IMU: tilt amplitude 30–100% of the stage cap, tilt rate β̇ 0.5–2.0 rad/s, IMU β-noise ±0.005 rad,
IMU rate fixed 200 Hz (`IMU_DECIM=1`).

### 2.6 Curriculum (8 stages) — `train_tqc.py` STAGES
0–4 learn the full task on **level** ground (balance ±10° → ±45° → ±90° → near-hanging+assist →
full swing-up), then 5–7 ramp tilt in (**±10° → ±20° → ±30°**). Advance when rolling success
(last 60 eps) **>0.6**, OR after a **700 k-step per-stage timeout** (anti-trap). Reward identical
across stages. DR off in stage 0 only.

---

## 3. Files (all under `tilt_pendulum/`)

| File | Role |
|---|---|
| `rl/furuta.xml` | MuJoCo model: **board (tilt) hinge about y** at the stand base + stiff position actuator tracking β_ref; floor/bearings/**yellow tilt-axis marker**/tan platform (visuals). Furuta params from sysid. |
| `rl/tilt.py` | bounded tilt generator: `triangle` (sweep) + `random` (training), `β̇` rate-capped. Shared sim↔firmware. |
| `rl/furuta_env.py` | 8-D Gym env: drives tilt each step, models BNO086 read (200 Hz + noise), `_true_up()` reward, tilt DR + curriculum knobs (`tilt_amp`, `tilt_betadot_max`). |
| `rl/train_tqc.py` | TQC training: 8-stage curriculum, soft-0.6 gate + timeout, `--seed`, checkpointing, EvalCallback→`best_model.zip`. |
| `rl/export_policy.py` | actor → `policy_weights.h` (**auto-detects obs dim**; verifies vs SB3 <1e-6; replicates gSDE `clip_mean`). |
| `rl/feasibility_tilt.py` | Phase-0 hand-LQR feasibility sweep (true-vertical FF, tilt rate × arm-orientation). |
| `rl/view_tilt.py` | interactive viewer (LQR balancing or `--nopolicy` raw physics; `--phi0/--theta0/--betadot/--mode`). |
| `rl/POLICY_RUBRIC.md` | acceptance rubric + **Tilt-project additions** (Pass 1-T/3-T/4-T/5-T). |
| `firmware/furuta_foc/` | project-#1 firmware (6-D `MODE_RL`). **Phase 3 must extend to 8-D + IMU.** |
| `sysid.json`, `config.py`, `pc_balance.py`, `step_response.py`, ID tools | inherited from project #1. |

---

## 4. Training run — UT server  (STOPPED 2026-06-26; resume per §0 — fix entropy, then relaunch)

- `ssh -i ~/.ssh/aere_codex_ed25519 tn22833@aere-a83514.ae.utexas.edu` (needs **UT VPN**).
- Project dir **`~/furuta_tilt/`** (code in `rl/`). **Reuses `~/furuta_rl/.venv`** (torch cu124 +
  sb3-contrib + mujoco) — do NOT touch `~/furuta_rl/` or `~/pendulum/`.
- 3 runs: `tilt_s0/s1/s2` (GPU 0/1/2), **`--nenv 8`** `--steps 8000000 --seed {0,1,2} --tag tilt_s{n}`,
  logs `train_tilt_s{n}.log`, models `rl/models/tilt_s{n}/best_model.zip` (+ `ckpt_*`).
- **NOTE (2026-06-26): first launched at nenv=16 → stage-0 was slow (0.28 @ 400k).** Investigation:
  a stage-0 bisect showed the tilt env at **nenv=8 matches project #1 (0.41 vs 0.43 @ 80k)**, board
  wobble negligible (0.012°) → the env is fine; **nenv=16 is just less sample-efficient** (matches the
  v2 post-mortem). Relaunched at **nenv=8** (old nenv16 logs → `*_nenv16.log`). Use nenv=8.
- Monitor: `grep -E 'curriculum|success_rate|ep_rew_mean' train_tilt_s0.log | tail`.
- Launch cmd (for reference / relaunch):
  `cd ~/furuta_tilt && CUDA_VISIBLE_DEVICES=0 nohup ~/furuta_rl/.venv/bin/python rl/train_tqc.py
  --steps 8000000 --nenv 16 --seed 0 --tag tilt_s0 > train_tilt_s0.log 2>&1 &`

**Selection:** pull each seed's `best_model.zip`, evaluate under random ±30° tilt + plant DR
(deterministic), **keep the best**, judge vs `POLICY_RUBRIC.md` (Tilt additions). Watch the
curriculum reach stage 7 and stage-7 success climb >0.6.

---

## 5. Phase-0 feasibility findings (sim, hand-LQR, no policy)

`python rl/feasibility_tilt.py`. ±30° tilt is **physically feasible on ±6 V**, but **orientation-
dependent**: at arm φ≈0 the swing plane is ⊥ the tilt axis → tilt barely disturbs the pole (pole
geometrically capped ~30° from vertical there, but stable); at **φ≈90° the tilt maximally drives the
pole** (the hard case). The linear LQR holds all orientations *moving* to ~2 rad/s and is marginal
on a sustained 30° hold at φ=90° (a linear-controller limit, not an authority wall). → **set
β̇_max≈2 rad/s** for training; RL (nonlinear + free to orient the arm) should match/exceed it.
**Tip:** physically orienting the rig so the arm's rest/center is near the benign φ=0 makes balance
easier, but the policy must still handle transits through φ=90°.

---

## 6. Phases 2–5 (hardware) — not started

- **Phase 2 — tilt subsystem:** mount rig on the board; LX-16A drives random ±30° (port `tilt.py`,
  cap β̇_max). Mount BNO086, read `β`/`β̇` over I²C @ 200 Hz (mag-free). Gate: smooth tilt; IMU β
  matches a protractor, low-noise.
- **Phase 3 — obs integration:** extend firmware `MODE_RL` to **8-D** (+β,β̇), keep the ±160° arm
  guard + auto-recovery. **Sign/scale check** at PC-in-loop: `+β` firmware = `+β` sim (β norm 0.6),
  plus the inherited `sinθ`/`θ̇`/action sign checks. Re-export `policy_weights.h` (8-D, verify <1e-6).
- **Phase 4 — deploy & staged test:** static tilt ±10/20/30° → slow random → full random ±30°.
- **Phase 5 — iterate:** log real `β/θ` + response, re-tune sim/DR, retrain.

---

## 7. Decisions / gotchas / open items

- **IMU read = mag-free fusion** (Game Rotation Vector / Gravity), NOT the mag-based Rotation Vector
  (motors disturb the magnetometer).
- **IMU @ 200 Hz over I²C** is assumed (`IMU_DECIM=1`). If hardware can't sustain it, fall back to
  UART-RVC 100 Hz and retrain with `IMU_DECIM=2` (env knob).
- **Servo sag does NOT corrupt β** — the IMU reads the *actual* board angle; sag only makes the tilt
  motion noisier/laggier.
- **USE `nenv=8`, NOT 16 (sample efficiency).** Confirmed twice (v2 post-mortem + the 2026-06-26
  stage-0 bisect): at the same *step* budget, nenv=8 learns markedly faster/more stably than nenv=16
  here. Why: `train_tqc.py` sets `gradient_steps = max(4, nenv//2)`, so nenv=16 does **8 gradient
  updates in a row on the same (staler) buffer snapshot** before refreshing data, vs 4 at nenv=8 —
  same updates/sample ratio (0.5) but bigger, staler update blocks → less effective *per env-step*,
  and learning is diluted over more steps per improvement cycle. nenv=16 is faster *wall-clock* data
  collection but worse *sample efficiency* — wrong trade for this small task. (Off-policy quirk: the
  PPO intuition "more envs = better" does NOT apply because `gradient_steps` scales with nenv. A
  cleaner future fix is to **decouple** them — fix `gradient_steps` independent of nenv.) Evidence:
  tilt env stage-0 reached 0.41@80k at nenv=8 (== project #1's 0.43) but only 0.28@400k at nenv=16.
- **ENTROPY COLLAPSE (2026-06-26 — RESOLVED).** At nenv=8 the seeds learned stage-0 to ~0.4–0.55 by
  ~160–200k then **oscillated/dropped** (s1 0.55→0.18) and never crossed the 0.6 gate. Root cause =
  **`ent_coef` ran away to ~0.77** under **gSDE + auto target-entropy** → policy too stochastic →
  noisy balance. NOT critic divergence (`critic_loss` stable ~4.7). **Fix:** constrain entropy by
  either route — `--no_sde` (chosen default) OR `--target_entropy -2` (gSDE on). Diag sweep proof:
  `--no_sde` crosses 0.6 @130k → 1.00 @240k, held to 300k; `--target_entropy -2` crosses 0.6 @100k →
  1.00 @240k, held. **`train_tqc.py` default is now gSDE-OFF** (`use_sde=args.use_sde`, default False;
  `--sde` re-enables for the contrast seed). Note: gSDE off is harmless for deployment — the exported
  policy is the deterministic mean action and CAPS already handles action smoothness.
- **Adaptive quantile dropping** was considered and **deferred** — not our bottleneck; keep fixed
  `top_quantiles_to_drop_per_net=2`. If overestimation instability appears, sweep the fixed value
  (3/5) before anything custom.
- **Tilt axis vs pole hinge geometry:** y-tilt × pole-hinge gives the φ-dependent coupling
  (`~cos`/`sin(φ−φ_axis)`); this is why `φ` (via `φ/π`) is in the obs — the policy needs arm
  orientation to know how the tilt projects onto the pole. Verified in the viewer.
- **Open:** confirm the LX-16A torque holds the actual board+rig without sag; confirm BNO086 200 Hz
  on the shared I²C bus; one-ESP32 200 Hz timing budget (FOC + servo + IMU + MLP) — split to the 2nd
  core if needed.

---

## 8. Git

Fresh repo in `tilt_pendulum/`. Commits: `9a234a2` seed · `d298041` Phase 0 · `6531f7e` viewer/base ·
`c02d5bc` Phase 1 env+train · `965c740` IMU 200 Hz. **No remote yet** (local only; `git push` not
run — ask the user before pushing).
