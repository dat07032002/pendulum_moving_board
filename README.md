# Furuta pendulum on a two-axis moving board — RL

Keep a Furuta (rotary inverted) pendulum balanced **upright to true gravity-vertical** while a
**two-axis board tilts it in roll and pitch**, using reinforcement learning (TQC). This is the
third project in a lineage: level ground → one-axis tilt → **two-axis tilt (this repo)**.

**Read these first** for the full story and method rationale:
- `PROJECT_REPORT.pdf` — 2-page summary (methods + result figures).
- `PROJECT_LINEAGE_METHODS_AND_LESSONS.md` — the complete method history, challenges, and lessons.
- `STEP5_CRITIC_DIAGNOSIS.md`, `STEP6_SERVO_ENVELOPE_AND_PRODUCTION.md` — key diagnoses.

---

## Current best model

- **`rl/models/v10_best_s3.zip`** — the deployment model. Trained at the 10 V motor limit;
  **verified ~96% sustained success at the hardest condition** (both-axis ±15°, 120°/s) and
  96–99% at all lower speeds, over 500 fresh episodes/condition.
- `rl/models/v11_reference.zip` — an 11 V variant (max motor voltage); slightly higher ceiling,
  kept for reference.
- `rl/models/clean20_master_2d_warmstart.zip` — the **warm start** every 2D run begins from
  (a 1D→2D weight transfer of the verified one-axis master).
- `rl/models/clean20_master_verified91p5.zip` — the source 1D master (for regenerating the warm start).
- `rl/teacher_2d_retention_100k.npz` — teacher dataset used by the retention path (6 V only).

> Motor voltage is set with the `FURUTA_VMAX` environment variable (6 = sim default, 10/11 =
> extended authority). **It must match between training and evaluation.** The best model was
> trained at `FURUTA_VMAX=10`.

---

## Setup (Python 3.12)

```bash
python -m venv .venv
source .venv/bin/activate            # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
# Install a platform-appropriate PyTorch separately, e.g. CUDA 12.4:
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

Quick sanity checks:
```bash
python -m py_compile rl/*.py
cd rl && python furuta_env_2d.py         # "Furuta2DEnv sanity check passed"
```

---

## Run training (the validated recipe)

The method (see the report): keep the warm-start **actor**, **re-initialize the critic**, do a
**frozen-actor critic warm-up**, use **γ=0.99** and **gradient clipping**, climb a **gentle ±15°
speed curriculum** with a **soft advancing gate**, and train **teacher-free at higher voltage**.

Launch **5 seeds** (one per GPU) at 10 V, from the `rl/` directory:
```bash
cd rl
for s in 0 1 2 3 4; do
  FURUTA_VMAX=10 CUDA_VISIBLE_DEVICES=$s python train_phaseb_2d.py \
    --tag v10_pm15_nt_s$s --seed $s --steps 900000 \
    --gamma 0.99 --warmup-steps 50000 --actor-lr 3e-5 \
    --ladder pm15 --no-teacher > train_v10_s$s.log 2>&1 &
done
```

Flags:
- `--ladder pm15` targets the ±15° envelope; `pm10` is the conservative ±10° envelope.
- `--no-teacher` for higher voltage (the 6 V teacher over-actuates at 10/11 V). At 6 V, **drop**
  `--no-teacher` and pass `--teacher-data teacher_2d_retention_100k.npz`.
- Each run writes `best_stage_N.zip` and `final_model.zip` under `rl/models/<tag>/`, and logs its
  per-stage success + critic calibration.

A seed "finishes" (`done`) when it soft-passes the final curriculum stage.

---

## Evaluate / verify

500-episode verification across the speed envelope (**match the training voltage**):
```bash
cd rl
FURUTA_VMAX=10 python verify_2d.py models/v10_best_s3.zip -n 500 --grid pm15 --out ../eval/verify_best
```
Other diagnostics:
- `probe_capability_2d.py <model>` — critic calibration (Q vs return-to-go) + action-saturation.
- `eval_dr_2d.py <model>` — robustness under plant domain randomization (isolates action delay).
- `stress_test_2d.py` / `eval_policy_2d.py` — additional stress and single-condition evals.

---

## Reproduce the figures and report

```bash
cd rl
FURUTA_VMAX=10 python make_plots_2d.py \
  --model models/v10_best_s3.zip --json ../eval/verify_best.json \
  --seed-jsons ../eval/verify_v10_s0.json ../eval/verify_v10_s1.json ../eval/verify_v10_s2.json \
               ../eval/verify_v10_s3.json ../eval/verify_v10_s4.json \
  --out ../figure_10V
cd .. && python make_report_pdf.py       # -> PROJECT_REPORT.pdf  (needs fpdf2 + matplotlib)
```

---

## Key files

| File | Role |
|---|---|
| `rl/furuta_2d.xml` | MuJoCo model: two-axis roll/pitch gimbal carrying the Furuta rig |
| `rl/furuta_env_2d.py` | 10-D Gym env (pole/arm state + BNO086 IMU roll/pitch/gyro); `FURUTA_VMAX` sets motor voltage |
| `rl/tilt_2d.py` | continuous + corner board-motion generators |
| `rl/bno086.py` | BNO086 IMU measurement model (latency, noise, sample-and-hold) |
| `rl/train_phaseb_2d.py` | **training entry point** (the validated recipe + curriculum) |
| `rl/retention_tqc.py` | `RetentionTQC`: critic reset, frozen-actor warm-up, separate LRs, teacher, grad clip |
| `rl/critic_warmup_2d.py` | frozen-actor critic-calibration diagnostic |
| `rl/verify_2d.py` | 500-episode verification harness (Wilson CIs, calibration gate) |
| `rl/probe_capability_2d.py`, `rl/eval_dr_2d.py` | capability / DR-robustness probes |
| `rl/transfer_1d_to_2d.py` | build the 10-D warm start from the 1D master |
| `rl/collect_teacher_data_2d.py` | build the retention teacher dataset |
| `rl/make_plots_2d.py`, `make_report_pdf.py` | figures + 2-page report |
| `rl/furuta_env.py`, `rl/tilt.py` | inherited 1D env/generator (base classes) |
| `firmware/` | ESP32 firmware (currently the level-ground 6-D policy; extend to 10-D + IMU for hardware) |
| `cad/` | CAD (IMU + motor mounts) |

Regenerate the warm start from the 1D master (optional):
```bash
cd rl
python transfer_1d_to_2d.py \
  --source models/clean20_master_verified91p5.zip \
  --output models/clean20_master_2d_warmstart.zip
```

---

## Not in the repo (regenerate with the scripts)

To keep the repo lean, the bulk **training-output run folders** (`rl/models/<run>/`, ~590 MB) and
most GIFs are gitignored. Everything needed to run training and verify results is included: the
warm start, teacher data, the board model, and the best/reference checkpoints. Re-run training or
`render_policy_gif_2d.py` to regenerate outputs.

## Server (optional)

Training is normally run on a shared UT AERE server (5× RTX 6000 Ada). Sync this repo to
`~/furuta_tilt_2d`, `cd rl`, and launch as above (use its venv or a fresh one). See
`PROJECT_LINEAGE_METHODS_AND_LESSONS.md` §3.4 for details and gotchas.
```
