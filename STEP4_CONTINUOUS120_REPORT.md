# Step 4 rerun: continuous 120°/s reference motion

Date: 2026-06-30

## Protocol

- Frozen checkpoint: `rl/models/clean20_master_2d_warmstart.zip`
- SHA-256 before and after:
  `DBAA19EEC48F25EBC44C09A28E780C65749D2052318E1553BE3AB2E65A34B5AD`
- No training or parameter updates
- 500 episodes per condition
- Fresh seeds 62000–62499
- Full ±15° cap for each moving condition
- Continuous random harmonic motion with no dwell
- Reference limits: 120°/s and 1200°/s²

## Results

| Condition | Sustained | Wilson 95% CI | Catch | Falls |
|---|---:|---:|---:|---:|
| Level | 497/500 (99.4%) | 98.3–99.8% | 497/500 | 3 |
| Continuous pitch only | 49/500 (9.8%) | 7.5–12.7% | 408/500 | 451 |
| Continuous roll only | 497/500 (99.4%) | 98.3–99.8% | 497/500 | 3 |
| Continuous roll + pitch | 49/500 (9.8%) | 7.5–12.7% | 393/500 | 451 |
| Four static ±15° corners | 478/500 (95.6%) | 93.4–97.1% | 494/500 | 9 |
| Aggressive diagonal reversals | 16/500 (3.2%) | 2.0–5.1% | 349/500 | 334 |
| Continuous motion + BNO stress | 49/500 (9.8%) | 7.5–12.7% | 385/500 | 451 |

## Realized motion

The reference generator is mathematically bounded at ±15° and 120°/s. The current simulated
position-servo tracking overshoots:

- Roll-only maximum realized rate: approximately 134°/s
- Pitch-only maximum realized rate: approximately 140°/s
- Maximum realized board angle: approximately 15.18°

These realized values, rather than the reference values, are what the BNO086 model reports to the
policy. Hardware measurements are still required to replace this simulated tracking response.

## Diagnosis

The new failure is specifically fast continuous pitch. Roll-only performance remains equal to the
level baseline even though the transferred roll and gyro-X input weights are zero. Simultaneous
performance matches pitch-only performance, so pitch is the limiting axis.

The high catch rates and low sustained rates show that the policy can usually swing up but cannot
maintain balance under continuous high-rate pitch excitation.

## Decision

- Preserve the frozen warm-start checkpoint.
- Do not broadly retrain level, roll, or static behavior.
- If 120°/s reference motion is required, train specifically on continuous pitch and compound
  motion using staged angle and speed curricula.
- Require level, roll-only, and static-corner retention checks before accepting any trained model.

Per-episode evidence is stored under `eval/step4_continuous120/`.
