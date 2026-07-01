# Step 4: frozen 2D policy stress test

Date: 2026-06-30

## Protocol

- Checkpoint: `rl/models/clean20_master_2d_warmstart.zip`
- Checkpoint SHA-256 before and after evaluation:
  `DBAA19EEC48F25EBC44C09A28E780C65749D2052318E1553BE3AB2E65A34B5AD`
- No training or parameter updates
- 500 episodes per condition
- Seeds 60000–60499
- Ten seconds per non-terminated episode
- Free-arm, clean Furuta plant
- Full ±15° board-amplitude cap for every non-level condition
- BNO086 observation model enabled

## Results

| Condition | Sustained | Wilson 95% CI | Catch | Falls | Mean upper RMS |
|---|---:|---:|---:|---:|---:|
| Level | 497/500 (99.4%) | 98.3–99.8% | 497/500 | 3 | 2.32° |
| Pitch only | 497/500 (99.4%) | 98.3–99.8% | 497/500 | 3 | 8.44° |
| Roll only | 496/500 (99.2%) | 98.0–99.7% | 496/500 | 4 | 8.34° |
| Random roll + pitch | 497/500 (99.4%) | 98.3–99.8% | 498/500 | 3 | 11.52° |
| Four static ±15° corners | 484/500 (96.8%) | 94.9–98.0% | 497/500 | 4 | 17.15° |
| Conservative BNO-error stress | 496/500 (99.2%) | 98.0–99.7% | 496/500 | 4 | 11.58° |
| Aggressive diagonal reversals | 8/500 (1.6%) | 0.8–3.1% | 337/500 | 344 | 17.36° |

Mean action saturation stayed below 0.33% in every normal/corner/BNO condition. It rose to
1.85% in the aggressive reversal condition.

## Interpretation

The transferred policy already handles normal two-axis ±15° motion, including roll despite its
new roll input weights being zero. Its pole/arm feedback generalizes to that disturbance.

Static compound corners are harder but remain strong at 96.8% sustained success. The difference
between 99.4% catch and 96.8% sustained success shows that several episodes catch the pole and
then lose stable occupancy near the end.

The aggressive reversal reference is bounded at ±15°, 80°/s, and 300°/s². The current simulated
position-servo dynamics overshoot the reference to approximately 87°/s and introduce sharper
acceleration transients. This condition is therefore an intentionally conservative, beyond-nominal
stress test until real BNO086 trajectories establish the hardware envelope. It identifies a useful
robustness target, but it is not evidence that the policy fails within a measured physical limit.

## Decision

- Preserve the frozen warm-start checkpoint as the current best model.
- Do not broadly retrain the already strong level/random-motion behavior.
- If additional robustness is required before hardware data is available, fine-tune against corner
  holds and progressively faster reversals while enforcing frozen-policy retention evaluations.
- Replace assumed motion limits with logged BNO086 trajectories when hardware is available.

Per-episode evidence is stored under `eval/step4/`.
