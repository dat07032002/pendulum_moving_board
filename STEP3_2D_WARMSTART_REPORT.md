# Step 3: 1D-to-2D warm start

Date: 2026-06-30

## Source

- Checkpoint: `rl/models/clean20_master_verified91p5.zip`
- Source observation size: 8
- Verified source task: free-arm, clean plant, one-axis Y/pitch tilt

## Output

- Checkpoint: `rl/models/clean20_master_2d_warmstart.zip`
- Output observation size: 10
- SHA-256: `DBAA19EEC48F25EBC44C09A28E780C65749D2052318E1553BE3AB2E65A34B5AD`

## Input mapping

The original board joint rotates about Y, so its tilt channels map to pitch:

- Original inputs 0–5 map unchanged.
- Original tilt angle maps to new pitch (input 7).
- Original tilt rate maps to new gyro-Y (input 9).
- New roll (input 6) and gyro-X (input 8) weights initialize to exactly zero.
- Pitch weights are rescaled to compensate for the changed angle normalization.
- Gyro-Y weights are rescaled to compensate for the changed rate normalization.
- The actor, both critics, and target critics use the same mapping.

## Numerical equivalence

Tested on 10,000 randomized physically equivalent observations:

- Maximum deterministic action difference: `2.146e-06`
- Maximum critic difference: `4.883e-04`
- The same results passed after saving and reloading the output checkpoint.
- Differences are float32 rounding from the normalization rescaling.

## No-training baseline

Each condition used 100 full 10-second episodes, seeds 50000–50099, free arm,
clean plant, deterministic policy inference, BNO086 observations, and no parameter updates.

| Board condition | Sustained success | Catch success |
|---|---:|---:|
| Pitch only, ±15° | 100/100 | 100/100 |
| Level, 0° | 100/100 | 100/100 |
| Roll + pitch, ±5° | 100/100 | 100/100 |
| Roll + pitch, ±10° | 100/100 | 100/100 |
| Roll + pitch, ±15° | 100/100 | 100/100 |

This is a baseline verification, not a statistically final model selection. Roll weights are
still zero; the policy succeeds through its existing pole/arm feedback and pitch response.
