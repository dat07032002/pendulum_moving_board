# Seed 0 Stage 1 diagnosis

Date: 2026-06-30

## Training outcome

- Run: `tilt2d_cont_local_s0`
- Stage 0, pitch ±5° at 40–60°/s: passed with two 100% target evaluations
- Stage 1, pitch ±10° at 60–90°/s: plateaued at 73–80%
- Automatic stop: 300k total steps after Stage 1 timeout
- No retention guard fired

## Paired independent Stage 1 verification

Condition: pitch only, full ±10°, 90°/s reference, 500 identical fresh seeds 90000–90499.

| Model | Sustained | Catch | Mean return |
|---|---:|---:|---:|
| Frozen warm-start | 417/500 (83.4%) | 465/500 (93.0%) | 4768.9 |
| Best Stage 1 | 428/500 (85.6%) | 465/500 (93.0%) | 4807.8 |

The trained checkpoint improved sustained success by 2.2 percentage points but did not improve
catch success.

## Independent retention and harder-target verification

Best Stage 1 checkpoint:

| Condition | Sustained | Catch |
|---|---:|---:|
| Level, 200 episodes | 99.5% | 99.5% |
| Continuous roll ±15° at 120°/s, 200 episodes | 99.5% | 99.5% |
| Slower simultaneous ±15° at 60°/s, 200 episodes | 99.0% | 99.5% |
| Four static corners, 200 episodes | 95.0% | 99.0% |
| Continuous pitch ±15° at 120°/s, 200 episodes | 14.0% | 80.0% |

Retention remained strong. The checkpoint is safe but does not solve fast pitch.

## Actor-change evidence

Compared with the frozen warm-start on 10,000 teacher observations:

- Mean absolute action change: `0.000193`
- 95th-percentile action change: `0.000392`
- Maximum action change: `0.0167`
- First-layer total weight-change norm: `0.00248`
- New roll-column norm: `0.000553`
- New gyro-X-column norm: `0.000438`

The actor remained almost identical to the warm-start. The training run therefore under-adapted
rather than forgetting.

## Why the training-buffer success looked better

The rollout buffer reported approximately 94% success because only 50% of training episodes used
the current hard stage. The other half were easier level, roll, slow-combined, and corner retention
profiles. Curriculum decisions correctly used a separate target-only evaluator, which remained
below the 90% gate.

## Diagnosis

1. Stage 1 is already difficult but not far from solvable: the warm-start scores 83.4%.
2. Training produced only a 2.2-point sustained improvement.
3. Retention is not the bottleneck; all retention conditions remain 95–99.5%.
4. The actor learning rate of `1e-5` is too conservative for this distribution shift.
5. The critic received substantial adaptation, but the actor barely moved.
6. Fast pitch remains the specific unresolved mode; roll is not limiting.

## Recommended next experiment

Continue from the Stage 1/full-optimizer checkpoint rather than restarting:

- Start directly at Stage 1
- Actor LR: increase from `1e-5` to `3e-5`
- Critic LR: keep `1e-4`
- Actor begins immediately because the critic has already adapted
- Increase current-stage episode share from 50% to 70%
- Keep 10% level, 10% roll, 5% slow-combined, 5% corners
- Retain the same independent safety gates
- Use a 150k Stage 1 ceiling
- Stop if level or roll drops below 95%

This tests whether more actor authority solves the plateau without committing to a long run.
