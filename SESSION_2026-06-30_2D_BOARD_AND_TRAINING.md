# Two-axis board, BNO086, warm start, and continuous-motion training

Date: 2026-06-30

This is the consolidated record for the new two-axis project. It supplements
`SESSION_2026-06-29_CLEAN91_RESIDUAL_DR.md` and supersedes informal intermediate plans from
the development thread.

## 1. Project separation and preserved baseline

The validated one-axis project remains preserved at:

`C:\Users\thanh\Desktop\tilt_pendulum`

Two-axis development is isolated at:

`C:\Users\thanh\Desktop\tilt_pendulum_2d`

The preserved one-axis master is:

- File: `rl/models/clean20_master_verified91p5.zip`
- SHA-256: `775AFBB5CF1553BECB347C422EDC6C03300990134D45C7D2B567F7A9DB849D3A`
- Task: clean plant, free arm, random one-axis pitch up to ±20°
- Independently verified sustained success: 91.5% over 1,000 episodes

No two-axis work overwrote this checkpoint or the original one-axis project.

## 2. Intended physical system

The Furuta pendulum will be mounted on a two-axis moving board, potentially the BRIO/CyberRunner
labyrinth. The balancing policy is intended to adapt to different board actuators rather than model
one specific servo.

The board drive is therefore treated as an external disturbance. The policy does not observe servo
commands. It observes the board's realized orientation and angular velocity through a board-mounted
SparkFun BNO086.

The CyberRunner board controller and Furuta balancing controller remain conceptually separate:

- CyberRunner actuators move the board to guide the marble.
- The Furuta motor balances the pendulum.
- The balancing policy reacts to the board motion measured by the IMU.

## 3. MuJoCo two-axis board

Primary model:

`rl/furuta_2d.xml`

The mechanism uses a nested gimbal:

1. Fixed base plate
2. Two support towers and roll bearings
3. Outer red roll frame, rotating about X
4. Pitch bearings fixed to the roll frame
5. Green pitch axle and inner board, rotating about local Y
6. Complete Furuta assembly mounted on the inner board

Corrections made during visual inspection:

- Enlarged the red frame to clear the moving board.
- Enabled explicit board/frame collision detection.
- Connected the base plate to both support towers.
- Replaced the fictitious full roll shaft with split axle stubs.
- Attached pitch-bearing housings to the outer red frame instead of the moving board.
- Extended the green pitch axle into those bearings.
- Verified the complete structural load path.

Current limits:

- Roll: ±15°
- Pitch: ±15°
- Maximum combined diagonal tilt: approximately 21.1° from vertical

Validation:

- Zero board/frame collisions across the tested joint envelope
- Finite compound dynamics
- Five-minute continuous-motion physics test remained stable and collision-free

The current red and green geometry is a functional conceptual model, not exact BRIO CAD.

## 4. Board-motion generator

Primary implementation:

`rl/tilt_2d.py`

### Previous motion

The first generator moved between independent random roll/pitch targets with smooth cubic
acceleration and a random 0.3–1.0 second dwell. The frozen warm-start performed very well under
that slower stop-and-go distribution.

### Current motion

The dwell-based generator was replaced at the user's request. Current motion is:

- Continuous, with no pauses
- Independent randomized roll and pitch
- Both axes moving simultaneously
- Three randomized harmonic components per axis
- Mathematically bounded reference angle, velocity, and acceleration
- Deterministic for a given seed

Current reference limits:

- Angle: ±15°
- Speed: 120°/s
- Acceleration: 1,200°/s²

Generator verification over five seeds and 50 aggregate simulated minutes:

- Maximum angle: 14.99–15.00°
- Maximum speed: 111.8–119.4°/s
- Maximum acceleration: 853–1,006°/s²
- Simultaneous zero-motion samples: none

The simulated position servos do not track the reference perfectly. Realized MuJoCo rates reached
approximately 127°/s in a five-minute no-policy test and up to approximately 140°/s in the full
policy stress suite. Training and evaluation must report realized board motion, not only reference
limits.

## 5. BNO086 observation model

Primary implementation:

`rl/bno086.py`

Manufacturer-based timing:

- Requested report rate: 200 Hz
- Report period: 5 ms
- Typical internal latency at 200 Hz: fixed 3.7 ms
- Effective sample age: 3.7–8.7 ms due to sample-and-hold
- Unsupported default timing jitter was removed

Outputs:

- Scalar-first orientation quaternion
- Roll and pitch derived from the quaternion
- Local X/Y/Z calibrated gyro rates
- Sensor timestamp and availability timestamp

Configurable installation and robustness effects:

- Mounting offset
- Tare offset
- Orientation error/noise
- Gyro bias/error

Mounting and tare offsets are installation effects, not manufacturer latency specifications.

Validation covered:

- Stationary board
- Single-axis motion
- Simultaneous motion
- Fast random motion
- Exact 200 Hz cadence
- Exact 3.7 ms internal latency
- Sample-and-hold behavior

## 6. Two-axis RL environment

Primary implementation:

`rl/furuta_env_2d.py`

The action remains one normalized Furuta-motor command mapped to ±6 V. Board motion is external
and is not controlled by the policy.

The ten-value observation is:

1. `cos(pole_angle_from_upright)`
2. `sin(pole_angle_from_upright)`
3. filtered pole rate / 15
4. arm angle / π
5. filtered arm rate / 25
6. previous normalized motor action
7. BNO086 roll / 15°
8. BNO086 pitch / 15°
9. BNO086 gyro-X / 80°/s
10. BNO086 gyro-Y / 80°/s

The original reward, success definition, cable/fall termination, and true-gravity vertical
calculation were retained. Current two-axis work uses free-arm evaluation/training because the
validated master is a free-arm model.

Environment validation:

- Gymnasium API passed
- Seeded resets and IMU/motion sequences are reproducible
- 0°, 5°, 10°, and 15° rollouts passed
- Cable-limit and fall-after-upright termination passed
- No board/frame contacts

## 7. Training-free 1D-to-2D warm start

Transfer implementation:

`rl/transfer_1d_to_2d.py`

Output checkpoint:

- File: `rl/models/clean20_master_2d_warmstart.zip`
- SHA-256: `DBAA19EEC48F25EBC44C09A28E780C65749D2052318E1553BE3AB2E65A34B5AD`
- Observation size: 10

The old one-axis board joint rotates about Y, so the old tilt channels are pitch, not roll.

Weight mapping:

- Original inputs 0–5 copied unchanged
- Old tilt-angle column mapped to new pitch input 7
- Old tilt-rate column mapped to new gyro-Y input 9
- New roll input 6 initialized to exactly zero
- New gyro-X input 8 initialized to exactly zero
- Pitch columns rescaled to preserve physical equivalence despite changed input normalization
- The same mapping applied to actor, both critics, and target critics
- Hidden and output layers copied unchanged

No gradients or replay were used for the transfer.

Equivalence over 10,000 physically matched observations:

- Maximum deterministic action difference: `2.146e-06`
- Maximum critic difference: `4.883e-04`
- Save/reload equivalence also passed

## 8. Evaluation findings

### Slower stop-and-go random motion

Five hundred fresh episodes per condition:

| Condition | Sustained success |
|---|---:|
| Level | 99.4% |
| Pitch only | 99.4% |
| Roll only | 99.2% |
| Random roll + pitch | 99.4% |
| Static ±15° corners | 96.8% |
| Conservative BNO-error stress | 99.2% |
| Aggressive diagonal reversals | 1.6% |

This established that the transferred policy can balance under ordinary simultaneous roll/pitch
motion without additional training. Roll worked through existing pole/arm feedback even though the
new roll input weights were zero.

Evidence:

- `STEP4_FROZEN_STRESS_REPORT.md`
- `eval/step4/`

### Continuous high-speed motion

The complete frozen-policy suite was repeated after removing dwell and increasing speed. Five
hundred fresh episodes were used per condition.

| Condition | Sustained | Catch | Falls |
|---|---:|---:|---:|
| Level | 99.4% | 99.4% | 3 |
| Continuous pitch only | 9.8% | 81.6% | 451 |
| Continuous roll only | 99.4% | 99.4% | 3 |
| Continuous roll + pitch | 9.8% | 78.6% | 451 |
| Static ±15° corners | 95.6% | 98.8% | 9 |
| Aggressive diagonal reversals | 3.2% | 69.8% | 334 |
| Continuous motion + BNO stress | 9.8% | 77.0% | 451 |

Key diagnosis:

- Fast continuous pitch is the limiting condition.
- Roll-only remains as strong as level balance.
- Simultaneous performance matches pitch-only performance, confirming pitch is the bottleneck.
- High catch but low sustained success means the model usually swings up but cannot maintain
  balance under continuous fast pitch excitation.
- The warm-start remains valuable because it preserves swing-up, level balance, slow board
  compensation, and most catches.

Evidence:

- `STEP4_CONTINUOUS120_REPORT.md`
- `eval/step4_continuous120/`

## 9. Current training method

Primary scripts:

- `rl/train_tqc_2d.py`
- `rl/retention_tqc.py`
- `rl/collect_teacher_data_2d.py`
- `rl/launch_2d_continuous5.sh`

### Teacher retention dataset

- File: `rl/teacher_2d_retention_100k.npz`
- Size: 100,000 transitions
- Observation shape: `(100000, 10)`
- Action shape: `(100000, 1)`

Only successful frozen-policy episodes were retained. Profiles include:

- Level
- Fast roll-only
- Slower simultaneous roll/pitch
- Static compound corners

### Curriculum

| Stage | Axes | Angle | Reference speed |
|---:|---|---:|---:|
| 0 | Pitch | ±5° | 40–60°/s |
| 1 | Pitch | ±10° | 60–90°/s |
| 2 | Pitch | ±15° | 90–120°/s |
| 3 | Roll + pitch | ±5° | 40–60°/s |
| 4 | Roll + pitch | ±10° | 60–90°/s |
| 5 | Roll + pitch | ±15° | 90–120°/s |
| 6 | Mixed deployment | ±15° | 30–120°/s |

Each training episode is sampled from:

- 50% current curriculum task
- 15% level retention
- 15% roll-only retention
- 10% slower simultaneous retention
- 10% static-corner retention

### Optimization and retention protection

- Algorithm: TQC
- Warm start: `clean20_master_2d_warmstart.zip`
- Actor LR: `1e-5`
- Critic LR: `1e-4`
- Critic adapts first; actor frozen until 25k steps
- Teacher replay fraction: 25%
- Adaptive teacher contribution target: 20% of absolute RL actor-loss magnitude
- This explicitly fixes the prior failure where retention contributed below 0.1% of actor loss
- Evaluation every 25k steps
- Two passing evaluations required after at least 50k stage steps
- Stage timeout: 250k steps
- Maximum seed budget: 1.75M steps

Advancement target:

- Current stage ≥90% sustained success

Retention stop floors:

- Level ≥90%
- Roll-only ≥90%
- Slower simultaneous ≥85%
- Static corners ≥85%

Candidate checkpoints are saved by stage. The frozen warm-start is never overwritten.

## 10. Active training state

Server launch was attempted after the UT VPN connected. The host resolves and accepts the public
key, but the SSH/PAM session closes immediately after key acceptance. Therefore no server seed was
started.

Local hardware:

- GPU: NVIDIA GeForce RTX 5070, 12 GB

Local CUDA smoke test passed:

- Vectorized MuJoCo environments
- Replay sampling
- Separate actor/critic rates
- Teacher retention logic
- Checkpoint save

Active local run:

- Tag: `tilt2d_cont_local_s0`
- Seed: 0
- Process at documentation time: PID 35928
- Output directory: `rl/models/tilt2d_cont_local_s0/`
- Log: `train_tilt2d_cont_local_s0.log`
- First formal gate at 25k steps:
  - Stage target: 100%
  - Level retention: 100%
  - Roll retention: 100%
  - Slower simultaneous retention: 95%
  - Static-corner retention: 90%
- The critic adapted first; the actor was frozen until the configured 25k release point

This snapshot is not a model-selection result. Selection must use independent evaluation
checkpoints and complete retention gates.

## 11. Visual artifacts

- `two_axis_warmstart_15deg.gif`
  - Successful frozen-policy episode under simultaneous randomized roll and pitch
  - Generated before the final continuous 120°/s change
- `rl/view_tilt_2d.py`
  - Interactive board viewer
- `rl/render_policy_gif_2d.py`
  - Reproducible two-axis policy GIF renderer

## 12. Important decisions

1. Preserve the one-axis master and keep two-axis development separate.
2. Use actual BNO086 observations, not servo commands.
3. Treat board actuation as an external disturbance.
4. Use the 2D warm-start rather than restarting from a no-tilt model.
5. Train first without plant domain randomization.
6. Target fast continuous pitch specifically; do not broadly relearn already strong level/roll
   behavior.
7. Use separate actor/critic learning rates and quantitatively significant retention loss.
8. Evaluate realized board rates because MuJoCo tracking can exceed reference limits.
9. Require independent multi-condition retention gates before accepting a trained checkpoint.

## 13. Next actions

1. Monitor local seed 0 through its first 25k evaluation and retention gate.
2. Stop immediately if retention floors fail.
3. If SSH/PAM access recovers, sync the isolated 2D project and launch seeds 0–4 on GPUs 0–4.
4. Do not select a model from training-buffer success or a single small evaluation.
5. Independently verify any candidate with at least 500 fresh episodes per condition.
6. Replace assumed motion envelopes with logged physical BNO086 trajectories when hardware is
   available.
