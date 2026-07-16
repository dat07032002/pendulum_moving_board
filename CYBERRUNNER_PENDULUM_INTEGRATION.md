# CyberRunner + Pendulum Integration Design

## Purpose

Combine two complementary controls:

1. Make CyberRunner's DreamerV3 maze agent pendulum-aware by adding measured
   pendulum state to its observation and a small penalty for pendulum falls.
2. Give the fast ESP32 pendulum policy CyberRunner's two commanded knob
   velocities as preview inputs, so it can react before the board motion is
   visible in the IMU.

The maze agent and pendulum agent remain separate. DreamerV3 decides how to
move through the maze at the camera/control rate. The ESP32 runs the pendulum
policy at 200 Hz and stabilizes the pendulum locally.

This design was checked against CyberRunner repository commit
`40cf614cb84a46bd8cf2da37ecc048166a4bd848` (2026-07-15 review). CyberRunner's
Gym environment publishes two `DynamixelVel` fields on
`cyberrunner_dynamixel/cmd` before waiting for its next camera observation,
which provides a suitable preview source.

## Architecture

```text
                     pendulum observation
                 +---------------------------+
                 |                           v
camera/state --> DreamerV3 maze agent --> knob velocity command --> Dynamixels
                                                |
                                                | preview command
                                                v
                                      ESP32 pendulum policy
                                                |
                                                v
                                          motor voltage
                                                |
                                                v
                                         Furuta pendulum
                                                |
                     /pendulum/state <----------+
```

The command path is intentionally split before physical board response. A
command appended only after the next camera frame would be action history, not
preview.

## Existing components

- `tools/ros2_pendulum_bridge/pendulum_bridge.py` publishes the ESP32's 200 Hz
  serial telemetry on `/pendulum/state`.
- `rl/furuta_env_2d.py` defines the simulation and the current 10/12-D
  pendulum-policy observation.
- `rl/expand_obs_warmstart.py` zero-expands TQC actor and critic input layers
  and verifies identical initial behavior.
- `firmware/furuta_foc/furuta_foc.ino` constructs the deployed observation and
  runs the policy locally.
- CyberRunner's `CyberrunnerGym._send_action()` publishes the two knob velocity
  commands before `_get_obs()` waits for a camera/state-estimation update.

The current expansion utility is configured through `FURUTA_ACT_HISTORY` and
therefore currently treats added inputs as action history. Before using it for
preview training, change the environment configuration and naming so the final
two entries explicitly represent CyberRunner command axes. Do not produce a
14-D checkpoint whose metadata says action history while deployment supplies
preview commands.

## Signal contracts

### CyberRunner to pendulum policy: preview

Preferred logical signal:

```text
preview = [cmd_axis_1, cmd_axis_2], float32, each nominally in [-1, 1]
```

The canonical values are CyberRunner's normalized policy outputs, before the
`max_angle_vel` scale and motor-specific sign factors. If only
`DynamixelVel.vel_1` and `vel_2` are available, convert them using the exact
CyberRunner scale and sign configuration and record the raw values as well.

Required conventions:

- Document which command axis produces positive board roll and pitch.
- Keep training and deployment order, scale, signs, and clipping identical.
- Hold the latest preview between CyberRunner updates.
- Set both preview values to zero on reset, terminal state, loss of command,
  and emergency stop.
- Mark preview stale and force zero when no command arrives for a configurable
  timeout; start testing with 100--150 ms.

The bridge should subscribe to `cyberrunner_dynamixel/cmd` using queue depth 1
and forward the latest normalized command to the ESP32. The serial protocol
should add a compact command such as `preview <axis1> <axis2>`. Protect serial
writes with a lock because telemetry reading and ROS callbacks use different
threads.

The ESP32 stores the last preview and its receive time. Its 200 Hz observation
builder appends the held preview values:

```text
obs[0:12]  = existing pendulum observation
obs[12]    = normalized CyberRunner command axis 1
obs[13]    = normalized CyberRunner command axis 2
```

### Pendulum to CyberRunner: measured state

Keep the existing `/pendulum/state` stream for bring-up. The current array is:

```text
[theta, theta_dot, phi, phi_dot, volts,
 imu_roll, imu_pitch, upright]
```

For training and synchronization, add a stamped interface or include the
ESP32 source timestamp and telemetry sequence. ROS receive time alone cannot
distinguish sensor age from USB/ROS delay.

At each CyberRunner observation, append the freshest valid pendulum vector:

```text
pendulum = [cos(theta), sin(theta), clip(theta_dot / 15, -2, 2), upright]
```

Also track telemetry age for diagnostics. If the sample is stale, do not claim
that a fall occurred. Report a separate telemetry fault and enter a safe mode.
The subscriber must use best-effort, depth-1 QoS to match the existing bridge
publisher and prevent queued old state.

CyberRunner's DreamerV3 `cyberrunner` configuration currently limits MLP keys
to `states|goal$`. Extend the encoder and decoder key expressions to include
`pendulum`, otherwise declaring the Gym observation alone will not feed the new
vector into the world model.

## Hardware data collection

### Why collect data

The command does not become board motion instantly:

```text
Dreamer command
  -> Dynamixel transport and servo dynamics
  -> linkage backlash/compliance
  -> board angle and angular rate
  -> pendulum disturbance
```

Hardware recordings capture real axis signs, scaling, delay, rate limiting,
backlash, and cross-axis coupling. They are used to identify the simulation
disturbance model and to provide realistic command sequences. Initial data
collection is not online pendulum-policy training.

### What to record

Record one timestamped stream containing or reconstructing:

| Field | Units / representation |
|---|---|
| source and ROS receive timestamps | seconds |
| telemetry and command sequence | integer |
| normalized command axes 1 and 2 | `[-1, 1]` |
| raw Dynamixel velocities 1 and 2 | native units |
| board roll and pitch | radians |
| board roll and pitch rates | rad/s |
| pendulum `theta` and `theta_dot` | radians, rad/s |
| arm `phi` and `phi_dot` | radians, rad/s |
| applied pendulum motor voltage | volts |
| upright, reset, terminal, and telemetry-valid flags | boolean |

Record the original ROS topics losslessly first:

```bash
ros2 bag record \
  /cyberrunner_dynamixel/cmd \
  /pendulum/state \
  /cyberrunner_state_estimation/estimate_subimg
```

If a normalized action topic and stamped pendulum topic are added, record
those as the preferred analysis sources. Export a synchronized CSV later; do
not discard the rosbag.

### Collection procedure

1. Run the existing frozen pendulum controller. Do not explore with an
   untrained 14-D controller on hardware.
2. Begin with CyberRunner velocity clamped below its normal limit.
3. Collect normal maze runs, starts/stops, both-axis reversals, diagonal
   commands, resets, and progressively more aggressive runs.
4. Collect approximately 20--30 minutes across the representative operating
   envelope, provided temperatures and cable behavior remain safe.
5. Exclude reset motion from the nominal identification set but retain and
   label it for safety testing.
6. Estimate per-axis delay and command-to-board transfer behavior, including
   cross-axis response. Validate the fitted model on held-out runs.

## Training design

### Pendulum policy: simulation first

Train the 14-D TQC pendulum policy primarily in simulation. Direct hardware
exploration is not the starting point because falls, cable limits, resets, and
rare timing faults make it slow and risky.

Simulation must model the causal chain from commanded knob velocity through
servo/board dynamics. Do not directly substitute the preview command for
measured board rate. Use a mixture of:

- replayed command traces from real CyberRunner runs;
- bounded synthetic commands covering the normal envelope;
- starts, stops, reversals, and simultaneous-axis commands;
- random command transport delay and sample-and-hold timing;
- servo time constant, acceleration/rate limits, backlash, and cross coupling;
- IMU/encoder delay, observation noise, dropped updates, and stale commands;
- existing Furuta plant, voltage, friction, and inertia randomization.

Warm-start from the current verified 12-D checkpoint by adding two zero input
columns to all affected actor and critic layers. Verify that, with both preview
inputs zero, the 14-D policy matches the source policy to less than `1e-6`.
Then fine-tune overnight in simulation.

### DreamerV3 maze policy: hardware online training

Add the measured pendulum vector only after the frozen 14-D stabilizer passes
hardware validation. Continue DreamerV3 online training on the real maze, as
CyberRunner already does. A changed observation dictionary affects the
Dreamer encoder/checkpoint structure; the SB3/TQC expansion script does not
apply to the JAX Dreamer model.

Use a small event-oriented reward term:

```text
combined_reward = maze_reward
                  - lambda_fall * confirmed_falling_edge
                  - lambda_down * confirmed_down_time
```

Start with `lambda_down = 0` or very small. A large continuous upright bonus
can reward inactivity and interfere with maze progress. A fall is confirmed
only from fresh telemetry; missing data is a separate fault.

## Deployment and validation

### Stage 0: passive logging

- Forward and log preview commands without changing either policy.
- Measure command arrival relative to board IMU response.
- Verify axis order, signs, normalization, command age, and reset behavior.

### Stage 1: simulated preview policy

- Fit/validate the board command-response model from hardware logs.
- Expand the 12-D checkpoint to 14-D with zero preview weights.
- Train with replayed and randomized command profiles.
- Require nominal and domain-randomized performance no worse than the current
  12-D controller before deployment.

### Stage 2: frozen hardware A/B test

- Clamp CyberRunner commands conservatively.
- Deploy the 14-D policy frozen; no hardware gradient updates.
- Run matched maze trials with preview forced to zero and preview enabled.
- Compare falls/minute, maximum `|theta|`, theta RMS/p95, recovery count, arm
  cable-limit events, maze progress, and completion rate.
- Increase command limits only after each envelope passes.

### Stage 3: pendulum-aware Dreamer

- Add fresh pendulum state to Dreamer's observation dictionary.
- Add the small confirmed-fall penalty.
- Continue online maze learning under the validated stabilizer.
- Compare four matched conditions: baseline, pendulum-aware Dreamer only,
  preview stabilizer only, and both combined.

### Optional Stage 4: guarded hardware fine-tuning

Hardware fine-tuning of the pendulum policy is optional and occurs only if a
measured sim-to-real gap remains. Use the simulation-trained policy, tight
action/board limits, automatic shutdown, short sessions, checkpoint rollback,
and a human emergency stop. Never begin from random exploration.

## Safety requirements

- Existing arm cable limits remain authoritative.
- CyberRunner reset and terminal paths publish zero preview and zero motor
  commands.
- Preview timeout forces zero and is visible in logs.
- Stale pendulum telemetry cannot generate a false training reward or penalty.
- A telemetry/serial fault causes a defined safe behavior rather than use of
  the last command indefinitely.
- USB devices are pinned by udev serial number so the Dynamixel and ESP32 ports
  cannot swap.
- Validate ESP32 serial bandwidth and parser behavior with telemetry at 200 Hz
  and preview updates at the full CyberRunner rate.
- Temperature, cable-limit, command-saturation, and reset counters are logged
  for every hardware evaluation.

## Acceptance criteria

The combined design is ready for unrestricted maze evaluation only when:

1. Preview age is normally below one CyberRunner control interval and timeout
   behavior has been physically tested.
2. Axis/sign tests pass for positive and negative commands on both axes.
3. The zero-preview 14-D checkpoint reproduces the 12-D source behavior.
4. Frozen preview-on hardware trials reduce pendulum falls or wobble without
   increasing cable-limit events.
5. Maze progress/completion does not regress outside a predeclared tolerance.
6. Pendulum-aware Dreamer improves the joint objective in matched trials, not
   merely the upright metric by learning to stop.

## Implementation checklist

- [ ] Add source timestamp and sequence to pendulum telemetry.
- [ ] Add normalized CyberRunner action publication or a documented raw-to-
      normalized conversion.
- [ ] Subscribe in the bridge and serial-forward preview with a write lock.
- [ ] Add ESP32 preview parsing, hold, age, timeout, and zero-on-reset behavior.
- [ ] Add 14-D preview observation support to simulation and firmware.
- [ ] Adapt and run the TQC zero-expansion/equivalence check.
- [ ] Add rosbag collection and offline synchronization/identification tools.
- [ ] Train and evaluate the preview policy in simulation.
- [ ] Complete frozen hardware preview-off/on A/B tests.
- [ ] Add the `pendulum` observation key and encoder/decoder configuration to
      CyberRunner DreamerV3.
- [ ] Add confirmed-fall reward logging and online Dreamer fine-tuning.
- [ ] Run the four-condition ablation and publish the results.
