# Two-axis tilting-board design

## 1. Objective

Build a board that can tilt about two orthogonal horizontal axes while the existing Furuta
pendulum performs swing-up and balances relative to true gravity.

The first target remains:

- free Furuta arm;
- total board tilt up to 20°;
- total board angular-rate cap near 2 rad/s;
- one GM3506 policy action;
- two independently commanded board actuators;
- BNO086 orientation and angular-rate feedback.

Cable-limited Furuta-arm behavior is a later phase and should not be mixed into initial 2D
development.

## 2. Mechanical architecture

Use a nested gimbal rather than placing two hinges on one rigid body:

1. A fixed outer base supports an outer roll frame.
2. The outer frame rotates about the world X axis.
3. An inner board rotates inside that frame about its local Y axis.
4. The Furuta stand, motor, electronics, and IMU mount to the inner board.

The two axes should intersect near the loaded board's center of mass. This minimizes gravity torque
and makes required actuator torque much smaller.

### Load path

- Bearings on both sides of each axis carry the board and Furuta weight.
- Servo shafts should provide torque, not serve as structural bearings.
- Use rigid aluminum, plywood, or composite frames rather than relying on a toy maze's thin pivots.
- Add counterweights or spring assistance if the loaded center of mass cannot lie near the axis
  intersection.

Approximate static torque for each axis:

```text
torque = total_mass × gravity × center_of_mass_offset
```

Example: 2 kg with a 50 mm offset already requires about 0.98 N·m before acceleration, friction,
and safety margin. Select actuators only after measuring total mass, center of mass, desired angular
acceleration, and linkage ratio. Use at least a 2–3× torque margin.

### Mechanical limits

- Commanded soft limit: 20° total tilt.
- Mechanical hard stops: approximately 23–25°.
- Do not permit ±20° independently on both axes unless a 28.3° diagonal tilt is acceptable.

For a true 20° cone, enforce:

```text
sqrt(roll² + pitch²) <= 20°
```

### Practical details

- Route Furuta, IMU, and actuator wiring through flexible loops near the gimbal axes.
- Ensure neither board axis pinches the Furuta arm-encoder cable.
- Mount the BNO086 rigidly near the inner board center and isolate only high-frequency vibration;
  it must follow board motion without flex.
- Include physical emergency stops and actuator-disable control.

## 3. MuJoCo structure

Use two nested bodies so the second hinge axis rotates with the first:

```xml
<body name="roll_frame" pos="0 0 0">
  <joint name="board_roll"
         type="hinge"
         axis="1 0 0"
         range="-25 25"/>

  <body name="board" pos="0 0 0">
    <joint name="board_pitch"
           type="hinge"
           axis="0 1 0"
           range="-25 25"/>

    <!-- board geometry and complete Furuta assembly -->
  </body>
</body>

<actuator>
  <position name="roll_servo" joint="board_roll" kp="..."/>
  <position name="pitch_servo" joint="board_pitch" kp="..."/>
</actuator>
```

Development files should be copied from, not substituted for, the 1D files:

- `furuta.xml` → `furuta_2d.xml`
- `tilt.py` → `tilt_2d.py`
- `furuta_env.py` → `furuta_env_2d.py`

The world-frame pole-vector calculation used by `_true_up()` remains valid automatically.

## 4. Two-axis motion generator

Do not use two fully independent scalar generators; independent saturation can create excessive
diagonal angle and rate.

Generate a 2D tilt vector:

```text
beta = [roll, pitch]
```

Choose random targets uniformly inside a disk and move toward each target with vector speed capped:

```text
norm(beta_target) <= beta_max
norm(beta_dot) <= beta_dot_max
```

Required test modes:

- roll only;
- pitch only;
- fixed diagonal directions;
- circular or elliptical motion;
- random targets with dwell;
- sudden reversals within actuator acceleration limits;
- static holds at the edge of the tilt disk.

Model the actual actuator trajectory, not an impossible ideal position jump. Include velocity and
acceleration limits, and later identify compliance, backlash, and servo lag from hardware.

## 5. Observation and BNO086 representation

The direct extension of the 8D observation is 10D:

```text
[cos(theta), sin(theta), theta_dot/15,
 phi/pi, phi_dot/25, previous_action,
 roll/0.6, roll_rate/3,
 pitch/0.6, pitch_rate/3]
```

For the first milestone, use this explicit roll/pitch representation because it allows a clean
warm-start mapping from the existing one-axis policy.

Longer term, the more robust representation is the board-frame gravity direction plus board
angular rates. It avoids Euler-angle ordering issues:

```text
[gravity_x, gravity_y, gyro_x/3, gyro_y/3]
```

Whichever representation is chosen must be identical in MuJoCo and ESP32 firmware. Define and test:

- positive roll and pitch signs;
- axis order;
- board-to-IMU mounting rotation;
- quaternion-to-angle conversion;
- gyro-frame convention;
- normalization and clipping.

Keep motor-action delay and IMU observation delay as separate simulation parameters.

## 6. Warm-start strategy

Expand the actor input from 8 to 10 values:

1. Copy all hidden and output layers from the verified 1D master.
2. Copy the original tilt-angle and tilt-rate input weights into the corresponding original-axis
   columns.
3. Initialize the two new second-axis columns to zero.
4. Preserve critic initialization only if its observation/action layout is expanded consistently;
   otherwise reinitialize critics and freeze the actor during early critic adaptation.

This makes the initial 10D policy exactly reproduce the 1D actor whenever the second axis is zero.

Do not begin with unrestricted 2D motion. First prove warm-start parity over the original 1,000
verification episodes with the new axis locked.

## 7. Training curriculum

### Phase 2D-0: geometry and warm-start parity

- Pitch/roll axis matching the original simulation moves up to ±20°.
- New axis locked at zero.
- No plant DR.
- Require performance statistically consistent with the 91.5% 1D master.

### Phase 2D-1: introduce the second axis

- Original axis up to ±20°.
- New axis starts at ±3–5°.
- Clean plant and measured/default delays.
- Increase the new axis only after sustained performance holds.

### Phase 2D-2: radial two-axis curriculum

Suggested total-tilt caps:

```text
5° → 10° → 15° → 20°
```

Each stage samples all directions inside the disk. Include dedicated axis and diagonal evaluations;
random sampling alone may under-test the hardest orientations.

### Phase 2D-3: hard motion

- Increase the minimum amplitude toward 70% of the cap.
- Increase vector rate toward 1.2–2.0 rad/s.
- Add static edge holds and reversals.

### Phase 2D-4: robustness

Only after clean 2D behavior is stable:

1. Add plant DR except unsupported action-delay ranges.
2. Add measured motor delay.
3. Add IMU noise and explicit IMU latency.
4. Add actuator mismatch, backlash, and axis misalignment.

Do not combine cable-limit training with initial 2D learning.

## 8. Evaluation matrix

Every candidate must be evaluated separately on:

| Condition | Purpose |
|---|---|
| original axis only | 1D skill retention |
| new axis only | new-axis competence |
| +45° and -45° diagonal directions | coupled-axis difficulty |
| random directions in the tilt disk | deployment distribution |
| static edge holds | gravity-load robustness |
| maximum-rate reversals | transient robustness |
| clean plant | task competence |
| target DR | robustness |
| fixed IMU-delay bins | latency sensitivity |

Continue using sustained success over at least 500 episodes for selection. Record success versus
tilt magnitude, direction, angular rate, and Furuta arm orientation.

## 9. Recommended implementation order

1. Freeze and document the 1D reference.
2. Create `furuta_2d.xml` with the second nested hinge and actuator.
3. Build `tilt_2d.py` with disk-bounded angle and rate.
4. Build `furuta_env_2d.py` and validate gravity/sign conventions.
5. Expand the policy input and prove exact one-axis warm-start parity.
6. Run hand-controller and random-policy physics sanity checks.
7. Train the clean second-axis curriculum with multiple seeds.
8. Add robustness only after clean 2D success.
9. Design/build hardware after torque and center-of-mass measurements confirm actuator sizing.
