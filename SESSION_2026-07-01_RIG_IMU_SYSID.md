# Rig bring-up, new pole bearing, and BNO086 characterization

Date: 2026-07-01

## Status

Phases 0–3 of the local hardware plan are complete through plant/IMU model calibration.
Server retraining has not started.

## Firmware and safety

- ESP32 base firmware compiles, flashes, and runs on COM5 at 921600 baud.
- Control and logging remain nominally 200 Hz.
- RL boot auto-start is disabled. Power-on/reset leaves the motor idle; `rl` is still available
  as an explicit command.
- Stored FOC calibration: direction `-1`, electrical offset `300 deg`.
- AS5600 hanging calibration: raw `2756`, upright reference `708`.
- Low-voltage motor sign check passed: `+V -> +phi`.
- Serial tools now open COM5 with DTR/RTS inactive so they do not reset the ESP32.

## Sensor buses

- AS5600: primary I2C `Wire`, SDA 21 / SCL 22, 400 kHz.
- BNO086: dedicated I2C `Wire1`, Qwiic SDA 32 / SCL 33, 400 kHz, address `0x4B`.
- AS5048A remains on SPI.
- Separating the buses prevents a stalled BNO086 from blocking the pole encoder.
- BNO086 `RST` and `INT` are not wired. If BNO initialization reports `valid=0`, a full power
  cycle is still required.

## BNO086 firmware and observations

- Magnetometer-free Game Rotation Vector plus calibrated gyro.
- Software roll/pitch level tare averages 500 ms on every boot.
- Firmware log fields:

  `t_ms, phi, theta, phi_dot, theta_dot, V, theta_raw, imu_roll, imu_pitch,`
  `gyro_x, gyro_y, gyro_z, imu_seq`

- Physical +roll maps to IMU +roll / gyro-X.
- Physical +pitch maps to IMU +pitch / gyro-Y.
- No firmware or simulation sign swap is required.

## Timing decision

Two 200 Hz BNO reports (orientation plus gyro) were not sustainable over polled I2C:

- usable orientation approximately 106 Hz;
- control approximately 195.6 Hz;
- worst observed control interval 14 ms.

Selected configuration:

- control/log loop approximately 199.7 Hz;
- BNO report cadence approximately 99.7–100.0 Hz;
- nominal two-control-step sample-and-hold;
- worst observed control interval 6 ms.

The orientation signal trails gyro by a repeatable 2–3 ms. Absolute physical-to-report gyro
latency was not measured independently; the simulation retains the previous 3.7 ms
manufacturer-based value and labels it as an assumption.

## IMU characterization

- Static orientation standard deviation:
  - roll: `0.0009 deg`;
  - pitch: `0.0116 deg`.
- Static drift over 60 s:
  - roll: `-0.0005 deg`;
  - pitch: `+0.0056 deg`.
- Calibrated gyro was exactly zero while stationary, so residual static bias/noise is below the
  report resolution.
- Two motion runs were repeatable:
  - gyro/orientation correlation: `0.996–0.998`;
  - rate scale: `0.986–1.001`;
  - relative orientation lag: `2–3 ms`;
  - tested rates reached approximately `92 deg/s`.
- Raw captures are under `eval/imu/`.

## New pole-bearing system identification

Latest accepted three-release fit (`20/35/45 deg`, 28 clean peak pairs):

- period: `440 ms`;
- `alpha`: `203.917 1/s^2`;
- viscous damping `b_theta`: `3.41989e-5 N*m*s/rad`;
- Coulomb friction `Tf`: `1.64973e-4 N*m`;
- decay ratio `rho`: `0.928385`;
- half-swing decrement `C`: `1.903 deg`;
- effective deadband: approximately `1.0 deg`.

Compared with the old bearing/model, viscous damping fell by roughly one third and Coulomb
friction by roughly one half. Pole period/alpha remained essentially unchanged. Arm parameters
were not modified.

## Simulation updates and validation

- `rl/furuta_2d.xml` pole damping: `3.41989e-5`.
- `rl/furuta_2d.xml` pole friction loss: `1.64973e-4`.
- `rl/bno086.py`:
  - 100 Hz reports;
  - independent gyro/orientation availability;
  - 0.012 deg orientation noise;
  - 0.06 deg/s gyro bias bound;
  - 0.50 deg/s gyro noise;
  - measured 2–3 ms orientation delay relative to gyro.
- Free-swing validation:
  - period error: `0.0%`;
  - decay-ratio error: `2.0%`;
  - Coulomb-decrement error: `7.6%`;
  - 15% gate: **PASS**.
- 2D pole DR ranges are centered on the new plant:
  - damping: `[1.4e-5, 5.4e-5]`;
  - friction: `[0.065e-3, 0.265e-3]`.
- Legacy 1D DR defaults and all arm/motor ranges remain unchanged.

## Next server phase

When the server is available:

1. Train at +/-10 deg board tilt, capped at 60 deg/s, with a +/-10 deg upright gate.
2. Warm-start from `rl/models/up15_best.zip`.
3. Use critic reset, gamma 0.99, gradient clipping, and no teacher.
4. Run three cable-aware seeds: hard +/-360 deg, success margin +/-330 deg.
5. Run two free-arm seeds as the control group.
6. Use arm-centering weight 0.02 in both groups.
7. Add near-vertical reward shaping only if the 10 deg gate still plateaus.
8. Screen candidates over 300 episodes per condition; verify the selected winner over 500 and
   report arm/cable metrics.

## Training-readiness audit

The full active 2D training/evaluation path was audited after calibration:

- canonical warm start: `rl/models/up15_best.zip`;
- observation/action shape: `10 -> 1`;
- checkpoint gamma: `0.99`;
- critic reset now clears critic weights, target, and stale Adam optimizer state;
- teacher-free 10 V training no longer loads or requires the old 6 V teacher dataset;
- all current curriculum and retention profiles are capped at a 60 deg/s reference;
- the `pm10_60` verification grid is capped at 60 deg/s;
- action-delay DR is capped at 1-2 control steps (5-10 ms); the measured 3-step regime fails and
  is outside the deployment target;
- default 2D generator/environment limits are 60 deg/s and 600 deg/s^2;
- production training uses `FURUTA_VMAX=10`, `FURUTA_UP_THRESH=0.984807753`,
  `--ladder pm10_60`, and `--no-teacher`;
- hard cable termination and success arm margin are independently configurable;
- cable-aware seeds use +/-360 deg hard termination and +/-330 deg success margin;
- free-arm seeds remove both limits; all seeds use arm-centering weight 0.02;
- `rl/preflight_training.py` passes;
- a teacher-free critic-plus-actor gradient smoke test passes;
- the exact production entry point passes a subprocess/callback/save smoke test.

The MuJoCo position servo may slightly exceed its reference speed. The curriculum is capped at
60 deg/s reference; report realized board rates in final evaluation.
