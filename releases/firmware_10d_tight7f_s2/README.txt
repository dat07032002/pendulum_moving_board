Furuta pendulum firmware — 10-D RL policy "tight7f_s2" (pre action-history generation)
=======================================================================================

Contents
--------
furuta_foc/furuta_foc.ino     ESP32 control firmware (200 Hz FOC + on-chip RL policy)
furuta_foc/policy_weights.h   Trained policy weights, baked in at compile time
                              (source checkpoint sha256 0c302f9a...cd  RL_OBS=10, RL_VMAX=10.0)

This is the firmware/policy pair as deployed 2026-07-04: balances a pendulum on a
two-axis moving board, swing-up + catch + hold, autonomous cable-guard recovery.

Hardware it expects
-------------------
- ESP32 (board target esp32:esp32:esp32)
- GM3506 gimbal motor + TMC6300 driver (pins in the .ino header comment)
- AS5048A arm encoder (SPI), AS5600 pendulum encoder (I2C, pins 21/22)
- BNO086 IMU on second I2C bus (SDA 32, SCL 33) — REQUIRED; the policy refuses
  to run without a healthy, tared IMU
- ~11 V motor supply

Build & flash
-------------
1. Install arduino-cli (or use the Arduino IDE)
2.   arduino-cli core install esp32:esp32
     arduino-cli lib install "SparkFun BNO08x Cortex Based IMU"
3.   arduino-cli compile --fqbn esp32:esp32:esp32 furuta_foc
     arduino-cli upload -p COM5 --fqbn esp32:esp32:esp32 furuta_foc   (adjust port)

First run (IMPORTANT, in order)
-------------------------------
Serial monitor at 921600 baud. Motor stays OFF at boot; only the `rl` command
enables the policy.

1. Board LEVEL during power-on (IMU does a level tare at boot)
2. `calfoc`  — one-time FOC motor calibration (saved to flash)
3. `calhang` — with the pendulum hanging DEAD STILL (pole encoder reference;
   redo after any pole/magnet work)
4. `health` and `imu` — check encoders + IMU are valid
5. `rlcheck` — prints the observation and the policy's action, motor off
6. Put the arm at the cable/wire center (the firmware zeroes the arm there),
   then `rl` to start; `s` stops instantly

Commands: rl | rlcheck | bal | t <V> | s | vlim <V> | log | nolog | imu |
imutare | calhang | calup | calfoc | clearcal | health | raw | params

Safety notes
------------
- Arm guard at +/-330 degrees: policy homes back to center and retries on its own
- Stale/dead IMU stops the motor immediately
- Start testing with `vlim 2` or `3`; full performance needs `vlim 10`
  (below 10 V the policy cannot complete swing-up — that is expected)
- If boot prints "BNO086 missing": full power cycle (USB out 10 s) — the IMU
  can wedge on warm reboots

Project: https://github.com/dat07032002/pendulum_moving_board
(there is also a newer 12-D "tight8_s2" policy in the repo with better
delay robustness; this 10-D pair is the simpler, well-tested baseline)
