# First hardware deployment

Do not flash a policy until it has passed the selected verification gate. Power-on/reset keeps the
motor idle; only the explicit `rl` command enables policy control.

## Build gate

1. Export the exact selected checkpoint:

   `python rl/export_policy.py --model <checkpoint.zip> --vmax 10 --out firmware/furuta_foc/policy_weights.h`

2. Confirm the exporter reports `OBS=10` and a forward-pass error below `1e-4`.
3. Compile the firmware and record the checkpoint SHA-256 embedded in `policy_weights.h`.
4. Keep the board level during boot so the BNO086 level tare is valid.

## Motor-off preflight

Start with motor power disconnected or the driver disabled.

1. Put the arm at the physical cable center before boot. The firmware treats the arm position at
   `rl` engagement as the center of the policy coordinate system.
2. Boot and run `health`, then `imu`. Require both encoders healthy, `valid=1`, changing GRV/gyro
   sequence counters, and approximately zero roll/pitch while level.
3. Run `rlcheck`. It prints the exact normalized 10-D observation and deterministic policy action
   while keeping the motor off.
4. At level, observation indices 6 and 7 should be near zero.
5. At approximately +10 degrees physical roll, index 6 should be approximately +0.667 and index 7
   should remain near zero.
6. At approximately +10 degrees physical pitch, index 7 should be approximately +0.667 and index 6
   should remain near zero.
7. Move roll and pitch in both directions. Gyro indices 8 and 9 must follow physical gyro-X and
   gyro-Y with the established signs. A BNO086 fault or stale report must make `rlcheck` fail.

## Powered stages

Keep one hand on the serial `s` command or a hardware power cutoff. Start with the board level and
the arm at cable center.

1. Set `vlim 1`, briefly run `rl`, verify motor direction, then `s`.
2. Repeat at `vlim 2`, then `vlim 3`, stopping immediately for wrong sign, violent motion, stale IMU,
   encoder discontinuity, or unexpected cable winding.
3. Use `vlim 6` and then `vlim 10` only after the low-voltage checks pass. The production policy
   is trained at a 10 V action scale (`RL_VMAX 10`), so every reduced-voltage stage is a safety
   check rather than a performance test: expect failed swing-ups and arm wind-up below 10 V.
   Proper balancing behavior only appears at `vlim 10`.
4. Test level ground first, then slow ±5-degree single-axis board motion, ±10-degree single-axis
   motion, and finally combined motion.
5. Stop before the arm approaches the ±330-degree firmware guard. Do not rely on automatic recovery
   during the first deployment.

Save the serial log from every powered stage. A successful simulation verification is necessary but
does not authorize skipping these hardware gates.
