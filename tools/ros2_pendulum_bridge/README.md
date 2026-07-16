# ROS2 pendulum bridge

Publishes the pendulum ESP32's 200 Hz serial telemetry as `/pendulum/state`
(`std_msgs/Float32MultiArray`), for feeding pendulum state into CyberRunner.

```
data = [theta, theta_dot, phi, phi_dot, V, imu_roll, imu_pitch, upright_flag]
```

## Quick start (no package needed)

On the CyberRunner machine (Linux, ROS2 sourced):

```bash
pip install pyserial
sudo usermod -aG dialout $USER        # once; re-login afterwards
python3 pendulum_bridge.py --port /dev/ttyUSB0
```

Verify in a second terminal:

```bash
ros2 topic hz /pendulum/state         # expect ~200 Hz
ros2 topic echo /pendulum/state --once
```

Find the port: `ls /dev/ttyUSB* /dev/ttyACM*` (unplug/replug the ESP32 to see
which appears). If CyberRunner's Dynamixels also use a USB serial adapter,
check both and udev-pin them by serial number to avoid swaps.

## As a proper package (optional)

Drop this folder into the CyberRunner workspace as `cyberrunner_pendulum/`,
add a minimal `setup.py`/`package.xml` (ament_python, entry point
`pendulum_bridge = cyberrunner_pendulum.pendulum_bridge:main`), then:

```bash
colcon build --packages-select cyberrunner_pendulum
ros2 run cyberrunner_pendulum pendulum_bridge --ros-args -p port:=/dev/ttyUSB0
```

## Wiring it into Dreamer (idea #4)

1. In CyberRunner's env/obs assembly node, subscribe to `/pendulum/state`
   (QoS best-effort, depth 1 — always take the freshest sample).
2. At each Dreamer step (camera frame, ~55 Hz), append
   `[theta, theta_dot, upright_flag]` to the observation dict as a small
   vector entry alongside the image.
3. Add the reward term, e.g. `r += 0.1 * upright_flag` (or a -1 penalty on
   the 1->0 transition = a fall). Start small so maze progress still dominates.
4. Keep training online as usual; the world model learns the coupling.

## Notes

- The bridge opens serial with DTR/RTS deasserted: connecting does NOT reboot
  the ESP32 (a reboot re-tares the IMU and can wedge the BNO086).
- The stream works in every firmware mode; the pendulum does not need to be
  running `rl` for the topic to be live.
- theta is 0 at upright, +/-pi hanging; upright_flag = |theta| < 10 deg.
