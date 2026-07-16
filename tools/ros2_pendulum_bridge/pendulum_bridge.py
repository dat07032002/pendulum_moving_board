#!/usr/bin/env python3
"""ROS2 bridge: publish the pendulum ESP32's 200 Hz serial telemetry as a topic.

Publishes std_msgs/Float32MultiArray on /pendulum/state at the firmware rate:
    data = [theta, theta_dot, phi, phi_dot, V, imu_roll, imu_pitch, upright]
      theta      pendulum angle from upright [rad]  (0 = balanced, +/-pi = hanging)
      theta_dot  pendulum rate [rad/s]
      phi        arm angle [rad] (relative to boot)
      phi_dot    arm rate [rad/s]
      V          motor voltage being applied
      imu_roll / imu_pitch  board tilt [rad] (from the pendulum's BNO086)
      upright    1.0 if |theta| < 10 deg else 0.0  (convenience flag)

Usage:
    ros2 run <your_pkg> pendulum_bridge --ros-args -p port:=/dev/ttyUSB0
or just:
    python3 pendulum_bridge.py --port /dev/ttyUSB0

Notes:
- Opens serial with DTR/RTS deasserted so the ESP32 does NOT reboot on connect
  (a reboot would re-run the IMU level tare and can wedge the BNO086).
- Sends `log` on connect to start the stream; sends `nolog` on shutdown.
- Linux serial permissions: sudo usermod -aG dialout $USER  (re-login after).
"""
import argparse
import math
import sys
import threading

import rclpy
import serial
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float32MultiArray

UP10 = math.cos(math.radians(10.0))


def open_noreset(port: str) -> serial.Serial:
    s = serial.Serial()
    s.port = port
    s.baudrate = 921600
    s.timeout = 0.05
    s.dtr = False
    s.rts = False
    s.open()
    return s


class PendulumBridge(Node):
    def __init__(self, port: str):
        super().__init__("pendulum_bridge")
        self.declare_parameter("port", port)
        port = self.get_parameter("port").get_parameter_value().string_value or port

        # Sensor-style QoS: best-effort, shallow queue — subscribers always get
        # the freshest state, never a backlog.
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.pub = self.create_publisher(Float32MultiArray, "/pendulum/state", qos)

        self.ser = open_noreset(port)
        self.ser.write(b"log\n")
        self.get_logger().info(f"streaming from {port} at 200 Hz -> /pendulum/state")

        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self):
        tail = ""
        while not self._stop.is_set():
            chunk = self.ser.read(4096).decode(errors="replace")
            if not chunk:
                continue
            tail += chunk
            lines = tail.split("\n")
            tail = lines[-1]
            for line in lines[:-1]:
                if not line.startswith("log=["):
                    continue
                try:
                    r = line.strip()[5:-1].split(",")
                    # log=[t_ms,phi,theta,phi_dot,theta_dot,V,theta_raw,
                    #      imu_roll,imu_pitch,gyro_x,gyro_y,gyro_z,seq...]
                    phi = float(r[1])
                    theta = float(r[2])
                    phi_dot = float(r[3])
                    theta_dot = float(r[4])
                    volts = float(r[5])
                    roll = float(r[7])
                    pitch = float(r[8])
                except (ValueError, IndexError):
                    continue
                msg = Float32MultiArray()
                msg.data = [
                    theta, theta_dot, phi, phi_dot, volts, roll, pitch,
                    1.0 if math.cos(theta) > UP10 else 0.0,
                ]
                self.pub.publish(msg)

    def destroy_node(self):
        self._stop.set()
        try:
            self.ser.write(b"nolog\n")
            self.ser.close()
        except Exception:
            pass
        super().destroy_node()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyUSB0")
    args, ros_args = ap.parse_known_args()
    rclpy.init(args=ros_args)
    node = PendulumBridge(args.port)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
