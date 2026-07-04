"""Open the two-axis board concept in MuJoCo and animate both gimbal axes."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from bno086 import BNO086Model
from tilt_2d import SmoothRandomTilt2D


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--amplitude", type=float, default=15.0, help="motion amplitude [deg]")
    parser.add_argument("--period", type=float, default=8.0, help="roll cycle period [s]")
    parser.add_argument("--mode", choices=("random", "sine", "static"), default="random")
    parser.add_argument("--seed", type=int, default=0, help="reproducible random sequence")
    parser.add_argument("--speed", type=float, default=60.0, help="maximum speed [deg/s]")
    parser.add_argument("--accel", type=float, default=600.0, help="maximum acceleration [deg/s^2]")
    args = parser.parse_args()

    xml_path = Path(__file__).with_name("furuta_2d.xml")
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    roll_ctrl = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "roll_servo")
    pitch_ctrl = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "pitch_servo")
    amplitude = np.deg2rad(np.clip(args.amplitude, 0.0, 20.0))
    random_tilt = SmoothRandomTilt2D(
        angle_max=amplitude,
        speed_max=np.deg2rad(args.speed),
        accel_max=np.deg2rad(args.accel),
        dt=0.01,
        seed=args.seed,
    )
    imu = BNO086Model(seed=args.seed)
    board_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pitch_frame")
    board_velocity = np.zeros(6)

    print("2D board viewer: RED = roll X axis, GREEN = pitch Y axis.")
    print(f"Motion mode: {args.mode}; amplitude: +/-{np.rad2deg(amplitude):.1f} deg.")
    print("Close the MuJoCo window to stop.")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -22
        viewer.cam.distance = 0.75
        viewer.cam.lookat[:] = (0.0, 0.0, 0.02)
        start = time.perf_counter()
        while viewer.is_running():
            frame_start = time.perf_counter()
            elapsed = frame_start - start
            if args.mode == "static":
                roll_ref = pitch_ref = 0.0
            elif args.mode == "sine":
                phase = 2.0 * np.pi * elapsed / args.period
                roll_ref = amplitude * np.sin(phase)
                pitch_ref = amplitude * np.sin(0.5 * phase)
            else:
                roll_ref, pitch_ref, _, _ = random_tilt.step()
            data.ctrl[roll_ctrl] = roll_ref
            data.ctrl[pitch_ctrl] = pitch_ref
            data.ctrl[0] = 0.0

            for _ in range(10):
                mujoco.mj_step(model, data)
                mujoco.mj_objectVelocity(
                    model,
                    data,
                    mujoco.mjtObj.mjOBJ_BODY,
                    board_body,
                    board_velocity,
                    1,
                )
                imu.update(data.time, data.xquat[board_body].copy(), board_velocity[:3].copy())
            viewer.sync()
            remaining = 0.01 - (time.perf_counter() - frame_start)
            if remaining > 0:
                time.sleep(remaining)


if __name__ == "__main__":
    main()
