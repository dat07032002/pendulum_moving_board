"""Two-axis Furuta environment using the approved board and BNO086 observation model."""
from __future__ import annotations

import os
from collections import deque

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

from bno086 import BNO086Model
from furuta_env import (
    ARM_LIMIT,
    DT,
    PHI_SCALE,
    TH_SCALE,
    V_MAX,
    FurutaEnv,
)
from tilt_2d import SmoothRandomTilt2D


HERE = os.path.dirname(__file__)
BOARD_ANGLE_SCALE = np.deg2rad(15.0)
BOARD_RATE_SCALE = np.deg2rad(80.0)


class Furuta2DEnv(FurutaEnv):
    """Furuta motor control with externally driven random roll/pitch board motion."""

    def __init__(self, randomize=True, render_mode=None, max_seconds=10.0):
        # Initialize common curriculum/reward settings, then replace only the physical model.
        super().__init__(randomize=randomize, render_mode=render_mode, max_seconds=max_seconds)
        self.model = mujoco.MjModel.from_xml_path(os.path.join(HERE, "furuta_2d.xml"))
        self.data = mujoco.MjData(self.model)
        self.sub = int(round(DT / self.model.opt.timestep))

        jp = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "pole")
        ja = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "arm")
        jr = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "board_roll")
        jpt = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "board_pitch")
        self.qadr_p, self.qadr_a = self.model.jnt_qposadr[jp], self.model.jnt_qposadr[ja]
        self.dadr_p, self.dadr_a = self.model.jnt_dofadr[jp], self.model.jnt_dofadr[ja]
        self.qadr_r, self.qadr_pt = self.model.jnt_qposadr[jr], self.model.jnt_qposadr[jpt]
        self.dadr_r, self.dadr_pt = self.model.jnt_dofadr[jr], self.model.jnt_dofadr[jpt]
        self.act_motor = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "motor"
        )
        self.act_roll = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "roll_servo"
        )
        self.act_pitch = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "pitch_servo"
        )
        self.bid_pole = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "pole"
        )
        self.bid_board = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "pitch_frame"
        )

        # Motor voltage limit. Hardware (TMC6300) can drive ~11 V; sim default is the
        # conservative 6 V. Override with FURUTA_VMAX to test the extra torque authority.
        self.v_max = float(os.environ.get("FURUTA_VMAX", V_MAX))
        self.model.actuator_ctrlrange[self.act_motor] = [-self.v_max, self.v_max]

        bp = self.bid_pole
        self.nom = dict(
            gear=float(self.model.actuator_gear[self.act_motor, 0]),
            dmp_a=float(self.model.dof_damping[self.dadr_a]),
            dmp_p=float(self.model.dof_damping[self.dadr_p]),
            fr_a=float(self.model.dof_frictionloss[self.dadr_a]),
            fr_p=float(self.model.dof_frictionloss[self.dadr_p]),
            inertia_p=self.model.body_inertia[bp].copy(),
        )
        self.tilt_speed_max = np.deg2rad(120.0)
        self.tilt_accel_max = np.deg2rad(1200.0)
        self.tilt_axis_mode = "both"  # "both", "pitch", or "roll"
        self.imu_kwargs = {}  # evaluation/training overrides for the BNO086 measurement model
        self._board_velocity = np.zeros(6)
        self._roll_meas = self._pitch_meas = 0.0
        self._gyro_x_meas = self._gyro_y_meas = 0.0
        self._imu_latest = None
        self.tilt_gen_2d = None
        self.imu = None

        self.observation_space = spaces.Box(-np.inf, np.inf, (10,), np.float32)

    def reset(self, *, seed=None, options=None):
        # Defaults are needed because FurutaEnv.reset dynamically calls our _obs().
        self._roll_meas = self._pitch_meas = 0.0
        self._gyro_x_meas = self._gyro_y_meas = 0.0
        self._imu_latest = None
        self.tilt_gen_2d = None
        self.imu = None
        super().reset(seed=seed, options=options)

        rng = self.np_random
        if self.tilt_amp > 1e-8:
            amp = rng.uniform(self.tilt_amp_min_fraction, 1.0) * self.tilt_amp
            self.tilt_gen_2d = SmoothRandomTilt2D(
                angle_max=amp,
                speed_max=self.tilt_speed_max,
                accel_max=self.tilt_accel_max,
                dt=DT,
                seed=int(rng.integers(0, 2**32 - 1)),
            )
        else:
            self.tilt_gen_2d = None
        self.tilt_gen = None
        self.imu = BNO086Model(
            seed=int(rng.integers(0, 2**32 - 1)),
            **self.imu_kwargs,
        )
        self.data.ctrl[self.act_roll] = 0.0
        self.data.ctrl[self.act_pitch] = 0.0
        return self._obs(), {}

    def _obs(self):
        q = self.data.qpos[self.qadr_p]
        th_up = q - np.pi
        phi = self.data.qpos[self.qadr_a]
        n = self._obs_noise * self.np_random.standard_normal(2)
        return np.array(
            [
                np.cos(th_up),
                np.sin(th_up),
                self.thd_f / TH_SCALE + n[0],
                np.clip(phi / np.pi, -2.0, 2.0),
                self.phid_f / PHI_SCALE + n[1],
                self.prev_action,
                np.clip(self._roll_meas / BOARD_ANGLE_SCALE, -2.0, 2.0),
                np.clip(self._pitch_meas / BOARD_ANGLE_SCALE, -2.0, 2.0),
                self._gyro_x_meas / BOARD_RATE_SCALE,
                self._gyro_y_meas / BOARD_RATE_SCALE,
            ],
            dtype=np.float32,
        )

    def step(self, action):
        a = float(np.clip(action[0], -1.0, 1.0))
        self.act_buf.append(a)
        a_eff = self.act_buf.pop(0)
        self.data.ctrl[self.act_motor] = a_eff * self.v_max
        if self.tilt_gen_2d is None:
            roll_ref = pitch_ref = 0.0
        else:
            roll_ref, pitch_ref, _, _ = self.tilt_gen_2d.step()
            if self.tilt_axis_mode == "pitch":
                roll_ref = 0.0
            elif self.tilt_axis_mode == "roll":
                pitch_ref = 0.0
            elif self.tilt_axis_mode != "both":
                raise ValueError(f"unknown tilt_axis_mode {self.tilt_axis_mode!r}")
        self.data.ctrl[self.act_roll] = roll_ref
        self.data.ctrl[self.act_pitch] = pitch_ref

        for _ in range(self.sub):
            mujoco.mj_step(self.model, self.data)
            mujoco.mj_objectVelocity(
                self.model,
                self.data,
                mujoco.mjtObj.mjOBJ_BODY,
                self.bid_board,
                self._board_velocity,
                1,
            )
            self._imu_latest = self.imu.update(
                self.data.time,
                self.data.xquat[self.bid_board].copy(),
                self._board_velocity[:3].copy(),
            )
        if self._imu_latest is not None:
            self._roll_meas = self._imu_latest.roll
            self._pitch_meas = self._imu_latest.pitch
            self._gyro_x_meas = float(self._imu_latest.gyro_xyz[0])
            self._gyro_y_meas = float(self._imu_latest.gyro_xyz[1])

        self.thd_f = 0.5 * self.thd_f + 0.5 * self.data.qvel[self.dadr_p]
        self.phid_f = 0.5 * self.phid_f + 0.5 * self.data.qvel[self.dadr_a]

        q = self.data.qpos[self.qadr_p]
        phi = self.data.qpos[self.qadr_a]
        thd = self.data.qvel[self.dadr_p]
        phid = self.data.qvel[self.dadr_a]
        up = self._true_up()

        # Deliberately identical to the validated 1D reward and termination logic.
        reward = (
            up
            - self.arm_center_w * (phi / np.pi) ** 2
            - 0.005 * a**2
            - 0.002 * phid**2
            - 0.02 * (a - self.prev_action) ** 2
        )
        reward -= self.arm_envelope_w * max(0.0, abs(phi) - np.pi / 2) ** 2
        if up > 0.5:
            reward -= 0.01 * thd**2
        arm_ok = self.arm_limit is None or abs(phi) < np.pi / 2
        if up > 0.92 and abs(thd) < 3.0 and arm_ok:
            reward += 2.0
        self.prev_action = a

        balanced = up > 0.9 and abs(thd) < 4.0 and arm_ok
        self._balance_window.append(balanced)
        if balanced:
            self._up_streak += 1
            self._best_up_streak = max(self._best_up_streak, self._up_streak)
        else:
            self._up_streak = 0
        if up > 0.9:
            self._was_up = True

        self.steps += 1
        terminated = False
        if self.arm_limit is not None and abs(phi) > self.arm_limit:
            reward -= 10.0
            terminated = True
        if self._was_up and up < 0.0:
            terminated = True
        truncated = self.steps >= self.max_steps
        info = {}
        if terminated or truncated:
            info["is_catch_success"] = bool(self._best_up_streak * DT > 0.5)
            occupancy = float(np.mean(self._balance_window)) if self._balance_window else 0.0
            info["final_balance_occupancy"] = occupancy
            info["is_success"] = bool(truncated and not terminated and occupancy >= 0.8)

        if self.render_mode == "human":
            self._render_human()
        return self._obs(), float(reward), terminated, truncated, info


if __name__ == "__main__":
    env = Furuta2DEnv(randomize=False, max_seconds=2.0)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (10,)
    for _ in range(400):
        obs, reward, terminated, truncated, _ = env.step(np.zeros(1))
        assert np.all(np.isfinite(obs)) and np.isfinite(reward)
        if terminated or truncated:
            break
    print("Furuta2DEnv sanity check passed")
