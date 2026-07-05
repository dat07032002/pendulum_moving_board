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

        # Upright tightness. `up` is cos(pole angle from true vertical); the success/balanced gate
        # and the +2 bonus fire above these thresholds. Default 0.90 (~26 deg) / 0.92 (~23 deg).
        # Override with FURUTA_UP_THRESH to demand a tighter upright hold (e.g. 0.95 ~= 18 deg).
        self.up_thresh = float(os.environ.get("FURUTA_UP_THRESH", 0.90))
        # Bonus gate at 80% of the success angle (angle space). The old
        # `min(0.99, up_thresh + 0.02)` saturated for gates tighter than ~8 deg,
        # collapsing the +2 balanced-bonus band; at the 10-deg production gate this
        # derivation is ~8.0 deg vs the old 8.1 deg (behavior-preserving).
        self.up_bonus = float(np.cos(0.8 * np.arccos(np.clip(self.up_thresh, -1.0, 1.0))))
        # Preserve the historical 2D free-arm default. Deployment constraints are
        # explicit environment variables so cable-aware and free-arm seeds are unambiguous.
        self.arm_limit = None
        self.success_arm_limit = None
        self.arm_center_w = 0.0
        if "FURUTA_CABLE_LIMIT_DEG" in os.environ:
            value = os.environ["FURUTA_CABLE_LIMIT_DEG"].strip().lower()
            self.arm_limit = None if value == "none" else np.deg2rad(float(value))
        if "FURUTA_SUCCESS_ARM_LIMIT_DEG" in os.environ:
            value = os.environ["FURUTA_SUCCESS_ARM_LIMIT_DEG"].strip().lower()
            self.success_arm_limit = (
                None if value == "none" else np.deg2rad(float(value))
            )
        if "FURUTA_ARM_CENTER_W" in os.environ:
            self.arm_center_w = float(os.environ["FURUTA_ARM_CENTER_W"])
        # Action-rate (smoothness) penalty weight; historical hard-coded value 0.02.
        # Raise (e.g. 0.06-0.10) to train visibly smoother motor output.
        self.action_rate_w = float(os.environ.get("FURUTA_ACTION_RATE_W", "0.02"))
        # Actuator slew limit in volts per control tick (0 = disabled, legacy
        # behavior). When >0 the applied motor voltage can change by at most this
        # much per 5 ms tick; the firmware applies the identical limit in rlStep,
        # so train and deploy see the same actuator. Breaks the +/-10 V 200 Hz
        # bang-bang limit cycle observed on hardware 2026-07-04.
        self.slew_v_per_tick = float(os.environ.get("FURUTA_SLEW_V_PER_TICK", "0"))
        self._applied_v = 0.0
        # First-order actuator lag [ms], modelling the real motor-electrical/FOC
        # lag chain that produced the 26 Hz hardware limit cycle (2026-07-04).
        # SIM-ONLY (the real rig already has the lag physically). "0" = off,
        # "5" = fixed tau, "2,8" = per-episode uniform DR over the range.
        lag_spec = os.environ.get("FURUTA_ACT_LAG_TAU_MS", "0")
        if "," in lag_spec:
            lo, hi = (float(x) for x in lag_spec.split(","))
            self.act_lag_range = (lo, hi)
        else:
            v = float(lag_spec)
            self.act_lag_range = (v, v)
        self._act_lag_tau = 0.0
        self._lag_v = 0.0
        # Extra past actions appended to the obs (a_{t-2}, a_{t-3}, ...). With
        # only prev_action (a_{t-1}) the delay-2 system is partially observed —
        # three delay campaigns collapsed on that (2026-07-03/04). 2 extra
        # actions restore full observability for delay <= 3. Default 0 keeps
        # the legacy 10-D obs and all existing checkpoints/tools working.
        self.act_history = int(os.environ.get("FURUTA_ACT_HISTORY", "0"))
        self._act_hist = [0.0] * self.act_history
        self.tight_upright_w = float(os.environ.get("FURUTA_TIGHT_UPRIGHT_W", "0"))
        self.tight_upright_scale = np.deg2rad(
            float(os.environ.get("FURUTA_TIGHT_UPRIGHT_SCALE_DEG", "10"))
        )
        self.cable_warning_w = float(os.environ.get("FURUTA_CABLE_WARNING_W", "0"))
        self.cable_warning_start = np.deg2rad(
            float(os.environ.get("FURUTA_CABLE_WARNING_START_DEG", "270"))
        )

        bp = self.bid_pole
        self.nom = dict(
            gear=float(self.model.actuator_gear[self.act_motor, 0]),
            dmp_a=float(self.model.dof_damping[self.dadr_a]),
            dmp_p=float(self.model.dof_damping[self.dadr_p]),
            fr_a=float(self.model.dof_frictionloss[self.dadr_a]),
            fr_p=float(self.model.dof_frictionloss[self.dadr_p]),
            inertia_p=self.model.body_inertia[bp].copy(),
        )
        self.tilt_speed_max = np.deg2rad(60.0)
        self.tilt_accel_max = np.deg2rad(600.0)
        self.tilt_axis_mode = "both"  # "both", "pitch", or "roll"
        # New low-friction pole bearing (2026-07-01). Keep broad multiplicative
        # coverage while centering the 2D plant below the legacy 0.35 mN*m model.
        self.dr_pole_damping_range = (1.4e-5, 5.4e-5)
        self.dr_pole_friction_range = (0.065e-3, 0.265e-3)
        self.imu_kwargs = {}  # evaluation/training overrides for the BNO086 measurement model
        self._board_velocity = np.zeros(6)
        self._roll_meas = self._pitch_meas = 0.0
        self._gyro_x_meas = self._gyro_y_meas = 0.0
        self._imu_latest = None
        self.tilt_gen_2d = None
        self.imu = None

        self.observation_space = spaces.Box(
            -np.inf, np.inf, (10 + self.act_history,), np.float32
        )

    def reset(self, *, seed=None, options=None):
        # Defaults are needed because FurutaEnv.reset dynamically calls our _obs().
        self._roll_meas = self._pitch_meas = 0.0
        self._gyro_x_meas = self._gyro_y_meas = 0.0
        self._imu_latest = None
        self.tilt_gen_2d = None
        self.imu = None
        self._applied_v = 0.0
        self._lag_v = 0.0
        self._act_hist = [0.0] * self.act_history
        super().reset(seed=seed, options=options)
        self._act_lag_tau = float(
            self.np_random.uniform(self.act_lag_range[0], self.act_lag_range[1])
        )

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
                *self._act_hist,
            ],
            dtype=np.float32,
        )

    def step(self, action):
        a = float(np.clip(action[0], -1.0, 1.0))
        self.act_buf.append(a)
        a_eff = self.act_buf.pop(0)
        v_cmd = a_eff * self.v_max
        if self.slew_v_per_tick > 0.0:
            # Actuator slew limit, mirrored in firmware (rlStep). Hidden actuator
            # state: prev_action in the obs stays the POLICY output on both sides.
            v_cmd = float(np.clip(
                v_cmd,
                self._applied_v - self.slew_v_per_tick,
                self._applied_v + self.slew_v_per_tick,
            ))
        self._applied_v = v_cmd
        if self._act_lag_tau > 0.0:
            # First-order motor lag AFTER the (digital) slew limiter, matching the
            # physical chain: policy -> delay -> firmware slew -> motor electrical lag.
            alpha = (DT * 1000.0) / (self._act_lag_tau + DT * 1000.0)
            self._lag_v += alpha * (v_cmd - self._lag_v)
            v_cmd = self._lag_v
        self.data.ctrl[self.act_motor] = v_cmd
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
            - self.action_rate_w * (a - self.prev_action) ** 2
        )
        reward -= self.arm_envelope_w * max(0.0, abs(phi) - np.pi / 2) ** 2
        if self.tight_upright_w > 0.0:
            pole_error = np.arccos(np.clip(up, -1.0, 1.0))
            reward += self.tight_upright_w * np.exp(
                -(pole_error / self.tight_upright_scale) ** 2
            )
        if up > 0.5:
            reward -= 0.01 * thd**2
        abs_phi = abs(phi)
        if (
            self.cable_warning_w > 0.0
            and self.success_arm_limit is not None
            and self.success_arm_limit > self.cable_warning_start
        ):
            cable_progress = np.clip(
                (abs_phi - self.cable_warning_start)
                / (self.success_arm_limit - self.cable_warning_start),
                0.0,
                1.0,
            )
            reward -= self.cable_warning_w * cable_progress**2
        self._max_abs_phi = max(self._max_abs_phi, abs_phi)
        arm_ok = self.success_arm_limit is None or abs_phi < self.success_arm_limit
        if not arm_ok:
            self._steps_outside_success_arm_limit += 1
        if up > self.up_bonus and abs(thd) < 3.0 and arm_ok:
            reward += 2.0
        # shift action history BEFORE prev_action updates: _act_hist becomes
        # [a_{t-2}, a_{t-3}, ...] as seen by the NEXT observation
        if self.act_history > 0:
            self._act_hist = [self.prev_action] + self._act_hist[:-1]
        self.prev_action = a

        balanced = up > self.up_thresh and abs(thd) < 4.0 and arm_ok
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
            self._cable_limit_hit = True
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
            info["max_abs_arm_deg"] = float(np.rad2deg(self._max_abs_phi))
            info["fraction_outside_success_arm_limit"] = (
                self._steps_outside_success_arm_limit / max(self.steps, 1)
            )
            info["cable_limit_hit"] = self._cable_limit_hit

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
