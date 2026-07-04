"""Frozen-policy 2D stress evaluation. This script never trains or updates model weights."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sb3_contrib import TQC

from furuta_env_2d import DT, Furuta2DEnv


CONDITIONS = (
    "level",
    "pitch_full",
    "roll_full",
    "both_full",
    "corners",
    "fast_diagonal",
    "bno_spec",
)


class FastDiagonalTilt:
    """Acceleration-limited diagonal reversal between opposite ±15° corners."""

    def __init__(self, angle, speed, accel, dt, dwell_s=0.20):
        self.angle = float(angle)
        self.speed_limit = float(speed)
        self.accel = float(accel)
        self.dt = float(dt)
        self.dwell_steps = int(dwell_s / dt)
        self.position = 0.0
        self.speed = 0.0
        self.target = self.angle
        self._dwell = 0
        self._plan_segment()

    def _plan_segment(self):
        self._start = self.position
        self._direction = float(np.sign(self.target - self._start))
        distance = abs(self.target - self._start)
        peak_speed = min(self.speed_limit, np.sqrt(distance * self.accel))
        self._accel_time = peak_speed / self.accel
        self._cruise_time = max(
            0.0, (distance - peak_speed**2 / self.accel) / peak_speed
        )
        self._total_time = 2.0 * self._accel_time + self._cruise_time
        self._elapsed = 0.0
        self._peak_speed = peak_speed

    def step(self):
        if self._elapsed >= self._total_time:
            self.position, self.speed = self.target, 0.0
            self._dwell += 1
            if self._dwell >= self.dwell_steps:
                self.target = -self.target
                self._dwell = 0
                self._plan_segment()
        else:
            self._elapsed = min(self._total_time, self._elapsed + self.dt)
            t = self._elapsed
            ta, tc, total = self._accel_time, self._cruise_time, self._total_time
            if t <= ta:
                distance = 0.5 * self.accel * t**2
                speed = self.accel * t
            elif t <= ta + tc:
                distance = 0.5 * self.accel * ta**2 + self._peak_speed * (t - ta)
                speed = self._peak_speed
            else:
                remaining = total - t
                total_distance = abs(self.target - self._start)
                distance = total_distance - 0.5 * self.accel * remaining**2
                speed = self.accel * remaining
            self.position = self._start + self._direction * distance
            self.speed = self._direction * speed
        return self.position, -self.position, self.speed, -self.speed


class CornerHoldTilt:
    """Smoothly ramp to one full-amplitude corner and hold it."""

    def __init__(self, angle, signs, dt, ramp_s=0.8):
        self.target = np.asarray(signs, dtype=float) * float(angle)
        self.dt = float(dt)
        self.steps = max(2, int(ramp_s / dt))
        self.index = 0

    def step(self):
        self.index += 1
        u = min(1.0, self.index / self.steps)
        blend = 3.0 * u**2 - 2.0 * u**3
        rate = (6.0 * u - 6.0 * u**2) / (self.steps * self.dt)
        position = blend * self.target
        velocity = rate * self.target if u < 1.0 else np.zeros(2)
        return position[0], position[1], velocity[0], velocity[1]


def configure(env: Furuta2DEnv, condition: str, episode: int) -> None:
    env.tilt_amp = np.deg2rad(15.0)
    env.tilt_amp_min_fraction = 1.0
    env.tilt_axis_mode = "both"
    env.imu_kwargs = {}
    if condition == "level":
        env.tilt_amp = 0.0
    elif condition == "pitch_full":
        env.tilt_axis_mode = "pitch"
    elif condition == "roll_full":
        env.tilt_axis_mode = "roll"
    elif condition == "bno_spec":
        # Conservative bounded-per-axis use of published dynamic orientation and gyro accuracy.
        # Gaussian report noise stays small; fixed episode errors exercise the specification limits.
        env.imu_kwargs = {
            "mounting_error_deg": 1.75,
            "tare_error_deg": 1.75,
            "orientation_noise_deg": 0.10,
            "gyro_bias_deg_s": 3.10,
            "gyro_noise_deg_s": 0.20,
        }
    elif condition not in ("both_full", "corners", "fast_diagonal"):
        raise ValueError(condition)


def run(model, condition, episodes, seed0):
    env = Furuta2DEnv(randomize=False, max_seconds=10.0)
    env.arm_limit = None
    env.success_arm_limit = None
    env.arm_center_w = 0.0
    env.init_angle_max = np.pi
    env.init_vel_assist = 0.0
    rows = []
    corner_signs = ((1, 1), (1, -1), (-1, 1), (-1, -1))
    for episode in range(episodes):
        configure(env, condition, episode)
        seed = seed0 + episode
        obs, _ = env.reset(seed=seed)
        if condition == "corners":
            env.tilt_gen_2d = CornerHoldTilt(
                np.deg2rad(15.0), corner_signs[episode % 4], DT
            )
        elif condition == "fast_diagonal":
            env.tilt_gen_2d = FastDiagonalTilt(
                np.deg2rad(15.0),
                np.deg2rad(80.0),
                np.deg2rad(300.0),
                DT,
            )

        terminated = truncated = False
        info = {}
        total_reward = 0.0
        upper_errors = []
        saturation = 0
        max_board = np.zeros(2)
        max_board_speed = np.zeros(2)
        steps = 0
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            saturation += int(abs(float(action[0])) >= 0.99)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            up = np.clip(env._true_up(), -1.0, 1.0)
            if up > 0.9:
                upper_errors.append(np.arccos(up))
            max_board = np.maximum(
                max_board,
                np.abs(env.data.qpos[[env.qadr_r, env.qadr_pt]]),
            )
            max_board_speed = np.maximum(
                max_board_speed,
                np.abs(env.data.qvel[[env.dadr_r, env.dadr_pt]]),
            )
        rows.append(
            (
                seed,
                int(info.get("is_success", False)),
                int(info.get("is_catch_success", False)),
                int(terminated and not truncated),
                total_reward,
                np.rad2deg(np.sqrt(np.mean(np.square(upper_errors))))
                if upper_errors
                else 180.0,
                saturation / steps,
                *np.rad2deg(max_board),
                *np.rad2deg(max_board_speed),
            )
        )
    env.close()
    names = (
        "seed",
        "success",
        "catch",
        "fall",
        "return",
        "upper_rms_deg",
        "saturation_fraction",
        "max_roll_deg",
        "max_pitch_deg",
        "max_roll_rate_deg_s",
        "max_pitch_rate_deg_s",
    )
    return names, np.asarray(rows, dtype=float)


def summarize(condition, names, rows):
    col = {name: rows[:, i] for i, name in enumerate(names)}
    order = np.argsort(col["return"])
    return {
        "condition": condition,
        "episodes": len(rows),
        "successes": int(col["success"].sum()),
        "success_rate": float(col["success"].mean()),
        "catches": int(col["catch"].sum()),
        "catch_rate": float(col["catch"].mean()),
        "falls": int(col["fall"].sum()),
        "mean_upper_rms_deg": float(col["upper_rms_deg"].mean()),
        "p95_upper_rms_deg": float(np.quantile(col["upper_rms_deg"], 0.95)),
        "mean_saturation_fraction": float(col["saturation_fraction"].mean()),
        "max_saturation_fraction": float(col["saturation_fraction"].max()),
        "worst_seeds_by_return": [int(x) for x in col["seed"][order[:10]]],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("-n", "--episodes", type=int, default=500)
    parser.add_argument("--seed0", type=int, default=60000)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    model = TQC.load(args.model, device="cpu")
    names, rows = run(model, args.condition, args.episodes, args.seed0)
    summary = summarize(args.condition, names, rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, names=np.asarray(names), rows=rows)
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
