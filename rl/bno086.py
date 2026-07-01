"""Configurable BNO086 observation model for a board-mounted IMU."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product for scalar-first (w, x, y, z) quaternions."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )


def quat_from_rotvec(rotvec: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(rotvec))
    if angle < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = rotvec / angle
    return np.r_[np.cos(0.5 * angle), axis * np.sin(0.5 * angle)]


def quat_to_roll_pitch(q: np.ndarray) -> tuple[float, float]:
    """Return intrinsic XYZ roll/pitch from a normalized scalar-first quaternion."""
    w, x, y, z = q / np.linalg.norm(q)
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sin_pitch = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
    return float(roll), float(np.arcsin(sin_pitch))


@dataclass(frozen=True)
class BNO086Reading:
    sensor_time: float
    available_time: float
    quaternion_wxyz: np.ndarray
    roll: float
    pitch: float
    gyro_xyz: np.ndarray


class BNO086Model:
    """Sample, delay, and perturb true board orientation/gyro measurements."""

    def __init__(
        self,
        report_hz: float = 200.0,
        latency_s: tuple[float, float] = (0.0037, 0.0037),
        timing_jitter_s: float = 0.0,
        mounting_error_deg: float = 0.5,
        tare_error_deg: float = 0.5,
        orientation_noise_deg: float = 0.10,
        gyro_bias_deg_s: float = 0.30,
        gyro_noise_deg_s: float = 0.20,
        seed: int = 0,
    ) -> None:
        self.period = 1.0 / float(report_hz)
        self.latency_s = latency_s
        self.timing_jitter_s = float(timing_jitter_s)
        self.orientation_noise = np.deg2rad(orientation_noise_deg)
        self.gyro_noise = np.deg2rad(gyro_noise_deg_s)
        self.rng = np.random.default_rng(seed)
        static_bound = np.deg2rad(mounting_error_deg + tare_error_deg)
        self._orientation_bias = self.rng.uniform(-static_bound, static_bound, size=3)
        gyro_bound = np.deg2rad(gyro_bias_deg_s)
        self._gyro_bias = self.rng.uniform(-gyro_bound, gyro_bound, size=3)
        self._next_sample_time = 0.0
        self._pending: deque[BNO086Reading] = deque()
        self.latest: BNO086Reading | None = None

    def update(
        self,
        now: float,
        true_quaternion_wxyz: np.ndarray,
        true_gyro_xyz: np.ndarray,
    ) -> BNO086Reading | None:
        """Generate due reports and release reports whose modeled latency has elapsed."""
        if now + 1e-12 >= self._next_sample_time:
            noise_rotvec = self.rng.normal(0.0, self.orientation_noise, size=3)
            error_q = quat_from_rotvec(self._orientation_bias + noise_rotvec)
            measured_q = quat_multiply(true_quaternion_wxyz, error_q)
            measured_q /= np.linalg.norm(measured_q)
            roll, pitch = quat_to_roll_pitch(measured_q)
            measured_gyro = (
                np.asarray(true_gyro_xyz)
                + self._gyro_bias
                + self.rng.normal(0.0, self.gyro_noise, size=3)
            )
            latency = self.rng.uniform(*self.latency_s)
            self._pending.append(
                BNO086Reading(
                    sensor_time=now,
                    available_time=now + latency,
                    quaternion_wxyz=measured_q,
                    roll=roll,
                    pitch=pitch,
                    gyro_xyz=measured_gyro,
                )
            )
            jitter = self.rng.uniform(-self.timing_jitter_s, self.timing_jitter_s)
            self._next_sample_time += max(0.5 * self.period, self.period + jitter)

        while self._pending and self._pending[0].available_time <= now:
            self.latest = self._pending.popleft()
        return self.latest
