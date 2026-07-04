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
    gyro_sensor_time: float
    gyro_available_time: float
    quaternion_wxyz: np.ndarray
    roll: float
    pitch: float
    gyro_xyz: np.ndarray


class BNO086Model:
    """Sample, delay, and perturb true board orientation/gyro measurements."""

    def __init__(
        self,
        report_hz: float = 100.0,
        gyro_latency_s: tuple[float, float] = (0.0037, 0.0037),
        orientation_extra_latency_s: tuple[float, float] = (0.002, 0.003),
        timing_jitter_s: float = 0.0,
        mounting_error_deg: float = 0.5,
        tare_error_deg: float = 0.5,
        orientation_noise_deg: float = 0.012,
        gyro_bias_deg_s: float = 0.06,
        gyro_noise_deg_s: float = 0.50,
        seed: int = 0,
    ) -> None:
        self.period = 1.0 / float(report_hz)
        # Absolute gyro latency is still the manufacturer-based assumption. Hardware
        # characterization measured orientation arriving 2-3 ms behind gyro.
        self.gyro_latency_s = gyro_latency_s
        self.orientation_extra_latency_s = orientation_extra_latency_s
        self.timing_jitter_s = float(timing_jitter_s)
        self.orientation_noise = np.deg2rad(orientation_noise_deg)
        self.gyro_noise = np.deg2rad(gyro_noise_deg_s)
        self.rng = np.random.default_rng(seed)
        static_bound = np.deg2rad(mounting_error_deg + tare_error_deg)
        self._orientation_bias = self.rng.uniform(-static_bound, static_bound, size=3)
        gyro_bound = np.deg2rad(gyro_bias_deg_s)
        self._gyro_bias = self.rng.uniform(-gyro_bound, gyro_bound, size=3)
        self._next_sample_time = 0.0
        self._pending_orientation: deque[tuple] = deque()
        self._pending_gyro: deque[tuple] = deque()
        self._latest_orientation: tuple | None = None
        self._latest_gyro: tuple | None = None
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
            measured_gyro = np.asarray(true_gyro_xyz) + self._gyro_bias + self.rng.normal(
                0.0, self.gyro_noise, size=3
            )
            gyro_latency = self.rng.uniform(*self.gyro_latency_s)
            orientation_extra = self.rng.uniform(*self.orientation_extra_latency_s)
            self._pending_orientation.append(
                (
                    now + gyro_latency + orientation_extra,
                    now,
                    measured_q,
                    roll,
                    pitch,
                )
            )
            self._pending_gyro.append(
                (
                    now + gyro_latency,
                    now,
                    measured_gyro,
                )
            )
            jitter = self.rng.uniform(-self.timing_jitter_s, self.timing_jitter_s)
            self._next_sample_time += max(0.5 * self.period, self.period + jitter)

        updated = False
        while self._pending_gyro and self._pending_gyro[0][0] <= now:
            self._latest_gyro = self._pending_gyro.popleft()
            updated = True
        while self._pending_orientation and self._pending_orientation[0][0] <= now:
            self._latest_orientation = self._pending_orientation.popleft()
            updated = True
        if updated and self._latest_orientation is not None and self._latest_gyro is not None:
            orientation_available, orientation_sensor, q, roll, pitch = self._latest_orientation
            gyro_available, gyro_sensor, gyro = self._latest_gyro
            self.latest = BNO086Reading(
                sensor_time=orientation_sensor,
                available_time=orientation_available,
                gyro_sensor_time=gyro_sensor,
                gyro_available_time=gyro_available,
                quaternion_wxyz=q,
                roll=roll,
                pitch=pitch,
                gyro_xyz=gyro,
            )
        return self.latest
