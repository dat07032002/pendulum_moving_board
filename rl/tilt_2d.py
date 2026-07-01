"""Continuous, smooth, bounded random roll/pitch references."""
from __future__ import annotations

import numpy as np


class SmoothRandomTilt2D:
    """Independent random harmonic motion on both axes with no dwell or target stops."""

    def __init__(
        self,
        angle_max: float = np.deg2rad(15.0),
        speed_max: float = np.deg2rad(120.0),
        accel_max: float = np.deg2rad(1200.0),
        dt: float = 0.005,
        seed: int = 0,
        harmonics: int = 3,
    ) -> None:
        self.angle_max = float(angle_max)
        self.speed_max = float(speed_max)
        self.accel_max = float(accel_max)
        self.dt = float(dt)
        self.rng = np.random.default_rng(seed)
        self.time = 0.0
        self._amplitudes = np.empty((2, harmonics))
        self._frequencies = np.empty((2, harmonics))

        for axis in range(2):
            weights = self.rng.uniform(0.25, 1.0, size=harmonics)
            weights /= weights.sum()
            amplitudes = self.angle_max * weights
            signs = self.rng.choice((-1.0, 1.0), size=harmonics)
            amplitudes *= signs

            raw_frequencies = self.rng.uniform(0.6, 1.4, size=harmonics)
            raw_speed_bound = np.sum(np.abs(amplitudes) * raw_frequencies)
            raw_accel_bound = np.sum(np.abs(amplitudes) * raw_frequencies**2)
            scale = min(
                self.speed_max / raw_speed_bound,
                np.sqrt(self.accel_max / raw_accel_bound),
            )
            # Randomize near the requested limit without ever exceeding it.
            scale *= self.rng.uniform(0.85, 1.0)
            self._amplitudes[axis] = amplitudes
            self._frequencies[axis] = raw_frequencies * scale

    def step(self) -> tuple[float, float, float, float]:
        self.time += self.dt
        phase = self._frequencies * self.time
        position = np.sum(self._amplitudes * np.sin(phase), axis=1)
        velocity = np.sum(
            self._amplitudes * self._frequencies * np.cos(phase), axis=1
        )
        return (
            float(position[0]),
            float(position[1]),
            float(velocity[0]),
            float(velocity[1]),
        )


class CornerHoldTilt2D:
    """Smoothly move to one roll/pitch corner and hold there."""

    def __init__(self, angle: float, signs: tuple[int, int], dt: float, ramp_s: float = 0.8):
        self.target = np.asarray(signs, dtype=float) * float(angle)
        self.dt = float(dt)
        self.steps = max(2, int(ramp_s / dt))
        self.index = 0

    def step(self) -> tuple[float, float, float, float]:
        self.index += 1
        u = min(1.0, self.index / self.steps)
        blend = 3.0 * u**2 - 2.0 * u**3
        blend_rate = (6.0 * u - 6.0 * u**2) / (self.steps * self.dt)
        position = blend * self.target
        velocity = blend_rate * self.target if u < 1.0 else np.zeros(2)
        return position[0], position[1], velocity[0], velocity[1]
