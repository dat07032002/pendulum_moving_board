"""Deterministic baseline evaluation for a 10-input two-axis TQC checkpoint."""
from __future__ import annotations

import argparse

import numpy as np
from sb3_contrib import TQC

from furuta_env_2d import Furuta2DEnv


def evaluate(
    model: TQC,
    episodes: int,
    seed0: int,
    tilt_deg: float,
    axis: str,
    speed_deg: float = 60.0,
    accel_deg: float | None = None,
    amp_min_fraction: float = 0.3,
) -> dict:
    env = Furuta2DEnv(randomize=False, max_seconds=10.0)
    env.init_angle_max = np.pi
    env.init_vel_assist = 0.0
    env.tilt_amp = np.deg2rad(tilt_deg)
    env.tilt_amp_min_fraction = amp_min_fraction
    env.tilt_axis_mode = axis
    env.tilt_speed_max = np.deg2rad(speed_deg)
    env.tilt_accel_max = np.deg2rad(
        accel_deg if accel_deg is not None else max(400.0, 10.0 * speed_deg)
    )
    successes = catches = cable_hits = 0
    returns = []
    max_arm, outside = [], []
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed0 + episode)
        terminated = truncated = False
        total = 0.0
        info = {}
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
        successes += int(info.get("is_success", False))
        catches += int(info.get("is_catch_success", False))
        cable_hits += int(info.get("cable_limit_hit", False))
        returns.append(total)
        max_arm.append(float(info.get("max_abs_arm_deg", np.nan)))
        outside.append(float(info.get("fraction_outside_success_arm_limit", 0.0)))
    env.close()
    return {
        "axis": axis,
        "tilt_deg": tilt_deg,
        "episodes": episodes,
        "successes": successes,
        "success_rate": successes / episodes,
        "catches": catches,
        "catch_rate": catches / episodes,
        "mean_return": float(np.mean(returns)),
        "cable_hits": cable_hits,
        "arm_max_mean": float(np.nanmean(max_arm)),
        "arm_max_p95": float(np.nanpercentile(max_arm, 95)),
        "arm_max": float(np.nanmax(max_arm)),
        "outside_success_margin": float(np.mean(outside)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("-n", "--episodes", type=int, default=100)
    parser.add_argument("--seed0", type=int, default=50000)
    parser.add_argument("--axis", choices=("pitch", "roll", "both"), default="both")
    parser.add_argument("--tilt-deg", type=float, default=15.0)
    parser.add_argument("--speed-deg", type=float, default=60.0)
    parser.add_argument("--accel-deg", type=float, default=None)
    parser.add_argument("--amp-min-fraction", type=float, default=0.3)
    args = parser.parse_args()
    model = TQC.load(args.model, device="cpu")
    result = evaluate(
        model,
        args.episodes,
        args.seed0,
        args.tilt_deg,
        args.axis,
        args.speed_deg,
        args.accel_deg,
        args.amp_min_fraction,
    )
    print(
        f"{result['axis']} +/-{result['tilt_deg']:g} deg: "
        f"success {result['successes']}/{result['episodes']} "
        f"({100*result['success_rate']:.1f}%), "
        f"catch {result['catches']}/{result['episodes']} "
        f"({100*result['catch_rate']:.1f}%), "
        f"return {result['mean_return']:.1f}, "
        f"arm p95/max {result['arm_max_p95']:.1f}/{result['arm_max']:.1f} deg, "
        f"cable hits {result['cable_hits']}, "
        f"outside success margin {100*result['outside_success_margin']:.1f}%"
    )


if __name__ == "__main__":
    main()
