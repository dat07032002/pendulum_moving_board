"""Evaluate a 2D model under plant domain randomization.

All 2D training was clean-plant, so this measures robustness to the inherited plant DR
(motor gear, damping, friction, inertia, obs noise, action delay). It reports each condition
under three settings so the action-delay bottleneck (the 1D finding) is isolated:
  clean      - no DR
  full DR    - all components (gear/damping/friction/inertia/obs-noise/action-delay 1-2)
  DR no-delay- full DR except action delay (held at 1 step)

Set FURUTA_VMAX to match the model's training voltage (e.g. 11).
"""
from __future__ import annotations

import argparse
import math

import numpy as np
from sb3_contrib import TQC

from furuta_env import DR_COMPONENTS
from furuta_env_2d import Furuta2DEnv

NO_DELAY = tuple(c for c in DR_COMPONENTS if c != "action_delay")

CONDS = [
    ("level",           "both", 0.0,   0.0),
    ("both +/-10 60",   "both", 10.0, 60.0),
    ("pitch +/-10 60",  "pitch", 10.0, 60.0),
]
MODES = [
    ("clean",       False, None),
    ("full DR",     True,  None),
    ("DR no-delay", True,  NO_DELAY),
]


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def run(model, axis, angle, speed, randomize, dr_components, episodes, seed0):
    env = Furuta2DEnv(randomize=randomize, max_seconds=10.0)
    env.init_angle_max = np.pi
    env.dr_probability = 1.0
    env.dr_components = dr_components
    env.tilt_axis_mode = axis
    env.tilt_amp = np.deg2rad(angle)
    env.tilt_amp_min_fraction = 1.0
    env.tilt_speed_max = np.deg2rad(speed)
    env.tilt_accel_max = np.deg2rad(max(400.0, 10.0 * speed))
    succ = catch = 0
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed0 + ep)
        if angle == 0:
            env.tilt_gen_2d = None
        term = trunc = False
        info = {}
        while not (term or trunc):
            a, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(a)
        succ += int(info.get("is_success", False))
        catch += int(info.get("is_catch_success", False))
    env.close()
    return succ, catch


def main():
    import os
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("-n", "--episodes", type=int, default=200)
    p.add_argument("--seed0", type=int, default=150000)
    args = p.parse_args()
    model = TQC.load(args.model, device="cpu")
    print(f"model={args.model}  FURUTA_VMAX={os.environ.get('FURUTA_VMAX','6')}  n={args.episodes}")
    print(f"{'condition':<16}{'mode':<13}{'success':>12}{'  95% CI':>15}{'catch':>8}")
    for label, axis, angle, speed in CONDS:
        for mode, rnd, comp in MODES:
            k, c = run(model, axis, angle, speed, rnd, comp, args.episodes, args.seed0)
            lo, hi = wilson(k, args.episodes)
            print(f"{label:<16}{mode:<13}{k:>4}/{args.episodes:<4}"
                  f"{100*k/args.episodes:>5.1f}%{f'[{100*lo:.0f}-{100*hi:.0f}%]':>15}"
                  f"{100*c/args.episodes:>7.1f}%", flush=True)
        print()


if __name__ == "__main__":
    main()
