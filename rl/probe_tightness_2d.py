"""Diagnose why a model can't hold a TIGHT upright angle: authority vs. training.

Runs the model on fast two-axis motion and, for each control step where the pole is still up,
pairs the pole's angle-from-vertical with the commanded action. The decisive number is the
action-saturation fraction *conditioned on the pole being outside the tight band*:
  high saturation when the pole is wide  -> motor is maxed out -> AUTHORITY-limited
  low saturation when the pole is wide   -> motor has headroom -> TRAINING/REWARD-limited
Run at the model's training voltage via FURUTA_VMAX.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
from sb3_contrib import TQC

from furuta_env_2d import Furuta2DEnv

SAT = 0.95


def run(model, episodes, seed0, angle_deg=15.0, speed=120.0):
    env = Furuta2DEnv(randomize=False, max_seconds=10.0)
    env.arm_limit = None
    env.arm_center_w = 0.0
    env.init_angle_max = 0.25  # start near upright: focus on HOLDING, not swing-up
    env.tilt_axis_mode = "both"
    env.tilt_amp = np.deg2rad(angle_deg)
    env.tilt_amp_min_fraction = 1.0
    env.tilt_speed_max = np.deg2rad(speed)
    env.tilt_accel_max = np.deg2rad(max(400.0, 10.0 * speed))
    angs, absa = [], []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed0 + ep)
        term = trunc = False
        while not (term or trunc):
            up_now = env._true_up()                      # state the action responds to
            a, _ = model.predict(obs, deterministic=True)
            if up_now > 0.0:                             # still up (not fallen)
                angs.append(np.rad2deg(np.arccos(np.clip(up_now, -1, 1))))
                absa.append(abs(float(a[0])))
            obs, _, term, trunc, info = env.step(a)
    env.close()
    return np.array(angs), np.array(absa)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("-n", "--episodes", type=int, default=60)
    p.add_argument("--seed0", type=int, default=180000)
    args = p.parse_args()
    model = TQC.load(args.model, device="cpu")
    angs, absa = run(model, args.episodes, args.seed0)
    sat = absa > SAT

    def fsat(mask):
        return float(sat[mask].mean()) if mask.sum() else float("nan")

    def mabs(mask):
        return float(absa[mask].mean()) if mask.sum() else float("nan")

    print(f"model={args.model}  FURUTA_VMAX={os.environ.get('FURUTA_VMAX','6')}  "
          f"steps(up)={len(angs)}")
    print(f"pole angle from vertical: mean {angs.mean():.1f} deg  p95 {np.percentile(angs,95):.1f}  "
          f"max {angs.max():.1f}")
    print(f"time within 10 deg: {(angs<10).mean()*100:.0f}%   within 15 deg: {(angs<15).mean()*100:.0f}%")
    print("-- action while the pole is wide (the diagnostic) --")
    for lo in (0, 8, 10, 12, 15):
        m = angs > lo
        print(f"  angle > {lo:>2} deg:  mean|a| {mabs(m):.2f}   saturated {100*fsat(m):.0f}%   "
              f"(n={int(m.sum())})")


if __name__ == "__main__":
    main()
