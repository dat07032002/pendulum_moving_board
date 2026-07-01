"""Selection / verification harness for 2D candidates.

Maps the +/-10 deg target envelope (success vs speed, pitch/roll/both), checks retention
(level, static corners, slow +/-15 deg), and runs the critic-calibration gate, for one or more
checkpoints head-to-head with Wilson 95% intervals. Frozen policy; never trains.

Two-pass use:
  screening : --episodes 150  across all candidates -> rank quickly
  final     : --episodes 500  on the winner only -> publishable numbers

Example:
  python verify_2d.py models/clean20_master_2d_warmstart.zip \
      models/prod2d_v1_s0/best_stage_6.zip models/prod2d_v1_s1/best_stage_6.zip \
      --episodes 150 --out eval/verify_v1
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
from sb3_contrib import TQC

from furuta_env_2d import DT, Furuta2DEnv
from probe_capability_2d import run_condition
from tilt_2d import CornerHoldTilt2D

# label, axis, angle_deg, speed_deg, kind, group
GRID_PM10 = (
    ("level",        "both",  0.0,   0.0, "level",  "retention"),
    ("roll 120",     "roll",  10.0, 120.0, "cont",  "retention"),
    ("pitch 60",     "pitch", 10.0,  60.0, "cont",  "envelope"),
    ("pitch 80",     "pitch", 10.0,  80.0, "cont",  "envelope"),
    ("pitch 100",    "pitch", 10.0, 100.0, "cont",  "envelope"),
    ("pitch 120",    "pitch", 10.0, 120.0, "cont",  "envelope"),
    ("both 60",      "both",  10.0,  60.0, "cont",  "deploy"),
    ("both 80",      "both",  10.0,  80.0, "cont",  "deploy"),
    ("both 100",     "both",  10.0, 100.0, "cont",  "deploy"),
    ("both 120",     "both",  10.0, 120.0, "cont",  "deploy"),
    ("corners 10",   "both",  10.0,  40.0, "corner", "retention"),
    ("slow 15",      "both",  15.0,  60.0, "cont",  "retention"),
    ("corners 15",   "both",  15.0,  40.0, "corner", "retention"),
)

# +/-15 deg envelope grid (for the 11 V model), with +/-10 deg 120 rows for comparison.
GRID_PM15 = (
    ("level",        "both",  0.0,   0.0, "level",  "retention"),
    ("roll 120",     "roll",  15.0, 120.0, "cont",  "retention"),
    ("pitch 60",     "pitch", 15.0,  60.0, "cont",  "envelope"),
    ("pitch 80",     "pitch", 15.0,  80.0, "cont",  "envelope"),
    ("pitch 100",    "pitch", 15.0, 100.0, "cont",  "envelope"),
    ("pitch 120",    "pitch", 15.0, 120.0, "cont",  "envelope"),
    ("both 60",      "both",  15.0,  60.0, "cont",  "deploy"),
    ("both 80",      "both",  15.0,  80.0, "cont",  "deploy"),
    ("both 100",     "both",  15.0, 100.0, "cont",  "deploy"),
    ("both 120",     "both",  15.0, 120.0, "cont",  "deploy"),
    ("corners 15",   "both",  15.0,  40.0, "corner", "retention"),
    ("pitch10 120",  "pitch", 10.0, 120.0, "cont",  "compare"),
    ("both10 120",   "both",  10.0, 120.0, "cont",  "compare"),
)

GRIDS = {"pm10": GRID_PM10, "pm15": GRID_PM15}
GRID = GRID_PM10  # default

CORNER_SIGNS = ((1, 1), (1, -1), (-1, 1), (-1, -1))


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def eval_cell(model, axis, angle_deg, speed_deg, kind, episodes, seed0):
    env = Furuta2DEnv(randomize=False, max_seconds=10.0)
    env.arm_limit = None
    env.arm_center_w = 0.0
    env.init_angle_max = np.pi
    env.init_vel_assist = 0.0
    env.tilt_axis_mode = axis
    env.tilt_amp = np.deg2rad(angle_deg)
    env.tilt_amp_min_fraction = 1.0
    env.tilt_speed_max = np.deg2rad(speed_deg)
    env.tilt_accel_max = np.deg2rad(max(400.0, 10.0 * speed_deg))
    succ = catch = 0
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed0 + ep)
        if kind == "level":
            env.tilt_gen_2d = None
        elif kind == "corner":
            env.tilt_gen_2d = CornerHoldTilt2D(
                np.deg2rad(angle_deg), CORNER_SIGNS[ep % 4], DT
            )
        terminated = truncated = False
        info = {}
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
        succ += int(info.get("is_success", False))
        catch += int(info.get("is_catch_success", False))
    env.close()
    return succ, catch


def verify_model(path, episodes, seed0, grid=GRID_PM10):
    model = TQC.load(path, device="cpu")
    gamma = float(model.gamma)
    rows = []
    for label, axis, angle, speed, kind, group in grid:
        k, c = eval_cell(model, axis, angle, speed, kind, episodes, seed0)
        lo, hi = wilson(k, episodes)
        rows.append({
            "label": label, "group": group, "axis": axis, "angle": angle,
            "speed": speed, "successes": k, "catches": c, "episodes": episodes,
            "success_rate": k / episodes, "ci_lo": lo, "ci_hi": hi,
            "catch_rate": c / episodes,
        })
    # calibration gate (Q vs return-to-go)
    calib = {}
    for ax, an, sp in (("both", 0.0, 0.0), ("pitch", 10.0, 120.0), ("both", 10.0, 120.0),
                       ("pitch", 15.0, 120.0), ("both", 15.0, 120.0)):
        r = run_condition(model, ax, an, sp, max(10, episodes // 10), seed0 + 5000, gamma)
        calib[f"{ax}{an:g}/{sp:g}"] = {"q": r["q_mean"], "rtg": r["rtg_mean"]}
    return {"model": path, "gamma": gamma, "rows": rows, "calibration": calib}


def print_report(result):
    print(f"\n=== {result['model']}  (gamma={result['gamma']:g}) ===")
    print(f"{'condition':<13}{'grp':<10}{'succ':>10}{'  95% CI':>16}{'catch':>8}")
    for r in result["rows"]:
        ci = f"[{100*r['ci_lo']:.0f}-{100*r['ci_hi']:.0f}%]"
        print(f"{r['label']:<13}{r['group']:<10}"
              f"{r['successes']:>4}/{r['episodes']:<4}{100*r['success_rate']:>5.1f}%"
              f"{ci:>16}{100*r['catch_rate']:>7.1f}%")
    cal = " | ".join(f"{k}:Q={v['q']:.0f}/RTG={v['rtg']:.0f}" for k, v in result["calibration"].items())
    print(f"calibration: {cal}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("models", nargs="+")
    p.add_argument("-n", "--episodes", type=int, default=150)
    p.add_argument("--seed0", type=int, default=120000)
    p.add_argument("--grid", choices=tuple(GRIDS), default="pm10")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    grid = GRIDS[args.grid]
    results = []
    for path in args.models:
        res = verify_model(path, args.episodes, args.seed0, grid=grid)
        print_report(res)
        results.append(res)

    # head-to-head comparison matrix (success %) for the decision-relevant rows
    print("\n=== comparison (success %) ===")
    labels = [r["label"] for r in results[0]["rows"]]
    names = [os.path.basename(os.path.dirname(m)) or os.path.basename(m) for m in args.models]
    print(f"{'condition':<13}" + "".join(f"{n[:14]:>15}" for n in names))
    for i, lab in enumerate(labels):
        cells = "".join(f"{100*res['rows'][i]['success_rate']:>14.1f}%" for res in results)
        print(f"{lab:<13}{cells}")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.with_suffix(".json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nsaved {out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
