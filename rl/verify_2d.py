"""Selection / verification harness for 2D candidates.

Maps the selected target envelope (default: +/-10 deg, 60 deg/s), checks dynamic retention,
arm/cable behavior, and critic calibration for one or more
checkpoints head-to-head with Wilson 95% intervals. Frozen policy; never trains.

Two-pass use:
  screening : --episodes 300  across all candidates -> rank with useful confidence
  final     : --episodes 500  on the winner only -> publishable numbers

Example:
  python verify_2d.py models/clean20_master_2d_warmstart.zip \
      models/prod2d_v1_s0/best_stage_6.zip models/prod2d_v1_s1/best_stage_6.zip \
      --episodes 300 --out eval/verify_v1
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
GRID_PM10_60 = (
    ("level",        "both",  0.0,  0.0, "level",  "retention"),
    ("roll 60",      "roll", 10.0, 60.0, "cont",   "retention"),
    ("pitch 30",     "pitch", 10.0, 30.0, "cont",  "envelope"),
    ("pitch 45",     "pitch", 10.0, 45.0, "cont",  "envelope"),
    ("pitch 60",     "pitch", 10.0, 60.0, "cont",  "envelope"),
    ("both 30",      "both", 10.0, 30.0, "cont",   "deploy"),
    ("both 45",      "both", 10.0, 45.0, "cont",   "deploy"),
    ("both 60",      "both", 10.0, 60.0, "cont",   "deploy"),
)

GRID_PM10 = (
    ("level",        "both",  0.0,   0.0, "level",  "retention"),
    ("roll 90",      "roll",  10.0,  90.0, "cont",  "retention"),
    ("pitch 60",     "pitch", 10.0,  60.0, "cont",  "envelope"),
    ("pitch 75",     "pitch", 10.0,  75.0, "cont",  "envelope"),
    ("pitch 90",     "pitch", 10.0,  90.0, "cont",  "envelope"),
    ("both 60",      "both",  10.0,  60.0, "cont",  "deploy"),
    ("both 75",      "both",  10.0,  75.0, "cont",  "deploy"),
    ("both 90",      "both",  10.0,  90.0, "cont",  "deploy"),
    ("corners 10",   "both",  10.0,  40.0, "corner", "retention"),
    ("slow 15",      "both",  15.0,  60.0, "cont",  "retention"),
    ("corners 15",   "both",  15.0,  40.0, "corner", "retention"),
)

# Legacy +/-15 deg / 90 deg/s grid retained for historical checkpoint comparisons.
GRID_PM15 = (
    ("level",        "both",  0.0,   0.0, "level",  "retention"),
    ("roll 90",      "roll",  15.0,  90.0, "cont",  "retention"),
    ("pitch 60",     "pitch", 15.0,  60.0, "cont",  "envelope"),
    ("pitch 75",     "pitch", 15.0,  75.0, "cont",  "envelope"),
    ("pitch 90",     "pitch", 15.0,  90.0, "cont",  "envelope"),
    ("both 60",      "both",  15.0,  60.0, "cont",  "deploy"),
    ("both 75",      "both",  15.0,  75.0, "cont",  "deploy"),
    ("both 90",      "both",  15.0,  90.0, "cont",  "deploy"),
    ("corners 15",   "both",  15.0,  40.0, "corner", "retention"),
    ("pitch10 90",   "pitch", 10.0,  90.0, "cont",  "compare"),
    ("both10 90",    "both",  10.0,  90.0, "cont",  "compare"),
)

GRIDS = {"pm10_60": GRID_PM10_60, "pm10": GRID_PM10, "pm15": GRID_PM15}
GRID = GRID_PM10_60  # default

CORNER_SIGNS = ((1, 1), (1, -1), (-1, 1), (-1, -1))


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def eval_cell(
    model, axis, angle_deg, speed_deg, kind, episodes, seed0,
    *, randomize=False, delay_steps=1,
):
    env = Furuta2DEnv(randomize=randomize, max_seconds=10.0)
    env.init_angle_max = np.pi
    env.init_vel_assist = 0.0
    env.tilt_axis_mode = axis
    env.tilt_amp = np.deg2rad(angle_deg)
    env.tilt_amp_min_fraction = 1.0
    env.tilt_speed_max = np.deg2rad(speed_deg)
    env.tilt_accel_max = np.deg2rad(max(400.0, 10.0 * speed_deg))
    succ = catch = cable_hits = 0
    max_arm, outside = [], []
    # tight-hold occupancy and smoothness over upright steps (up > cos(10 deg))
    up_steps = occ7_steps = occ5_steps = 0
    abs_da_sum = 0.0
    abs_da_n = 0
    up10, up7, up5 = np.cos(np.deg2rad([10.0, 7.0, 5.0]))
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed0 + ep)
        # FurutaEnv uses a FIFO whose length is the control-step delay. Override after reset so
        # nominal and DR evaluations can be compared at an exact latency.
        env._delay = int(delay_steps)
        env.act_buf = [0.0] * env._delay
        if kind == "level":
            env.tilt_gen_2d = None
        elif kind == "corner":
            env.tilt_gen_2d = CornerHoldTilt2D(
                np.deg2rad(angle_deg), CORNER_SIGNS[ep % 4], DT
            )
        terminated = truncated = False
        info = {}
        prev_a = None
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            up = env._true_up()
            if up > up10:
                up_steps += 1
                occ7_steps += int(up > up7)
                occ5_steps += int(up > up5)
                if prev_a is not None:
                    abs_da_sum += abs(float(action[0]) - prev_a)
                    abs_da_n += 1
            prev_a = float(action[0])
        succ += int(info.get("is_success", False))
        catch += int(info.get("is_catch_success", False))
        cable_hits += int(info.get("cable_limit_hit", False))
        max_arm.append(float(info.get("max_abs_arm_deg", np.nan)))
        outside.append(float(info.get("fraction_outside_success_arm_limit", 0.0)))
    env.close()
    return {
        "successes": succ,
        "catches": catch,
        "cable_hits": cable_hits,
        "arm_max_mean": float(np.nanmean(max_arm)),
        "arm_max_p95": float(np.nanpercentile(max_arm, 95)),
        "arm_max": float(np.nanmax(max_arm)),
        "outside_success_margin": float(np.mean(outside)),
        "occ7": occ7_steps / up_steps if up_steps else float("nan"),
        "occ5": occ5_steps / up_steps if up_steps else float("nan"),
        "mean_abs_da": abs_da_sum / abs_da_n if abs_da_n else float("nan"),
    }


def verify_model(
    path, episodes, seed0, grid=GRID_PM10_60, *, randomize=False, delay_steps=1
):
    model = TQC.load(path, device="cpu")
    gamma = float(model.gamma)
    rows = []
    for label, axis, angle, speed, kind, group in grid:
        metrics = eval_cell(
            model, axis, angle, speed, kind, episodes, seed0,
            randomize=randomize, delay_steps=delay_steps,
        )
        k, c = metrics["successes"], metrics["catches"]
        lo, hi = wilson(k, episodes)
        rows.append({
            "label": label, "group": group, "axis": axis, "angle": angle,
            "speed": speed, "successes": k, "catches": c, "episodes": episodes,
            "success_rate": k / episodes, "ci_lo": lo, "ci_hi": hi,
            "catch_rate": c / episodes,
            **metrics,
        })
    # The existing calibration probe is a nominal one-step environment. Do not label it as
    # calibration for DR or forced-delay evaluations, where it would describe a different MDP.
    calib = {}
    if not randomize and delay_steps == 1:
        angle_cap = max(row[2] for row in grid)
        speed_cap = max(row[3] for row in grid)
        for ax, an, sp in (
            ("both", 0.0, 0.0),
            ("pitch", angle_cap, speed_cap),
            ("both", angle_cap, speed_cap),
        ):
            r = run_condition(model, ax, an, sp, max(10, episodes // 10), seed0 + 5000, gamma)
            calib[f"{ax}{an:g}/{sp:g}"] = {"q": r["q_mean"], "rtg": r["rtg_mean"]}
    return {
        "model": path,
        "gamma": gamma,
        "randomize": bool(randomize),
        "delay_steps": int(delay_steps),
        "rows": rows,
        "calibration": calib,
    }


def print_report(result):
    plant = "DR" if result.get("randomize", False) else "nominal"
    delay = result.get("delay_steps", 1)
    print(
        f"\n=== {result['model']}  (gamma={result['gamma']:g}, "
        f"plant={plant}, delay={delay} step) ==="
    )
    print(f"{'condition':<13}{'grp':<10}{'succ':>10}{'  95% CI':>16}"
          f"{'catch':>8}{'armP95':>9}{'cable':>7}{'occ7':>7}{'occ5':>7}{'m|da|':>8}")
    for r in result["rows"]:
        ci = f"[{100*r['ci_lo']:.0f}-{100*r['ci_hi']:.0f}%]"
        print(f"{r['label']:<13}{r['group']:<10}"
              f"{r['successes']:>4}/{r['episodes']:<4}{100*r['success_rate']:>5.1f}%"
              f"{ci:>16}{100*r['catch_rate']:>7.1f}%"
              f"{r['arm_max_p95']:>8.1f}°{r['cable_hits']:>7}"
              f"{100*r['occ7']:>6.0f}%{100*r['occ5']:>6.0f}%{r['mean_abs_da']:>8.3f}")
    if result["calibration"]:
        cal = " | ".join(
            f"{k}:Q={v['q']:.0f}/RTG={v['rtg']:.0f}"
            for k, v in result["calibration"].items()
        )
        print(f"calibration: {cal}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("models", nargs="+")
    p.add_argument("-n", "--episodes", type=int, default=300)
    p.add_argument("--seed0", type=int, default=120000)
    p.add_argument("--grid", choices=tuple(GRIDS), default="pm10_60")
    p.add_argument("--delay-steps", type=int, choices=(1, 2), default=1)
    p.add_argument("--randomize", action="store_true", help="evaluate the full randomized plant")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    grid = GRIDS[args.grid]
    results = []
    for path in args.models:
        res = verify_model(
            path,
            args.episodes,
            args.seed0,
            grid=grid,
            randomize=args.randomize,
            delay_steps=args.delay_steps,
        )
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
