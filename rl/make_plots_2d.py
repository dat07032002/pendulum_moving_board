"""Two representative figures for the best 2D model:
  1. Time trace of one successful episode: board roll/pitch vs pole angle from vertical.
  2. Capability envelope: sustained success vs board-tilt speed (both axes), with Wilson CI band.

Run at the model's training voltage via FURUTA_VMAX.
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sb3_contrib import TQC

from furuta_env_2d import DT, Furuta2DEnv

BOARD = "#c86e19"   # board disturbance
BOARD2 = "#e0a765"
POLE = "#1a5cb0"    # controlled variable
FILL = "#1a8c58"


def make_trace(model, out_prefix, angle_deg=15.0, speed=120.0, seed0=50000, max_try=8):
    """Record one SUCCESSFUL episode and plot board motion vs pole-from-vertical."""
    env = Furuta2DEnv(randomize=False, max_seconds=10.0)
    env.arm_limit = None
    env.arm_center_w = 0.0
    env.init_angle_max = 0.25  # start near upright: a clean 'balancing through motion' trace
    env.tilt_axis_mode = "both"
    env.tilt_amp = np.deg2rad(angle_deg)
    env.tilt_amp_min_fraction = 1.0
    env.tilt_speed_max = np.deg2rad(speed)
    env.tilt_accel_max = np.deg2rad(max(400.0, 10.0 * speed))
    chosen = None
    for k in range(max_try):
        obs, _ = env.reset(seed=seed0 + k)
        t, roll, pitch, polev = [], [], [], []
        term = trunc = False
        info = {}
        step = 0
        while not (term or trunc):
            a, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(a)
            t.append(step * DT)
            roll.append(np.rad2deg(env.data.qpos[env.qadr_r]))
            pitch.append(np.rad2deg(env.data.qpos[env.qadr_pt]))
            polev.append(np.rad2deg(np.arccos(np.clip(env._true_up(), -1, 1))))
            step += 1
        if info.get("is_success", False):
            chosen = (np.array(t), np.array(roll), np.array(pitch), np.array(polev), seed0 + k)
            break
    env.close()
    if chosen is None:
        raise RuntimeError("no successful episode found for the trace")
    t, roll, pitch, polev, sd = chosen

    fig, ax = plt.subplots(figsize=(8.8, 3.2))
    ax.plot(t, roll, color=BOARD, lw=1.4, label="board roll")
    ax.plot(t, pitch, color=BOARD2, lw=1.4, label="board pitch")
    ax.plot(t, polev, color=POLE, lw=2.2, label="pole angle from vertical")
    ax.axhline(0, color="#999", lw=0.6, zorder=0)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("angle (deg)")
    ax.set_title("Balancing through two-axis board motion", fontsize=12, weight="bold")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.25)
    ax.set_xlim(t[0], t[-1])
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_prefix}_trace.{ext}", dpi=160)
    plt.close(fig)
    print(f"trace: seed {sd}, pole-from-vertical mean {polev.mean():.1f} deg / "
          f"p95 {np.percentile(polev,95):.1f} deg")


def make_envelope(json_path, out_prefix):
    """Plot sustained success vs board speed (both axes) from a verify_2d JSON."""
    data = json.loads(open(json_path, encoding="utf-8").read())
    rows = data[0]["rows"] if isinstance(data, list) else data["rows"]
    pts = []
    for r in rows:
        if (r["group"] == "deploy" or r["label"] == "level"):
            pts.append((r["speed"], r["success_rate"], r["ci_lo"], r["ci_hi"]))
    pts.sort()
    sp = np.array([p[0] for p in pts])
    sr = np.array([p[1] for p in pts]) * 100
    lo = np.array([p[2] for p in pts]) * 100
    hi = np.array([p[3] for p in pts]) * 100

    fig, ax = plt.subplots(figsize=(7.8, 3.2))
    ax.fill_between(sp, lo, hi, color=FILL, alpha=0.18)
    ax.plot(sp, sr, "-o", color=FILL, lw=2.2, ms=6, label="sustained success")
    ax.set_xlabel("board tilt speed (deg/s)")
    ax.set_ylabel("sustained success (%)")
    ax.set_title("Capability envelope (two-axis ±15° board motion)",
                 fontsize=12, weight="bold")
    ax.set_ylim(0, 103)
    ax.set_xlim(-3, max(sp) + 3)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_prefix}_envelope.{ext}", dpi=160)
    plt.close(fig)
    print(f"envelope: speeds {sp.tolist()} -> success {sr.round(1).tolist()}")


def _both_envelope(json_path):
    data = json.loads(open(json_path, encoding="utf-8").read())
    rows = data[0]["rows"] if isinstance(data, list) else data["rows"]
    pts = sorted((r["speed"], r["success_rate"] * 100) for r in rows
                 if (r["group"] == "deploy" or r["label"] == "level"))
    return np.array([p[0] for p in pts]), np.array([p[1] for p in pts])


def make_seed_robustness(json_paths, out_prefix):
    """Overlay every seed's both-axis envelope to show the method's run-to-run consistency."""
    fig, ax = plt.subplots(figsize=(7.8, 3.2))
    curves = []
    sp = None
    for jp in json_paths:
        sp, sr = _both_envelope(jp)
        curves.append(sr)
        ax.plot(sp, sr, "-", color=FILL, lw=1.0, alpha=0.35)
    curves = np.array(curves)
    ax.fill_between(sp, curves.min(0), curves.max(0), color=FILL, alpha=0.15,
                    label="seed min-max")
    ax.plot(sp, curves.mean(0), "-o", color="#0d5e3a", lw=2.4, ms=6,
            label=f"mean of {len(json_paths)} seeds")
    ax.set_xlabel("board tilt speed (deg/s)")
    ax.set_ylabel("sustained success (%)")
    ax.set_title(f"Method robustness: {len(json_paths)} independent seeds",
                 fontsize=12, weight="bold")
    ax.set_ylim(0, 103)
    ax.set_xlim(-3, max(sp) + 3)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_prefix}_seeds.{ext}", dpi=160)
    plt.close(fig)
    spread = curves.max(0) - curves.min(0)
    print(f"seed robustness: {len(json_paths)} seeds, max spread {spread.max():.1f} pts "
          f"at {sp[spread.argmax()]:.0f} deg/s")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--json", required=True, help="verify_2d JSON for the best seed")
    p.add_argument("--seed-jsons", nargs="+", default=None,
                   help="verify JSONs for all seeds (for the robustness plot)")
    p.add_argument("--out", default="../figure_10V")
    args = p.parse_args()
    model = TQC.load(args.model, device="cpu")
    print(f"FURUTA_VMAX={os.environ.get('FURUTA_VMAX','6')}")
    make_trace(model, args.out)
    make_envelope(args.json, args.out)
    if args.seed_jsons:
        make_seed_robustness(args.seed_jsons, args.out)
    print(f"wrote {args.out}_trace / _envelope"
          f"{' / _seeds' if args.seed_jsons else ''} .[png/pdf]")


if __name__ == "__main__":
    main()
