"""Render one deterministic Furuta policy episode to a GIF."""
from __future__ import annotations

import argparse
import os
import sys

import mujoco
import numpy as np
from PIL import Image
from sb3_contrib import TQC

sys.path.insert(0, os.path.dirname(__file__))
from furuta_env import DR_COMPONENTS, FurutaEnv  # noqa: E402
from residual_env import ResidualActionWrapper  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("output")
    ap.add_argument("--seed", type=int, default=40003)
    ap.add_argument("--tilt_deg", type=float, default=20.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--dr", action="store_true")
    ap.add_argument("--dr_components", default="all")
    ap.add_argument("--p_corner", type=float, default=0.10)
    ap.add_argument("--residual_base")
    ap.add_argument("--residual_scale", type=float, default=0.05)
    args = ap.parse_args()

    model = TQC.load(args.model, device="cpu")
    if args.dr_components == "all":
        dr_component_set = None
    elif args.dr_components == "none":
        dr_component_set = frozenset()
    else:
        dr_component_set = frozenset(
            part.strip() for part in args.dr_components.split(",") if part.strip()
        )
        unknown = dr_component_set.difference(DR_COMPONENTS)
        if not dr_component_set or unknown:
            ap.error(f"invalid --dr_components; unknown={','.join(sorted(unknown)) or 'none'}")

    base_env = FurutaEnv(randomize=args.dr)
    base_env.init_angle_max = np.pi
    base_env.tilt_amp = float(np.deg2rad(args.tilt_deg))
    base_env.p_corner = args.p_corner
    base_env.dr_components = dr_component_set
    base_env.arm_limit = None
    base_env.success_arm_limit = None
    env = (
        ResidualActionWrapper(base_env, args.residual_base, args.residual_scale)
        if args.residual_base
        else base_env
    )
    obs, _ = env.reset(seed=args.seed)

    renderer = mujoco.Renderer(base_env.model, height=480, width=640)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = (0.0, 0.0, 0.04)
    camera.distance = 0.38
    camera.azimuth = 135
    camera.elevation = -18

    frames = []
    frame_interval = max(1, int(round(200 / args.fps)))
    terminated = truncated = False
    step = 0
    info = {}
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        if step % frame_interval == 0:
            renderer.update_scene(base_env.data, camera=camera)
            frames.append(renderer.render().copy())
        step += 1

    renderer.close()
    env.close()
    images = [Image.fromarray(frame) for frame in frames]
    images[0].save(
        args.output,
        save_all=True,
        append_images=images[1:],
        duration=int(round(1000 / args.fps)),
        loop=0,
        optimize=False,
    )
    print(
        f"wrote {args.output}: seed={args.seed}, frames={len(frames)}, "
        f"sustained={bool(info.get('is_success', False))}"
    )


if __name__ == "__main__":
    main()
