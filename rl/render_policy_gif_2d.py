"""Render one simultaneous roll/pitch policy episode to a GIF."""
from __future__ import annotations

import argparse

import mujoco
import numpy as np
from PIL import Image
from sb3_contrib import TQC

from furuta_env_2d import Furuta2DEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("output")
    parser.add_argument("--seed", type=int, default=50000)
    parser.add_argument("--tilt-deg", type=float, default=15.0)
    parser.add_argument("--speed-deg", type=float, default=60.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--find-success", type=int, default=0,
                        help="try up to N consecutive seeds and render the first success")
    args = parser.parse_args()

    model = TQC.load(args.model, device="cpu")
    env = Furuta2DEnv(randomize=False, max_seconds=10.0)
    env.init_angle_max = np.pi
    env.init_vel_assist = 0.0
    env.tilt_amp = np.deg2rad(args.tilt_deg)
    env.tilt_amp_min_fraction = 1.0
    env.tilt_axis_mode = "both"
    env.tilt_speed_min = np.deg2rad(args.speed_deg)
    env.tilt_speed_max = np.deg2rad(args.speed_deg)
    render_seed = args.seed
    if args.find_success:
        for candidate in range(args.seed, args.seed + args.find_success):
            obs, _ = env.reset(seed=candidate)
            terminated = truncated = False
            info = {}
            while not (terminated or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = env.step(action)
            if info.get("is_success", False):
                render_seed = candidate
                print(f"selected successful seed {render_seed}", flush=True)
                break
        else:
            raise SystemExit(f"no successful rollout in {args.find_success} seeds")
    obs, _ = env.reset(seed=render_seed)

    renderer = mujoco.Renderer(env.model, height=480, width=640)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = (0.0, 0.0, 0.02)
    camera.distance = 0.72
    camera.azimuth = 135
    camera.elevation = -22

    frames = []
    frame_interval = max(1, int(round(200 / args.fps)))
    terminated = truncated = False
    info = {}
    step = 0
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        if step % frame_interval == 0:
            renderer.update_scene(env.data, camera=camera)
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
        f"wrote {args.output}: frames={len(frames)}, "
        f"success={bool(info.get('is_success', False))}"
    )


if __name__ == "__main__":
    main()
