"""Collect successful frozen-policy transitions for 2D retention training."""
from __future__ import annotations

import argparse

import mujoco
import numpy as np
from sb3_contrib import TQC

from furuta_env_2d import DT, Furuta2DEnv
from tilt_2d import CornerHoldTilt2D, SmoothRandomTilt2D


def configure(env: Furuta2DEnv, profile: str, episode: int) -> None:
    env.tilt_amp_min_fraction = 1.0
    env.tilt_axis_mode = "both"
    if profile == "level":
        env.tilt_amp = 0.0
    else:
        env.tilt_amp = np.deg2rad(15.0)
    if profile == "roll":
        env.tilt_axis_mode = "roll"
        env.tilt_speed_max = np.deg2rad(60.0)
        env.tilt_accel_max = np.deg2rad(600.0)
    elif profile == "slow_both":
        env.tilt_speed_max = np.deg2rad(60.0)
        env.tilt_accel_max = np.deg2rad(400.0)
    elif profile not in ("level", "corners"):
        raise ValueError(profile)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("output")
    parser.add_argument("--transitions", type=int, default=100_000)
    parser.add_argument("--seed0", type=int, default=70000)
    args = parser.parse_args()

    model = TQC.load(args.model, device="cpu")
    env = Furuta2DEnv(randomize=False, max_seconds=10.0)
    env.arm_limit = None
    env.success_arm_limit = None
    env.arm_center_w = 0.0
    env.init_angle_max = np.pi
    profiles = ("level", "roll", "slow_both", "corners")
    corner_signs = ((1, 1), (1, -1), (-1, 1), (-1, -1))
    accepted = []
    episode = 0
    while sum(len(x) for x in accepted) < args.transitions:
        profile = profiles[episode % len(profiles)]
        configure(env, profile, episode)
        obs, _ = env.reset(seed=args.seed0 + episode)
        if profile == "corners":
            env.tilt_gen_2d = CornerHoldTilt2D(
                np.deg2rad(15.0), corner_signs[(episode // len(profiles)) % 4], DT
            )
        trajectory = []
        terminated = truncated = False
        info = {}
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            next_obs, reward, terminated, truncated, info = env.step(action)
            trajectory.append(
                (
                    obs.copy(),
                    np.asarray(action, dtype=np.float32).copy(),
                    next_obs.copy(),
                    np.float32(reward),
                    np.float32(terminated),
                )
            )
            obs = next_obs
        if info.get("is_success", False):
            accepted.append(trajectory)
        episode += 1
        if episode % 20 == 0:
            count = sum(len(x) for x in accepted)
            print(f"episodes={episode} accepted={len(accepted)} transitions={count}", flush=True)

    flat = [item for trajectory in accepted for item in trajectory][: args.transitions]
    observations, actions, next_observations, rewards, dones = zip(*flat)
    np.savez_compressed(
        args.output,
        observations=np.asarray(observations, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.float32),
        next_observations=np.asarray(next_observations, dtype=np.float32),
        rewards=np.asarray(rewards, dtype=np.float32).reshape(-1, 1),
        dones=np.asarray(dones, dtype=np.float32).reshape(-1, 1),
    )
    env.close()
    print(f"saved {args.output}: {len(flat)} transitions from {len(accepted)} episodes")


if __name__ == "__main__":
    main()
