"""Collect successful, cable-conservative S1 transitions for policy rehearsal."""
from __future__ import annotations

import argparse

import numpy as np
from sb3_contrib import TQC

from furuta_env_2d import Furuta2DEnv
from train_c360_stationary_2d import dynamic_generator


PROFILES = (
    ("level", "both", 0.0, 0.0),
    ("roll60", "roll", 10.0, 60.0),
    ("pitch60", "pitch", 10.0, 60.0),
    ("both30", "both", 10.0, 30.0),
    ("both45", "both", 10.0, 45.0),
    ("both60", "both", 10.0, 60.0),
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("output")
    parser.add_argument("--transitions", type=int, default=100_000)
    parser.add_argument("--seed0", type=int, default=370_000)
    parser.add_argument("--max-arm-deg", type=float, default=270.0)
    args = parser.parse_args()

    model = TQC.load(args.model, device="cpu")
    env = Furuta2DEnv(randomize=False, max_seconds=10.0)
    env.init_angle_max = np.pi
    target_per_profile = int(np.ceil(args.transitions / len(PROFILES)))
    accepted = {name: [] for name, *_ in PROFILES}
    accepted_steps = {name: 0 for name, *_ in PROFILES}
    episode = 0
    while min(accepted_steps.values()) < target_per_profile:
        name, axis, angle, speed = PROFILES[episode % len(PROFILES)]
        env.tilt_amp = np.deg2rad(angle)
        env.tilt_axis_mode = axis
        obs, _ = env.reset(seed=args.seed0 + episode)
        env.tilt_gen_2d = None if angle == 0 else dynamic_generator(env, angle, speed)
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
        if (
            info.get("is_success", False)
            and not info.get("cable_limit_hit", False)
            and info.get("max_abs_arm_deg", np.inf) <= args.max_arm_deg
            and accepted_steps[name] < target_per_profile
        ):
            accepted[name].append(trajectory)
            accepted_steps[name] += len(trajectory)
        episode += 1
        if episode % 20 == 0:
            print(
                f"episodes={episode} transitions={accepted_steps}",
                flush=True,
            )

    flat = []
    for name, *_ in PROFILES:
        profile_flat = [
            item for trajectory in accepted[name] for item in trajectory
        ][:target_per_profile]
        flat.extend(profile_flat)
    flat = flat[: args.transitions]
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
    print(
        f"saved {args.output}: {len(flat)} balanced transitions; "
        f"profile_steps={accepted_steps}; attempted_episodes={episode}"
    )


if __name__ == "__main__":
    main()
