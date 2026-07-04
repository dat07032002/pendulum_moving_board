"""Retention-protected TQC fine-tuning for continuous two-axis board motion."""
from __future__ import annotations

import argparse
import os

import numpy as np
from sb3_contrib import TQC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from furuta_env_2d import DT, Furuta2DEnv
from retention_tqc import RetentionTQC
from tilt_2d import CornerHoldTilt2D, SmoothRandomTilt2D


HERE = os.path.dirname(__file__)
# axis, angle deg, speed range deg/s
STAGES = (
    ("pitch", 5.0, 25.0, 40.0),
    ("pitch", 10.0, 40.0, 50.0),
    ("pitch", 10.0, 50.0, 60.0),
    ("both", 5.0, 25.0, 40.0),
    ("both", 10.0, 40.0, 50.0),
    ("both", 10.0, 50.0, 60.0),
    ("both", 10.0, 25.0, 60.0),
)


class CurriculumMix2DEnv(Furuta2DEnv):
    """50% current task plus fixed retention profiles on every curriculum stage."""

    def __init__(self):
        super().__init__(randomize=False, max_seconds=10.0)
        self.init_angle_max = np.pi
        self.curriculum_axis = "pitch"
        self.curriculum_angle_deg = 5.0
        self.curriculum_speed_lo = 25.0
        self.curriculum_speed_hi = 40.0
        self.retention_angle_deg = 10.0
        self.retention_speed_deg = 60.0
        self._training_profile = "current"

    def _generator(self, angle_deg, speed_deg):
        return SmoothRandomTilt2D(
            angle_max=np.deg2rad(angle_deg),
            speed_max=np.deg2rad(speed_deg),
            accel_max=np.deg2rad(max(400.0, 10.0 * speed_deg)),
            dt=DT,
            seed=int(self.np_random.integers(0, 2**32 - 1)),
        )

    def reset(self, *, seed=None, options=None):
        self.tilt_amp = np.deg2rad(self.curriculum_angle_deg)
        obs, info = super().reset(seed=seed, options=options)
        draw = self.np_random.random()
        if draw < 0.50:
            self._training_profile = "current"
            self.tilt_axis_mode = self.curriculum_axis
            speed = self.np_random.uniform(
                self.curriculum_speed_lo, self.curriculum_speed_hi
            )
            self.tilt_gen_2d = self._generator(self.curriculum_angle_deg, speed)
        elif draw < 0.65:
            self._training_profile = "level"
            self.tilt_axis_mode = "both"
            self.tilt_gen_2d = None
        elif draw < 0.80:
            self._training_profile = "roll_retention"
            self.tilt_axis_mode = "roll"
            self.tilt_gen_2d = self._generator(
                self.retention_angle_deg, self.retention_speed_deg
            )
        elif draw < 0.90:
            self._training_profile = "slow_both_retention"
            self.tilt_axis_mode = "both"
            self.tilt_gen_2d = self._generator(
                self.retention_angle_deg, min(40.0, self.retention_speed_deg)
            )
        else:
            self._training_profile = "corner_retention"
            signs = (
                1 if self.np_random.random() < 0.5 else -1,
                1 if self.np_random.random() < 0.5 else -1,
            )
            self.tilt_axis_mode = "both"
            self.tilt_gen_2d = CornerHoldTilt2D(
                np.deg2rad(self.retention_angle_deg), signs, DT
            )
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        if terminated or truncated:
            info["training_profile"] = self._training_profile
        return obs, reward, terminated, truncated, info


def make_env():
    def factory():
        return CurriculumMix2DEnv()

    return factory


def evaluate(model, axis, angle_deg, speed_deg, episodes, seed0, corner=False):
    env = Furuta2DEnv(randomize=False, max_seconds=10.0)
    env.init_angle_max = np.pi
    successes = 0
    for episode in range(episodes):
        env.tilt_amp = np.deg2rad(angle_deg)
        env.tilt_amp_min_fraction = 1.0
        env.tilt_axis_mode = axis
        env.tilt_speed_max = np.deg2rad(speed_deg)
        env.tilt_accel_max = np.deg2rad(max(400.0, 10.0 * speed_deg))
        obs, _ = env.reset(seed=seed0 + episode)
        if angle_deg == 0:
            env.tilt_gen_2d = None
        elif corner:
            signs = ((1, 1), (1, -1), (-1, 1), (-1, -1))[episode % 4]
            env.tilt_gen_2d = CornerHoldTilt2D(np.deg2rad(angle_deg), signs, DT)
        terminated = truncated = False
        info = {}
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
        successes += int(info.get("is_success", False))
    env.close()
    return successes / episodes


class CurriculumAndRetentionEval(BaseCallback):
    def __init__(self, save_dir, eval_freq=25_000, n_target=30, n_guard=20):
        super().__init__()
        self.save_dir = save_dir
        self.eval_freq = eval_freq
        self.n_target = n_target
        self.n_guard = n_guard
        self.stage = 0
        self.stage_start = 0
        self.last_eval = 0
        self.passes = 0
        self.best_target = -1.0
        self.angle_cap = max(stage[1] for stage in STAGES)
        self.speed_cap = max(stage[3] for stage in STAGES)

    def _apply(self):
        axis, angle, speed_lo, speed_hi = STAGES[self.stage]
        self.training_env.env_method(
            "set_params",
            curriculum_axis=axis,
            curriculum_angle_deg=angle,
            curriculum_speed_lo=speed_lo,
            curriculum_speed_hi=speed_hi,
            retention_angle_deg=self.angle_cap,
            retention_speed_deg=self.speed_cap,
        )
        print(
            f"[2d_curriculum] stage={self.stage} axis={axis} angle={angle:g} "
            f"speed={speed_lo:g}-{speed_hi:g}",
            flush=True,
        )

    def _on_training_start(self):
        self._apply()

    def _on_step(self):
        if self.num_timesteps - self.last_eval < self.eval_freq:
            return True
        self.last_eval = self.num_timesteps
        axis, angle, _, speed_hi = STAGES[self.stage]
        seed = 80000 + 1000 * self.stage
        target = evaluate(
            self.model, axis, angle, speed_hi, self.n_target, seed
        )
        level = evaluate(self.model, "both", 0.0, 0.0, self.n_guard, seed + 200)
        roll = evaluate(
            self.model, "roll", self.angle_cap, self.speed_cap,
            self.n_guard, seed + 400,
        )
        slow = evaluate(
            self.model, "both", self.angle_cap, min(40.0, self.speed_cap),
            self.n_guard, seed + 600,
        )
        corners = evaluate(
            self.model, "both", self.angle_cap, min(40.0, self.speed_cap),
            self.n_guard, seed + 800, corner=True
        )
        print(
            f"[2d_eval] t={self.num_timesteps} stage={self.stage} target={target:.2f} "
            f"level={level:.2f} roll={roll:.2f} slow={slow:.2f} corners={corners:.2f}",
            flush=True,
        )
        if min(level, roll) < 0.90 or slow < 0.85 or corners < 0.85:
            print("[retention_guard] failed -> stopping seed", flush=True)
            return False
        if target > self.best_target:
            self.best_target = target
            self.model.save(os.path.join(self.save_dir, f"best_stage_{self.stage}"))
        in_stage = self.num_timesteps - self.stage_start
        self.passes = self.passes + 1 if target >= 0.90 else 0
        if in_stage >= 50_000 and self.passes >= 2:
            if self.stage == len(STAGES) - 1:
                self.model.save(os.path.join(self.save_dir, "best_success_model"))
                print("[2d_curriculum] final stage passed -> stopping", flush=True)
                return False
            self.stage += 1
            self.stage_start = self.num_timesteps
            self.passes = 0
            self.best_target = -1.0
            self.model.ep_info_buffer.clear()
            self._apply()
        elif in_stage >= 250_000:
            print("[2d_curriculum] stage timeout -> stopping seed", flush=True)
            return False
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmstart", required=True)
    parser.add_argument("--teacher-data", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=1_750_000)
    parser.add_argument("--nenv", type=int, default=8)
    parser.add_argument("--eval-freq", type=int, default=25_000)
    parser.add_argument("--n-target", type=int, default=30)
    parser.add_argument("--n-guard", type=int, default=20)
    parser.add_argument("--actor-start", type=int, default=25_000)
    args = parser.parse_args()

    output = os.path.join(HERE, "models", args.tag)
    os.makedirs(output, exist_ok=True)
    env = VecMonitor(
        SubprocVecEnv([make_env() for _ in range(args.nenv)]),
        info_keywords=("is_success", "is_catch_success"),
    )
    model = RetentionTQC.load(args.warmstart, env=env, device="cuda")
    model.tensorboard_log = os.path.join(HERE, "tb", args.tag)
    model.verbose = 1
    model.learning_starts = 10_000
    model.batch_size = 512
    model.gradient_steps = max(4, args.nenv // 2)
    model.seed = args.seed
    model.set_random_seed(args.seed)
    model.configure_retention(
        args.teacher_data,
        actor_lr=1e-5,
        critic_lr=1e-4,
        actor_start_steps=args.actor_start,
        teacher_coef=1.0,
        teacher_fraction=0.25,
        teacher_target_ratio=0.20,
    )
    curriculum = CurriculumAndRetentionEval(
        output,
        eval_freq=args.eval_freq,
        n_target=args.n_target,
        n_guard=args.n_guard,
    )
    checkpoint = CheckpointCallback(
        save_freq=max(100_000 // args.nenv, 1),
        save_path=output,
        name_prefix="ckpt",
    )
    model.learn(
        total_timesteps=args.steps,
        callback=[curriculum, checkpoint],
        progress_bar=False,
    )
    model.save(os.path.join(output, "tqc_final"))
    env.close()
    print("done", flush=True)


if __name__ == "__main__":
    main()
