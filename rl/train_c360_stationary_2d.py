"""Stationary nominal-plant fine-tuning of the verified S1 policy.

The episode distribution never changes. Training and evaluation use the physical +/-360 deg hard
boundary and +/-330 deg success margin. Domain randomization is a later, separate experiment.
"""
from __future__ import annotations

import argparse
import os

import mujoco
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from furuta_env_2d import DT, Furuta2DEnv
from probe_capability_2d import run_condition
from retention_tqc import RetentionTQC
from tilt_2d import SmoothRandomTilt2D


HERE = os.path.dirname(__file__)


def dynamic_generator(env: Furuta2DEnv, angle_deg: float, speed_deg: float):
    return SmoothRandomTilt2D(
        angle_max=np.deg2rad(angle_deg),
        speed_max=np.deg2rad(speed_deg),
        accel_max=np.deg2rad(max(400.0, 10.0 * speed_deg)),
        dt=DT,
        seed=int(env.np_random.integers(0, 2**32 - 1)),
    )


class StationaryCableEnv(Furuta2DEnv):
    """Fixed mixture of deployment motion, retention, and cable-recovery states."""

    def __init__(self):
        super().__init__(randomize=False, max_seconds=10.0)
        self.init_angle_max = np.pi
        self._training_profile = "both_fast"

    def reset(self, *, seed=None, options=None):
        self.tilt_amp = np.deg2rad(10.0)
        obs, info = super().reset(seed=seed, options=options)
        draw = self.np_random.random()
        if draw < 0.35:
            self._training_profile = "both_fast"
            self.tilt_axis_mode = "both"
            self.tilt_gen_2d = dynamic_generator(
                self, 10.0, self.np_random.uniform(50.0, 60.0)
            )
        elif draw < 0.60:
            self._training_profile = "both_mixed"
            self.tilt_axis_mode = "both"
            self.tilt_gen_2d = dynamic_generator(
                self, 10.0, self.np_random.uniform(25.0, 50.0)
            )
        elif draw < 0.70:
            self._training_profile = "axis_retention"
            self.tilt_axis_mode = "roll" if self.np_random.random() < 0.5 else "pitch"
            self.tilt_gen_2d = dynamic_generator(
                self, 10.0, self.np_random.uniform(40.0, 60.0)
            )
        elif draw < 0.80:
            self._training_profile = "level"
            self.tilt_axis_mode = "both"
            self.tilt_gen_2d = None
        elif draw < 0.90:
            self._training_profile = "both_slow"
            self.tilt_axis_mode = "both"
            self.tilt_gen_2d = dynamic_generator(
                self, 10.0, self.np_random.uniform(25.0, 40.0)
            )
        else:
            self._training_profile = "cable_recovery"
            self.tilt_axis_mode = "both"
            self.tilt_gen_2d = dynamic_generator(
                self, 10.0, self.np_random.uniform(40.0, 60.0)
            )
            magnitude = np.deg2rad(self.np_random.uniform(220.0, 285.0))
            sign = 1.0 if self.np_random.random() < 0.5 else -1.0
            self.data.qpos[self.qadr_a] = sign * magnitude
            self.data.qvel[self.dadr_a] = self.np_random.uniform(-0.5, 0.5)
            mujoco.mj_forward(self.model, self.data)
            self._max_abs_phi = magnitude
            obs = self._obs()
        info["training_profile"] = self._training_profile
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        if terminated or truncated:
            info["training_profile"] = self._training_profile
        return obs, reward, terminated, truncated, info


def make_env():
    def factory():
        return StationaryCableEnv()

    return factory


def evaluate_metrics(model, axis, angle_deg, speed_deg, episodes, seed0):
    env = Furuta2DEnv(randomize=False, max_seconds=10.0)
    env.init_angle_max = np.pi
    successes = catches = cable_hits = 0
    outside, arm_max = [], []
    for ep in range(episodes):
        env.tilt_amp = np.deg2rad(angle_deg)
        env.tilt_axis_mode = axis
        obs, _ = env.reset(seed=seed0 + ep)
        if angle_deg <= 0.0:
            env.tilt_gen_2d = None
        else:
            env.tilt_gen_2d = dynamic_generator(env, angle_deg, speed_deg)
        terminated = truncated = False
        info = {}
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
        successes += int(info.get("is_success", False))
        catches += int(info.get("is_catch_success", False))
        cable_hits += int(info.get("cable_limit_hit", False))
        outside.append(float(info.get("fraction_outside_success_arm_limit", 0.0)))
        arm_max.append(float(info.get("max_abs_arm_deg", 0.0)))
    env.close()
    return {
        "success": successes / episodes,
        "catch": catches / episodes,
        "cable_hits": cable_hits,
        "outside": float(np.mean(outside)),
        "arm_p95": float(np.percentile(arm_max, 95)),
    }


class StationarySafetyEval(BaseCallback):
    def __init__(
        self, save_dir, eval_freq=50_000, consecutive_passes=2, early_stop=True
    ):
        super().__init__()
        self.save_dir = save_dir
        self.eval_freq = int(eval_freq)
        self.consecutive_passes = int(consecutive_passes)
        self.early_stop = bool(early_stop)
        self.last_eval = 0
        self.eval_round = 0
        self.pass_streak = 0
        self.best_safe_target = -1.0

    def _on_step(self):
        if self.num_timesteps - self.last_eval < self.eval_freq:
            return True
        self.last_eval = self.num_timesteps
        seed = 280_000 + 3_000 * self.eval_round
        self.eval_round += 1
        target = evaluate_metrics(self.model, "both", 10.0, 60.0, 60, seed)
        both45 = evaluate_metrics(self.model, "both", 10.0, 45.0, 30, seed + 500)
        pitch = evaluate_metrics(self.model, "pitch", 10.0, 60.0, 20, seed + 1000)
        level = evaluate_metrics(self.model, "both", 0.0, 0.0, 20, seed + 1500)
        total_hits = (
            target["cable_hits"] + both45["cable_hits"]
            + pitch["cable_hits"] + level["cable_hits"]
        )
        gamma = float(self.model.gamma)
        q = run_condition(self.model, "both", 10.0, 60.0, 5, seed + 5000, gamma)
        print(
            f"[stationary_eval] t={self.num_timesteps} target={target['success']:.2f} "
            f"catch={target['catch']:.2f} both45={both45['success']:.2f} "
            f"pitch60={pitch['success']:.2f} level={level['success']:.2f} "
            f"hits={total_hits} outside330={target['outside']:.3f} "
            f"armP95={target['arm_p95']:.1f} Q/RTG={q['q_mean']:.0f}/{q['rtg_mean']:.0f}",
            flush=True,
        )
        safe = total_hits == 0 and target["outside"] <= 0.02
        if safe and target["success"] > self.best_safe_target:
            self.best_safe_target = target["success"]
            self.model.save(os.path.join(self.save_dir, "best_safe"))
        passed = (
            target["success"] >= 0.85
            and both45["success"] >= 0.85
            and pitch["success"] >= 0.90
            and level["success"] >= 0.90
            and safe
        )
        self.pass_streak = self.pass_streak + 1 if passed else 0
        if passed:
            print(
                f"[stationary] gate pass {self.pass_streak}/{self.consecutive_passes}",
                flush=True,
            )
        if self.pass_streak >= self.consecutive_passes:
            self.model.save(os.path.join(self.save_dir, "eligible_model"))
            print("[stationary] two-pass deployment gate reached -> done", flush=True)
            return not self.early_stop
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--warmstart", default="models/c360_s0ft_s1/best_stage_3.zip"
    )
    parser.add_argument("--teacher-data", default="teacher_s1_safe_dynamic_100k.npz")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=400_000)
    parser.add_argument("--nenv", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=25_000)
    parser.add_argument("--actor-lr", type=float, default=5e-6)
    parser.add_argument("--critic-lr", type=float, default=1e-4)
    parser.add_argument("--rehearsal-fraction", type=float, default=0.15)
    parser.add_argument("--no-rehearsal", action="store_true")
    parser.add_argument("--eval-freq", type=int, default=50_000)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    output = os.path.join(HERE, "models", args.tag)
    os.makedirs(output, exist_ok=True)
    env = VecMonitor(
        SubprocVecEnv([make_env() for _ in range(args.nenv)]),
        info_keywords=("is_success", "is_catch_success", "training_profile"),
    )
    warmstart = (
        args.warmstart
        if os.path.isabs(args.warmstart)
        else os.path.join(HERE, args.warmstart)
    )
    teacher = (
        args.teacher_data
        if os.path.isabs(args.teacher_data)
        else os.path.join(HERE, args.teacher_data)
    )
    model = RetentionTQC.load(warmstart, env=env, device=args.device)
    model.verbose = 1
    model.learning_starts = 5_000
    model.batch_size = 512
    model.gradient_steps = max(4, args.nenv // 2)
    model.gamma = 0.99
    model.seed = args.seed
    model.set_random_seed(args.seed)
    model.configure_retention(
        teacher,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        actor_start_steps=args.warmup_steps,
        teacher_coef=1.0,
        teacher_fraction=args.rehearsal_fraction,
        teacher_target_ratio=0.10,
        use_teacher=not args.no_rehearsal,
    )
    print(
        f"[stationary] actor_lr={args.actor_lr:g} critic_lr={args.critic_lr:g} "
        f"warmup={args.warmup_steps} rehearsal={not args.no_rehearsal}",
        flush=True,
    )
    evaluator = StationarySafetyEval(output, eval_freq=args.eval_freq)
    checkpoint = CheckpointCallback(
        save_freq=max(50_000 // args.nenv, 1),
        save_path=output,
        name_prefix="ckpt",
    )
    model.learn(
        total_timesteps=args.steps,
        callback=[evaluator, checkpoint],
        progress_bar=False,
    )
    model.save(os.path.join(output, "tqc_final"))
    env.close()
    print("done", flush=True)


if __name__ == "__main__":
    main()
