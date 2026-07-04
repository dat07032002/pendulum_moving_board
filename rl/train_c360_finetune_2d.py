"""Targeted cable-aware fine-tuning for the +/-10 deg, 60 deg/s deployment task.

This run starts from a critic already trained on the corrected plant. It deliberately keeps
that critic, freezes the actor briefly while a fresh replay buffer fills, and advances only
after the current stage is mastered. There is no timeout-based advancement.
"""
from __future__ import annotations

import argparse
import os

import mujoco
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from probe_capability_2d import run_condition
from retention_tqc import RetentionTQC
from train_tqc_2d import CurriculumMix2DEnv, evaluate


HERE = os.path.dirname(__file__)

# kind, angle [deg], speed low/high [deg/s], mastery threshold
STAGES_C360 = (
    ("both", 5.0, 25.0, 40.0, 0.85),
    ("both", 7.5, 30.0, 45.0, 0.80),
    ("both", 10.0, 35.0, 45.0, 0.80),
    ("both", 10.0, 45.0, 55.0, 0.80),
    ("both", 10.0, 50.0, 60.0, 0.78),
    ("both", 10.0, 25.0, 60.0, 0.80),
)


class Cable360FineTuneEnv(CurriculumMix2DEnv):
    """Targeted mixture: current 50%, level 15%, axis 15%, slow-both 10%, recovery 10%."""

    def __init__(self):
        super().__init__()
        self.curriculum_kind = "both"

    def set_training_stage(self, kind, angle, speed_lo, speed_hi):
        self.curriculum_kind = kind
        self.curriculum_angle_deg = float(angle)
        self.curriculum_speed_lo = float(speed_lo)
        self.curriculum_speed_hi = float(speed_hi)

    def _set_current_motion(self):
        speed = self.np_random.uniform(
            self.curriculum_speed_lo, self.curriculum_speed_hi
        )
        self.tilt_axis_mode = "both"
        self.tilt_gen_2d = self._generator(self.curriculum_angle_deg, speed)

    def reset(self, *, seed=None, options=None):
        # Call the physical env reset directly; this class supplies its own profile mixture.
        self.tilt_amp = np.deg2rad(self.curriculum_angle_deg)
        obs, info = super(CurriculumMix2DEnv, self).reset(seed=seed, options=options)
        draw = self.np_random.random()
        if draw < 0.50:
            self._training_profile = "current"
            self._set_current_motion()
        elif draw < 0.65:
            self._training_profile = "level"
            self.tilt_axis_mode = "both"
            self.tilt_gen_2d = None
        elif draw < 0.80:
            self._training_profile = "axis_retention"
            self.tilt_axis_mode = "roll" if self.np_random.random() < 0.5 else "pitch"
            self.tilt_gen_2d = self._generator(10.0, 60.0)
        elif draw < 0.90:
            self._training_profile = "slow_both_retention"
            self.tilt_axis_mode = "both"
            self.tilt_gen_2d = self._generator(10.0, 40.0)
        else:
            self._training_profile = "cable_recovery"
            self._set_current_motion()
            magnitude = np.deg2rad(self.np_random.uniform(240.0, 300.0))
            self.data.qpos[self.qadr_a] = magnitude * (
                1.0 if self.np_random.random() < 0.5 else -1.0
            )
            self.data.qvel[self.dadr_a] = 0.0
            mujoco.mj_forward(self.model, self.data)
            self._max_abs_phi = abs(float(self.data.qpos[self.qadr_a]))
            obs = self._obs()
        info["training_profile"] = self._training_profile
        return obs, info


def make_env():
    def factory():
        return Cable360FineTuneEnv()

    return factory


class MasteryCurriculum(BaseCallback):
    def __init__(
        self,
        save_dir,
        eval_freq=50_000,
        n_target=60,
        n_guard=20,
        consecutive_passes=2,
    ):
        super().__init__()
        self.save_dir = save_dir
        self.eval_freq = int(eval_freq)
        self.n_target = int(n_target)
        self.n_guard = int(n_guard)
        self.consecutive_passes = int(consecutive_passes)
        self.stage = 0
        self.last_eval = 0
        self.best_target = -1.0
        self.pass_streak = 0
        self.eval_round = 0

    def _apply(self):
        kind, angle, lo, hi, threshold = STAGES_C360[self.stage]
        self.training_env.env_method("set_training_stage", kind, angle, lo, hi)
        print(
            f"[c360] stage={self.stage} kind={kind} angle={angle:g} "
            f"speed={lo:g}-{hi:g} mastery={threshold:.2f}",
            flush=True,
        )

    def _on_training_start(self):
        self._apply()

    def _on_step(self):
        if self.num_timesteps - self.last_eval < self.eval_freq:
            return True
        self.last_eval = self.num_timesteps
        kind, angle, _, speed_hi, threshold = STAGES_C360[self.stage]
        frozen = self.num_timesteps < self.model.actor_start_steps
        seed = 180_000 + 20_000 * self.stage + 2_000 * self.eval_round
        self.eval_round += 1
        target = evaluate(self.model, "both", angle, speed_hi, self.n_target, seed)
        level = evaluate(self.model, "both", 0.0, 0.0, self.n_guard, seed + 300)
        pitch = evaluate(self.model, "pitch", 10.0, 60.0, self.n_guard, seed + 600)
        both60 = evaluate(self.model, "both", 10.0, 60.0, self.n_guard, seed + 900)
        both45 = evaluate(self.model, "both", 10.0, 45.0, self.n_guard, seed + 1200)
        gamma = float(self.model.gamma)
        q = run_condition(self.model, "both", 10.0, 60.0, 5, seed + 5000, gamma)
        print(
            f"[c360_eval] t={self.num_timesteps} stage={self.stage} "
            f"{'(warmup)' if frozen else ''} target={target:.2f} level={level:.2f} "
            f"pitch60={pitch:.2f} both45={both45:.2f} both60={both60:.2f} "
            f"Q/RTG={q['q_mean']:.0f}/{q['rtg_mean']:.0f}",
            flush=True,
        )
        if frozen:
            return True
        if target > self.best_target:
            self.best_target = target
            self.model.save(os.path.join(self.save_dir, f"best_stage_{self.stage}"))
        if target < threshold:
            self.pass_streak = 0
            return True
        self.pass_streak += 1
        print(
            f"[c360] mastery pass {self.pass_streak}/{self.consecutive_passes} "
            f"target={target:.2f}",
            flush=True,
        )
        if self.pass_streak < self.consecutive_passes:
            return True
        if self.stage == len(STAGES_C360) - 1:
            self.model.save(os.path.join(self.save_dir, "final_model"))
            print(f"[c360] final mastery target={target:.2f} -> done", flush=True)
            return False
        print(f"[c360] stage {self.stage} mastered target={target:.2f}", flush=True)
        self.stage += 1
        self.best_target = -1.0
        self.pass_streak = 0
        self.model.ep_info_buffer.clear()
        self._apply()
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--warmstart",
        default="models/pm10_60_up10_cable_s0/best_stage_5.zip",
    )
    parser.add_argument("--tag", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=800_000)
    parser.add_argument("--nenv", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=25_000)
    parser.add_argument("--actor-lr", type=float, default=1e-5)
    parser.add_argument("--critic-lr", type=float, default=1e-4)
    parser.add_argument("--eval-freq", type=int, default=50_000)
    parser.add_argument("--eval-target-episodes", type=int, default=60)
    parser.add_argument("--eval-guard-episodes", type=int, default=20)
    parser.add_argument("--consecutive-passes", type=int, default=2)
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
    model = RetentionTQC.load(warmstart, env=env, device=args.device)
    model.verbose = 1
    model.learning_starts = 5_000
    model.batch_size = 512
    model.gradient_steps = max(4, args.nenv // 2)
    model.gamma = 0.99
    model.seed = args.seed
    model.set_random_seed(args.seed)
    # Deliberately retain the corrected-plant critic and its optimizer state.
    model.configure_retention(
        "",
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        actor_start_steps=args.warmup_steps,
        teacher_coef=1.0,
        teacher_fraction=0.25,
        teacher_target_ratio=0.20,
        use_teacher=False,
    )
    print(
        f"[c360] retained critic warmup={args.warmup_steps} "
        f"actor_lr={args.actor_lr:g} critic_lr={args.critic_lr:g}",
        flush=True,
    )
    curriculum = MasteryCurriculum(
        output,
        eval_freq=args.eval_freq,
        n_target=args.eval_target_episodes,
        n_guard=args.eval_guard_episodes,
        consecutive_passes=args.consecutive_passes,
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
