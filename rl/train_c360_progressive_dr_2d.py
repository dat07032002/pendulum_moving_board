"""Progressive domain-randomization fine-tuning from the verified nominal policy."""
from __future__ import annotations

import argparse
import os

from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from furuta_env import DR_COMPONENTS
from retention_tqc import RetentionTQC
from train_c360_stationary_2d import StationaryCableEnv, StationarySafetyEval


HERE = os.path.dirname(__file__)
MECHANICAL = (
    "motor_gear",
    "arm_damping",
    "pole_damping",
    "arm_friction",
    "pole_friction",
    "pole_inertia",
)
DR_STAGES = (
    (0, 0.25, MECHANICAL, "mechanical_25"),
    (100_000, 0.50, MECHANICAL, "mechanical_50"),
    (200_000, 1.00, MECHANICAL, "mechanical_100"),
    (300_000, 1.00, MECHANICAL + ("obs_noise",), "mechanical_obs"),
    (400_000, 1.00, DR_COMPONENTS, "full_delay_1_2"),
)


class ProgressiveDRCableEnv(StationaryCableEnv):
    def __init__(self):
        super().__init__()
        self.randomize = True
        self.dr_probability = 1.0
        self.dr_scale = 0.25
        self.dr_components = MECHANICAL

    def set_dr_stage(self, scale, components):
        self.randomize = True
        self.dr_probability = 1.0
        self.dr_scale = float(scale)
        self.dr_components = tuple(components)


def make_env():
    def factory():
        return ProgressiveDRCableEnv()

    return factory


class ProgressiveDRCallback(BaseCallback):
    def __init__(self, save_dir):
        super().__init__()
        self.save_dir = save_dir
        self.stage = -1

    def _apply(self, stage):
        _, scale, components, name = DR_STAGES[stage]
        self.training_env.env_method("set_dr_stage", scale, components)
        self.model.save(os.path.join(self.save_dir, f"dr_stage_{stage}_start"))
        print(
            f"[progressive_dr] stage={stage} name={name} scale={scale:g} "
            f"components={','.join(components)}",
            flush=True,
        )
        self.stage = stage

    def _on_training_start(self):
        self._apply(0)

    def _on_step(self):
        next_stage = self.stage
        for index, (start, *_rest) in enumerate(DR_STAGES):
            if self.num_timesteps >= start:
                next_stage = index
        if next_stage != self.stage:
            self._apply(next_stage)
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--warmstart",
        default="models/stationary_c360_nominal_primary_s1/best_safe.zip",
    )
    parser.add_argument("--teacher-data", default="teacher_s1_safe_nominal_100k.npz")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=500_000)
    parser.add_argument("--nenv", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=25_000)
    parser.add_argument("--actor-lr", type=float, default=3e-6)
    parser.add_argument("--critic-lr", type=float, default=1e-4)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    output = os.path.join(HERE, "models", args.tag)
    os.makedirs(output, exist_ok=True)
    env = VecMonitor(
        SubprocVecEnv([make_env() for _ in range(args.nenv)]),
        info_keywords=("is_success", "is_catch_success", "training_profile"),
    )
    warmstart = (
        args.warmstart if os.path.isabs(args.warmstart)
        else os.path.join(HERE, args.warmstart)
    )
    teacher = (
        args.teacher_data if os.path.isabs(args.teacher_data)
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
        teacher_fraction=0.15,
        teacher_target_ratio=0.10,
        use_teacher=True,
    )
    dr_callback = ProgressiveDRCallback(output)
    # Nominal evaluation remains a retention guard; DR validation is performed separately.
    nominal_guard = StationarySafetyEval(output, eval_freq=50_000, early_stop=False)
    checkpoint = CheckpointCallback(
        save_freq=max(50_000 // args.nenv, 1),
        save_path=output,
        name_prefix="ckpt",
    )
    model.learn(
        total_timesteps=args.steps,
        callback=[dr_callback, nominal_guard, checkpoint],
        progress_bar=False,
    )
    model.save(os.path.join(output, "tqc_final"))
    env.close()
    print("done", flush=True)


if __name__ == "__main__":
    main()
