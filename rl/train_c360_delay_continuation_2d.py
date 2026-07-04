"""Conservative 1-2 step action-delay continuation from the best progressive-DR policy."""
from __future__ import annotations

import argparse
import os

from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from critic_warmup_2d import reinit_critic
from retention_tqc import RetentionTQC
from train_c360_progressive_dr_2d import MECHANICAL
from train_c360_stationary_2d import StationaryCableEnv, StationarySafetyEval


HERE = os.path.dirname(__file__)
BASE_COMPONENTS = MECHANICAL + ("obs_noise",)
DELAY_STAGES = (
    (0, "fixed1", BASE_COMPONENTS, "critic_rebuild_delay1"),
    (50_000, "mixed12", BASE_COMPONENTS + ("action_delay",), "mixed_delay_1_2"),
    (125_000, "fixed2", BASE_COMPONENTS + ("action_delay",), "worst_case_delay_2"),
)


class DelayContinuationEnv(StationaryCableEnv):
    def __init__(self):
        super().__init__()
        self.randomize = True
        self.dr_probability = 1.0
        self.dr_scale = 1.0
        self.dr_components = BASE_COMPONENTS
        self.delay_mode = "fixed1"

    def set_delay_stage(self, mode, components):
        self.delay_mode = str(mode)
        self.dr_components = tuple(components)

    def _randomize(self):
        super()._randomize()
        if self.delay_mode == "fixed1":
            self._delay = 1
        elif self.delay_mode == "fixed2":
            self._delay = 2
        elif self.delay_mode != "mixed12":
            raise ValueError(f"unknown delay mode {self.delay_mode!r}")


def make_env():
    def factory():
        return DelayContinuationEnv()

    return factory


class DelaySchedule(BaseCallback):
    def __init__(self, save_dir):
        super().__init__()
        self.save_dir = save_dir
        self.stage = -1

    def _apply(self, stage):
        _, mode, components, name = DELAY_STAGES[stage]
        self.training_env.env_method("set_delay_stage", mode, components)
        self.model.save(os.path.join(self.save_dir, f"delay_stage_{stage}_start"))
        print(
            f"[delay_dr] stage={stage} name={name} mode={mode} "
            f"components={','.join(components)}",
            flush=True,
        )
        self.stage = stage

    def _on_training_start(self):
        self._apply(0)

    def _on_step(self):
        next_stage = self.stage
        for index, (start, *_rest) in enumerate(DELAY_STAGES):
            if self.num_timesteps >= start:
                next_stage = index
        if next_stage != self.stage:
            self._apply(next_stage)
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmstart", default="models/progressive_dr_s1/best_safe.zip")
    parser.add_argument("--teacher-data", default="teacher_s1_safe_nominal_100k.npz")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--nenv", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=50_000)
    parser.add_argument("--actor-lr", type=float, default=5e-7)
    parser.add_argument("--critic-lr", type=float, default=3e-5)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    output = os.path.join(HERE, "models", args.tag)
    os.makedirs(output, exist_ok=True)
    env = VecMonitor(
        SubprocVecEnv([make_env() for _ in range(args.nenv)]),
        info_keywords=("is_success", "is_catch_success", "training_profile"),
    )
    warmstart = args.warmstart if os.path.isabs(args.warmstart) else os.path.join(HERE, args.warmstart)
    teacher = (
        args.teacher_data
        if os.path.isabs(args.teacher_data)
        else os.path.join(HERE, args.teacher_data)
    )
    model = RetentionTQC.load(warmstart, env=env, device=args.device)
    reinit_critic(model)
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
    print(
        f"[delay_continuation] actor_lr={args.actor_lr:g} critic_lr={args.critic_lr:g} "
        f"actor_freeze={args.warmup_steps} steps={args.steps}",
        flush=True,
    )
    schedule = DelaySchedule(output)
    nominal_guard = StationarySafetyEval(output, eval_freq=25_000, early_stop=False)
    checkpoint = CheckpointCallback(
        save_freq=max(25_000 // args.nenv, 1),
        save_path=output,
        name_prefix="ckpt",
    )
    model.learn(
        total_timesteps=args.steps,
        callback=[schedule, nominal_guard, checkpoint],
        progress_bar=False,
    )
    model.save(os.path.join(output, "tqc_final"))
    env.close()
    print("done", flush=True)


if __name__ == "__main__":
    main()
