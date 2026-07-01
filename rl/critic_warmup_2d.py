"""Phase A: frozen-actor critic warmup + divergence instrumentation.

The 2D fine-tune stalled because the critic diverges to large-negative Q on good states
(see STEP5_CRITIC_DIAGNOSIS.md). This isolates the cause: keep the warm-start ACTOR frozen
(pure policy evaluation, no actor/critic feedback loop) and watch whether the critic converges
to the true discounted return-to-go or still drifts negative.

Two switches probe the two hypotheses:
  --reinit-critic : start the critic from fresh random weights (vs the OOD transferred critic).
                    If a fresh critic converges but the transferred one does not, the transfer
                    is unrecoverable and Phase B must re-init.
  the soft-value instrumentation in retention_tqc (train/soft_value_penalty, train/next_log_prob)
                    tests whether a low-entropy actor + fixed ent_coef is dragging the target down.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch.nn as nn
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from probe_capability_2d import run_condition
from retention_tqc import RetentionTQC
from train_tqc_2d import STAGES, CurriculumMix2DEnv

HERE = os.path.dirname(__file__)

PROBE_CONDS = [
    ("both", 0.0, 0.0),
    ("pitch", 10.0, 90.0),
    ("pitch", 15.0, 120.0),
]


def make_env():
    def factory():
        return Monitor(
            CurriculumMix2DEnv(),
            info_keywords=("is_success", "is_catch_success"),
        )

    return factory


def reinit_critic(model: RetentionTQC) -> None:
    for module in model.critic.modules():
        if isinstance(module, nn.Linear):
            module.reset_parameters()
    model.critic_target.load_state_dict(model.critic.state_dict())
    print("[warmup] critic re-initialized to fresh random weights", flush=True)


class CalibrationLogger(BaseCallback):
    def __init__(self, eval_freq=20_000, episodes=10, seed0=95000):
        super().__init__()
        self.eval_freq = eval_freq
        self.episodes = episodes
        self.seed0 = seed0
        self.last = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last < self.eval_freq:
            return True
        self.last = self.num_timesteps
        gamma = float(self.model.gamma)
        parts = []
        for axis, angle, speed in PROBE_CONDS:
            r = run_condition(
                self.model, axis, angle, speed, self.episodes, self.seed0, gamma
            )
            parts.append(
                f"{axis}{angle:g}/{speed:g}: Q={r['q_mean']:.0f} RTG={r['rtg_mean']:.0f}"
            )
        print(f"[calib] t={self.num_timesteps} " + " | ".join(parts), flush=True)
        return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--warmstart", default="models/clean20_master_2d_warmstart.zip")
    p.add_argument("--teacher-data", default="teacher_2d_retention_100k.npz")
    p.add_argument("--tag", required=True)
    p.add_argument("--stage", type=int, default=1)
    p.add_argument("--steps", type=int, default=120_000)
    p.add_argument("--nenv", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--reinit-critic", action="store_true")
    p.add_argument("--gamma", type=float, default=None,
                   help="override discount; smaller = shorter bootstrap, smaller/stabler targets")
    p.add_argument("--tqd", type=int, default=None,
                   help="override top_quantiles_to_drop_per_net (lower = less pessimism)")
    args = p.parse_args()

    output = os.path.join(HERE, "models", args.tag)
    os.makedirs(output, exist_ok=True)
    env = VecMonitor(
        SubprocVecEnv([make_env() for _ in range(args.nenv)]),
        info_keywords=("is_success", "is_catch_success"),
    )
    model = RetentionTQC.load(args.warmstart, env=env, device="cuda")
    model.verbose = 1
    model.learning_starts = 5_000
    model.batch_size = 512
    model.gradient_steps = max(4, args.nenv // 2)
    model.seed = args.seed
    model.set_random_seed(args.seed)
    if args.gamma is not None:
        model.gamma = args.gamma
    if args.tqd is not None:
        model.top_quantiles_to_drop_per_net = args.tqd
    if args.reinit_critic:
        reinit_critic(model)
    print(f"[warmup] gamma={model.gamma:g} tqd={model.top_quantiles_to_drop_per_net}", flush=True)
    # actor frozen for the whole run: pure policy evaluation of the warm-start actor
    model.configure_retention(
        os.path.join(HERE, args.teacher_data) if not os.path.isabs(args.teacher_data)
        else args.teacher_data,
        actor_lr=1e-5,
        critic_lr=1e-4,
        actor_start_steps=args.steps + 10_000,  # never reached -> actor never updates
        teacher_coef=1.0,
        teacher_fraction=0.25,
        teacher_target_ratio=0.20,
    )
    # pin the curriculum stage so the critic evaluates one fixed data distribution
    axis, angle, lo, hi = STAGES[args.stage]
    env.env_method(
        "set_params",
        curriculum_axis=axis,
        curriculum_angle_deg=angle,
        curriculum_speed_lo=lo,
        curriculum_speed_hi=hi,
    )
    print(f"[warmup] stage={args.stage} axis={axis} angle={angle:g} speed={lo:g}-{hi:g} "
          f"reinit_critic={args.reinit_critic}", flush=True)
    model.learn(
        total_timesteps=args.steps,
        callback=[CalibrationLogger()],
        progress_bar=False,
    )
    model.save(os.path.join(output, "critic_warmup_final"))
    env.close()
    print("done", flush=True)


if __name__ == "__main__":
    main()
