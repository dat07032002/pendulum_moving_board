"""Phase B: critic-reset, low-gamma, gentle-ladder co-training of the 2D warm start.

Built from the Step-5 diagnosis (STEP5_CRITIC_DIAGNOSIS.md) and the Phase-A warmups:
  - The warm-start ACTOR is good; the transferred CRITIC is the failure (OOD, diverges).
  - A re-initialized critic does honest policy evaluation and stays positive.
  - gamma=0.99 calibrates ~3x faster than 0.998 with the same stability.

Method:
  1. Keep warm-start actor; re-initialize the critic.
  2. gamma=0.99 (shorter, lower-variance targets); optional reduced quantile truncation.
  3. Short frozen-actor critic warmup (just until the critic is positive/shaped).
  4. Unfreeze actor; co-train on a GENTLE pitch-speed ladder (small speed steps so the
     policy never collapses and the critic always has a mostly-succeeding regime).
  5. SOFT, ADVANCING curriculum gate (advance on a soft threshold OR timeout -> advance,
     never kill the seed). Retention and Q-vs-RTG calibration are logged every eval.

The decisive question this answers: with a healthy critic, does fast-pitch success actually
climb above the frozen warm-start baseline (pitch15/120 ~0.15, pitch10/120 ~0.55)?
"""
from __future__ import annotations

import argparse
import os

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from critic_warmup_2d import reinit_critic
from probe_capability_2d import run_condition
from retention_tqc import RetentionTQC
from train_tqc_2d import CurriculumMix2DEnv, evaluate, make_env

HERE = os.path.dirname(__file__)

# Target envelope (decided 2026-06-30): +/-10 deg up to 120 deg/s, both axes. The +/-15 deg
# fast regime is at the +/-6 V feasibility edge (Phase B stalled at ~0.33 there), so it is NOT
# a training target; +/-15 deg robustness margin is preserved only via the retention guards
# (slow +/-15 deg and static +/-15 deg corners). Gentle speed increments keep success high so
# the critic always has a mostly-succeeding regime to anchor on (the 1D stability trick).
STAGES_B = (
    ("pitch", 10.0, 40.0, 60.0),
    ("pitch", 10.0, 60.0, 80.0),
    ("pitch", 10.0, 80.0, 100.0),
    ("pitch", 10.0, 100.0, 120.0),
    ("both", 10.0, 60.0, 90.0),
    ("both", 10.0, 90.0, 120.0),
    ("both", 10.0, 30.0, 120.0),
)

# +/-15 deg ladder for the higher-authority (11 V) experiment: does extra torque reclaim the
# +/-15 deg fast-pitch regime that was infeasible at 6 V?
STAGES_PM15 = (
    ("pitch", 10.0, 40.0, 60.0),
    ("pitch", 15.0, 60.0, 85.0),
    ("pitch", 15.0, 85.0, 100.0),
    ("pitch", 15.0, 100.0, 120.0),
    ("both", 15.0, 60.0, 90.0),
    ("both", 15.0, 90.0, 120.0),
    ("both", 15.0, 30.0, 120.0),
)

LADDERS = {"pm10": STAGES_B, "pm15": STAGES_PM15}

CALIB_CONDS = [("both", 0.0, 0.0), ("pitch", 10.0, 120.0), ("pitch", 15.0, 120.0)]


class PhaseBCurriculum(BaseCallback):
    def __init__(self, save_dir, stages=STAGES_B, eval_freq=25_000, n_target=30, n_guard=20,
                 soft_thresh=0.78, stage_timeout=200_000):
        super().__init__()
        self.stages = stages
        self.save_dir = save_dir
        self.eval_freq = eval_freq
        self.n_target = n_target
        self.n_guard = n_guard
        self.soft_thresh = soft_thresh
        self.stage_timeout = stage_timeout
        self.stage = 0
        self.stage_start = 0
        self.last_eval = 0
        self.best_target = -1.0

    def _apply(self):
        axis, angle, lo, hi = self.stages[self.stage]
        self.training_env.env_method(
            "set_params",
            curriculum_axis=axis,
            curriculum_angle_deg=angle,
            curriculum_speed_lo=lo,
            curriculum_speed_hi=hi,
        )
        print(f"[phaseb] stage={self.stage} axis={axis} angle={angle:g} "
              f"speed={lo:g}-{hi:g}", flush=True)

    def _on_training_start(self):
        self._apply()

    def _on_step(self):
        if self.num_timesteps - self.last_eval < self.eval_freq:
            return True
        self.last_eval = self.num_timesteps
        axis, angle, _, speed_hi = self.stages[self.stage]
        frozen = self.num_timesteps < self.model.actor_start_steps
        seed = 80_000 + 1000 * self.stage
        target = evaluate(self.model, axis, angle, speed_hi, self.n_target, seed)
        level = evaluate(self.model, "both", 0.0, 0.0, self.n_guard, seed + 200)
        roll = evaluate(self.model, "roll", 15.0, 120.0, self.n_guard, seed + 400)
        slow = evaluate(self.model, "both", 15.0, 60.0, self.n_guard, seed + 600)
        corners = evaluate(self.model, "both", 15.0, 40.0, self.n_guard, seed + 800, corner=True)
        gamma = float(self.model.gamma)
        calib = " ".join(
            f"{ax}{an:g}/{sp:g}:Q={run_condition(self.model, ax, an, sp, 8, 96000, gamma)['q_mean']:.0f}"
            for ax, an, sp in CALIB_CONDS
        )
        print(f"[phaseb_eval] t={self.num_timesteps} stage={self.stage} "
              f"{'(warmup)' if frozen else ''} target={target:.2f} level={level:.2f} "
              f"roll={roll:.2f} slow={slow:.2f} corners={corners:.2f} | {calib}", flush=True)

        if frozen:
            return True  # don't advance or score while the actor is frozen
        if target > self.best_target:
            self.best_target = target
            self.model.save(os.path.join(self.save_dir, f"best_stage_{self.stage}"))
        in_stage = self.num_timesteps - self.stage_start
        advance = target >= self.soft_thresh or in_stage >= self.stage_timeout
        if advance:
            reason = "soft-pass" if target >= self.soft_thresh else "timeout-advance"
            if self.stage == len(self.stages) - 1:
                self.model.save(os.path.join(self.save_dir, "final_model"))
                print(f"[phaseb] final stage {reason} target={target:.2f} -> done", flush=True)
                return False
            print(f"[phaseb] stage {self.stage} {reason} target={target:.2f} -> advance", flush=True)
            self.stage += 1
            self.stage_start = self.num_timesteps
            self.best_target = -1.0
            self.model.ep_info_buffer.clear()
            self._apply()
        return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--warmstart", default="models/clean20_master_2d_warmstart.zip")
    p.add_argument("--teacher-data", default="teacher_2d_retention_100k.npz")
    p.add_argument("--tag", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=1_000_000)
    p.add_argument("--nenv", type=int, default=8)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tqd", type=int, default=None)
    p.add_argument("--warmup-steps", type=int, default=50_000)
    p.add_argument("--actor-lr", type=float, default=3e-5)
    p.add_argument("--critic-lr", type=float, default=1e-4)
    p.add_argument("--eval-freq", type=int, default=25_000)
    p.add_argument("--soft-thresh", type=float, default=0.78)
    p.add_argument("--stage-timeout", type=int, default=200_000)
    p.add_argument("--ladder", choices=tuple(LADDERS), default="pm10")
    p.add_argument("--no-teacher", action="store_true",
                   help="disable teacher retention (needed when the teacher's 6 V actions "
                        "don't match the training dynamics, e.g. the 11 V experiment)")
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
    model.gamma = args.gamma
    if args.tqd is not None:
        model.top_quantiles_to_drop_per_net = args.tqd
    model.seed = args.seed
    model.set_random_seed(args.seed)
    reinit_critic(model)
    model.configure_retention(
        os.path.join(HERE, args.teacher_data) if not os.path.isabs(args.teacher_data)
        else args.teacher_data,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        actor_start_steps=args.warmup_steps,
        teacher_coef=1.0,
        teacher_fraction=0.25,
        teacher_target_ratio=0.20,
        use_teacher=not args.no_teacher,
    )
    vmax = float(os.environ.get("FURUTA_VMAX", 6.0))
    print(f"[phaseb] gamma={model.gamma:g} tqd={model.top_quantiles_to_drop_per_net} "
          f"warmup={args.warmup_steps} actor_lr={args.actor_lr:g} ladder={args.ladder} "
          f"v_max={vmax:g}", flush=True)
    curriculum = PhaseBCurriculum(
        output, stages=LADDERS[args.ladder], eval_freq=args.eval_freq,
        soft_thresh=args.soft_thresh, stage_timeout=args.stage_timeout,
    )
    checkpoint = CheckpointCallback(
        save_freq=max(100_000 // args.nenv, 1), save_path=output, name_prefix="ckpt"
    )
    model.learn(total_timesteps=args.steps, callback=[curriculum, checkpoint], progress_bar=False)
    model.save(os.path.join(output, "tqc_final"))
    env.close()
    print("done", flush=True)


if __name__ == "__main__":
    main()
