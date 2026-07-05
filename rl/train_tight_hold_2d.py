"""Tight-hold fine-tuning: tighter upright gate + smoother action, robustness-first.

Clone of the validated stationary nominal-plant recipe (train_c360_stationary_2d.py):
warm start, brief actor freeze, retained critic, teacher rehearsal, zero-hit safety gate,
fixed delay=1 (measured hardware latency). Differences:

- Training envs use a tighter success gate / tight-upright bonus / higher action-rate
  penalty via --up-thresh-deg / --tight-scale-deg / --tight-w / --action-rate-w
  (exported as FURUTA_* env vars before env construction).
- Evaluation envs are pinned to the canonical +/-10 deg success gate so the safety /
  deployment gate stays comparable with all recorded baselines (robustness-first).
- The eval gate additionally reports occ7/occ5 (fraction of upright time within 7/5 deg)
  and mean |delta action| (vibration proxy); best_safe ties are broken by occ5.

Stage A (7 deg): warm start models/progressive_dr_s1/best_safe.zip.
Stage B (5 deg): warm start the verified Stage A winner. Never start a tighter gate
from scratch (TIGHT_UPRIGHT_RESULTS.md).
"""
from __future__ import annotations

import argparse
import os

import mujoco
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

HERE = os.path.dirname(__file__)

# Canonical +/-10 deg evaluation gate (must match the recorded baselines).
EVAL_UP_THRESH = float(np.cos(np.deg2rad(10.0)))
UP10, UP7, UP5 = np.cos(np.deg2rad([10.0, 7.0, 5.0]))


def _pin_eval_gate(env):
    """Force the canonical +/-10 deg success gate on an eval env regardless of the
    tighter training env vars."""
    env.up_thresh = EVAL_UP_THRESH
    env.up_bonus = float(np.cos(0.8 * np.arccos(EVAL_UP_THRESH)))
    return env


def dynamic_generator(env, angle_deg: float, speed_deg: float):
    from tilt_2d import SmoothRandomTilt2D
    from furuta_env_2d import DT

    return SmoothRandomTilt2D(
        angle_max=np.deg2rad(angle_deg),
        speed_max=np.deg2rad(speed_deg),
        accel_max=np.deg2rad(max(400.0, 10.0 * speed_deg)),
        dt=DT,
        seed=int(env.np_random.integers(0, 2**32 - 1)),
    )


def make_env():
    def factory():
        # Import inside the subprocess so FURUTA_* env vars set in main() apply.
        from furuta_env_2d import Furuta2DEnv

        class TightStationaryCableEnv(Furuta2DEnv):
            """Fixed mixture of deployment motion, retention, and cable recovery
            (identical to StationaryCableEnv in train_c360_stationary_2d.py)."""

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
                    self.tilt_axis_mode = (
                        "roll" if self.np_random.random() < 0.5 else "pitch"
                    )
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

        return TightStationaryCableEnv()

    return factory


def evaluate_metrics(model, axis, angle_deg, speed_deg, episodes, seed0):
    """Canonical +/-10 deg gate metrics plus tight-hold occupancy and smoothness."""
    from furuta_env_2d import Furuta2DEnv

    env = _pin_eval_gate(Furuta2DEnv(randomize=False, max_seconds=10.0))
    env.init_angle_max = np.pi
    successes = catches = cable_hits = 0
    outside, arm_max = [], []
    up_steps = occ7_steps = occ5_steps = 0
    abs_da_sum = 0.0
    abs_da_n = 0
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
        prev_a = None
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            up = env._true_up()
            if up > UP10:
                up_steps += 1
                occ7_steps += int(up > UP7)
                occ5_steps += int(up > UP5)
                if prev_a is not None:
                    abs_da_sum += abs(float(action[0]) - prev_a)
                    abs_da_n += 1
            prev_a = float(action[0])
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
        "occ7": occ7_steps / up_steps if up_steps else float("nan"),
        "occ5": occ5_steps / up_steps if up_steps else float("nan"),
        "mean_abs_da": abs_da_sum / abs_da_n if abs_da_n else float("nan"),
    }


class TightSafetyEval(BaseCallback):
    """StationarySafetyEval with tight-hold metrics; best_safe tie-broken by occ5."""

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
        self.best_key = (-1.0, -1.0)  # (target success, occ5)

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
        print(
            f"[tight_eval] t={self.num_timesteps} target={target['success']:.2f} "
            f"catch={target['catch']:.2f} both45={both45['success']:.2f} "
            f"pitch60={pitch['success']:.2f} level={level['success']:.2f} "
            f"hits={total_hits} outside330={target['outside']:.3f} "
            f"armP95={target['arm_p95']:.1f} "
            f"occ7={target['occ7']:.2f}/{level['occ7']:.2f} "
            f"occ5={target['occ5']:.2f}/{level['occ5']:.2f} "
            f"m|da|={target['mean_abs_da']:.3f}",
            flush=True,
        )
        safe = total_hits == 0 and target["outside"] <= 0.02
        key = (target["success"], target["occ5"])
        if safe and key > self.best_key:
            self.best_key = key
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
                f"[tight] gate pass {self.pass_streak}/{self.consecutive_passes}",
                flush=True,
            )
        if self.pass_streak >= self.consecutive_passes:
            self.model.save(os.path.join(self.save_dir, "eligible_model"))
            print("[tight] two-pass deployment gate reached -> done", flush=True)
            return not self.early_stop
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmstart", default="models/progressive_dr_s1/best_safe.zip")
    parser.add_argument("--teacher-data", default="teacher_s1_safe_nominal_100k.npz")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=400_000)
    parser.add_argument("--nenv", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=25_000)
    parser.add_argument("--actor-lr", type=float, default=5e-6)
    parser.add_argument("--critic-lr", type=float, default=1e-4)
    parser.add_argument("--rehearsal-fraction", type=float, default=0.15)
    parser.add_argument("--no-rehearsal", action="store_true")
    # Behavior-clone anchor strength. 1.0 froze the actor's style entirely in the
    # first tight7 campaign; lower (~0.2) keeps replay rehearsal for stability while
    # letting the reward actually reshape the policy.
    parser.add_argument("--teacher-coef", type=float, default=1.0)
    # "none" disables the adaptive rescaling in retention_tqc, which otherwise
    # drives the effective BC coefficient to its 1e6 cap as the student converges
    # to the teacher (an infinitely stiff anchor that froze Stage A/A' actors).
    parser.add_argument("--teacher-ratio", default="0.10")
    parser.add_argument("--eval-freq", type=int, default=50_000)
    parser.add_argument("--device", default="auto")
    # tight-hold knobs -> FURUTA_* env vars for the TRAINING envs only
    parser.add_argument("--up-thresh-deg", type=float, default=7.0)
    parser.add_argument("--tight-scale-deg", type=float, default=7.0)
    parser.add_argument("--tight-w", type=float, default=0.35)
    parser.add_argument("--action-rate-w", type=float, default=0.06)
    # Actuator slew limit [V/tick]; must match the RL_SLEW_V_PER_TICK the policy
    # will be exported/deployed with (export_policy.py --slew-v).
    parser.add_argument("--slew-v", type=float, default=0.0)
    # Sim-only first-order actuator lag [ms]: "0" off, "5" fixed, "2,8" DR range.
    parser.add_argument("--act-lag-ms", default="0")
    # Action-delay steps: "1" fixed, "1,2" per-episode choice (nominal-plant DR).
    parser.add_argument("--delay-steps", default="1")
    # Extra past actions in the obs (0 = legacy 10-D; 2 = 12-D action history,
    # restores observability for delay <= 3; warm start must match, see
    # expand_obs_warmstart.py).
    parser.add_argument("--act-history", type=int, default=0)
    args = parser.parse_args()

    # Export the tight-training reward configuration BEFORE any env construction.
    # SubprocVecEnv workers inherit these; eval envs are re-pinned to +/-10 deg.
    os.environ["FURUTA_UP_THRESH"] = f"{np.cos(np.deg2rad(args.up_thresh_deg)):.9f}"
    os.environ["FURUTA_TIGHT_UPRIGHT_SCALE_DEG"] = f"{args.tight_scale_deg:g}"
    os.environ["FURUTA_TIGHT_UPRIGHT_W"] = f"{args.tight_w:g}"
    os.environ["FURUTA_ACTION_RATE_W"] = f"{args.action_rate_w:g}"
    os.environ["FURUTA_SLEW_V_PER_TICK"] = f"{args.slew_v:g}"
    os.environ["FURUTA_ACT_LAG_TAU_MS"] = str(args.act_lag_ms)
    os.environ["FURUTA_DELAY_STEPS"] = str(args.delay_steps)
    os.environ["FURUTA_ACT_HISTORY"] = str(args.act_history)

    from retention_tqc import RetentionTQC

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
        teacher_coef=args.teacher_coef,
        teacher_fraction=args.rehearsal_fraction,
        teacher_target_ratio=(
            None if str(args.teacher_ratio).lower() == "none"
            else float(args.teacher_ratio)
        ),
        use_teacher=not args.no_rehearsal,
    )
    print(
        f"[tight] gate={args.up_thresh_deg:g}deg scale={args.tight_scale_deg:g}deg "
        f"tight_w={args.tight_w:g} action_rate_w={args.action_rate_w:g} "
        f"slew_v={args.slew_v:g} act_lag_ms={args.act_lag_ms} "
        f"delay_steps={args.delay_steps} act_history={args.act_history} "
        f"actor_lr={args.actor_lr:g} critic_lr={args.critic_lr:g} "
        f"warmup={args.warmup_steps} rehearsal={not args.no_rehearsal}",
        flush=True,
    )
    evaluator = TightSafetyEval(output, eval_freq=args.eval_freq)
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
