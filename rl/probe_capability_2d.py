"""Capability vs training probe for the 2D warm start.

Answers two questions on a FROZEN checkpoint, no training:
  1. Authority: when the policy loses the pole under fast continuous pitch, is its
     commanded action saturated (|a|>0.95) in the window just before the loss?
       saturated  -> ran out of actuator authority (feasibility wall)
       unsaturated-> had authority left, used it wrong (a training/strategy problem)
  2. Critic calibration (#5): on genuinely balanced states (true_up>0.9), does the
     critic's Q(s, a_det) match the empirical discounted return-to-go?
"""
from __future__ import annotations

import argparse

import numpy as np
import torch as th
from sb3_contrib import TQC

from furuta_env_2d import DT, Furuta2DEnv

SAT = 0.95
WINDOW = 100  # steps (~0.5 s) before episode end


def critic_q(model: TQC, obs: np.ndarray, action: np.ndarray) -> float:
    obs_t = model.policy.obs_to_tensor(obs)[0]
    act_t = th.as_tensor(action, dtype=th.float32, device=model.device).reshape(1, -1)
    with th.no_grad():
        quantiles = model.critic(obs_t, act_t)  # (1, n_critics, n_quantiles)
    return float(quantiles.mean().item())


def run_condition(model, axis, angle_deg, speed_deg, episodes, seed0, gamma):
    env = Furuta2DEnv(randomize=False, max_seconds=10.0)
    env.init_angle_max = np.pi
    env.tilt_axis_mode = axis
    env.tilt_amp = np.deg2rad(angle_deg)
    env.tilt_amp_min_fraction = 1.0
    env.tilt_speed_max = np.deg2rad(speed_deg)
    env.tilt_accel_max = np.deg2rad(max(400.0, 10.0 * speed_deg))

    n_success = n_catch = n_fall = 0
    sat_before_loss = []            # frac saturated in WINDOW before end (non-success eps)
    cal_q, cal_rtg = [], []         # critic Q vs return-to-go on balanced states
    max_abs_a_all = []
    # tightness/smoothness accumulators over upright steps (up > cos(10 deg))
    up_steps = occ5_steps = occ7_steps = 0
    abs_da_sum = 0.0
    abs_da_n = 0
    UP10, UP7, UP5 = np.cos(np.deg2rad([10.0, 7.0, 5.0]))

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed0 + ep)
        if angle_deg == 0:
            env.tilt_gen_2d = None
        actions, ups, rewards, qs = [], [], [], []
        terminated = truncated = False
        info = {}
        while not (terminated or truncated):
            a, _ = model.predict(obs, deterministic=True)
            qs.append(critic_q(model, obs, a))
            obs, r, terminated, truncated, info = env.step(a)
            actions.append(float(a[0]))
            ups.append(env._true_up())
            rewards.append(r)

        actions = np.asarray(actions)
        ups = np.asarray(ups)
        rewards = np.asarray(rewards)
        qs = np.asarray(qs)
        max_abs_a_all.append(float(np.max(np.abs(actions))))

        # tight-hold occupancy: of the steps spent upright (<10 deg), what fraction
        # stayed within 7 / 5 deg. Smoothness: mean |da| over those upright steps.
        up_mask = ups > UP10
        up_steps += int(up_mask.sum())
        occ7_steps += int((ups > UP7).sum())
        occ5_steps += int((ups > UP5).sum())
        if len(actions) > 1:
            da = np.abs(np.diff(actions))
            m = up_mask[1:]
            abs_da_sum += float(da[m].sum())
            abs_da_n += int(m.sum())

        success = bool(info.get("is_success", False))
        n_success += success
        n_catch += bool(info.get("is_catch_success", False))
        if not success:
            n_fall += 1
            w = actions[-WINDOW:]
            sat_before_loss.append(float(np.mean(np.abs(w) > SAT)))

        # discounted return-to-go, then calibration pairs on balanced states
        rtg = np.zeros_like(rewards)
        g = 0.0
        for i in range(len(rewards) - 1, -1, -1):
            g = rewards[i] + gamma * g
            rtg[i] = g
        mask = ups > 0.9
        cal_q.extend(qs[mask].tolist())
        cal_rtg.extend(rtg[mask].tolist())

    env.close()
    cal_q = np.asarray(cal_q)
    cal_rtg = np.asarray(cal_rtg)
    return {
        "success": n_success / episodes,
        "catch": n_catch / episodes,
        "falls": n_fall,
        "sat_before_loss": float(np.mean(sat_before_loss)) if sat_before_loss else float("nan"),
        "mean_max_abs_a": float(np.mean(max_abs_a_all)),
        "q_mean": float(cal_q.mean()) if len(cal_q) else float("nan"),
        "rtg_mean": float(cal_rtg.mean()) if len(cal_rtg) else float("nan"),
        "n_balanced": int(len(cal_q)),
        "occ7": occ7_steps / up_steps if up_steps else float("nan"),
        "occ5": occ5_steps / up_steps if up_steps else float("nan"),
        "mean_abs_da": abs_da_sum / abs_da_n if abs_da_n else float("nan"),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("-n", "--episodes", type=int, default=40)
    p.add_argument("--seed0", type=int, default=90000)
    args = p.parse_args()

    model = TQC.load(args.model, device="cpu")
    gamma = float(model.gamma)
    conditions = [
        ("both", 0.0, 0.0),       # level: critic sanity
        ("pitch", 10.0, 30.0),
        ("pitch", 10.0, 45.0),
        ("pitch", 10.0, 60.0),
        ("both", 10.0, 60.0),
    ]
    print(f"model={args.model} gamma={gamma:g} episodes/cond={args.episodes}")
    print(f"{'cond':<18}{'succ':>6}{'catch':>6}{'falls':>6}"
          f"{'sat<loss':>9}{'maxA':>6}{'Q_bal':>9}{'RTG_bal':>9}{'n':>7}")
    for axis, angle, speed in conditions:
        r = run_condition(model, axis, angle, speed, args.episodes, args.seed0, gamma)
        label = f"{axis} {angle:g}d {speed:g}/s"
        print(f"{label:<18}{r['success']:>6.2f}{r['catch']:>6.2f}{r['falls']:>6}"
              f"{r['sat_before_loss']:>9.2f}{r['mean_max_abs_a']:>6.2f}"
              f"{r['q_mean']:>9.1f}{r['rtg_mean']:>9.1f}{r['n_balanced']:>7}", flush=True)


if __name__ == "__main__":
    main()
