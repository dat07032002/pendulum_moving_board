"""Does a stiffer board mount make pitch +/-15 deg 120 deg/s feasible?

The bare position servo already tracks the reference (servo_id_2d.py). The ~140 deg/s realized
rate in the stress reports is reaction-driven: the Furuta motor's torque pushes the compliant
(kp=80) board mount. This measures whether stiffening the mount (so the policy can't shove the
board) reduces realized board motion and raises sustained success at the hard regime.
"""
from __future__ import annotations

import argparse

import mujoco
import numpy as np
from sb3_contrib import TQC

from furuta_env_2d import Furuta2DEnv


def set_stiffness(env: Furuta2DEnv, kp: float, kv: float) -> None:
    for name in ("roll_servo", "pitch_servo"):
        act = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        env.model.actuator_gainprm[act, 0] = kp
        env.model.actuator_biasprm[act, 1] = -kp
        env.model.actuator_biasprm[act, 2] = -kv


def run(model, kp, kv, episodes, seed0, angle_deg=15.0, speed=120.0, axis="pitch"):
    env = Furuta2DEnv(randomize=False, max_seconds=10.0)
    env.arm_limit = None
    env.success_arm_limit = None
    env.arm_center_w = 0.0
    env.init_angle_max = np.pi
    env.tilt_axis_mode = axis
    env.tilt_amp = np.deg2rad(angle_deg)
    env.tilt_amp_min_fraction = 1.0
    env.tilt_speed_max = np.deg2rad(speed)
    env.tilt_accel_max = np.deg2rad(max(400.0, 10.0 * speed))
    set_stiffness(env, kp, kv)

    succ = catch = 0
    max_rates = []
    dadr_pt = env.dadr_pt
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed0 + ep)
        set_stiffness(env, kp, kv)  # reset doesn't touch actuator gains, but be explicit
        terminated = truncated = False
        info = {}
        ep_rate = 0.0
        while not (terminated or truncated):
            a, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(a)
            ep_rate = max(ep_rate, abs(float(env.data.qvel[dadr_pt])))
        succ += int(info.get("is_success", False))
        catch += int(info.get("is_catch_success", False))
        max_rates.append(np.rad2deg(ep_rate))
    env.close()
    return succ / episodes, catch / episodes, float(np.mean(max_rates)), float(np.max(max_rates))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("-n", "--episodes", type=int, default=40)
    p.add_argument("--seed0", type=int, default=97000)
    args = p.parse_args()
    model = TQC.load(args.model, device="cpu")
    print(f"model={args.model}  pitch +/-15 deg 120 deg/s")
    print(f"{'mount':<18}{'succ':>6}{'catch':>7}{'meanMaxRate':>13}{'absMaxRate':>12}")
    for name, kp, kv in [("kp80 (compliant)", 80.0, 0.0), ("kp800 (stiff)", 800.0, 12.0)]:
        s, c, mr, xr = run(model, kp, kv, args.episodes, args.seed0)
        print(f"{name:<18}{s:>6.2f}{c:>7.2f}{mr:>13.0f}{xr:>12.0f}", flush=True)


if __name__ == "__main__":
    main()
