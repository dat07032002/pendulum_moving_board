"""Measure board position-servo tracking vs the tilt reference, and tune kp/kv.

The Step-4/Step-5 reports flagged that the board position servo (kp=80, no derivative)
overshoots: a 120 deg/s reference realizes ~140 deg/s, so the policy is asked to handle a
harder disturbance than the label. This script quantifies the overshoot for several
(kp, kv, damping) settings on the pitch axis (the bottleneck) so we can pick a config whose
realized motion tracks the reference.
"""
from __future__ import annotations

import os

import mujoco
import numpy as np

from tilt_2d import SmoothRandomTilt2D

HERE = os.path.dirname(__file__)
DT = 0.005


def measure(kp, kv, damping, seconds=40.0, angle_deg=15.0, speed=120.0, seed=0):
    model = mujoco.MjModel.from_xml_path(os.path.join(HERE, "furuta_2d.xml"))
    data = mujoco.MjData(model)
    sub = int(round(DT / model.opt.timestep))

    jpt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "board_pitch")
    jr = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "board_roll")
    qadr = model.jnt_qposadr[jpt]
    dadr = model.jnt_dofadr[jpt]
    dadr_r = model.jnt_dofadr[jr]
    act_p = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "pitch_servo")
    act_r = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "roll_servo")

    for act in (act_p, act_r):
        model.actuator_gainprm[act, 0] = kp
        model.actuator_biasprm[act, 1] = -kp
        model.actuator_biasprm[act, 2] = -kv  # derivative term: force -= kv*qvel
    if damping is not None:
        model.dof_damping[dadr] = damping
        model.dof_damping[dadr_r] = damping

    gen = SmoothRandomTilt2D(
        angle_max=np.deg2rad(angle_deg),
        speed_max=np.deg2rad(speed),
        accel_max=np.deg2rad(max(400.0, 10.0 * speed)),
        dt=DT,
        seed=seed,
    )
    n = int(seconds / DT)
    ref_ang, ref_rate, real_ang, real_rate, accel = [], [], [], [], []
    prev_rate = 0.0
    for _ in range(n):
        _, pp, _, vp = gen.step()  # pitch reference angle/rate
        data.ctrl[act_p] = pp
        data.ctrl[act_r] = 0.0
        for _ in range(sub):
            mujoco.mj_step(model, data)
        ref_ang.append(pp)
        ref_rate.append(vp)
        real_ang.append(float(data.qpos[qadr]))
        rr = float(data.qvel[dadr])
        real_rate.append(rr)
        accel.append((rr - prev_rate) / DT)
        prev_rate = rr

    ref_ang = np.rad2deg(ref_ang)
    ref_rate = np.rad2deg(ref_rate)
    real_ang = np.rad2deg(real_ang)
    real_rate = np.rad2deg(real_rate)
    accel = np.rad2deg(accel)
    track_rms = float(np.sqrt(np.mean((real_ang - ref_ang) ** 2)))
    return {
        "ref_rate_max": float(np.max(np.abs(ref_rate))),
        "real_rate_max": float(np.max(np.abs(real_rate))),
        "overshoot": float(np.max(np.abs(real_rate)) / np.max(np.abs(ref_rate))),
        "real_ang_max": float(np.max(np.abs(real_ang))),
        "real_accel_max": float(np.max(np.abs(accel))),
        "track_rms_deg": track_rms,
    }


def main():
    configs = [
        ("baseline kp80 kv0 d1.2", 80.0, 0.0, 1.2),
        ("kp80 kv2.5 d1.2", 80.0, 2.5, 1.2),
        ("kp200 kv4 d0.5", 200.0, 4.0, 0.5),
        ("kp400 kv8 d0.2", 400.0, 8.0, 0.2),
        ("kp800 kv12 d0.1", 800.0, 12.0, 0.1),
    ]
    print(f"{'config':<24}{'refRate':>8}{'realRate':>9}{'oversh':>7}"
          f"{'realAng':>8}{'realAcc':>9}{'trackRMS':>9}")
    for name, kp, kv, d in configs:
        # average over a few seeds for stability
        rows = [measure(kp, kv, d, seed=s) for s in range(3)]
        agg = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        print(f"{name:<24}{agg['ref_rate_max']:>8.0f}{agg['real_rate_max']:>9.0f}"
              f"{agg['overshoot']:>7.2f}{agg['real_ang_max']:>8.1f}"
              f"{agg['real_accel_max']:>9.0f}{agg['track_rms_deg']:>9.2f}", flush=True)


if __name__ == "__main__":
    main()
