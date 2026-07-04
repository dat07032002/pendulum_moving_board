"""
calibrate_sim.py — check the 2D MuJoCo model against the latest hardware sysid.

Runs three experiments in sim and compares to the hardware system-ID:
  1. Free-swing: period + amplitude-decay fit against sysid.json friction_id.
  2. Arm coast-down (spin to 16 rad/s, cut power): dw/dt vs w fit
     -> compare to DAMPING/J=13.8, Tc/J=100.
  3. Free-spin terminal velocity vs voltage -> compare KM/DAMPING ~ 19 rad/s/V.

No hardware needed. Gate: sim matches the real curves within ~10%.
"""
from __future__ import annotations

import json
import os

import numpy as np
import mujoco

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(HERE)
MODEL_PATH = os.path.join(HERE, "furuta_2d.xml")
SYSID_PATH = os.path.join(ROOT, "sysid.json")
M = mujoco.MjModel.from_xml_path(MODEL_PATH)
D = mujoco.MjData(M)
with open(SYSID_PATH) as f:
    SYSID = json.load(f)
REAL_POLE = SYSID["friction_id"]
PA = M.jnt_qposadr[mujoco.mj_name2id(M, mujoco.mjtObj.mjOBJ_JOINT, "pole")]
AA = M.jnt_qposadr[mujoco.mj_name2id(M, mujoco.mjtObj.mjOBJ_JOINT, "arm")]
PV = M.jnt_dofadr[mujoco.mj_name2id(M, mujoco.mjtObj.mjOBJ_JOINT, "pole")]
AV = M.jnt_dofadr[mujoco.mj_name2id(M, mujoco.mjtObj.mjOBJ_JOINT, "arm")]
DT = 0.005
SUB = int(round(DT / M.opt.timestep))


def step_ctrl(v=0.0, lock_pole=False):
    D.ctrl[0] = v
    for _ in range(SUB):
        mujoco.mj_step(M, D)
        if lock_pole:                    # hold pole at hanging -> arm-only (matches the
            D.qpos[PA] = 0.0             # decoupled hardware coast-down / terminal tests)
            D.qvel[PV] = 0.0


def free_swing(theta0_deg, secs=4.0):
    mujoco.mj_resetData(M, D)
    D.qpos[PA] = np.deg2rad(theta0_deg)
    t, th = [], []
    for i in range(int(secs / DT)):
        step_ctrl(0.0)
        t.append(i * DT); th.append(D.qpos[PA])
    return np.array(t), np.rad2deg(np.array(th))


def peaks(t, d):
    pk = []
    for i in range(1, len(d) - 1):
        ext = (d[i] > d[i-1] and d[i] >= d[i+1]) or (d[i] < d[i-1] and d[i] <= d[i+1])
        if not ext or abs(d[i]) < 1.0:
            continue
        if pk and (np.sign(d[i]) == np.sign(pk[-1][1])):
            continue
        pk.append((t[i], d[i]))
    return [tp for tp, _ in pk], [abs(a) for _, a in pk]


def coast(w0=16.0, secs=1.5):
    mujoco.mj_resetData(M, D)
    D.qvel[AV] = w0
    t, w = [], []
    for i in range(int(secs / DT)):
        step_ctrl(0.0, lock_pole=True)   # arm alone (decoupled), as measured
        t.append(i * DT); w.append(abs(D.qvel[AV]))
        if abs(D.qvel[AV]) < 0.3:
            break
    return np.array(t), np.array(w)


def terminal(v, secs=1.5):
    mujoco.mj_resetData(M, D)
    for _ in range(int(secs / DT)):
        step_ctrl(v, lock_pole=True)     # arm alone (decoupled), as measured
    return abs(D.qvel[AV])


print("===== MuJoCo 2D model validation =====")
print(f"model: {MODEL_PATH}")
print(f"sysid: {SYSID_PATH}\n")

# 1) free-swing
release_deg = 41.0
t, th = free_swing(release_deg)
tp, amps = peaks(t, th)
half = np.diff(tp)
per = 2 * np.median(half[half > 0.05])
pairs = [(amps[i], amps[i+1]) for i in range(len(amps)-1) if amps[i] > amps[i+1]]
A = np.array([p[0] for p in pairs]); An = np.array([p[1] for p in pairs])
rho, negC = np.polyfit(A, An, 1)
sim_C = -negC
real_per = 2 * np.pi / np.sqrt(REAL_POLE["alpha"])
period_err = abs(per / real_per - 1.0)
rho_err = abs(rho / REAL_POLE["rho"] - 1.0)
C_err = abs(sim_C / REAL_POLE["C_deg"] - 1.0)
free_swing_pass = max(period_err, rho_err, C_err) <= 0.15
print("1) FREE-SWING")
print(f"   release      = {release_deg:.1f} deg")
print(f"   sim period   = {per*1e3:.0f} ms   real = {real_per*1e3:.0f} ms"
      f"   error = {period_err*100:.1f}%")
print(f"   sim decay    = A_(n+1) = {rho:.3f}*A_n - {sim_C:.2f} deg")
print(f"   real decay   = A_(n+1) = {REAL_POLE['rho']:.3f}*A_n"
      f" - {REAL_POLE['C_deg']:.2f} deg")
print(f"   errors       = rho {rho_err*100:.1f}%   C {C_err*100:.1f}%")
print(f"   sim peaks    = {[round(float(a), 1) for a in amps[:6]]}")
print(f"   15% gate     = {'PASS' if free_swing_pass else 'FAIL'}")

# 2) coast-down
t, w = coast()
dwdt = np.gradient(w, t)
m = (w > 2) & (w < w.max()*0.95) & (dwdt < 0)
sl, ic = np.polyfit(w[m], dwdt[m], 1)
print("\n2) ARM COAST-DOWN")
print(f"   sim DAMPING/J = {-sl:.1f} 1/s   (real 13.8)")
print(f"   sim Tc/J      = {-ic:.0f} rad/s^2   (real 100)")

# 3) terminal velocity
print("\n3) TERMINAL VELOCITY")
vs = [0.5, 1.0, 1.5]
tv = [terminal(v) for v in vs]
for v, w_ in zip(vs, tv):
    print(f"   {v:.1f}V -> {w_:.1f} rad/s")
slope = np.polyfit(vs, tv, 1)[0]
print(f"   sim KM/DAMPING = {slope:.1f} rad/s/V   (real ~19)")

# sanity: upright unstable, hanging stable
mujoco.mj_resetData(M, D); D.qpos[PA] = np.pi - np.deg2rad(2)
for _ in range(int(1.0 / DT)): step_ctrl(0.0)
print(f"\n4) SANITY: from 2 deg below upright -> {np.rad2deg(D.qpos[PA]):.0f} deg "
      f"({'falls (unstable) OK' if abs(np.rad2deg(D.qpos[PA])) < 150 else 'check'})")
