"""Sweep sim actuator lag and measure a policy's chatter + hold quality at level.

Finds the lag at which the policy limit-cycles in sim -- calibrates the real
rig's effective phase burden through the policy's own behavior.
"""
import os
import sys

import numpy as np

rl_dir = r"c:\Users\thanh\Desktop\tilt_pendulum_2d\rl"
sys.path.insert(0, rl_dir)
os.chdir(rl_dir)

os.environ.update({
    "FURUTA_VMAX": "10",
    "FURUTA_UP_THRESH": "0.984807753",
    "FURUTA_CABLE_LIMIT_DEG": "360",
    "FURUTA_SUCCESS_ARM_LIMIT_DEG": "330",
    "FURUTA_ARM_CENTER_W": "0.02",
    "FURUTA_TIGHT_UPRIGHT_W": "0.25",
    "FURUTA_TIGHT_UPRIGHT_SCALE_DEG": "10",
    "FURUTA_CABLE_WARNING_W": "0.20",
    "FURUTA_CABLE_WARNING_START_DEG": "270",
    "FURUTA_ACT_LAG_TAU_MS": "0",
})

MODEL = sys.argv[1] if len(sys.argv) > 1 else "models/tight7f_s2/best_safe.zip"
UP10 = float(np.cos(np.deg2rad(10.0)))

from sb3_contrib import TQC  # noqa: E402

model = TQC.load(MODEL, device="cpu")

print(f"{'tau_ms':>7}{'delay':>6}{'m|da|':>8}{'theta_std':>10}{'in-band':>9}{'flips':>7}")
for delay in (1, 2):
    for tau in (0.0, 6.0, 9.0, 12.0, 15.0, 20.0):
        os.environ["FURUTA_ACT_LAG_TAU_MS"] = f"{tau:g}"
        from importlib import reload  # env reads env var at construction; fresh env each time
        from furuta_env_2d import Furuta2DEnv

        env = Furuta2DEnv(randomize=False, max_seconds=10.0)
        env.init_angle_max = 0.3          # start near upright: probe the HOLD, not swing-up
        env.init_vel_assist = 0.0
        das, ths, flips, n_up = [], [], 0, 0
        prev_a = prev_sign = None
        for ep in range(4):
            obs, _ = env.reset(seed=900000 + ep)
            env.tilt_gen_2d = None
            env._delay = delay
            env.act_buf = [0.0] * delay
            term = trunc = False
            prev_a = None
            while not (term or trunc):
                a, _ = model.predict(obs, deterministic=True)
                obs, _, term, trunc, _ = env.step(a)
                up = env._true_up()
                if up > UP10:
                    n_up += 1
                    ths.append(np.degrees(np.arccos(min(1.0, up))))
                    if prev_a is not None:
                        das.append(abs(float(a[0]) - prev_a))
                        s = np.sign(float(a[0]))
                        if prev_sign is not None and s * prev_sign < 0:
                            flips += 1
                        prev_sign = s
                prev_a = float(a[0])
        env.close()
        total = 4 * 2000
        print(f"{tau:>7g}{delay:>6}{np.mean(das) if das else float('nan'):>8.3f}"
              f"{np.std(ths) if ths else float('nan'):>10.2f}"
              f"{n_up / total:>9.2f}{flips / max(1, n_up):>7.2f}")
