"""What the policy sees while holding: theta bias, theta_dot noise, V distribution."""
import math
import sys

import numpy as np

path = sys.argv[1]
rows = []
for line in open(path, encoding="utf-8"):
    if line.startswith("log=["):
        try:
            r = line.strip()[5:-1].split(",")
            rows.append((float(r[2]), float(r[3]), float(r[4]), float(r[5])))
        except (ValueError, IndexError):
            pass

th = np.array([r[0] for r in rows])
phid = np.array([r[1] for r in rows])
thd = np.array([r[2] for r in rows])
V = np.array([r[3] for r in rows])
up = np.cos(th) > 0.9848

th_up = np.degrees(np.arctan2(np.sin(th[up]), np.cos(th[up])))
print(f"upright ticks: {up.sum()}")
print(f"theta while upright: mean={th_up.mean():+.2f}deg  std={th_up.std():.2f}deg")
print(f"theta_dot while upright: mean={thd[up].mean():+.2f}  std={thd[up].std():.2f} rad/s")
print(f"phi_dot  while upright: mean={phid[up].mean():+.2f}  std={phid[up].std():.2f} rad/s")
absV = np.abs(V[up])
print(f"|V| while upright: mean={absV.mean():.2f}  median={np.median(absV):.2f}  "
      f"frac>9V={float((absV > 9).mean()):.2f}  frac<2V={float((absV < 2).mean()):.2f}")
# sign flip rate
sgn = np.sign(V[up])
flips = float((sgn[1:] * sgn[:-1] < 0).mean())
print(f"V sign-flip rate between consecutive ticks: {flips:.2f}")
