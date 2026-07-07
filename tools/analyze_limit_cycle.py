"""Spectral analysis of the balance limit cycle: dominant frequency of theta and V
during the longest hold segment, plus V<->theta_dot phase relation."""
import math
import sys

import numpy as np

path = sys.argv[1]
rows = []
for line in open(path, encoding="utf-8"):
    if line.startswith("log=["):
        try:
            r = line.strip()[5:-1].split(",")
            rows.append((int(r[0]), float(r[2]), float(r[4]), float(r[5])))
        except (ValueError, IndexError):
            pass

t = np.array([r[0] for r in rows])
th = np.array([r[1] for r in rows])
thd = np.array([r[2] for r in rows])
V = np.array([r[3] for r in rows])
up = np.cos(th) > 0.9848

# longest contiguous upright run
best = (0, 0)
start = None
for i, u in enumerate(up):
    if u and start is None:
        start = i
    elif not u and start is not None:
        if i - start > best[1] - best[0]:
            best = (start, i)
        start = None
if start is not None and len(up) - start > best[1] - best[0]:
    best = (start, len(up))
a, b = best
seg_th = th[a:b] - th[a:b].mean()
seg_V = V[a:b] - V[a:b].mean()
seg_thd = thd[a:b]
n = len(seg_th)
print(f"longest hold: {n} ticks = {n * 0.005:.1f}s")

fs = 200.0
freqs = np.fft.rfftfreq(n, 1 / fs)
for name, sig in (("theta", seg_th), ("V", seg_V), ("theta_dot", seg_thd - seg_thd.mean())):
    P = np.abs(np.fft.rfft(sig * np.hanning(n))) ** 2
    # ignore <0.5 Hz drift
    m = freqs > 0.5
    top = freqs[m][np.argsort(P[m])[-3:]][::-1]
    print(f"{name}: top spectral peaks at {', '.join(f'{f:.1f}' for f in top)} Hz")

# V sign-flip period distribution
sgn = np.sign(seg_V)
flips = np.where(sgn[1:] * sgn[:-1] < 0)[0]
if len(flips) > 2:
    gaps = np.diff(flips) * 5.0  # ms
    print(f"V sign-flip half-periods: median={np.median(gaps):.0f} ms "
          f"-> flip frequency ~{1000 / (2 * np.median(gaps)):.0f} Hz")
print(f"theta osc amplitude (std) = {np.degrees(seg_th.std()):.2f} deg")
