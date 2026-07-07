"""Analyze a powered rl_stage log: V->phi_dot direction gain, encoder continuity."""
import math
import sys

path = sys.argv[1]
rows = []
for line in open(path, encoding="utf-8"):
    if line.startswith("log=["):
        try:
            r = line.strip()[5:-1].split(",")
            rows.append((int(r[0]), float(r[1]), float(r[2]), float(r[3]),
                         float(r[4]), float(r[5])))
        except (ValueError, IndexError):
            pass

t = [r[0] for r in rows]
phi = [r[1] for r in rows]
theta = [r[2] for r in rows]
phi_dot = [r[3] for r in rows]
theta_dot = [r[4] for r in rows]
V = [r[5] for r in rows]

# direction: correlation between V and subsequent delta(phi_dot) (2-tick lead)
num = den_v = den_d = 0.0
for i in range(len(rows) - 2):
    if abs(V[i]) > 0.05:
        dpd = phi_dot[i + 2] - phi_dot[i]
        num += V[i] * dpd
        den_v += V[i] * V[i]
        den_d += dpd * dpd
corr = num / math.sqrt(den_v * den_d) if den_v > 0 and den_d > 0 else float("nan")
print(f"ticks={len(rows)}  V-to-phi_ddot correlation: {corr:+.3f} "
      f"(positive = +V accelerates +phi)")

# encoder continuity: max per-tick jumps
max_dphi = max(abs(phi[i + 1] - phi[i]) for i in range(len(rows) - 1))
dth = [abs(theta[i + 1] - theta[i]) for i in range(len(rows) - 1)]
dth = [min(d, 2 * math.pi - d) for d in dth]  # unwrap
print(f"max per-tick |dphi|={math.degrees(max_dphi):.2f}deg "
      f"|dtheta|={math.degrees(max(dth)):.2f}deg (5 ms tick)")

# timeline: first nonzero V, peak phi, when
iv = next((i for i, v in enumerate(V) if abs(v) > 1e-6), None)
ip = max(range(len(rows)), key=lambda i: abs(phi[i]))
print(f"V active from tick {iv}; peak |phi|={math.degrees(abs(phi[ip])):.1f}deg "
      f"at t={(t[ip]-t[0])/1000:.2f}s; final phi={math.degrees(phi[-1]):.1f}deg")
print(f"peak |theta_dot|={max(abs(x) for x in theta_dot):.1f} rad/s; "
      f"peak |phi_dot|={max(abs(x) for x in phi_dot):.1f} rad/s")
