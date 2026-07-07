"""Analyze furuta ESP32 log capture: tick jitter, IMU cadence, and (if present) V-step response."""
import statistics as st
import sys

path = sys.argv[1]
lines = [l for l in open(path, encoding="utf-8") if l.startswith("log=[")]
rows = [l.strip()[5:-1].split(",") for l in lines]
t = [int(r[0]) for r in rows]
V = [float(r[5]) for r in rows]
theta_dot = [float(r[4]) for r in rows]
phi_dot = [float(r[3]) for r in rows]
grv = [int(r[-2]) for r in rows]
gyr = [int(r[-1]) for r in rows]

dt = [b - a for a, b in zip(t, t[1:])]
print(f"ticks={len(t)}  dt ms: min={min(dt)} max={max(dt)} mean={st.mean(dt):.3f}")

for name, seq in (("gyro", gyr), ("grv", grv)):
    gaps, last = [], 0
    for i in range(1, len(seq)):
        if seq[i] != seq[i - 1]:
            gaps.append(i - last)
            last = i
    if gaps:
        print(f"{name} seq gaps (ticks): min={min(gaps)} max={max(gaps)} mean={st.mean(gaps):.2f} n={len(gaps)}")

# torque-step response: first tick where V leaves 0, then first tick where |phi_dot| responds
iv = next((i for i, v in enumerate(V) if abs(v) > 1e-6), None)
if iv is not None:
    base = [abs(x) for x in phi_dot[max(0, iv - 60):iv]] or [0.0]
    thresh = max(0.3, 4 * max(base))
    ir = next((i for i in range(iv, len(V)) if abs(phi_dot[i]) > thresh), None)
    print(f"V step at tick {iv} (t={t[iv]} ms, V={V[iv]:.2f}); |phi_dot| baseline max={max(base):.3f}, thresh={thresh:.2f}")
    if ir is not None:
        print(f"phi_dot response at tick {ir} (t={t[ir]} ms) -> latency {t[ir]-t[iv]} ms = {(t[ir]-t[iv])/5:.1f} control steps")
    else:
        print("no phi_dot response found above threshold")
else:
    print("no V step in capture")
