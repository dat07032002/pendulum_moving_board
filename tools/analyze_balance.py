"""Analyze balance quality in a powered rl_stage log: hold segments, drops, recoveries.

Upright band uses the training success threshold cos(theta) > 0.9848 (|theta| < 10 deg).
"""
import math
import sys

path = sys.argv[1]
rows = []
for line in open(path, encoding="utf-8"):
    if line.startswith("log=["):
        try:
            r = line.strip()[5:-1].split(",")
            rows.append((int(r[0]), float(r[1]), float(r[2]), float(r[4]),
                         float(r[5]), float(r[7]), float(r[8])))
        except (ValueError, IndexError):
            pass
if not rows:
    print("no log rows"); sys.exit(1)

t0 = rows[0][0]
UP = 0.9848  # cos(10 deg)
up = [math.cos(r[2]) > UP for r in rows]

# segments
segs, start = [], None
for i, u in enumerate(up):
    if u and start is None:
        start = i
    elif not u and start is not None:
        segs.append((start, i)); start = None
if start is not None:
    segs.append((start, len(up)))
# merge blips: ignore gaps < 100 ms (20 ticks)
merged = []
for s in segs:
    if merged and s[0] - merged[-1][1] < 20:
        merged[-1] = (merged[-1][0], s[1])
    else:
        merged.append(list(s))
holds = [(rows[a][0] - t0, (rows[b - 1][0] - rows[a][0]) / 1000.0) for a, b in merged]
long_holds = [h for h in holds if h[1] >= 0.5]

total_s = (rows[-1][0] - t0) / 1000.0
up_s = sum(1 for u in up if u) * 0.005
first_catch = holds[0][0] / 1000.0 if holds else None
drops = max(0, len(long_holds) - 1)
max_phi = max(abs(r[1]) for r in rows)
max_tilt = max(max(abs(r[5]), abs(r[6])) for r in rows)

print(f"{path}")
print(f"  duration {total_s:.1f}s | upright(|th|<10deg) {up_s:.1f}s ({100*up_s/total_s:.0f}%)")
if first_catch is not None:
    print(f"  first catch at t={first_catch:.1f}s")
print(f"  hold segments >=0.5s: {len(long_holds)} | drop->recover events: {drops}")
for t, d in long_holds:
    print(f"    hold at t={t/1000:6.1f}s for {d:5.1f}s")
print(f"  max |arm|={math.degrees(max_phi):.0f}deg | max |board tilt| obs={max_tilt:.2f} (~{max_tilt*15:.1f}deg)")
