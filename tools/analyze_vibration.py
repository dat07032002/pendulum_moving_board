"""Vibration proxy from firmware logs: RMS of per-tick delta-V while upright."""
import math
import sys

for path in sys.argv[1:]:
    rows = []
    for line in open(path, encoding="utf-8"):
        if line.startswith("log=["):
            try:
                r = line.strip()[5:-1].split(",")
                rows.append((float(r[2]), float(r[5])))  # theta, V
            except (ValueError, IndexError):
                pass
    dv2 = dv_n = 0.0
    v2 = 0.0
    for i in range(1, len(rows)):
        th, v = rows[i]
        if math.cos(th) > 0.9848:            # upright <10 deg
            dv = v - rows[i - 1][1]
            dv2 += dv * dv
            v2 += v * v
            dv_n += 1
    if dv_n:
        print(f"{path}: upright ticks={int(dv_n)}  "
              f"dV RMS={math.sqrt(dv2/dv_n):.2f} V/tick  "
              f"V RMS={math.sqrt(v2/dv_n):.2f} V")
    else:
        print(f"{path}: no upright ticks")
