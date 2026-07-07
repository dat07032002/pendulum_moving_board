"""One powered RL bring-up stage: vlim <V>, log, rl for <sec>, s, nolog.

Usage: python rl_stage.py <vlim> <rl_seconds> <out_log_path>
Board resets on port open (motor idle); KEEP BOARD LEVEL during first ~6 s
(IMU tare) and have the arm at physical cable center ('rl' zeroes phi there).
's' is always sent in the finally block; total rl time is hard-capped.
"""
import sys
import time

import serial


def open_noreset(port):
    """Open serial WITHOUT toggling DTR/RTS so the ESP32 does not reboot."""
    s = serial.Serial()
    s.port = port
    s.baudrate = 921600
    s.timeout = 0.05
    s.dtr = False
    s.rts = False
    s.open()
    return s

VLIM = float(sys.argv[1])
RL_SEC = float(sys.argv[2])
OUT = sys.argv[3]
assert 0.5 <= VLIM <= 10.0 and 0.5 <= RL_SEC <= 120.0

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ser = open_noreset("COM5")
buf = []


def pump(seconds):
    t0 = time.time()
    while time.time() - t0 < seconds:
        chunk = ser.read(16384)
        if chunk:
            buf.append(chunk)


try:
    pump(6.0)                                   # boot + level tare
    ser.write(f"vlim {VLIM}\n".encode()); pump(0.3)
    ser.write(b"imu\n");    pump(0.3)
    ser.write(b"log\n");    pump(0.4)
    ser.write(b"rl\n")
    pump(RL_SEC)                                # policy active
finally:
    try:
        ser.write(b"s\n"); pump(0.4)
        ser.write(b"nolog\n"); pump(0.3)
        ser.write(b"s\n"); pump(0.2)            # belt and suspenders
    finally:
        ser.close()

raw = b"".join(buf).decode(errors="replace")
with open(OUT, "w", encoding="utf-8") as f:
    f.write(raw)

# quick summary: peak |V|, peak |phi|, peak |theta_dot|, any mode/guard messages
import math
peak_v = peak_phi = peak_td = 0.0
n = 0
for line in raw.splitlines():
    if line.startswith("log=["):
        try:
            r = line.strip()[5:-1].split(",")
            peak_phi = max(peak_phi, abs(float(r[1])))
            peak_td = max(peak_td, abs(float(r[4])))
            peak_v = max(peak_v, abs(float(r[5])))
            n += 1
        except (ValueError, IndexError):
            pass
    elif line.startswith("#"):
        print(line)
print(f"ticks={n} peak|V|={peak_v:.2f} peak|phi|={math.degrees(peak_phi):.1f}deg "
      f"peak|theta_dot|={peak_td:.1f}rad/s  log->{OUT}")
