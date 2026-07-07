"""Live AS5600 RAW counts (bypasses the firmware glitch filter).

Usage: python raw_live.py [seconds]. A full pendulum turn must sweep 0..4095.
"""
import re
import sys
import time

import serial


def open_noreset(port):
    s = serial.Serial()
    s.port = port
    s.baudrate = 921600
    s.timeout = 0.05
    s.dtr = False
    s.rts = False
    s.open()
    return s


sys.stdout.reconfigure(encoding="utf-8", errors="replace")
DUR = float(sys.argv[1]) if len(sys.argv) > 1 else 120.0

ser = open_noreset("COM5")
try:
    t0 = time.time()
    while time.time() - t0 < DUR:
        ser.write(b"raw\n")
        time.sleep(0.2)
        out = ser.read(8192).decode(errors="replace")
        m = re.search(r"raw=(\d+) UP=(\d+) th=([-\d.]+)", out)
        if m:
            raw = int(m.group(1))
            print(f"raw={raw:4d}/4095  ({raw*360/4096:5.1f} deg raw)  "
                  f"th={m.group(3)}", flush=True)
        elif out.strip():
            print(out.strip().splitlines()[-1], flush=True)
finally:
    ser.close()
