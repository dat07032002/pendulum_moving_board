"""Sample pendulum theta via rlcheck. Usage: python theta_watch.py [seconds]"""
import math
import re
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

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
DUR = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0

ser = open_noreset("COM5")
try:
    t0 = time.time()
    while time.time() - t0 < 6.0:
        ser.read(4096)
    t0 = time.time()
    while time.time() - t0 < DUR:
        ser.write(b"rlcheck\n")
        time.sleep(0.4)
        out = ser.read(8192).decode(errors="replace")
        m = re.search(r"obs=\[([-\d.]+),([-\d.]+)", out)
        if m:
            th = math.degrees(math.atan2(float(m.group(2)), float(m.group(1))))
            print(f"theta = {th:+.1f} deg  (0=up, +/-180=down)", flush=True)
        elif "FAILED" in out:
            print("rlcheck FAILED (IMU?)", flush=True)
finally:
    ser.close()
