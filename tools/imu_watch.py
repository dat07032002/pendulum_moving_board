"""Watch BNO086 health while the user toggles motor power. Motor never driven.

Usage: python imu_watch.py [seconds]. Samples `imu` twice per second and prints
valid flag + seq counters so a dropout is visible within ~1 s.
"""
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
DUR = float(sys.argv[1]) if len(sys.argv) > 1 else 25.0

ser = open_noreset("COM5")
try:
    t0 = time.time()
    while time.time() - t0 < 5.0:   # boot (board level)
        ser.read(4096)
    print("=== WATCHING - flip motor power ON in ~5 s ===", flush=True)
    t0 = time.time()
    while time.time() - t0 < DUR:
        ser.write(b"imu\n")
        time.sleep(0.5)
        out = ser.read(8192).decode(errors="replace")
        m = re.search(r"valid=(\d).*?grv_seq=(\d+) gyro_seq=(\d+)", out)
        t = time.time() - t0
        if m:
            print(f"t={t:5.1f}s valid={m.group(1)} grv={m.group(2)} gyro={m.group(3)}", flush=True)
        else:
            print(f"t={t:5.1f}s NO RESPONSE", flush=True)
finally:
    ser.close()
