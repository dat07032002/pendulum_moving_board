"""Live pendulum angle from the 200 Hz log stream, ~10 updates/s.

Usage: python theta_live.py [seconds]   (Ctrl+C safe; sends nolog on exit)
"""
import math
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
DUR = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0

ser = open_noreset("COM5")
try:
    ser.write(b"log\n")
    t0 = time.time()
    last_print = 0.0
    tail = ""
    while time.time() - t0 < DUR:
        tail += ser.read(16384).decode(errors="replace")
        lines = tail.split("\n")
        tail = lines[-1]
        latest = None
        for line in lines[:-1]:
            if line.startswith("log=["):
                try:
                    r = line.strip()[5:-1].split(",")
                    latest = (float(r[2]), float(r[4]))  # theta, theta_dot
                except (ValueError, IndexError):
                    pass
        now = time.time()
        if latest and now - last_print > 0.1:
            th = math.degrees(math.atan2(math.sin(latest[0]), math.cos(latest[0])))
            print(f"theta = {th:+7.1f} deg   theta_dot = {latest[1]:+6.2f} rad/s",
                  flush=True)
            last_print = now
        time.sleep(0.02)
finally:
    try:
        ser.write(b"nolog\n")
        time.sleep(0.2)
    finally:
        ser.close()
