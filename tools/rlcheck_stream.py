"""Sample rlcheck repeatedly (motor stays off) and print the board-tilt obs channels.

Usage: python rlcheck_stream.py [seconds]
Opens COM5 (board resets + 500 ms level tare -- KEEP BOARD LEVEL for first ~6 s),
then samples rlcheck ~4x/s and prints obs[6](roll) obs[7](pitch) obs[8](gyro_x) obs[9](gyro_y).
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
DUR = float(sys.argv[1]) if len(sys.argv) > 1 else 15.0

ser = open_noreset("COM5")
try:
    t0 = time.time()
    while time.time() - t0 < 5.0:   # boot + tare, board must be level
        ser.read(4096)
    print("=== ROCK THE AXIS NOW ===", flush=True)
    t0 = time.time()
    buf = b""
    while time.time() - t0 < DUR:
        ser.write(b"rlcheck\n")
        time.sleep(0.25)
        buf += ser.read(16384)
    for line in buf.decode(errors="replace").splitlines():
        m = re.search(r"obs=\[([^\]]+)\]", line)
        if m:
            o = [float(x) for x in m.group(1).split(",")]
            print(f"roll={o[6]:+.3f} pitch={o[7]:+.3f} gx={o[8]:+.3f} gy={o[9]:+.3f}")
finally:
    ser.close()
