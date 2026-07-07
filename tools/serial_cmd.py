"""Send serial commands to the furuta ESP32 (COM5, 921600) and print replies.

Usage: python serial_cmd.py [--port COM5] [--boot-wait 4] "health" "imu" ...
Each command is sent, then output is drained until quiet for --quiet seconds.
Motor-off safe: does not send motion commands unless the user passes them.
"""
import argparse
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


def drain(ser, quiet_s=1.0, max_s=8.0):
    out = []
    t_last = time.time()
    t0 = t_last
    while True:
        chunk = ser.read(4096)
        now = time.time()
        if chunk:
            out.append(chunk)
            t_last = now
        if now - t0 > max_s or (not chunk and now - t_last > quiet_s):
            break
    return b"".join(out).decode(errors="replace")


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("commands", nargs="+")
    ap.add_argument("--port", default="COM5")
    ap.add_argument("--boot-wait", type=float, default=4.0,
                    help="seconds to wait after open (DTR reset -> boot + tare)")
    ap.add_argument("--quiet", type=float, default=1.0)
    ap.add_argument("--max", type=float, default=8.0)
    args = ap.parse_args()

    ser = open_noreset(args.port)
    try:
        time.sleep(args.boot_wait)
        boot = drain(ser, quiet_s=0.5, max_s=args.boot_wait)
        if boot.strip():
            print("=== boot ===")
            print(boot)
        for cmd in args.commands:
            print(f"=== > {cmd} ===")
            ser.write((cmd + "\n").encode())
            print(drain(ser, quiet_s=args.quiet, max_s=args.max))
    finally:
        ser.close()


if __name__ == "__main__":
    main()
