"""Guided BNO086 in-motion characterization with the motor disabled."""
from __future__ import annotations

import argparse
import csv
import time

import numpy as np
import serial

import config


def open_without_reset(port: str) -> serial.Serial:
    ser = serial.Serial()
    ser.port = port
    ser.baudrate = config.BAUD
    ser.timeout = 0.03
    ser.dtr = False
    ser.rts = False
    ser.open()
    return ser


def capture(ser: serial.Serial, phase: str, seconds: float) -> list[dict]:
    rows: list[dict] = []
    last_print = 0.0
    ser.reset_input_buffer()
    ser.write(b"log\n")
    ser.flush()
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        row = config.parse_log(ser.readline().decode("ascii", errors="ignore"))
        if row is None or "imu_seq" not in row:
            continue
        rows.append(dict(row, phase=phase))
        now = time.perf_counter()
        if now - last_print >= 0.1:
            print(
                f"\r    {phase:12s}  roll={np.rad2deg(row['imu_roll']):+6.1f}°  "
                f"pitch={np.rad2deg(row['imu_pitch']):+6.1f}°  "
                f"gx={np.rad2deg(row['gyro_x']):+7.1f}°/s  "
                f"gy={np.rad2deg(row['gyro_y']):+7.1f}°/s",
                end="",
                flush=True,
            )
            last_print = now
    ser.write(b"nolog\n")
    ser.flush()
    print()
    if not rows:
        raise RuntimeError(f"No IMU data during {phase}")
    return rows


def summarize(phase: str, rows: list[dict], primary: str) -> None:
    angle_key = "imu_roll" if primary == "roll" else "imu_pitch"
    gyro_key = "gyro_x" if primary == "roll" else "gyro_y"
    cross_angle = "imu_pitch" if primary == "roll" else "imu_roll"
    cross_gyro = "gyro_y" if primary == "roll" else "gyro_x"
    angle = np.rad2deg([r[angle_key] for r in rows])
    gyro = np.rad2deg([r[gyro_key] for r in rows])
    angle_cross = np.rad2deg([r[cross_angle] for r in rows])
    gyro_cross = np.rad2deg([r[cross_gyro] for r in rows])
    print(
        f"{phase:12s}: {primary}=[{angle.min():+.1f},{angle.max():+.1f}]°  "
        f"gyro=[{gyro.min():+.1f},{gyro.max():+.1f}]°/s  "
        f"cross-angle-rms={np.sqrt(np.mean(angle_cross**2)):.2f}°  "
        f"cross-gyro-rms={np.sqrt(np.mean(gyro_cross**2)):.2f}°/s"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Guided BNO086 in-motion test")
    ap.add_argument("--port", default=config.PORT)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--save", default="imu_motion_test.csv")
    args = ap.parse_args()

    tests = [
        ("roll_slow", "roll", "rock ROLL ±10° slowly"),
        ("roll_medium", "roll", "rock ROLL ±10° at a moderate rate"),
        ("roll_fast", "roll", "rock ROLL ±10° quickly but safely"),
        ("pitch_slow", "pitch", "rock PITCH ±10° slowly"),
        ("pitch_medium", "pitch", "rock PITCH ±10° at a moderate rate"),
        ("pitch_fast", "pitch", "rock PITCH ±10° quickly but safely"),
    ]

    print("Motor stays OFF. Each segment starts only after you press Enter.")
    print(f"Move continuously for {args.seconds:g} seconds during each segment.")
    ser = open_without_reset(args.port)
    all_rows: list[dict] = []
    try:
        ser.write(b"s\n")
        for phase, primary, instruction in tests:
            input(f"\nReturn LEVEL. Press Enter, then {instruction}...")
            rows = capture(ser, phase, args.seconds)
            all_rows.extend(rows)
            summarize(phase, rows, primary)
    finally:
        ser.write(b"nolog\ns\n")
        ser.flush()
        ser.close()

    fields = ["phase", *config.LOG_FIELDS_IMU]
    with open(args.save, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nSaved raw capture -> {args.save}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
