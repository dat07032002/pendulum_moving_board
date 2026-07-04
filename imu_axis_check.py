"""Guided BNO086 roll/pitch axis and sign verification.

Keep the motor off. The script captures:
  1. level,
  2. a smooth move from level to known physical +roll,
  3. level again,
  4. a smooth move from level to known physical +pitch.

Use the right-hand rule about the board-frame +X axis for +roll and +Y for
+pitch. Hold the final angle with a protractor so magnitude can be checked.
"""
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


def capture(ser: serial.Serial, seconds: float, label: str) -> list[dict]:
    rows: list[dict] = []
    last_print = 0.0
    ser.reset_input_buffer()
    ser.write(b"log\n")
    ser.flush()
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        row = config.parse_log(ser.readline().decode("ascii", errors="ignore"))
        if row is not None and "imu_seq" in row:
            rows.append(row)
            now = time.perf_counter()
            if now - last_print >= 0.1:
                print(
                    f"\r    {label}: roll={np.rad2deg(row['imu_roll']):+6.2f}°  "
                    f"pitch={np.rad2deg(row['imu_pitch']):+6.2f}°  "
                    f"gyro-X={np.rad2deg(row['gyro_x']):+6.1f}°/s  "
                    f"gyro-Y={np.rad2deg(row['gyro_y']):+6.1f}°/s",
                    end="",
                    flush=True,
                )
                last_print = now
    ser.write(b"nolog\n")
    ser.flush()
    print()
    if not rows:
        raise RuntimeError("No IMU log rows received")
    return rows


def capture_to_target(
    ser: serial.Serial,
    target_deg: float,
    label: str,
    baseline: tuple[float, float],
    timeout_s: float = 60.0,
) -> list[dict]:
    """Capture until either orientation axis reaches the target and is held."""
    rows: list[dict] = []
    hold_start: float | None = None
    accepted = False
    last_print = 0.0
    ser.reset_input_buffer()
    ser.write(b"log\n")
    ser.flush()
    end = time.perf_counter() + timeout_s
    while time.perf_counter() < end:
        row = config.parse_log(ser.readline().decode("ascii", errors="ignore"))
        if row is None or "imu_seq" not in row:
            continue
        rows.append(row)
        roll = np.rad2deg(row["imu_roll"])
        pitch = np.rad2deg(row["imu_pitch"])
        gyro_x = np.rad2deg(row["gyro_x"])
        gyro_y = np.rad2deg(row["gyro_y"])
        delta = max(abs(roll - baseline[0]), abs(pitch - baseline[1]))
        on_target = target_deg - 2.0 <= delta <= target_deg + 5.0
        steady = max(abs(gyro_x), abs(gyro_y)) < 3.0
        now = time.perf_counter()
        if on_target and steady:
            hold_start = hold_start or now
        else:
            hold_start = None
        held_s = 0.0 if hold_start is None else now - hold_start
        if now - last_print >= 0.1:
            print(
                f"\r    {label}: roll={roll:+6.2f}°  pitch={pitch:+6.2f}°  "
                f"gyro-X={gyro_x:+6.1f}°/s  gyro-Y={gyro_y:+6.1f}°/s  "
                f"hold={held_s:3.1f}/1.0 s",
                end="",
                flush=True,
            )
            last_print = now
        if held_s >= 1.0:
            accepted = True
            break
    ser.write(b"nolog\n")
    ser.flush()
    print()
    if not rows or not accepted:
        raise RuntimeError(f"{label} target was not reached and held within {timeout_s:.0f} s")
    print(f"    {label} target held — accepted")
    return rows


def steady_mean(rows: list[dict], tail_s: float = 0.75) -> tuple[float, float]:
    newest = rows[-1]["t_ms"]
    tail = [r for r in rows if newest - r["t_ms"] <= tail_s * 1000]
    return tuple(np.rad2deg(np.mean([r[k] for r in tail])) for k in ("imu_roll", "imu_pitch"))


def summarize(name: str, rows: list[dict], baseline: tuple[float, float]) -> None:
    roll, pitch = steady_mean(rows)
    gx = np.rad2deg([r["gyro_x"] for r in rows])
    gy = np.rad2deg([r["gyro_y"] for r in rows])
    print(
        f"{name:8s}: final Δroll={roll-baseline[0]:+6.2f}°  "
        f"Δpitch={pitch-baseline[1]:+6.2f}°  "
        f"gyro-X=[{gx.min():+6.1f},{gx.max():+6.1f}]°/s  "
        f"gyro-Y=[{gy.min():+6.1f},{gy.max():+6.1f}]°/s"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Guided BNO086 axis/sign check")
    ap.add_argument("--port", default=config.PORT)
    ap.add_argument("--angle-deg", type=float, default=10.0)
    ap.add_argument("--save", default="imu_axis_check.csv")
    args = ap.parse_args()

    print("Motor stays OFF. Use right-hand-rule physical +roll and +pitch.")
    print(f"Use the live IMU display to reach approximately {args.angle_deg:g}° and hold.")
    ser = open_without_reset(args.port)
    all_rows: list[dict] = []
    try:
        ser.write(b"s\n")
        input("\nHold the board LEVEL and still, then press Enter...")
        level = capture(ser, 1.5, "LEVEL ")
        all_rows += [dict(r, phase="level") for r in level]
        baseline = steady_mean(level)
        print(f"Level baseline: roll={baseline[0]:+.2f}°, pitch={baseline[1]:+.2f}°")

        input("\nReturn level. Press Enter, then smoothly move to physical +ROLL and hold...")
        roll = capture_to_target(ser, args.angle_deg, "+ROLL ", baseline)
        all_rows += [dict(r, phase="plus_roll") for r in roll]
        summarize("+ROLL", roll, baseline)

        input("\nReturn LEVEL and still, then press Enter...")
        level2 = capture(ser, 1.5, "LEVEL ")
        all_rows += [dict(r, phase="level_2") for r in level2]

        input("\nPress Enter, then smoothly move to physical +PITCH and hold...")
        pitch = capture_to_target(ser, args.angle_deg, "+PITCH", baseline)
        all_rows += [dict(r, phase="plus_pitch") for r in pitch]
        summarize("+PITCH", pitch, baseline)
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
