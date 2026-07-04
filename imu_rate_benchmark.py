"""Measure delivered BNO086 rates and 200 Hz control-loop timing with the motor off."""

from __future__ import annotations

import argparse
import time

import numpy as np

from config import Link


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="COM5")
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--requested-hz", type=float, default=100.0)
    args = parser.parse_args()

    rows: list[dict] = []
    with Link(args.port) as link:
        link.send("s")
        link.send("nolog")
        time.sleep(0.2)
        link.ser.reset_input_buffer()
        link.send("log")
        deadline = time.perf_counter() + args.seconds
        while time.perf_counter() < deadline:
            row = link.read_log(timeout=0.5)
            if row is not None:
                rows.append(row)
        link.send("nolog")
        link.send("s")

    if len(rows) < 100:
        raise SystemExit(f"FAIL: only {len(rows)} log rows received")
    if "imu_gyro_seq" not in rows[0]:
        raise SystemExit("FAIL: firmware does not expose imu_gyro_seq; flash benchmark build")

    t_ms = np.asarray([r["t_ms"] for r in rows], dtype=np.int64)
    dt_ms = np.diff(t_ms).astype(float)
    elapsed_s = (t_ms[-1] - t_ms[0]) / 1000.0
    grv = np.asarray([r["imu_seq"] for r in rows], dtype=np.int64)
    gyro = np.asarray([r["imu_gyro_seq"] for r in rows], dtype=np.int64)
    grv_hz = (grv[-1] - grv[0]) / elapsed_s
    gyro_hz = (gyro[-1] - gyro[0]) / elapsed_s
    loop_hz = (len(rows) - 1) / elapsed_s
    missed = int(np.count_nonzero(dt_ms > 7.5))

    print(f"\n===== BNO086 {args.requested_hz:g} Hz request benchmark =====")
    print(f"device window : {elapsed_s:.3f} s, {len(rows)} control ticks")
    print(f"control loop  : {loop_hz:.2f} Hz")
    print(
        "loop dt       : "
        f"p50={np.percentile(dt_ms, 50):.1f} "
        f"p95={np.percentile(dt_ms, 95):.1f} "
        f"p99={np.percentile(dt_ms, 99):.1f} "
        f"max={np.max(dt_ms):.1f} ms"
    )
    print(f"deadline gaps : {missed} intervals >7.5 ms ({100*missed/len(dt_ms):.3f}%)")
    print(f"GRV delivered : {grv_hz:.2f} Hz")
    print(f"gyro delivered: {gyro_hz:.2f} Hz")

    passed = (
        loop_hz >= 195.0
        and grv_hz >= 0.90 * args.requested_hz
        and gyro_hz >= 0.85 * args.requested_hz
        and missed / len(dt_ms) <= 0.001
    )
    print("result        :", "PASS" if passed else "FAIL for requested rate")


if __name__ == "__main__":
    main()
