"""
config.py — serial link + shared helpers for the system-ID scripts (foc branch).

Mirrors main:config.py: one home for the serial port/baud, a small Link wrapper
around the ESP32, the log=[...] line parser, and load/save of measured parameters
to sysid.json. The ID scripts (freeswing.py, arm_id.py, sign_check.py) all import
from here so the link behaviour and the log format live in exactly one place.

The firmware streams, when `log` is enabled, one line per control tick:
    log=[t_ms, phi, theta, phi_dot, theta_dot, V, theta_raw,
         imu_roll, imu_pitch, gyro_x, gyro_y, gyro_z, imu_seq, imu_gyro_seq]
"""
from __future__ import annotations

import json
import os
import re
import time

import serial
import serial.tools.list_ports

# --- serial link to the ESP32 ---
PORT = "COM5"            # override with --port on any script
BAUD = 921600            # must match Serial.begin() in furuta_foc.ino
CONTROL_DT = 1.0 / 200.0  # firmware loop period (200 Hz)

# Legacy firmware produced 7 fields. BNO086 firmware appends 6 IMU fields.
LOG_RE = re.compile(r"log=\[([^\]]+)\]")
LOG_FIELDS_LEGACY = ("t_ms", "phi", "theta", "phi_dot", "theta_dot", "V", "theta_raw")
LOG_FIELDS_IMU = LOG_FIELDS_LEGACY + (
    "imu_roll", "imu_pitch", "gyro_x", "gyro_y", "gyro_z", "imu_seq",
)
LOG_FIELDS_IMU_DUAL_SEQ = LOG_FIELDS_IMU + ("imu_gyro_seq",)
AS5600_CPR = 4096

_SYSID_FILE = os.path.join(os.path.dirname(__file__), "sysid.json")


def autodetect_port() -> str | None:
    """Best-effort USB-serial autodetect (same heuristic as main:check_rate.py)."""
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        desc = f"{p.description} {p.manufacturer or ''}".lower()
        if any(k in desc for k in ("cp210", "ch340", "ch910", "silicon labs", "usb-serial", "uart")):
            return p.device
    return ports[0].device if ports else None


def parse_log(line: str) -> dict | None:
    """Parse one 'log=[...]' line into a dict, or None if it isn't one / is malformed."""
    m = LOG_RE.search(line)
    if not m:
        return None
    parts = m.group(1).split(",")
    if len(parts) == len(LOG_FIELDS_LEGACY):
        fields = LOG_FIELDS_LEGACY
    elif len(parts) == len(LOG_FIELDS_IMU):
        fields = LOG_FIELDS_IMU
    elif len(parts) == len(LOG_FIELDS_IMU_DUAL_SEQ):
        fields = LOG_FIELDS_IMU_DUAL_SEQ
    else:
        return None
    try:
        vals = [float(p) for p in parts]
    except ValueError:
        return None
    d = dict(zip(fields, vals))
    d["t_ms"] = int(d["t_ms"])
    d["theta_raw"] = int(d["theta_raw"])
    if "imu_seq" in d:
        d["imu_seq"] = int(d["imu_seq"])
    if "imu_gyro_seq" in d:
        d["imu_gyro_seq"] = int(d["imu_gyro_seq"])
    return d


class Link:
    """Thin serial wrapper: open without resetting the ESP32, send commands, read logs.

    Always call stop() (or use as a context manager) so the motor is disabled and
    the log stream is turned off on exit — even on Ctrl-C / exception.
    """

    def __init__(self, port: str | None = None, baud: int = BAUD, boot_timeout: float = 16.0):
        self.port = port or PORT
        # Open with DTR/RTS inactive. Toggling those lines resets common ESP32
        # dev boards; an MCU-only reset can leave a powered BNO086 out of sync.
        self.ser = serial.Serial()
        self.ser.port = self.port
        self.ser.baudrate = baud
        self.ser.timeout = 0.2
        self.ser.dtr = False
        self.ser.rts = False
        self.ser.open()
        self._wait_ready(boot_timeout)

    def _wait_ready(self, timeout: float) -> bool:
        """Wait for a live parameter response or a post-boot banner."""
        print(f"   connecting to live ESP32 (up to {timeout:.0f}s)...")
        self.ser.reset_input_buffer()
        self.ser.write(b"s\nparams\n")
        self.ser.flush()
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < timeout:
            raw = self.ser.readline()
            if not raw:
                continue
            line = raw.decode("utf-8", errors="replace")
            if "cmds:" in line or "Furuta FOC" in line or line.startswith("# K="):
                self.ser.reset_input_buffer()
                return True
        self.ser.reset_input_buffer()    # proceed; caller will report if logging is unavailable
        return False

    # -- commands --
    def send(self, line: str) -> None:
        self.ser.write((line.rstrip("\n") + "\n").encode("ascii"))
        self.ser.flush()

    def torque(self, volts: float) -> None:
        self.send(f"t {volts:.4f}")

    def stop_motor(self) -> None:
        self.send("s")

    def log_on(self) -> None:
        self.send("log")

    def log_off(self) -> None:
        self.send("nolog")

    # -- reads --
    def read_log(self, timeout: float = 0.2) -> dict | None:
        """Return the next valid log sample within `timeout`, else None."""
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            raw = self.ser.readline()
            if not raw:
                continue
            d = parse_log(raw.decode("utf-8", errors="replace"))
            if d is not None:
                return d
        return None

    def drain_until_logging(self, timeout: float = 3.0) -> bool:
        """Wait for the first valid log line (skips the boot banner)."""
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < timeout:
            if self.read_log(timeout=0.2) is not None:
                return True
        return False

    def capture(self, seconds: float) -> list[dict]:
        """Collect every log sample for `seconds` (host-clock windowed)."""
        out: list[dict] = []
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < seconds:
            d = self.read_log(timeout=0.2)
            if d is not None:
                out.append(d)
        return out

    # -- lifecycle --
    def stop(self) -> None:
        try:
            self.log_off()
            self.stop_motor()
            time.sleep(0.05)
            self.stop_motor()
        finally:
            self.ser.close()

    def __enter__(self) -> "Link":
        return self

    def __exit__(self, *exc) -> None:
        self.stop()


# --- measured-parameter persistence (mirrors main:config.save_calibration) ---
def load_sysid() -> dict:
    if os.path.exists(_SYSID_FILE):
        with open(_SYSID_FILE) as f:
            return json.load(f)
    return {}


def save_sysid(updates: dict) -> str:
    """Merge `updates` into sysid.json and write it back. Returns the path."""
    d = load_sysid()
    d.update(updates)
    with open(_SYSID_FILE, "w") as f:
        json.dump(d, f, indent=2)
    return _SYSID_FILE


if __name__ == "__main__":
    print(f"PORT={PORT}  BAUD={BAUD}  CONTROL_DT={CONTROL_DT:.5f}s")
    found = [p.device for p in serial.tools.list_ports.comports()]
    print(f"serial ports: {found or '(none)'}   autodetect -> {autodetect_port()}")
    s = load_sysid()
    print(f"sysid.json: {s if s else '(empty - run the ID scripts)'}")
