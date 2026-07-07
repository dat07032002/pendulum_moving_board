"""Arm/motor friction identification: breakaway (stiction) + steady-speed curve.

Usage: python arm_friction_id.py [outfile_prefix]
Setup: motor power ON, pendulum hanging, ARM AT CABLE CENTER, hand on cutoff.
Safety: every burst is time-boxed; a position guard sends 's' if the arm strays
more than GUARD_DEG from where the script started; 't 0' + 's' in finally.

Test A - breakaway: at the current position, ramp V slowly (0.05 V / 100 ms)
until the arm moves; record the breakaway voltage. 8 ramps, alternating sign.
Test B - steady speed: constant-V bursts (+/-0.8..3.0 V), record steady phi_dot.
Fit: gear*V = b*w + Tf*sign(w) -> viscous b and Coulomb Tf, vs sim nominals.
"""
import math
import sys
import time

import serial

GEAR = 0.0127          # N*m per volt (sim motor constant)
GUARD_DEG = 280.0      # hard stop if |phi - phi0| exceeds this
BREAK_DPHI_DEG = 2.0   # arm moved this much => breakaway
BREAK_PHID = 0.30      # or arm speed exceeds this [rad/s]

PREFIX = sys.argv[1] if len(sys.argv) > 1 else "eval/hw_bringup/arm_friction"
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def open_noreset(port):
    s = serial.Serial()
    s.port = port
    s.baudrate = 921600
    s.timeout = 0.02
    s.dtr = False
    s.rts = False
    s.open()
    return s


ser = open_noreset("COM5")
tail = ""
state = {"phi": None, "phi_dot": None, "t": None}
rawlog = open(f"{PREFIX}_raw.txt", "w", encoding="utf-8")


def pump(seconds=0.05):
    """Read the log stream, update latest state."""
    global tail
    t0 = time.time()
    while True:
        chunk = ser.read(16384).decode(errors="replace")
        if chunk:
            rawlog.write(chunk)
            tail += chunk
            lines = tail.split("\n")
            tail = lines[-1]
            for line in lines[:-1]:
                if line.startswith("log=["):
                    try:
                        r = line.strip()[5:-1].split(",")
                        state["t"] = int(r[0])
                        state["phi"] = float(r[1])
                        state["phi_dot"] = float(r[3])
                    except (ValueError, IndexError):
                        pass
        if time.time() - t0 >= seconds:
            return


def cmd(c):
    ser.write((c + "\n").encode())


def wait_state(timeout=3.0):
    t0 = time.time()
    while state["phi"] is None and time.time() - t0 < timeout:
        pump(0.05)
    if state["phi"] is None:
        raise RuntimeError("no log data - is the board on COM5 alive?")


def guard_ok(phi0):
    dev = abs(math.degrees(state["phi"] - phi0))
    if dev > GUARD_DEG:
        cmd("t 0"); cmd("s")
        raise RuntimeError(f"position guard: {dev:.0f} deg from start")
    return True


def settle(phi0, timeout=4.0):
    cmd("t 0")
    t0 = time.time()
    while time.time() - t0 < timeout:
        pump(0.1)
        guard_ok(phi0)
        if abs(state["phi_dot"]) < 0.15:
            return


try:
    pump(0.5)
    cmd("nolog"); pump(0.2)
    cmd("s"); pump(0.2)
    cmd("vlim 3"); pump(0.2)
    cmd("log"); pump(0.5)
    wait_state()
    phi0 = state["phi"]
    print(f"start phi={math.degrees(phi0):+.1f} deg (guard at +/-{GUARD_DEG:.0f})")

    # ---- Test A: breakaway ramps ----
    breakaways = []
    sign = 1.0
    for ramp in range(8):
        settle(phi0)
        pump(0.3)
        p_start = state["phi"]
        v = 0.0
        vbreak = None
        while v < 3.0:
            v = round(v + 0.05, 2)
            cmd(f"t {sign * v:.2f}")
            pump(0.10)
            guard_ok(phi0)
            moved = abs(math.degrees(state["phi"] - p_start)) > BREAK_DPHI_DEG
            if moved or abs(state["phi_dot"]) > BREAK_PHID:
                vbreak = v
                break
        cmd("t 0")
        pos = math.degrees(p_start - phi0)
        breakaways.append((pos, sign, vbreak))
        print(f"ramp {ramp + 1}: pos={pos:+7.1f} deg dir={sign:+.0f} "
              f"breakaway={'%.2f V' % vbreak if vbreak else '>3 V'}")
        sign = -sign

    # ---- Test B: steady-speed bursts ----
    bursts = []
    for v in (0.8, 1.2, 1.7, 2.3, 3.0):
        for sign in (1.0, -1.0):
            settle(phi0)
            cmd(f"t {sign * v:.2f}")
            t0 = time.time()
            speeds = []
            while time.time() - t0 < 2.0:
                pump(0.05)
                guard_ok(phi0)
                if time.time() - t0 > 0.8:          # past acceleration transient
                    speeds.append(state["phi_dot"])
            cmd("t 0")
            if speeds:
                mid = sorted(speeds)[len(speeds) // 2]
                bursts.append((sign * v, mid))
                print(f"burst V={sign * v:+.1f}: steady phi_dot={mid:+.2f} rad/s")
    settle(phi0)
finally:
    try:
        cmd("t 0"); cmd("s"); pump(0.3)
        cmd("nolog"); pump(0.3)
        cmd("s")
    finally:
        ser.close()
        rawlog.close()

# ---- fits ----
import numpy as np

with open(f"{PREFIX}_results.csv", "w", encoding="utf-8") as f:
    f.write("test,value1,value2,value3\n")
    for pos, sign, vb in breakaways:
        f.write(f"breakaway,{pos:.1f},{sign:.0f},{vb if vb else ''}\n")
    for v, w in bursts:
        f.write(f"burst,{v:.2f},{w:.4f},\n")

vb = [b[2] for b in breakaways if b[2]]
if vb:
    print(f"\nbreakaway: mean={np.mean(vb):.2f} V  range={min(vb):.2f}-{max(vb):.2f} V")
    print(f"  -> stiction torque ~{np.mean(vb) * GEAR * 1e3:.2f} mN*m "
          f"(sim frictionloss=6.0 mN*m predicts {6e-3 / GEAR:.2f} V)")
data = [(v, w) for v, w in bursts if abs(w) > 0.5]
if len(data) >= 4:
    V = np.array([d[0] for d in data])
    W = np.array([d[1] for d in data])
    # gear*V = b*w + Tf*sign(w)
    A = np.column_stack([W, np.sign(W)])
    (b_fit, tf_fit), *_ = np.linalg.lstsq(A, GEAR * V, rcond=None)
    print(f"steady-speed fit: b={b_fit:.2e} N*m*s/rad (sim 9.4e-4), "
          f"Tf={tf_fit * 1e3:.2f} mN*m (sim 6.0)")
print(f"saved {PREFIX}_results.csv and {PREFIX}_raw.txt")
