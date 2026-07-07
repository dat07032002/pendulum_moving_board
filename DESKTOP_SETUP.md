# Setting up on a new desktop

## 1. Clone

```powershell
git clone https://github.com/dat07032002/pendulum_moving_board.git tilt_pendulum_2d
cd tilt_pendulum_2d
```

## 2. Python (3.10+; 3.12 known-good)

```powershell
pip install numpy gymnasium mujoco sb3-contrib torch pyserial
```

(`sb3-contrib` pulls stable-baselines3; GPU torch optional — everything rig-side runs on CPU.)

## 3. Firmware toolchain (only for flashing/building)

```powershell
winget install ArduinoSA.CLI          # or download arduino-cli manually
arduino-cli core install esp32:esp32
arduino-cli lib install "SparkFun BNO08x Cortex Based IMU"
```

Build and flash the CURRENT policy (already in `firmware/furuta_foc/policy_weights.h`,
tight8_s2 sha 61e27050, RL_OBS=12):

```powershell
arduino-cli compile --fqbn esp32:esp32:esp32 firmware/furuta_foc
arduino-cli upload -p COM5 --fqbn esp32:esp32:esp32 firmware/furuta_foc   # adjust COM port
```

Find the COM port: `python -c "import serial.tools.list_ports as l; [print(p) for p in l.comports()]"`

## 4. Rig commands (tools/ — all open serial WITHOUT resetting the ESP32)

All default to COM5; edit the constant or pass --port where supported. Motor stays off for
everything except `rl_stage.py`.

```powershell
# health / IMU / one-shot obs+action check (motor off)
python tools/serial_cmd.py --port COM5 --boot-wait 1 "health" "imu" "rlcheck"

# live pendulum angle (hanging=+/-180, upright=0); use for calibration checks
python tools/theta_live.py 60

# recalibrate the pole reference (pendulum hanging DEAD STILL)
python tools/serial_cmd.py --port COM5 --boot-wait 1 "calhang"

# raw AS5600 counts (bypasses the software glitch filter; full turn must sweep 0-4095)
python tools/raw_live.py 60

# POWERED 30 s policy run, logs archived (motor power ON, arm at cable center,
# pendulum hanging, board level, hand near stop)
python tools/rl_stage.py 10 30 eval\hw_bringup\my_run.txt

# analyze a run: holds/drops, vibration, oscillation spectrum
python tools/analyze_balance.py eval\hw_bringup\my_run.txt
python tools/analyze_vibration.py eval\hw_bringup\my_run.txt
python tools/analyze_limit_cycle.py eval\hw_bringup\my_run.txt
```

Note: `rl_stage.py` and some analyzers keep a hardcoded COM5 / scratch conventions — grep for
"COM5" if your port differs.

## 5. Training server (optional)

Requires the UT VPN and the SSH key (copy `~/.ssh/aere_codex_ed25519` manually — it is NOT in
the repo):

```powershell
ssh -i $HOME/.ssh/aere_codex_ed25519 tn22833@aere-a83514.ae.utexas.edu
# project: ~/furuta_tilt_2d/rl   python: ~/furuta_rl/.venv/bin/python
```

Sync files with scp (the server is not a git checkout). Launch training from `~/furuta_tilt_2d/rl`
via the `launch_*.sh` scripts.

## 6. Known rig gotchas (see SESSION_* docs for detail)

- **BNO086 warm-boot wedge**: if boot says "BNO086 missing", full power cycle (USB out 10 s).
  The `rl` command refuses to run without a healthy IMU, so this fails safe.
- **Pole calibration**: after ANY pole/magnet/bearing work, `calhang` + verify with
  `theta_live.py` (hanging +/-180, upright ~0, full turn sweeps smoothly).
- **AS5600 magnet gap is marginal** (AGC ~120/128): if theta compresses (full turn reads
  a few degrees), check the encoder cable isn't twisted/dragging and the gap is 0.5-2 mm.
- Keep the board LEVEL during boot (IMU level tare) and the ARM AT CABLE CENTER before `rl`.
- Policies and firmware are paired through `policy_weights.h` (RL_OBS/RL_VMAX/sha embedded).
  Always regenerate via `python rl/export_policy.py --model <ckpt.zip> --vmax 10 --out
  firmware/furuta_foc/policy_weights.h` — never hand-edit.
