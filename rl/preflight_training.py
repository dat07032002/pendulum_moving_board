"""Fast, non-training preflight for corrected-plant 2D retraining."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from sb3_contrib import TQC

from critic_warmup_2d import reinit_critic
from furuta_env_2d import Furuta2DEnv
from retention_tqc import RetentionTQC
from train_c360_finetune_2d import STAGES_C360
from train_phaseb_2d import STAGES_PM10_60
from verify_2d import GRID_PM10_60


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WARMSTART = HERE / "models" / "up15_best.zip"
SYSID = ROOT / "sysid.json"


def close(a: float, b: float, rtol: float = 1e-5) -> bool:
    return abs(a - b) <= rtol * max(abs(a), abs(b), 1e-12)


def main() -> None:
    os.environ["FURUTA_VMAX"] = "10"
    os.environ["FURUTA_UP_THRESH"] = str(np.cos(np.deg2rad(10.0)))
    os.environ["FURUTA_CABLE_LIMIT_DEG"] = "360"
    os.environ["FURUTA_SUCCESS_ARM_LIMIT_DEG"] = "330"
    os.environ["FURUTA_ARM_CENTER_W"] = "0.02"
    os.environ["FURUTA_TIGHT_UPRIGHT_W"] = "0.25"
    os.environ["FURUTA_TIGHT_UPRIGHT_SCALE_DEG"] = "10"
    os.environ["FURUTA_CABLE_WARNING_W"] = "0.20"
    os.environ["FURUTA_CABLE_WARNING_START_DEG"] = "270"

    with SYSID.open() as f:
        pole = json.load(f)["friction_id"]

    assert WARMSTART.exists(), WARMSTART
    assert max(stage[3] for stage in STAGES_PM10_60) <= 60.0
    assert max(row[3] for row in GRID_PM10_60) <= 60.0
    assert max(stage[1] for stage in STAGES_PM10_60) <= 10.0
    assert max(stage[1] for stage in STAGES_C360) <= 10.0
    assert max(stage[3] for stage in STAGES_C360) <= 60.0
    assert all(0.0 < stage[4] <= 1.0 for stage in STAGES_C360)

    env = Furuta2DEnv(randomize=False, max_seconds=0.05)
    obs, _ = env.reset(seed=123)
    assert obs.shape == (10,) and np.isfinite(obs).all()
    assert close(env.v_max, 10.0)
    assert close(env.up_thresh, np.cos(np.deg2rad(10.0)))
    assert close(env.arm_limit, 2 * np.pi)
    assert close(env.success_arm_limit, np.deg2rad(330.0))
    assert close(env.arm_center_w, 0.02)
    assert close(env.tight_upright_w, 0.25)
    assert close(env.cable_warning_w, 0.20)
    assert close(env.cable_warning_start, np.deg2rad(270.0))
    assert close(env.imu.period, 0.01)
    assert close(env.nom["dmp_p"], pole["b_theta"])
    assert close(env.nom["fr_p"], pole["Tf"])
    assert close(sum(env.dr_pole_damping_range) / 2, env.nom["dmp_p"], rtol=0.01)
    assert close(sum(env.dr_pole_friction_range) / 2, env.nom["fr_p"], rtol=0.01)

    model = TQC.load(WARMSTART, device="cpu")
    assert model.observation_space.shape == (10,)
    assert model.action_space.shape == (1,)
    assert close(float(model.gamma), 0.99)
    action, _ = model.predict(obs, deterministic=True)
    for _ in range(10):
        obs, reward, terminated, truncated, _ = env.step(action)
        assert np.isfinite(obs).all() and np.isfinite(reward)
        if terminated or truncated:
            break
    env.close()

    os.environ["FURUTA_CABLE_LIMIT_DEG"] = "none"
    os.environ["FURUTA_SUCCESS_ARM_LIMIT_DEG"] = "none"
    free_env = Furuta2DEnv(randomize=False, max_seconds=0.01)
    assert free_env.arm_limit is None and free_env.success_arm_limit is None
    assert close(free_env.arm_center_w, 0.02)
    free_env.close()

    dr_env = Furuta2DEnv(randomize=True, max_seconds=0.01)
    delays = set()
    for seed in range(100):
        dr_env.reset(seed=seed)
        delays.add(dr_env._delay)
    assert delays == {1, 2}, delays
    dr_env.close()

    retained = RetentionTQC.load(WARMSTART, device="cpu")
    assert len(retained.critic.optimizer.state) > 0
    reinit_critic(retained)
    assert len(retained.critic.optimizer.state) == 0

    print("training preflight: PASS")
    print(f"warmstart={WARMSTART.name} obs=10 action=1 gamma={model.gamma:g}")
    print("envelope=+/-10 deg, reference speed <=60 deg/s, upright=+/-10 deg")
    print("A/B=cable 360 deg + success 330 deg versus free arm; center weight=0.02")
    print("fine-tune=tight-up 0.25; cable warning 0.20 from 270 deg; mastery only")
    print("action-delay DR=1-2 control steps (5-10 ms); 3 steps excluded by hardware test")
    print(
        f"pole damping={env.nom['dmp_p']:.7g} friction={env.nom['fr_p']:.7g}; "
        "IMU=100 Hz"
    )


if __name__ == "__main__":
    main()
