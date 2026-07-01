# Preserved 1D baseline

The original one-axis project remains at:

`C:\Users\thanh\Desktop\tilt_pendulum`

Do not convert that folder's MuJoCo model or observation space to two axes. It is the reproducible
1D reference.

## Recommended preserved policy

- Model: `rl/models/clean20_master_verified91p5.zip`
- SHA-256: `775afbb5cf1553becb347c422edc6c03300990134d45c7d2b567f7a9db849d3a`
- Task: free-arm, clean plant, random one-axis board tilt up to ±20°
- Verified sustained success: 91.5% over 1,000 episodes
- Verified performance with all DR except action-delay randomization: approximately 93%

The five fixed-action-delay residual policies scored 91.4–92.8% under the same non-delay DR
condition. They did not materially improve on the simpler clean master, so the clean master is the
preferred warm start for two-axis work.

Visual references:

- `clean20_verified91p5.gif`
- `residual_nodelay_s4_verified.gif`

## 2D development rule

Create new 2D-specific files instead of replacing the 1D implementation:

- `rl/furuta_2d.xml`
- `rl/tilt_2d.py`
- `rl/furuta_env_2d.py`
- `rl/train_tqc_2d.py`
- `rl/eval_policy_2d.py`

The first 2D milestone must reproduce the existing one-axis result with the second axis locked at
zero. This parity test catches geometry, sign, observation-order, and warm-start errors before true
two-axis training begins.
