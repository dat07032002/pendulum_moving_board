# Tighter upright-tolerance experiment

Date: 2026-07-01

## Setup

All 5 seeds trained on the **±15° two-axis board at 120°/s** (the deployment envelope), at
**10 V**, teacher-free, with the validated Phase-B recipe (re-init critic, γ=0.99, gradient
clipping, gentle curriculum). The only thing varied was the **upright success tolerance**
(`FURUTA_UP_THRESH`) — how close to true gravity-vertical the pole must stay for ≥80% of the
final 2 s to count as success:

- **3 seeds at a 15° gate** (`FURUTA_UP_THRESH=0.966`) — `up15_s0/s1/s2`
- **2 seeds at a 10° gate** (`FURUTA_UP_THRESH=0.985`) — `up10_s3/s4`

For reference, all prior deployment work used a **~26° gate** (`up>0.90`).

## Results (best training-eval success, 30 ep, at the deploy condition both ±15° 90–120°/s)

| Upright tolerance | Seeds | Success |
|---|---|---:|
| ~26° (deployment standard) | — | ~96% (verified) |
| **15°** | up15_s0/s1/s2 | 0.70 / **0.80** / 0.70 |
| **10°** | up10_s3/s4 | 0.17 / **0.23** |

Best models: **`rl/models/up15_best.zip`** (15° gate, from up15_s1) and
**`rl/models/up10_best.zip`** (10° gate, from up10_s4).

## Takeaway

- **The pole can be held reliably within ~15° of true vertical through fast two-axis motion**
  (~0.75–0.80 at the hard deploy condition, 0.80–0.90 at easier ones).
- **~10° is too tight at high speed** — it drops to ~0.20 at the deploy condition (fine only at
  low speed).

## Why 10° fails — diagnosed (not an authority wall)

`rl/probe_tightness_2d.py` on `up10_best` (fast ±15° 120°/s, 10 V) shows the motor has huge
unused headroom exactly when the pole is wide:

| Pole angle from vertical | mean \|action\| | saturated | steps |
|---|---:|---:|---:|
| > 10° (outside the gate) | 0.24 | **1%** | 37,419 |
| > 15° | 0.25 | **2%** | 10,986 |

The pole sits inside 10° only ~65% of the time (mean 8.7°, p95 17.3°), but when it drifts wide the
motor commands only ~0.24 of max and saturates ~1% — it isn't *trying* to pull tighter. Root
cause: the main reward is `cos(angle)`, which is **flat near vertical** (0.996 at 5° vs 0.966 at
15°), so there is almost no reward gradient rewarding a tighter hold. **Training/reward-limited,
not authority-limited — more voltage would not help.**

## Plan to improve the 10° gate (not yet run)

1. **Reward shaping** — add a steep near-vertical bonus (e.g. `exp(−(angle/σ)²)`, σ≈8°) to create
   the gradient `cos` lacks.
2. **Warm-start from the 15° model** (`up15_best`) and **curriculum the tolerance 15° → 12° → 10°**.
3. Same recipe otherwise (critic reset, γ=0.99, clipping); verify at 500 episodes.
