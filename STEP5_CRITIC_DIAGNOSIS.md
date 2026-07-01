# Step 5: capability-vs-training diagnosis of the stalled continuous-pitch run

Date: 2026-06-30

## Question

The local seed `tilt2d_cont_local_s0` stalled at stage 1 (pitch ±10°, 60–90°/s),
plateauing at 0.73–0.80 target success and self-terminating on the 250k stage timeout.
Before committing more compute we needed to know whether fast continuous pitch is a
**capability/authority limit** (training cannot help) or a **training/strategy limit**
(fixable). A secondary question was the suspicious training signal `rl_actor_loss ≈ +1050`,
implying the critic valued the actor's own actions at ≈ −1000.

## Method

`rl/probe_capability_2d.py`, frozen policy, no training, 40 episodes/condition. Per step it
records the commanded action, the critic's Q(s, a_det), the true-vertical cosine, and the
reward. It reports:

- **Authority:** fraction of the 0.5 s (100-step) window before a loss where |a| > 0.95.
  Saturated → out of actuator authority; unsaturated → authority left, used wrong.
- **Critic calibration (#5):** mean critic Q vs mean empirical discounted return-to-go on
  genuinely balanced states (true_up > 0.9), γ = 0.998.

## Results

### Authority — fast pitch is NOT an actuator wall

`sat<loss` stays at 0.15–0.39 in every failing condition (warm-start and the 300k checkpoint
alike). The policy is not pinned at saturation when it loses the pole; it has torque to spare
and still falls. Fast continuous pitch is therefore a **control-strategy failure within the
actuator's authority** — learnable.

### Critic calibration — the critic is broken, and fine-tuning made it diverge

Mean critic Q vs actual discounted return-to-go on balanced states:

| Condition | Warm-start Q | After 300k Q | Actual return |
|---|---:|---:|---:|
| Level | 398 | −202 | 1120 |
| Pitch 10° 60/s | 65 | −1313 | 1073 |
| Pitch 10° 90/s | −63 | −1523 | 1049 |
| Pitch 10° 120/s | −172 | −1646 | 1013 |
| Pitch 15° 120/s | −263 | −1799 | 779 |

1. The **transferred warm-start critic is already miscalibrated**: it underestimates good
   states ≈ 3× on level and goes *negative* under fast pitch. The 1D master never saw
   continuous fast pitch, so its critic is out-of-distribution there.
2. **Fine-tuning made the critic diverge catastrophically negative** — ≈ −200 to −1800 for
   states that actually earn +780 to +1120, **including level**, despite 25% teacher replay of
   successful level episodes. The "critic adapts first for 25k steps" mechanism did not fix the
   critic; the critic got worse.

## Verdict

The stall is a **training problem, specifically a critic problem** — not a capability limit and
not a defect of the warm-start actor.

- Fast pitch is learnable (authority headroom confirmed).
- The warm-start **actor** is good and should be kept.
- The **critic** is the failure point. The transferred critic is OOD-miscalibrated, and joint
  fine-tuning drives it into a negative divergence that starves the actor of a usable gradient
  (`rl_actor_loss ≈ −qf_pi` is large-positive precisely because Q collapsed negative). Retention
  "held" only because the actor barely moved.

## Implication for the next plan

Center the next method on the critic: discard the transferred critic and do a **frozen-actor
critic warmup (policy evaluation of the good actor) gated on Q tracking return-to-go**, before
any joint actor update. Investigate the negative-divergence driver (γ = 0.998 long horizon,
fixed `ent_coef` soft-value term with a low-entropy warm-start actor, conservative top-quantile
dropping, repeated fast-pitch fall penalties). Also fix the previously-identified loop defects:
the inert adaptive teacher term (1e6 clamp leaves it at ≈ 0.01% of actor loss), and the brittle
hard-0.90 curriculum gate whose timeout kills the seed instead of advancing the stage.

Evidence: `rl/probe_capability_2d.py` output, this run, 2026-06-30.
