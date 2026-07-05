"""Expand a 10-D TQC checkpoint to 10+H dims with ZERO-initialized new input columns.

The expanded model is functionally IDENTICAL to the source on any state (the new
obs entries are multiplied by zero weights), so fine-tuning starts from the
source policy's exact behavior and gradually learns to use the action history.

Usage (from rl/):
    FURUTA_ACT_HISTORY=2 python expand_obs_warmstart.py \
        --model models/tight7f_s2/best_safe.zip --out models/tight7f_s2_h2.zip

Verifies: expanded.predict([obs, zeros]) == source.predict(obs) to <1e-6.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch as th
from sb3_contrib import TQC

from furuta_env_2d import Furuta2DEnv


def expand_linear_in(weight: th.Tensor, insert_at: int, extra: int) -> th.Tensor:
    """Insert `extra` zero input-columns at position `insert_at`."""
    out_f, in_f = weight.shape
    new = th.zeros((out_f, in_f + extra), dtype=weight.dtype)
    new[:, :insert_at] = weight[:, :insert_at]
    new[:, insert_at + extra:] = weight[:, insert_at:]
    return new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    hist = int(os.environ.get("FURUTA_ACT_HISTORY", "0"))
    if hist <= 0:
        raise SystemExit("set FURUTA_ACT_HISTORY>0 (the target extra obs dims)")

    src = TQC.load(args.model, device="cpu")
    n_src = src.observation_space.shape[0]
    n_dst = n_src + hist

    env = Furuta2DEnv(randomize=False)
    assert env.observation_space.shape[0] == n_dst, (
        f"env obs {env.observation_space.shape[0]} != expected {n_dst}"
    )

    # Mirror the source architecture exactly (net_arch, n_quantiles, gSDE, ...).
    dst = TQC(
        "MlpPolicy",
        env,
        policy_kwargs=dict(src.policy_kwargs),
        device="cpu",
        gamma=src.gamma,
    )

    src_sd = src.policy.state_dict()
    dst_sd = dst.policy.state_dict()
    copied = expanded = 0
    for k, v in src_sd.items():
        if k not in dst_sd:
            raise KeyError(f"missing key in target policy: {k}")
        if dst_sd[k].shape == v.shape:
            dst_sd[k] = v.clone()
            copied += 1
        elif v.dim() == 2 and dst_sd[k].shape[0] == v.shape[0]:
            extra = dst_sd[k].shape[1] - v.shape[1]
            assert extra == hist, f"{k}: unexpected width delta {extra}"
            # Actor input layout: [obs(n_src)] -> new obs cols appended at END of obs.
            # Critic input layout: [obs(n_src), action(1)] -> insert BEFORE the action.
            insert_at = n_src
            dst_sd[k] = expand_linear_in(v, insert_at, extra)
            expanded += 1
        else:
            raise ValueError(f"{k}: cannot map {v.shape} -> {dst_sd[k].shape}")
    dst.policy.load_state_dict(dst_sd)
    print(f"copied {copied} tensors, expanded {expanded} input layers "
          f"({n_src}-D -> {n_dst}-D)")

    # equivalence check: padded obs must reproduce source actions exactly
    rng = np.random.default_rng(0)
    obs10 = rng.uniform(-1, 1, size=(256, n_src)).astype(np.float32)
    obs12 = np.concatenate(
        [obs10, np.zeros((256, hist), dtype=np.float32)], axis=1
    )
    a_src, _ = src.predict(obs10, deterministic=True)
    a_dst, _ = dst.predict(obs12, deterministic=True)
    err = float(np.max(np.abs(a_src - a_dst)))
    print(f"max |a_src - a_expanded| on zero history = {err:.2e}")
    assert err < 1e-6, "expanded model does not reproduce the source policy"

    dst.save(args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
