"""Transfer the verified 8-input 1D TQC checkpoint into the 10-input 2D environment."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sb3_contrib import TQC

from furuta_env import BETA_SCALE, BETADOT_SCALE
from furuta_env_2d import BOARD_ANGLE_SCALE, BOARD_RATE_SCALE, Furuta2DEnv


# Old: [pole(3), arm(2), action, beta, betadot]
# New: [pole(3), arm(2), action, roll, pitch, gyro_x, gyro_y]
OLD_TO_NEW = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 7, 7: 9}
INPUT_SCALE = {
    0: 1.0,
    1: 1.0,
    2: 1.0,
    3: 1.0,
    4: 1.0,
    5: 1.0,
    6: BOARD_ANGLE_SCALE / BETA_SCALE,
    7: BOARD_RATE_SCALE / BETADOT_SCALE,
}


def _new_model(old: TQC) -> TQC:
    env = Furuta2DEnv(randomize=False)
    return TQC(
        "MlpPolicy",
        env,
        learning_rate=old.learning_rate,
        buffer_size=old.buffer_size,
        learning_starts=old.learning_starts,
        batch_size=old.batch_size,
        tau=old.tau,
        gamma=old.gamma,
        train_freq=(old.train_freq.frequency, old.train_freq.unit.value),
        gradient_steps=old.gradient_steps,
        action_noise=None,
        replay_buffer_class=old.replay_buffer_class,
        replay_buffer_kwargs=old.replay_buffer_kwargs,
        optimize_memory_usage=old.optimize_memory_usage,
        ent_coef=old.ent_coef,
        target_update_interval=old.target_update_interval,
        target_entropy=old.target_entropy,
        top_quantiles_to_drop_per_net=old.top_quantiles_to_drop_per_net,
        policy_kwargs=old.policy_kwargs,
        seed=old.seed,
        device="cpu",
        verbose=0,
    )


def _transfer_actor(old: TQC, new: TQC) -> None:
    old_state = old.actor.state_dict()
    new_state = new.actor.state_dict()
    for key, value in old_state.items():
        if key != "latent_pi.0.weight":
            new_state[key].copy_(value)
    old_weight = old_state["latent_pi.0.weight"]
    new_weight = new_state["latent_pi.0.weight"]
    new_weight.zero_()
    for old_index, new_index in OLD_TO_NEW.items():
        new_weight[:, new_index] = old_weight[:, old_index] * INPUT_SCALE[old_index]
    new.actor.load_state_dict(new_state)


def _transfer_critic(old_critic, new_critic) -> None:
    old_state = old_critic.state_dict()
    new_state = new_critic.state_dict()
    first_layers = {"qf0.0.weight", "qf1.0.weight"}
    for key, value in old_state.items():
        if key not in first_layers:
            new_state[key].copy_(value)
    for key in first_layers:
        old_weight = old_state[key]
        new_weight = new_state[key]
        new_weight.zero_()
        for old_index, new_index in OLD_TO_NEW.items():
            new_weight[:, new_index] = old_weight[:, old_index] * INPUT_SCALE[old_index]
        new_weight[:, 10] = old_weight[:, 8]  # action follows the ten observation values
    new_critic.load_state_dict(new_state)


def transfer(old: TQC) -> TQC:
    new = _new_model(old)
    with torch.no_grad():
        _transfer_actor(old, new)
        _transfer_critic(old.critic, new.critic)
        _transfer_critic(old.critic_target, new.critic_target)
    return new


def physical_equivalence(old: TQC, new: TQC, samples: int = 10_000) -> tuple[float, float]:
    rng = np.random.default_rng(20260630)
    old_obs = rng.normal(0.0, 0.5, size=(samples, 8)).astype(np.float32)
    beta = rng.uniform(-np.deg2rad(15.0), np.deg2rad(15.0), size=samples)
    betadot = rng.uniform(-BOARD_RATE_SCALE, BOARD_RATE_SCALE, size=samples)
    old_obs[:, 6] = beta / BETA_SCALE
    old_obs[:, 7] = betadot / BETADOT_SCALE
    new_obs = np.zeros((samples, 10), dtype=np.float32)
    new_obs[:, :6] = old_obs[:, :6]
    new_obs[:, 7] = beta / BOARD_ANGLE_SCALE
    new_obs[:, 9] = betadot / BOARD_RATE_SCALE

    old_action, _ = old.predict(old_obs, deterministic=True)
    new_action, _ = new.predict(new_obs, deterministic=True)
    action_error = float(np.max(np.abs(old_action - new_action)))

    with torch.no_grad():
        old_tensor = torch.as_tensor(old_obs)
        new_tensor = torch.as_tensor(new_obs)
        action_tensor = torch.as_tensor(old_action)
        old_q = old.critic(old_tensor, action_tensor)
        new_q = new.critic(new_tensor, action_tensor)
    critic_error = float(torch.max(torch.abs(old_q - new_q)).item())
    return action_error, critic_error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", default="rl/models/clean20_master_verified91p5.zip"
    )
    parser.add_argument(
        "--output", default="rl/models/clean20_master_2d_warmstart.zip"
    )
    args = parser.parse_args()

    old = TQC.load(args.source, device="cpu")
    if old.observation_space.shape != (8,):
        raise ValueError(f"expected 8-input source policy, got {old.observation_space.shape}")
    new = transfer(old)
    action_error, critic_error = physical_equivalence(old, new)
    # Rescaling changes floating-point multiplication order; tolerate only float32 roundoff.
    if action_error > 5e-6 or critic_error > 1e-3:
        raise RuntimeError(
            f"transfer equivalence failed: action={action_error}, critic={critic_error}"
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    new.save(output)

    reloaded = TQC.load(output, device="cpu")
    reload_action_error, reload_critic_error = physical_equivalence(old, reloaded)
    if reload_action_error > 5e-6 or reload_critic_error > 1e-3:
        raise RuntimeError(
            "saved checkpoint equivalence failed: "
            f"action={reload_action_error}, critic={reload_critic_error}"
        )
    print(f"saved {output}")
    print(f"observation shape: {old.observation_space.shape} -> {new.observation_space.shape}")
    print(f"max action error: {action_error:.3e}")
    print(f"max critic error: {critic_error:.3e}")
    print(f"reload max action error: {reload_action_error:.3e}")
    print(f"reload max critic error: {reload_critic_error:.3e}")


if __name__ == "__main__":
    main()
