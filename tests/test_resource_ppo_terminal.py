"""Safe diversion is terminal for feedforward PPO even in a live world."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from naigos.env.config import (
    RESOURCE_WORLD_L1,
    DetectionConfig,
    EnvConfig,
    ResourceWorldConfig,
    TerrainConfig,
)
from naigos.env.flight_env import NaigosEnv
from naigos.rl.ppo import PPOConfig, active_agent_mask, init_learner, init_rollout, make_train
from naigos.rl.reward import RewardWeights


CFG = EnvConfig(
    n_blue=2,
    n_threat=2,
    n_threat_active=0,
    k_threat_obs=1,
    k_friend_obs=1,
    max_steps=40,
    detection=DetectionConfig(los_samples=8),
    terrain=TerrainConfig(nx=32, ny=32, cell=1_000.0),
    resource=ResourceWorldConfig(
        model=RESOURCE_WORLD_L1,
        fuel_budget_ratio=(5.0, 5.0),
        reserve_ratio=(0.1, 0.1),
        burn_multiplier=(1.0, 1.0),
        gate_arrival_radius=1_000.0,
    ),
)
PPO = PPOConfig(n_envs=2, n_steps=4, n_epochs=1, n_minibatches=1)


def _setup():
    env = NaigosEnv(CFG, hmap=jnp.zeros((CFG.terrain.ny, CFG.terrain.nx)))
    k_init, k_roll, k_run = jax.random.split(jax.random.PRNGKey(13), 3)
    _, sample_obs = env.reset(k_init)
    learner = init_learner(k_init, CFG, PPO, sample_obs)
    train = make_train(env, PPO, RewardWeights())
    state, obs = init_rollout(env, PPO, k_roll)
    # Aircraft zero reaches a gate on its next step. Aircraft one keeps each
    # world alive, exposing any post-diversion samples the collector retains.
    gates = state.resource.diversion_pos.at[:, 0, 0].set(state.air.pos[:, 0])
    state = state._replace(resource=state.resource._replace(diversion_pos=gates))
    return learner, train, (state, obs), k_run


def test_active_agent_mask_excludes_every_terminal_kind():
    learner, train, (state, _), _ = _setup()
    del learner, train
    marked = state._replace(
        alive=state.alive.at[:, 0].set(False),
        reached=state.reached.at[:, 1].set(True),
        resource=state.resource._replace(
            diverted=state.resource.diverted.at[:, 0].set(True)
        ),
    )
    np.testing.assert_array_equal(active_agent_mask(marked), False)


def test_diversion_step_is_trained_once_and_post_terminal_samples_are_zero():
    learner, train, rollout, key = _setup()
    traj, *_ = jax.jit(train.collect)(learner, rollout, key)
    live = np.asarray(traj["live"])
    completed = np.asarray(traj["completed"])

    np.testing.assert_array_equal(live[0, :, 0], True)
    np.testing.assert_array_equal(live[1:, :, 0], False)
    np.testing.assert_array_equal(completed[0, :, 0], 1.0)
    np.testing.assert_array_equal(completed[1:, :, 0], 0.0)
    np.testing.assert_array_equal(np.asarray(traj["r"])[1:, :, 0], 0.0)
    np.testing.assert_array_equal(np.asarray(traj["c"])[1:, :, 0], 0.0)
    # The other aircraft keeps the world open; no auto-reset hides the bug.
    np.testing.assert_array_equal(np.asarray(traj["reset"]), 0.0)


def test_episode_accounting_counts_each_diversion_once():
    learner, train, rollout, key = _setup()
    updated, _, _ = jax.jit(train)(learner, rollout, key)
    assert float(updated.episode_accum) == PPO.n_envs
    assert float(updated.cost_accum) == 0.0
