"""Focused lifecycle, batching and reset tests for the GRU policy rung."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from naigos.env.config import DetectionConfig, EnvConfig, TerrainConfig
from naigos.env.flight_env import NaigosEnv
from naigos.rl import policies
from naigos.rl.ppo import (
    PPOConfig,
    greedy_policy,
    init_learner,
    init_rollout,
    make_train,
)
from naigos.rl.reward import RewardWeights


CFG = EnvConfig(
    n_blue=2,
    n_threat=2,
    n_threat_active=0,
    max_steps=3,
    terrain=TerrainConfig(nx=9, ny=9, cell=2_000.0),
    detection=DetectionConfig(los_samples=8),
)
PPO = PPOConfig(
    n_envs=2,
    n_steps=4,
    n_epochs=1,
    n_minibatches=2,
    policy_family="recurrent",
    recurrent_seq_len=2,
)


def _setup(cfg=CFG, ppo=PPO):
    env = NaigosEnv(cfg, hmap=jnp.zeros((cfg.terrain.ny, cfg.terrain.nx)))
    k_init, k_roll, k_run = jax.random.split(jax.random.PRNGKey(31), 3)
    _, obs = env.reset(k_init)
    learner = init_learner(k_init, cfg, ppo, obs)
    rollout = init_rollout(env, ppo, k_roll)
    return env, learner, rollout, k_run


def test_memory_lifecycle_is_explicit_and_batched_per_world_and_aircraft():
    _, _, rollout, _ = _setup()
    memory = rollout[2]
    assert memory.shape == (PPO.n_envs, CFG.n_blue, policies.RECURRENT_HIDDEN)
    np.testing.assert_array_equal(np.asarray(memory), 0.0)

    filled = jnp.ones_like(memory)
    done = jnp.array([[False, True], [True, False]])
    reset = policies.reset_memory(filled, done)
    np.testing.assert_array_equal(np.asarray(reset[0, 0]), 1.0)
    np.testing.assert_array_equal(np.asarray(reset[0, 1]), 0.0)
    np.testing.assert_array_equal(np.asarray(reset[1, 0]), 0.0)
    np.testing.assert_array_equal(np.asarray(reset[1, 1]), 1.0)


def test_collection_resets_memory_before_an_auto_reset_episode_can_use_it():
    env, learner, rollout, key = _setup()
    step = make_train(env, PPO, RewardWeights())
    traj, _, _, next_rollout = jax.jit(step.collect)(learner, rollout, key)

    # max_steps=3: index 2 ends the old world and index 3 is the new world's
    # first policy call. Its stored pre-call memory must be exactly fresh.
    np.testing.assert_array_equal(np.asarray(traj["reset"][2]), 1.0)
    np.testing.assert_array_equal(np.asarray(traj["memory"][3]), 0.0)
    assert next_rollout[2].shape == rollout[2].shape


def test_recurrent_ppo_uses_sequence_batches_and_updates_finitely():
    env, learner, rollout, key = _setup()
    step = jax.jit(make_train(env, PPO, RewardWeights()))
    updated, rollout2, metrics = step(learner, rollout, key)
    assert int(updated.actor.step) == PPO.n_epochs * PPO.n_minibatches
    assert rollout2[2].shape == rollout[2].shape
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(metrics))


def test_recurrent_sequence_geometry_is_validated_before_tracing():
    bad = dataclasses.replace(PPO, n_steps=5)
    env, _, _, _ = _setup(ppo=bad)
    try:
        make_train(env, bad, RewardWeights())
    except ValueError as exc:
        assert "must divide n_steps" in str(exc)
    else:  # pragma: no cover - protects a fail-closed configuration invariant
        raise AssertionError("invalid truncated-BPTT geometry was accepted")


def test_greedy_recurrent_rollout_resets_memory_at_episode_boundary():
    env, learner, _, key = _setup()
    policy = greedy_policy(learner.actor.params, CFG, family="recurrent")
    assert policy.initial_memory.shape == (CFG.n_blue, policies.RECURRENT_HIDDEN)

    def visible_memory_policy(obs, unused_key, memory):
        action = jnp.zeros((CFG.n_blue, 3)).at[:, 0].set(memory[:, 0])
        return action, memory + 0.25

    visible_memory_policy.initial_memory = policy.initial_memory
    final, traj = jax.jit(lambda k: env.rollout(
        k, visible_memory_policy, n_steps=CFG.max_steps + 1
    ))(key)
    assert traj["action"].shape == (CFG.max_steps + 1, CFG.n_blue, 3)
    assert bool(np.asarray(traj["done"][CFG.max_steps - 1]))
    # The scan continues one diagnostic step after done. If evaluation leaked
    # memory across that boundary, this would be 0.75 instead of fresh zero.
    np.testing.assert_array_equal(np.asarray(traj["action"][-1, :, 0]), 0.0)
    assert int(final.t) == CFG.max_steps + 1

