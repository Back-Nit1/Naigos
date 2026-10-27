"""Fixed-shape and compatibility contract for resource-world L1."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from naigos.env import airframe
from naigos.env.config import (
    RESOURCE_WORLD_BASELINE,
    RESOURCE_WORLD_L1,
    AirframeConfig,
    DetectionConfig,
    EnvConfig,
    ResourceWorldConfig,
    TerrainConfig,
)
from naigos.env.flight_env import NaigosEnv


def _cfg(**resource_overrides) -> EnvConfig:
    return EnvConfig(
        n_blue=2,
        n_threat=4,
        n_threat_active=0,
        k_threat_obs=2,
        k_friend_obs=1,
        max_steps=12,
        detection=DetectionConfig(los_samples=8),
        terrain=TerrainConfig(nx=32, ny=32, cell=1_000.0),
        resource=ResourceWorldConfig(model=RESOURCE_WORLD_L1, **resource_overrides),
    )


def test_baseline_v0_remains_the_default_and_keeps_the_legacy_actor_width():
    cfg = EnvConfig(n_blue=2, n_threat=4, n_threat_active=0)
    state, obs = NaigosEnv(cfg).reset(jax.random.PRNGKey(7))
    assert cfg.resource.model == RESOURCE_WORLD_BASELINE
    assert cfg.action_dim == 3
    assert cfg.ego_dim == 10
    assert obs.ego.shape == (2, 10)
    np.testing.assert_array_equal(state.air.fuel, np.full(2, cfg.airframe.fuel_init))
    np.testing.assert_array_equal(state.resource.burn_multiplier, np.ones(2))
    assert not bool(state.resource.diversion_active.any())


def test_airframe_default_multiplier_is_numerically_identical_to_one():
    cfg = AirframeConfig()
    state = airframe.AircraftState(
        pos=jnp.array([[0.0, 0.0, 2_000.0]]),
        speed=jnp.array([180.0]),
        psi=jnp.array([0.2]),
        gamma=jnp.array([0.0]),
        phi=jnp.array([0.0]),
        fuel=jnp.array([3000.0]),
    )
    action = jnp.array([[0.3, -0.1, 0.8]])
    implicit = airframe.step(cfg, state, action, 1.0)
    explicit = airframe.step(cfg, state, action, 1.0, 1.0)
    for a, b in zip(implicit, explicit):
        np.testing.assert_array_equal(a, b)


def test_l1_reset_has_fixed_shapes_and_only_three_controls():
    cfg = _cfg()
    state, obs = NaigosEnv(cfg).reset(jax.random.PRNGKey(1))
    assert cfg.action_dim == 3
    assert state.resource.diversion_pos.shape == (cfg.n_blue, 2, 3)
    assert state.resource.diversion_active.shape == (cfg.n_blue, 2)
    assert obs.ego.shape == (cfg.n_blue, cfg.ego_dim)
    assert cfg.ego_dim == 30
    assert bool(jnp.all(jnp.isfinite(obs.ego)))


def test_resource_draws_are_deterministic_and_use_independent_local_keys():
    key = jax.random.PRNGKey(22)
    base_cfg = _cfg()
    base, _ = NaigosEnv(base_cfg).reset(key)
    again, _ = NaigosEnv(base_cfg).reset(key)
    for a, b in zip(base.resource, again.resource):
        np.testing.assert_array_equal(a, b)

    changed_cfg = _cfg(burn_multiplier=(1.7, 1.7))
    changed, _ = NaigosEnv(changed_cfg).reset(key)
    np.testing.assert_array_equal(base.resource.initial_fuel, changed.resource.initial_fuel)
    np.testing.assert_array_equal(base.resource.reserve_fuel, changed.resource.reserve_fuel)
    np.testing.assert_array_equal(base.resource.diversion_pos, changed.resource.diversion_pos)
    np.testing.assert_array_equal(
        changed.resource.burn_multiplier, np.full(2, 1.7, dtype=np.float32)
    )


def test_l1_reset_step_and_rollout_are_jit_vmap_compatible():
    cfg = _cfg()
    env = NaigosEnv(cfg)
    keys = jax.random.split(jax.random.PRNGKey(0), 3)
    states, observations = jax.jit(jax.vmap(env.reset))(keys)
    assert states.resource.initial_fuel.shape == (3, cfg.n_blue)
    assert observations.ego.shape == (3, cfg.n_blue, cfg.ego_dim)

    action = jnp.zeros((3, cfg.n_blue, 3))
    next_states, next_obs, *_ = jax.jit(jax.vmap(env.step))(states, action)
    assert next_states.air.pos.shape == (3, cfg.n_blue, 3)
    assert next_obs.ego.shape == (3, cfg.n_blue, cfg.ego_dim)

    def straight(obs, _key):
        return jnp.zeros((cfg.n_blue, 3))

    final, traj = jax.jit(jax.vmap(lambda k: env.rollout(k, straight)))(keys)
    assert final.resource.outcome.shape == (3, cfg.n_blue)
    assert traj["resource_observation"].shape == (
        3,
        cfg.max_steps,
        cfg.n_blue,
        cfg.resource.observation_dim,
    )


def test_hidden_outcome_label_is_not_in_the_actor_observation():
    cfg = _cfg()
    env = NaigosEnv(cfg)
    state, obs = env.reset(jax.random.PRNGKey(5))
    poisoned = state._replace(
        resource=state.resource._replace(
            outcome=jnp.full_like(state.resource.outcome, 255)
        )
    )
    poisoned_obs = env.observe(poisoned)
    np.testing.assert_array_equal(obs.ego, poisoned_obs.ego)
    np.testing.assert_array_equal(obs.threats, poisoned_obs.threats)
    np.testing.assert_array_equal(obs.friends, poisoned_obs.friends)


def test_unknown_resource_model_is_refused_instead_of_falling_back():
    with pytest.raises(ValueError, match="resource.model"):
        NaigosEnv(_cfg().replace(resource=ResourceWorldConfig(model="resource-world-v9")))

