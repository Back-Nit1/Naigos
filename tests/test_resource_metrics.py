"""L1 evaluation does not report a timeout as a recovery."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from naigos.env.config import (
    RESOURCE_WORLD_L1,
    DetectionConfig,
    EnvConfig,
    ResourceWorldConfig,
    TerrainConfig,
)
from naigos.env.flight_env import NaigosEnv
from naigos.rl.train import rollout_metrics


def test_live_timeout_is_not_l1_survival_or_recovery():
    cfg = EnvConfig(
        n_blue=1,
        n_threat=2,
        n_threat_active=0,
        k_threat_obs=1,
        k_friend_obs=1,
        max_steps=1,
        detection=DetectionConfig(los_samples=8),
        terrain=TerrainConfig(nx=32, ny=32, cell=1_000.0),
        resource=ResourceWorldConfig(
            model=RESOURCE_WORLD_L1,
            fuel_budget_ratio=(10.0, 10.0),
            reserve_ratio=(0.1, 0.1),
            burn_multiplier=(1.0, 1.0),
        ),
    )
    env = NaigosEnv(cfg)

    def policy(_obs, _key):
        return jnp.zeros((1, 3))

    final, traj = jax.vmap(lambda key: env.rollout(key, policy))(
        jax.random.split(jax.random.PRNGKey(0), 2)
    )
    metrics = rollout_metrics(final, traj, n_worlds=2, n_blue=1)
    assert metrics["timeout_rate"] == 1.0
    assert metrics["survival_rate"] == 0.0
    assert metrics["recovery_rate"] == 0.0
    assert metrics["objective_rate"] == 0.0
    assert metrics["diversion_rate"] == 0.0

