"""L1 consumes only the reserved fuel stream; sensor/comms IDs stay free."""

from __future__ import annotations

import jax
import numpy as np

from naigos.env import resource as resource_mod
from naigos.env.config import (
    RESOURCE_WORLD_L1,
    EnvConfig,
    ResourceWorldConfig,
    TerrainConfig,
)
from naigos.env.flight_env import NaigosEnv


def test_stream_ids_are_unique_and_match_the_plan():
    ids = resource_mod.RNG_STREAM_IDS
    assert len(set(ids.values())) == len(ids)
    assert ids["fuel"] == 101 and ids["sensor"] == 102 and ids["comms"] == 103


def test_l1_only_folds_the_fuel_stream_into_the_episode_key(monkeypatch):
    cfg = EnvConfig(
        n_blue=2, n_threat=2, n_threat_active=0,
        terrain=TerrainConfig(nx=16, ny=16, cell=1_000.0),
        resource=ResourceWorldConfig(model=RESOURCE_WORLD_L1),
    )
    env = NaigosEnv(cfg)
    key = jax.random.PRNGKey(5)
    state = env.reset(key)[0]
    top_level = []
    real = jax.random.fold_in

    def spy(k, data):
        if np.array_equal(np.asarray(k), np.asarray(key)):
            top_level.append(int(data))
        return real(k, data)

    monkeypatch.setattr(jax.random, "fold_in", spy)
    resource_mod.initial_state(cfg, key, state.hmap, state.air.pos, state.objective, state.bounds)
    assert top_level == [resource_mod.RNG_STREAM_IDS["fuel"]]
