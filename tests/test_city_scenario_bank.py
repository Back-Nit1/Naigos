"""Frozen NYC/LA weather, mission and resource cross-product scenarios."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _theatre_fixtures import offline_theatre
from naigos.env.config import EnvConfig
from naigos.env.flight_env import NaigosEnv
from naigos.env.scenario_bank import (
    SCENARIO_BANK_V1_SHA256,
    SCENARIO_SEED_BASE,
    SCENARIO_SEED_SPAN,
    assert_scenario_seeds_disjoint,
    config_for_scenario,
    scenario_bank,
    scenario_bank_digest,
    scenario_protocol,
)


def test_bank_digest_cases_and_seed_splits_are_frozen():
    rows = scenario_bank()
    assert scenario_bank_digest(rows) == SCENARIO_BANK_V1_SHA256
    nyc = {r.case for r in scenario_bank("nyc_harbor")}
    la = {r.case for r in scenario_bank("los_angeles_basin")}
    assert {"calm", "harbor_crosswind", "strong_building_wake", "low_density_high_demand", "marginal_fuel_headwind"} <= nyc
    assert {"calm", "coastal_onshore", "basin_crosswind", "mountain_wave_rotor_surrogate", "marginal_fuel_headwind", "reduced_authority_crosswind"} <= la
    train = {r.master_seed for r in scenario_bank(split="train_reference")}
    held = {r.master_seed for r in scenario_bank(split="heldout")}
    assert train.isdisjoint(held)
    assert scenario_protocol()["bank_sha256"] == SCENARIO_BANK_V1_SHA256


def test_training_seed_overlap_fails_closed():
    assert_scenario_seeds_disjoint([0, 1, 900_001])
    for base in SCENARIO_SEED_BASE.values():
        with pytest.raises(ValueError, match="overlap"):
            assert_scenario_seeds_disjoint([base + SCENARIO_SEED_SPAN - 1])


@pytest.mark.parametrize("theatre", ["nyc_harbor", "los_angeles_basin"])
def test_every_city_scenario_builds_and_heldout_cases_step(theatre):
    from naigos.env.theatre_bridge import env_from_theatre

    with offline_theatre(theatre):
        base, hmap, notes = env_from_theatre(aoi=theatre, n_blue=1, n_threat=2, cell_m=1500.0)
    rows = scenario_bank(theatre)
    for row in rows:
        cfg = config_for_scenario(base, row)
        kwargs = {}
        if cfg.urban.enabled:
            kwargs = {
                "urban_obstacles": notes["urban_obstacles"],
                "env_origin_utm_m": notes["env_origin_utm_m"],
            }
        env = NaigosEnv(cfg, hmap=hmap, **kwargs)
        state, obs = env.reset(jax.random.PRNGKey(row.master_seed))
        state, obs, terms, *_ = env.step(state, jnp.zeros((1, 3)))
        assert cfg.environment_model_id.startswith("resource-world-v1/")
        assert np.isfinite(np.asarray(state.air.pos)).all()
        assert np.isfinite(np.asarray(obs.ego)).all()
        assert state.mission.stage_mask.shape == (1, cfg.mission.max_stages)


def test_notional_scenarios_make_no_current_conditions_claim():
    for row in scenario_bank():
        assert "notional" in row.note and "not current" in row.note
        cfg = config_for_scenario(EnvConfig(), row)
        assert cfg.weather.scenario_id == row.scenario_id
