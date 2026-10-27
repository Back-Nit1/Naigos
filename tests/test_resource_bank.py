"""Frozen resource-bank provenance and the observation-only L1 baseline."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from naigos.env.config import DetectionConfig, EnvConfig, TerrainConfig
from naigos.env.flight_env import NaigosEnv
from naigos.env.resource_bank import (
    RESOURCE_BANK_SEED_BASE,
    RESOURCE_BANK_SEED_SPAN,
    RESOURCE_BANK_V1_SHA256,
    ResourceBankOverlap,
    assert_resource_bank_disjoint,
    config_for_scenario,
    resource_bank,
    resource_bank_digest,
    resource_protocol,
)
from naigos.rl.train import direct_route_policy, make_reserve_divert_policy


def _small_base():
    return EnvConfig(
        n_blue=1,
        n_threat=2,
        n_threat_active=0,
        k_threat_obs=1,
        k_friend_obs=0,
        max_steps=200,
        terrain=TerrainConfig(nx=32, ny=32, cell=1_000.0),
        detection=DetectionConfig(los_samples=8),
    )


def test_resource_bank_digest_and_minimum_combinations_are_frozen():
    rows = resource_bank()
    assert resource_bank_digest(rows) == RESOURCE_BANK_V1_SHA256
    assert {row.combination for row in rows} == {
        "nominal",
        "marginal_fuel",
        "burn_multiplier",
        "farther_gate_only",
        "threat_detour",
    }
    assert {row.split for row in rows} == {"train_reference", "heldout"}
    assert len({row.scenario_id for row in rows}) == len(rows)
    assert len({(row.seed, row.fuel_budget_ratio, row.reserve_ratio, row.burn_multiplier)
                for row in rows}) == len(rows)


def test_resource_bank_reserved_seed_ranges_are_disjoint_from_training():
    starts = sorted(RESOURCE_BANK_SEED_BASE.values())
    assert starts[0] + RESOURCE_BANK_SEED_SPAN <= starts[1]
    assert_resource_bank_disjoint([0, 1, 900_001])
    for base in starts:
        with pytest.raises(ResourceBankOverlap):
            assert_resource_bank_disjoint([base + RESOURCE_BANK_SEED_SPAN - 1])


def test_resource_scenario_reproduces_exact_state_and_observation():
    row = resource_bank("heldout")[2]
    cfg = config_for_scenario(_small_base(), row)
    env = NaigosEnv(cfg)
    key = jax.random.PRNGKey(row.seed)
    route = jnp.array([jnp.deg2rad(row.bearing_deg), row.family_id])
    first = env.reset_route(key, route)
    second = env.reset_route(key, route)
    for a, b in zip(jax.tree.leaves(first), jax.tree.leaves(second)):
        np.testing.assert_array_equal(a, b)
    assert resource_protocol()["bank_sha256"] == RESOURCE_BANK_V1_SHA256


def test_farther_gate_case_has_exactly_one_active_fixed_capacity_gate():
    row = next(r for r in resource_bank("heldout") if r.combination == "farther_gate_only")
    cfg = config_for_scenario(_small_base(), row)
    state, _ = NaigosEnv(cfg).reset(jax.random.PRNGKey(row.seed))
    assert state.resource.diversion_pos.shape == (1, 2, 3)
    np.testing.assert_array_equal(state.resource.diversion_active[0], [False, True])


def test_reserve_divert_reaches_recovery_when_direct_route_runs_dry():
    """Constructive marginal-fuel acceptance case on the frozen bank."""
    row = next(r for r in resource_bank("train_reference") if r.combination == "marginal_fuel")
    cfg = config_for_scenario(_small_base(), row)
    env = NaigosEnv(cfg)
    key = jax.random.PRNGKey(row.seed)
    route = jnp.array([jnp.deg2rad(row.bearing_deg), row.family_id])

    direct_final, direct_trace = env.rollout(key, direct_route_policy, route=route)
    divert_final, divert_trace = env.rollout(
        key, make_reserve_divert_policy(cfg), route=route
    )
    assert int(direct_trace["terms"].out_of_fuel.sum()) == 1
    assert not bool(direct_final.alive[0])
    assert int(divert_trace["terms"].out_of_fuel.sum()) == 0
    assert bool(divert_final.resource.diverted[0])
    assert not bool(divert_final.reached[0]), "safe diversion must not receive primary credit"

    nominal = next(r for r in resource_bank("train_reference") if r.combination == "nominal")
    nominal_cfg = config_for_scenario(_small_base(), nominal)
    nominal_env = NaigosEnv(nominal_cfg)
    nominal_route = jnp.array([jnp.deg2rad(nominal.bearing_deg), nominal.family_id])
    high_final, high_trace = nominal_env.rollout(
        jax.random.PRNGKey(nominal.seed), direct_route_policy, route=nominal_route
    )
    assert int(high_trace["terms"].out_of_fuel.sum()) == 0
    assert bool(high_final.reached[0])
