"""The NumPy L1 verifier agrees with JAX and detects trace tampering."""

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
from naigos.env.resource_bank import config_for_scenario, resource_bank
from naigos.rl import verifier
from naigos.rl.train import make_reserve_divert_policy


CFG = EnvConfig(
    n_blue=1,
    n_threat=2,
    n_threat_active=0,
    k_threat_obs=1,
    k_friend_obs=0,
    max_steps=8,
    detection=DetectionConfig(los_samples=8),
    terrain=TerrainConfig(nx=32, ny=32, cell=1_000.0),
    resource=ResourceWorldConfig(
        model=RESOURCE_WORLD_L1,
        fuel_budget_ratio=(0.001, 0.001),
        reserve_ratio=(0.1, 0.1),
        burn_multiplier=(1.1, 1.1),
    ),
)


def _trace():
    env = NaigosEnv(CFG)

    def policy(_obs, _key):
        return jnp.array([[0.2, 0.0, 0.7]])

    final, traj = env.rollout(jax.random.PRNGKey(3), policy)
    out = {name: np.asarray(value) for name, value in traj.items() if name != "terms"}
    out["terms"] = type(traj["terms"])(*[np.asarray(x) for x in traj["terms"]])
    return final, out


def test_resource_verifier_agrees_and_counts_one_shot_fuel_loss():
    final, trace = _trace()
    report = verifier.verify_resource_trace(CFG, trace)
    assert report.ok, report.mismatches
    assert report.fuel_losses == 1
    assert int(np.asarray(trace["terms"].out_of_fuel).sum()) == 1

    combined = verifier.verify_trace(
        CFG,
        np.asarray(final.hmap),
        trace,
        np.asarray(final.threats.kind),
        np.asarray(final.threats.active),
    )
    assert combined.ok, str(combined)
    assert combined.fuel_losses == 1
    assert combined.total_cost == 1.0


def test_resource_verifier_rejects_tampered_fuel_transition():
    _, trace = _trace()
    bad = {**trace, "fuel": trace["fuel"].copy()}
    bad["fuel"][0, 0] += 10.0
    report = verifier.verify_resource_trace(CFG, bad)
    assert not report.ok
    assert any("fuel mismatch" in message for message in report.mismatches)


def test_resource_verifier_rejects_tampered_outcome_and_observation():
    _, trace = _trace()
    bad = {
        **trace,
        "outcome": trace["outcome"].copy(),
        "resource_observation": trace["resource_observation"].copy(),
    }
    bad["outcome"][:] = 0
    bad["resource_observation"][0, 0, 0] += 0.5
    report = verifier.verify_resource_trace(CFG, bad)
    assert not report.ok
    assert any("outcome label" in message for message in report.mismatches)
    assert any("observation mismatch" in message for message in report.mismatches)


def test_resource_verifier_checks_first_diversion_crossing():
    row = next(
        item for item in resource_bank("train_reference")
        if item.combination == "marginal_fuel"
    )
    cfg = config_for_scenario(
        CFG.replace(max_steps=200), row
    )
    env = NaigosEnv(cfg)
    route = jnp.array([jnp.deg2rad(row.bearing_deg), row.family_id])
    _final, traj = env.rollout(
        jax.random.PRNGKey(row.seed), make_reserve_divert_policy(cfg), route=route
    )
    trace = {name: np.asarray(value) for name, value in traj.items() if name != "terms"}
    trace["terms"] = type(traj["terms"])(*[np.asarray(x) for x in traj["terms"]])
    report = verifier.verify_resource_trace(cfg, trace)
    assert report.ok, report.mismatches
    assert report.diversions == 1

    bad = {**trace, "terms": trace["terms"]._replace(diverted=np.zeros_like(trace["terms"].diverted))}
    tampered = verifier.verify_resource_trace(cfg, bad)
    assert not tampered.ok
    assert any("diversion" in message for message in tampered.mismatches)


def test_resource_verifier_does_not_import_jax_resource_implementation():
    import inspect

    source = inspect.getsource(verifier)
    assert "from ..env import resource" not in source
    assert "import jax" not in source
