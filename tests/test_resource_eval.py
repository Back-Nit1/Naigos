"""evaluate_resource_bank: production bank evaluation with seed/provenance gates."""

from __future__ import annotations

import pytest

from naigos.env.config import DetectionConfig, EnvConfig, TerrainConfig, RESOURCE_WORLD_L1, ResourceWorldConfig
from naigos.env.flight_env import NaigosEnv
from naigos.env.resource_bank import (
    RESOURCE_BANK_SEED_BASE,
    ResourceBankOverlap,
    base_config_digest,
    config_for_scenario,
    provenance_problems,
    resource_bank,
)
from naigos.rl.train import (
    baseline_policies,
    direct_route_policy,
    evaluate_resource_bank,
    make_reserve_divert_policy,
)

COMBINATIONS = {"nominal", "marginal_fuel", "burn_multiplier", "farther_gate_only", "threat_detour"}


def _env():
    cfg = EnvConfig(
        n_blue=1, n_threat=2, n_threat_active=0, k_threat_obs=1, k_friend_obs=0, max_steps=200,
        terrain=TerrainConfig(nx=32, ny=32, cell=1_000.0), detection=DetectionConfig(los_samples=8),
        resource=ResourceWorldConfig(model=RESOURCE_WORLD_L1),
    )
    return NaigosEnv(cfg)


@pytest.fixture(scope="module")
def results():
    env = _env()
    return {
        "direct": evaluate_resource_bank(env, direct_route_policy, training_seeds=[1, 2]),
        "reserve_divert": evaluate_resource_bank(env, make_reserve_divert_policy(env.cfg)),
    }


def test_per_combination_heldout_and_reference_metrics_are_verified(results):
    for res in results.values():
        assert res["protocol"] == "naigos-resource-eval-v1"
        assert res["verified"], (res["problems"], res["heldout"]["by_combination"])
        for split in ("heldout", "train_reference"):
            assert set(res[split]["by_combination"]) == COMBINATIONS
            row = res[split]["by_combination"]["marginal_fuel"]
            assert {"survival_rate", "diversion_rate", "timeout_rate", "fuel_loss_rate"} <= set(row)
        assert res["base_config_sha256"] and res["bank_sha256"]


def test_reserve_divert_beats_direct_on_marginal_fuel_in_both_splits(results):
    for split in ("heldout", "train_reference"):
        direct = results["direct"][split]["by_combination"]["marginal_fuel"]
        divert = results["reserve_divert"][split]["by_combination"]["marginal_fuel"]
        assert direct["fuel_loss_rate"] == 1.0 and direct["diversion_rate"] == 0.0
        assert divert["fuel_loss_rate"] == 0.0 and divert["diversion_rate"] == 1.0
        assert divert["survival_rate"] == 0.0  # a safe diversion is not ordinary survival


def test_training_seed_inside_reserved_range_is_refused():
    base = RESOURCE_BANK_SEED_BASE["heldout"]
    with pytest.raises(ResourceBankOverlap):
        evaluate_resource_bank(_env(), direct_route_policy, training_seeds=[base + 3])


def test_l1_registers_reserve_divert_and_legacy_defaults_are_untouched():
    assert set(baseline_policies(_env().cfg)) == {"direct", "avoid_nap", "reserve_divert"}
    assert set(baseline_policies(EnvConfig())) == {"direct", "avoid_nap"}


def test_provenance_detects_edited_config_or_foreign_base():
    env = _env()
    row = resource_bank("heldout")[0]
    cfg = config_for_scenario(env.cfg, row)
    digest = base_config_digest(env.cfg)
    assert provenance_problems(cfg, row, digest) == []
    assert provenance_problems(cfg.replace(max_steps=99), row, digest)  # different base world
    other = config_for_scenario(env.cfg, resource_bank("heldout")[1])
    assert provenance_problems(other, row, digest)  # wrong row's resource parameters
    moved = type(row)(**{**row.__dict__, "seed": 1})
    assert provenance_problems(cfg, moved, digest)  # seed outside the reserved range
