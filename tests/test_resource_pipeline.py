"""Production pipeline evaluation exercises the L1 resource gates."""

from __future__ import annotations

import jax
import pytest

from naigos.env.config import (
    RESOURCE_WORLD_L1,
    DetectionConfig,
    EnvConfig,
    ResourceWorldConfig,
    TerrainConfig,
)
from naigos.env.flight_env import NaigosEnv
from naigos.env.resource_bank import RESOURCE_BANK_SEED_BASE, ResourceBankOverlap
from naigos.pipeline import evaluation
from naigos.rl.ppo import PPOConfig, init_learner

CFG = EnvConfig(
    n_blue=1, n_threat=2, n_threat_active=0, k_threat_obs=1, k_friend_obs=1, max_steps=30,
    terrain=TerrainConfig(nx=32, ny=32, cell=1_000.0), detection=DetectionConfig(los_samples=8),
    resource=ResourceWorldConfig(model=RESOURCE_WORLD_L1),
)
PLAN = {"seeds": [900001], "worlds_per_seed": 2, "red_level": 0.0, "use_cbf": False,
        "verifier_episodes": 1, "training_seeds": [0, 1]}


@pytest.fixture(scope="module")
def setup():
    env = NaigosEnv(CFG)
    key = jax.random.PRNGKey(0)
    _, obs = env.reset(key)
    return env, init_learner(key, CFG, PPOConfig(n_envs=2, n_steps=4), obs).actor.params


def test_l1_run_evaluation_runs_bank_and_resource_verifier(setup):
    env, params = setup
    out = evaluation.run_evaluation(env, PLAN, params)
    assert {"direct", "avoid_nap", "reserve_divert"} == set(out["baselines"])
    bank = out["resource_bank"]
    assert set(bank["baselines"]) == {"direct", "avoid_nap", "reserve_divert"}
    assert len(bank["candidate"]["heldout"]["by_combination"]) == 5
    report = out["verifier"]["reports"][0]
    assert report["resource_ok"] is True and "total_cost" in report
    assert out["verifier"]["ok"], out["verifier"]["mismatches"]


def test_l1_run_evaluation_refuses_training_seeds_in_the_bank_range(setup):
    env, params = setup
    bad = {**PLAN, "training_seeds": [RESOURCE_BANK_SEED_BASE["train_reference"] + 7]}
    with pytest.raises(ResourceBankOverlap):
        evaluation.run_evaluation(env, bad, params)
