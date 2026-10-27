"""L1 checkpoint metadata, legacy fallback, and cross-schema refusal."""

from __future__ import annotations

import dataclasses

import jax
import numpy as np
import pytest

from naigos.env.config import (
    RESOURCE_WORLD_BASELINE,
    RESOURCE_WORLD_L1,
    DetectionConfig,
    EnvConfig,
    ResourceWorldConfig,
    TerrainConfig,
)
from naigos.env.flight_env import NaigosEnv
from naigos.rl import checkpoint as ck
from naigos.rl.ppo import PPOConfig, init_learner
from naigos.rl.red_team import RedCurriculum
from naigos.rl.reward import RewardCurriculum, RewardWeights


CFG = EnvConfig(
    n_blue=1,
    n_threat=2,
    n_threat_active=0,
    k_threat_obs=1,
    k_friend_obs=1,
    max_steps=4,
    detection=DetectionConfig(los_samples=8),
    terrain=TerrainConfig(nx=16, ny=16, cell=1_000.0),
    resource=ResourceWorldConfig(
        model=RESOURCE_WORLD_L1,
        fuel_budget_ratio=(1.2, 1.4),
        reserve_ratio=(0.1, 0.2),
        burn_multiplier=(0.9, 1.1),
    ),
)


def test_missing_environment_fields_mean_baseline_only():
    legacy = {"schema": 4, "actor": {}, "env_cfg": {"obs_edge_features": False}}
    assert ck.environment_model_from_blob(legacy) == RESOURCE_WORLD_BASELINE
    assert ck.obs_config_from_blob(legacy) == {"obs_edge_features": False}
    ck.require_environment_compatible(legacy, EnvConfig(), "legacy")
    with pytest.raises(ck.CheckpointError, match="cross-schema"):
        ck.require_environment_compatible(legacy, CFG, "legacy")


def test_schema4_baseline_normalizes_the_unreachable_diversion_weight():
    saved_weights = dataclasses.asdict(RewardWeights())
    saved_weights.pop("diverted")
    legacy = {
        "schema": 4,
        "recovery": {
            "red_curriculum": dataclasses.asdict(RedCurriculum()),
            "reward_curriculum": dataclasses.asdict(RewardCurriculum()),
            "reward_weights_base": saved_weights,
        },
    }
    assert ck.curriculum_compatibility(
        legacy,
        red_curriculum=RedCurriculum(),
        reward_curriculum=RewardCurriculum(),
        reward_weights_base=RewardWeights(),
    ) == []

    resource = {**legacy, "environment_model_id": RESOURCE_WORLD_L1}
    problems = ck.curriculum_compatibility(
        resource,
        red_curriculum=RedCurriculum(),
        reward_curriculum=RewardCurriculum(),
        reward_weights_base=RewardWeights(),
    )
    assert any("diverted" in problem for problem in problems)


def test_resource_checkpoint_requires_explicit_matching_resource_config():
    incomplete = {
        "environment_model_id": RESOURCE_WORLD_L1,
        "observation_schema_id": ck.RESOURCE_OBSERVATION_SCHEMA,
        "actor_schema_id": ck.RESOURCE_ACTOR_SCHEMA,
    }
    with pytest.raises(ck.CheckpointError, match="missing its resource_config"):
        ck.resource_config_from_blob(incomplete)


def test_resource_actor_checkpoint_round_trips_and_refuses_baseline(tmp_path):
    ppo = PPOConfig(n_envs=1, n_steps=2, n_epochs=1, n_minibatches=1)
    key = jax.random.PRNGKey(0)
    _, observation = NaigosEnv(CFG).reset(key)
    learner = init_learner(key, CFG, ppo, observation)
    path = ck.save(
        tmp_path / "ckpt_000001",
        learner=learner,
        key=key,
        iteration=1,
        red_level=0.0,
        shootdown_rate=0.0,
        survival_rate=1.0,
        survival_w=1.0,
        efficiency_w=0.0,
        history=[{"iter": 0}],
        env_cfg=CFG,
        ppo_cfg=ppo,
        red_curriculum=RedCurriculum(),
        reward_curriculum=RewardCurriculum(),
        reward_weights_base=RewardWeights(),
        wall_s=0.0,
    )
    blob = ck.load(path)
    assert blob["environment_model_id"] == RESOURCE_WORLD_L1
    assert blob["observation_schema_id"] == ck.RESOURCE_OBSERVATION_SCHEMA
    assert blob["actor_schema_id"] == ck.RESOURCE_ACTOR_SCHEMA
    assert blob["resource_bank"]["sha256"]
    assert blob["recovery"]["resource_curriculum"]["kind"] == "fixed-v1"
    assert ck.obs_config_from_blob(blob)["resource"] == CFG.resource
    ck.require_environment_compatible(blob, CFG, path)
    ck.require_actor_ego_dim(blob["actor"], CFG, path)

    template = init_learner(key, CFG, ppo, observation)
    restored = ck.restore_learner(template, blob)
    for expected, actual in zip(
        jax.tree.leaves(learner.actor.params), jax.tree.leaves(restored.actor.params)
    ):
        np.testing.assert_array_equal(expected, actual)

    with pytest.raises(ck.CheckpointError, match="cross-schema"):
        ck.require_environment_compatible(blob, EnvConfig(), path)


def test_same_shape_but_different_resource_ranges_are_refused():
    blob = {
        "environment_model_id": RESOURCE_WORLD_L1,
        "observation_schema_id": ck.RESOURCE_OBSERVATION_SCHEMA,
        "actor_schema_id": ck.RESOURCE_ACTOR_SCHEMA,
        "resource_config": {
            **vars(CFG.resource),
            "fuel_budget_ratio": (9.0, 10.0),
        },
    }
    with pytest.raises(ck.CheckpointError, match="resource_config differs"):
        ck.require_environment_compatible(blob, CFG, "different-ranges")
