"""Schema-6 environment identity and fail-closed physics resume checks."""

from __future__ import annotations

import copy
import dataclasses

import jax
import jax.numpy as jnp
import pytest

import _urban_fixtures as fx
from naigos.data.urban_obstacles import load_urban_obstacles
from naigos.env.config import (
    MISSION_V1,
    RESOURCE_WORLD_L1,
    URBAN_OBSTACLE_V1,
    WEATHER_AR1,
    DetectionConfig,
    EnvConfig,
    MissionConfig,
    ResourceWorldConfig,
    TerrainConfig,
    UrbanPhysicsConfig,
    WeatherConfig,
)
from naigos.env.flight_env import NaigosEnv
from naigos.rl import checkpoint as ck
from naigos.rl.ppo import PPOConfig, init_learner
from naigos.rl.red_team import RedCurriculum
from naigos.rl.reward import RewardCurriculum, RewardWeights


def _cfg():
    return EnvConfig(
        n_blue=1,
        n_threat=2,
        n_threat_active=0,
        k_friend_obs=1,
        max_steps=4,
        terrain=TerrainConfig(nx=64, ny=64, cell=100.0),
        detection=DetectionConfig(los_samples=8),
        resource=ResourceWorldConfig(model=RESOURCE_WORLD_L1),
        weather=WeatherConfig(model=WEATHER_AR1, scenario_id="checkpoint", turbulence_intensity_mps=(2.0, 2.0)),
        mission=MissionConfig(model=MISSION_V1, family="multi_leg", scenario_id="checkpoint"),
        urban=UrbanPhysicsConfig(model=URBAN_OBSTACLE_V1, los_surface="terrain_plus_urban"),
    )


def _save(tmp_path, cfg, digest):
    grid = load_urban_obstacles(fx.committed_dir())
    env = NaigosEnv(
        cfg, hmap=jnp.zeros((64, 64)), urban_obstacles=grid,
        env_origin_utm_m=grid.meta.env_origin_utm_m,
    )
    key = jax.random.PRNGKey(0)
    ppo = PPOConfig(n_envs=1, n_steps=2, n_epochs=1, n_minibatches=1)
    learner = init_learner(key, cfg, ppo, env.reset(key)[1])
    path = ck.save(
        tmp_path / "ckpt_000001", learner=learner, key=key, iteration=1,
        red_level=0.0, shootdown_rate=0.0, survival_rate=1.0,
        survival_w=1.0, efficiency_w=0.0, history=[], env_cfg=cfg, ppo_cfg=ppo,
        red_curriculum=RedCurriculum(), reward_curriculum=RewardCurriculum(),
        reward_weights_base=RewardWeights(), wall_s=0.0,
        urban_obstacle_digest=digest, code_commit="a" * 40, code_dirty=False,
    )
    return ck.load(path)


def test_urban_checkpoint_requires_artifact_digest(tmp_path):
    cfg = _cfg()
    with pytest.raises(ck.CheckpointError, match="artifact digest"):
        _save(tmp_path, cfg, None)


def test_schema6_records_and_enforces_every_new_physics_identity(tmp_path):
    cfg = _cfg()
    grid = load_urban_obstacles(fx.committed_dir())
    digest = grid.hashes.content_sha256
    blob = _save(tmp_path, cfg, digest)
    assert blob["schema"] == 6
    assert blob["environment_model_id"] == "resource-world-v1/urban-vehicle-weather"
    identity = blob["environment_identity"]
    assert identity["urban_obstacle_sha256"] == digest
    assert identity["weather_config_sha256"] and identity["turbulence_config_sha256"]
    assert identity["mission_config_sha256"] and identity["mission_bank_sha256"]
    assert identity["aircraft_physics_model"] == "fast-point-mass-weather-v1"
    assert identity["seed_stream_version"] == "named-streams-v1"
    assert identity["policy_family"] == "deepsets@1"
    assert identity["resource_model"] == RESOURCE_WORLD_L1
    assert identity["code_commit"] == "a" * 40 and identity["code_dirty"] is False
    ck.require_environment_compatible(blob, cfg, "ok", urban_obstacle_digest=digest)

    variants = (
        cfg.replace(weather=dataclasses.replace(cfg.weather, scenario_id="tampered")),
        cfg.replace(weather=dataclasses.replace(cfg.weather, turbulence_intensity_mps=(9.0, 9.0))),
        cfg.replace(mission=dataclasses.replace(cfg.mission, family="transit")),
        cfg.replace(urban=dataclasses.replace(cfg.urban, collision_samples=256)),
    )
    for changed in variants:
        with pytest.raises(ck.CheckpointError, match="identity mismatch"):
            ck.require_environment_compatible(blob, changed, "changed", urban_obstacle_digest=digest)
    with pytest.raises(ck.CheckpointError, match="identity mismatch"):
        ck.require_environment_compatible(blob, cfg, "wrong-artifact", urban_obstacle_digest="0" * 64)

    bad = copy.deepcopy(blob)
    bad["environment_identity"]["cbf_contract_id"] = "other-cbf"
    with pytest.raises(ck.CheckpointError, match="cbf_contract_id"):
        ck.require_environment_compatible(bad, cfg, "changed-cbf", urban_obstacle_digest=digest)
