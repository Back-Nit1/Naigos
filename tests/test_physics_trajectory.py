"""Versioned weather/mission/urban evidence at the trajectory boundary."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

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
from naigos.rl import trajectory_recording as trec
from naigos.trajectory import TERMINATION_CAUSES, TrajectoryReader


def test_physics_rollout_records_explicit_versioned_evidence(tmp_path):
    grid = load_urban_obstacles(fx.committed_dir())
    cfg = EnvConfig(
        n_blue=1,
        n_threat=2,
        n_threat_active=0,
        k_friend_obs=1,
        max_steps=3,
        terrain=TerrainConfig(nx=64, ny=64, cell=100.0),
        detection=DetectionConfig(los_samples=8),
        resource=ResourceWorldConfig(model=RESOURCE_WORLD_L1),
        weather=WeatherConfig(
            model=WEATHER_AR1,
            scenario_id="trajectory-physics-test",
            wind_enu_mps=((7.0, -3.0, 0.0), (9.0, -2.0, 0.0)),
            turbulence_intensity_mps=(1.0, 2.0),
        ),
        mission=MissionConfig(model=MISSION_V1, family="multi_leg", scenario_id="trajectory-physics-test"),
        urban=UrbanPhysicsConfig(model=URBAN_OBSTACLE_V1, los_surface="terrain_plus_urban"),
    )
    env = NaigosEnv(
        cfg,
        hmap=jnp.zeros((64, 64)),
        urban_obstacles=grid,
        env_origin_utm_m=grid.meta.env_origin_utm_m,
    )
    policy = lambda obs, key: jnp.zeros((1, 3), dtype=jnp.float32)  # noqa: E731
    _, trace = env.rollout(jax.random.PRNGKey(17), policy, record=True)
    evidence = jax.tree.map(lambda value: value[None, ...], trace["evidence"])
    identity = trec.policy_record("physics-test", cfg, family="deepsets", version="1")
    entry = trec.record_rollouts(
        trec.RecordingConfig(enabled=True, root=str(tmp_path), chunk_steps=2),
        env,
        evidence,
        label="physics-evidence",
        seed=17,
        policy=identity,
    )

    reader = TrajectoryReader(entry["path"])
    arrays = reader.read_all()
    fields = reader.metadata.schema.field_map
    expected = {
        "extension/weather_observation",
        "extension/cbf_reason_bits",
        "privileged/mean_wind_enu",
        "privileged/gust_enu",
        "privileged/density_ratio",
        "extension/air_velocity",
        "extension/ground_velocity",
        "extension/mission_observation",
        "extension/mission_stage_before",
        "extension/mission_stage_after",
        "privileged/mission_stage_pos",
        "extension/obstacle_clearance",
        "reward/building_collision",
    }
    assert expected <= set(fields)
    assert np.isfinite(arrays["privileged/gust_enu"]).all()
    np.testing.assert_allclose(
        arrays["extension/ground_velocity"],
        arrays["extension/air_velocity"]
        + arrays["privileged/mean_wind_enu"]
        + arrays["privileged/gust_enu"],
        rtol=1e-5,
        atol=1e-5,
    )
    assert fields["status/termination_cause"].normalization["values"]["building"] == TERMINATION_CAUSES["building"]
    environment = reader.metadata.provenance.environment_configuration
    assert environment["environment_model_id"] == "resource-world-v1/urban-vehicle-weather"
    assert environment["environment_identity"]["urban_obstacle_sha256"] == grid.hashes.content_sha256
