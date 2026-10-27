"""Independent NumPy verification and tamper rejection for new physics evidence."""

from __future__ import annotations

import copy

import jax
import jax.numpy as jnp
import numpy as np

import _urban_fixtures as fx
from naigos.data.urban_obstacles import load_urban_obstacles
from naigos.env import weather
from naigos.env.config import (
    MISSION_V1,
    URBAN_OBSTACLE_V1,
    WEATHER_AR1,
    DetectionConfig,
    EnvConfig,
    MissionConfig,
    TerrainConfig,
    UrbanPhysicsConfig,
    WeatherConfig,
)
from naigos.env.flight_env import NaigosEnv
from naigos.rl.physics_verifier import counter_normal, verify_physics_trace


def _case():
    grid = load_urban_obstacles(fx.committed_dir())
    cfg = EnvConfig(
        n_blue=1,
        n_threat=2,
        n_threat_active=0,
        k_friend_obs=0,
        max_steps=4,
        terrain=TerrainConfig(nx=64, ny=64, cell=100.0),
        detection=DetectionConfig(los_samples=16),
        weather=WeatherConfig(
            model=WEATHER_AR1,
            scenario_id="verifier",
            wind_enu_mps=((8.0, -3.0, 0.0), (8.0, -3.0, 0.0)),
            turbulence_intensity_mps=(5.0, 5.0),
            estimate_noise_scale=0.0,
        ),
        mission=MissionConfig(model=MISSION_V1, family="transit", scenario_id="verifier", deadline_s=20.0),
        urban=UrbanPhysicsConfig(model=URBAN_OBSTACLE_V1, los_surface="terrain_plus_urban"),
    )
    hmap = jnp.zeros((64, 64), dtype=jnp.float32)
    env = NaigosEnv(
        cfg, hmap=hmap, urban_obstacles=grid, env_origin_utm_m=grid.meta.env_origin_utm_m
    )
    policy = lambda obs, key: jnp.zeros((1, 3))  # noqa: E731
    _, trace = env.rollout(jax.random.PRNGKey(6), policy, n_steps=3)
    return cfg, np.asarray(hmap), grid, jax.tree.map(np.asarray, trace)


def test_numpy_counter_noise_matches_jax_without_calling_transition():
    key = jnp.array([123, 456], dtype=jnp.uint32)
    got = counter_normal(np.asarray(key), 7, (3, 3))
    want = np.asarray(weather.counter_normal(key, 7, (3, 3)))
    np.testing.assert_array_equal(got, want)


def test_independent_verifier_accepts_an_untampered_trace():
    cfg, hmap, grid, trace = _case()
    report = verify_physics_trace(
        cfg, hmap, trace, urban_grid=grid, env_origin_utm_m=grid.meta.env_origin_utm_m,
        expected_urban_digest=grid.hashes.content_sha256,
    )
    assert report.ok, report.mismatches
    assert report.max_gust_error < 4e-3 and report.max_velocity_error < 4e-3


def test_weather_collision_mission_and_identity_tampering_are_rejected():
    cfg, hmap, grid, trace = _case()
    edits = (
        ("gust_enu", lambda a: a.__setitem__((0, 0, 0), a[0, 0, 0] + 2.0)),
        ("ground_velocity", lambda a: a.__setitem__((0, 0, 1), a[0, 0, 1] + 3.0)),
        ("density_ratio", lambda a: a.__setitem__((0, 0), a[0, 0] * 0.8)),
        ("obstacle_clearance", lambda a: a.__setitem__((0, 0), a[0, 0] + 20.0)),
        ("mission_stage", lambda a: a.__setitem__((0, 0), 3)),
    )
    for name, edit in edits:
        bad = {k: copy.deepcopy(v) for k, v in trace.items()}
        edit(bad[name])
        report = verify_physics_trace(
            cfg, hmap, bad, urban_grid=grid, env_origin_utm_m=grid.meta.env_origin_utm_m,
            expected_urban_digest=grid.hashes.content_sha256,
        )
        assert not report.ok, name
    report = verify_physics_trace(
        cfg, hmap, trace, urban_grid=grid, env_origin_utm_m=grid.meta.env_origin_utm_m,
        expected_urban_digest="0" * 64,
    )
    assert not report.ok and any("digest" in x for x in report.mismatches)
