"""Simulation consumption of the validated urban-obstacle artifact."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import _urban_fixtures as fx
from _theatre_fixtures import offline_theatre
from naigos.data.urban_obstacles import load_urban_obstacles
from naigos.env import terrain, urban
from naigos.env.config import (
    URBAN_OBSTACLE_V1,
    DetectionConfig,
    EnvConfig,
    TerrainConfig,
    UrbanPhysicsConfig,
)
from naigos.env.flight_env import NaigosEnv


@pytest.fixture(scope="module")
def grid():
    return load_urban_obstacles(fx.committed_dir())


@pytest.fixture(scope="module")
def cfg():
    return EnvConfig(
        n_blue=1,
        n_threat=2,
        n_threat_active=0,
        k_friend_obs=0,
        max_steps=8,
        terrain=TerrainConfig(nx=64, ny=64, cell=100.0),
        detection=DetectionConfig(los_samples=64),
        urban=UrbanPhysicsConfig(
            model=URBAN_OBSTACLE_V1,
            los_surface="terrain_plus_urban",
            collision_samples=128,
        ),
    )


def _surface(cfg, grid):
    return urban.from_artifact(grid, grid.meta.env_origin_utm_m)


def test_urban_model_refuses_absent_artifact(cfg):
    with pytest.raises(ValueError, match="validated artifact"):
        NaigosEnv(cfg)


def test_sampling_preserves_ground_top_building_and_explicit_outside_semantics(cfg, grid):
    s = _surface(cfg, grid)
    hmap = jnp.full((cfg.terrain.ny, cfg.terrain.nx), 7.0)
    x = s.origin_x_m + 5.5 * s.cell_m
    y = s.origin_y_m + 4.5 * s.cell_m
    top, building, covered = urban.sample(s, hmap, cfg.terrain, jnp.array(x), jnp.array(y))
    assert bool(building) and bool(covered) and float(top) > 80.0
    outside, building, covered = urban.sample(s, hmap, cfg.terrain, jnp.array(100.0), jnp.array(100.0))
    assert not bool(building) and not bool(covered) and float(outside) == pytest.approx(7.0)


def test_swept_collision_prevents_tunneling_and_separates_terrain(cfg, grid):
    s = _surface(cfg, grid)
    hmap = jnp.zeros((cfg.terrain.ny, cfg.terrain.nx))
    y = s.origin_y_m + 4.5 * s.cell_m
    p0 = jnp.array([[s.origin_x_m + 3.5 * s.cell_m, y, 50.0]])
    p1 = jnp.array([[s.origin_x_m + 9.5 * s.cell_m, y, 50.0]])
    th, bh, _, first = urban.swept_collisions(s, hmap, cfg.terrain, cfg.urban, p0, p1, 30.0)
    assert bool(bh[0]) and not bool(th[0]) and 0 < int(first[0]) < cfg.urban.collision_samples
    # Both endpoints are surveyed open cells: endpoint-only sampling would miss the tower.
    assert not bool(urban.sample(s, hmap, cfg.terrain, p0[0, 0], p0[0, 1])[1])
    assert not bool(urban.sample(s, hmap, cfg.terrain, p1[0, 0], p1[0, 1])[1])

    low0 = jnp.array([[100.0, 100.0, 20.0]])
    low1 = jnp.array([[200.0, 100.0, 20.0]])
    th, bh, *_ = urban.swept_collisions(s, hmap, cfg.terrain, cfg.urban, low0, low1, 30.0)
    assert bool(th[0]) and not bool(bh[0])


def test_environment_reports_building_collision_as_distinct_terminal_cause(cfg, grid):
    hmap = jnp.zeros((cfg.terrain.ny, cfg.terrain.nx))
    env = NaigosEnv(cfg, hmap=hmap, urban_obstacles=grid, env_origin_utm_m=grid.meta.env_origin_utm_m)
    st, _ = env.reset(jax.random.PRNGKey(2))
    s = env.urban_surface
    y = s.origin_y_m + 4.5 * s.cell_m
    pos = jnp.array([[s.origin_x_m + 1.0 * s.cell_m, y, 50.0]])
    air = st.air._replace(pos=pos, psi=jnp.zeros(1), gamma=jnp.zeros(1), phi=jnp.zeros(1))
    st = st._replace(air=air, prev_dist=jnp.linalg.norm((st.objective - pos)[:, :2], axis=-1))
    throttle_for_180 = 2.0 * (180.0 - cfg.airframe.v_stall) / (cfg.airframe.v_max - cfg.airframe.v_stall) - 1.0
    after, _, terms, done, info = env.step(st, jnp.array([[0.0, 0.0, throttle_for_180]]))
    assert bool(done) and not bool(after.alive[0])
    assert float(terms.building_collision[0]) == 1.0
    assert float(terms.terrain_violation[0]) == 0.0
    assert bool(info["building_collision"][0])


def test_urban_los_masks_a_ray_that_bare_terrain_does_not(cfg, grid):
    s = _surface(cfg, grid)
    hmap = jnp.zeros((cfg.terrain.ny, cfg.terrain.nx))
    y = s.origin_y_m + 5.0 * s.cell_m
    p0 = jnp.array([s.origin_x_m + 2.0 * s.cell_m, y, 50.0])
    p1 = jnp.array([s.origin_x_m + 10.0 * s.cell_m, y, 50.0])
    bare = terrain.los_clearance(hmap, cfg.terrain, cfg.detection, p0, p1)
    city = urban.los_clearance(s, hmap, cfg.terrain, cfg.detection, p0, p1)
    assert float(bare) > 40.0 and float(city) < 0.0


def test_building_wake_surrogate_uses_only_validated_surface(cfg, grid):
    s = _surface(cfg, grid)
    hmap = jnp.zeros((cfg.terrain.ny, cfg.terrain.nx))
    y = s.origin_y_m + 5.5 * s.cell_m
    downwind = jnp.array([[s.origin_x_m + 9.5 * s.cell_m, y, 100.0]])
    sigma = urban.mechanical_wake_intensity(s, hmap, cfg.terrain, downwind, jnp.array([[25.0, 0.0, 0.0]]))
    clear = urban.mechanical_wake_intensity(
        s, hmap, cfg.terrain, jnp.array([[100.0, 100.0, 100.0]]), jnp.array([[25.0, 0.0, 0.0]])
    )
    assert float(sigma[0]) > 0.0 and float(sigma[0]) <= 25.0
    assert float(clear[0]) == 0.0


def test_presentation_payload_is_ignored_when_urban_physics_is_disabled():
    base = EnvConfig(n_blue=1, n_threat=2, n_threat_active=0, k_friend_obs=0, max_steps=2)
    a = NaigosEnv(base)
    b = NaigosEnv(base, urban_obstacles={"presentation": "only"}, env_origin_utm_m=(0.0, 0.0))
    key = jax.random.PRNGKey(31)
    sa, oa = a.reset(key)
    sb, ob = b.reset(key)
    for x, y in zip(jax.tree.leaves(sa), jax.tree.leaves(sb)):
        np.testing.assert_array_equal(x, y)
    np.testing.assert_array_equal(oa.ego, ob.ego)


@pytest.mark.parametrize("aoi", ["nyc_harbor", "los_angeles_basin"])
def test_city_theatre_constructs_with_its_validated_artifact(aoi):
    from naigos.env.theatre_bridge import env_from_theatre

    with offline_theatre(aoi):
        cfg, hmap, notes = env_from_theatre(aoi=aoi, n_blue=1, n_threat=2, cell_m=1500.0)
    cfg = cfg.replace(urban=UrbanPhysicsConfig(
        model=URBAN_OBSTACLE_V1,
        los_surface="terrain_plus_urban",
        collision_samples=128,
    ))
    env = NaigosEnv(
        cfg,
        hmap=hmap,
        urban_obstacles=notes["urban_obstacles"],
        env_origin_utm_m=notes["env_origin_utm_m"],
    )
    st, obs = env.reset(jax.random.PRNGKey(0))
    st, obs, terms, *_ = env.step(st, jnp.zeros((1, 3)))
    assert env.urban_surface.content_sha256 == notes["urban_obstacle_evidence"]["content_sha256"]
    assert np.isfinite(np.asarray(st.air.pos)).all() and np.isfinite(np.asarray(obs.ego)).all()
