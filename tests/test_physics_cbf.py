"""Weather/urban CBF diagnostics and independent reference-solver checks."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import _urban_fixtures as fx
from naigos.data.urban_obstacles import load_urban_obstacles
from naigos.env import urban
from naigos.env.config import (
    URBAN_OBSTACLE_V1,
    WEATHER_AR1,
    EnvConfig,
    TerrainConfig,
    UrbanPhysicsConfig,
    WeatherConfig,
)
from naigos.env.flight_env import NaigosEnv
from naigos.rl.cbf import (
    CBF_REASON_BOUNDS,
    CBF_REASON_URBAN,
    CBFConfig,
    _project_qp,
    filter_action_diagnostics,
    make_policy_filter,
    reference_project_qp_numpy,
)


def _none():
    return jnp.zeros((1, 3)), jnp.zeros(1), jnp.zeros(1, dtype=bool)


def test_reference_solver_matches_jax_projection_on_held_out_convex_cells():
    cases = (
        (np.array([0.2, -0.3]), np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([0.5, -0.1])),
        (np.array([-0.7, 0.8]), np.array([[1.0, 1.0], [-1.0, 2.0]]), np.array([0.4, -0.2])),
        (np.array([0.9, 0.9]), np.array([[-1.0, 0.0], [0.0, -1.0]]), np.array([-0.25, -0.4])),
    )
    lo, hi, weight = np.array([-1.0, -1.0]), np.array([1.0, 1.0]), np.array([4.0, 1.0])
    for u0, A, b in cases:
        got, feasible = _project_qp(
            jnp.asarray(u0), jnp.asarray(A), jnp.asarray(b), jnp.asarray(lo), jnp.asarray(hi), 96,
            jnp.asarray(weight),
        )
        ref, ref_feasible = reference_project_qp_numpy(u0, A, b, lo, hi, weight)
        assert bool(feasible) == ref_feasible
        np.testing.assert_allclose(np.asarray(got), ref, atol=2e-3, rtol=2e-3)


def test_tailwind_estimate_activates_current_play_area_bounds_reason():
    cfg = EnvConfig(
        n_blue=1,
        n_threat=1,
        n_threat_active=0,
        weather=WeatherConfig(model=WEATHER_AR1),
    )
    centers, radii, active = _none()
    pos = jnp.array([95_500.0, 50_000.0, 5_000.0])
    action = jnp.array([0.0, 0.0, 0.8])
    kwargs = dict(
        cbf=CBFConfig(), cfg=cfg, hmap=jnp.zeros((128, 128)), action=action, pos=pos,
        psi=jnp.float32(0.0), gamma=jnp.float32(0.0), phi=jnp.float32(0.0), speed=jnp.float32(180.0),
        known_centers=centers, known_radii=radii, known_active=active,
        estimated_density_ratio=jnp.float32(1.0), bounds=jnp.array([10_000.0, 100_000.0, 0.0, 120_000.0]),
    )
    _, _, calm_bits = filter_action_diagnostics(
        estimated_wind_enu=jnp.zeros(3), **kwargs
    )
    _, _, tailwind_bits = filter_action_diagnostics(
        estimated_wind_enu=jnp.array([80.0, 0.0, 0.0]), **kwargs
    )
    assert int(tailwind_bits) & CBF_REASON_BOUNDS
    assert int(tailwind_bits) >= int(calm_bits)


def test_urban_surface_reason_and_artifact_fail_closed():
    grid = load_urban_obstacles(fx.committed_dir())
    cfg = EnvConfig(
        n_blue=1,
        n_threat=1,
        n_threat_active=0,
        k_friend_obs=1,
        terrain=TerrainConfig(nx=64, ny=64, cell=100.0),
        urban=UrbanPhysicsConfig(model=URBAN_OBSTACLE_V1, los_surface="terrain_plus_urban"),
    )
    surface = urban.from_artifact(grid, grid.meta.env_origin_utm_m)
    y = surface.origin_y_m + 4.5 * surface.cell_m
    pos = jnp.array([surface.origin_x_m + 3.5 * surface.cell_m, y, 70.0])
    centers, radii, active = _none()
    cbf = CBFConfig(enable_bounds=False)
    safe, feasible, bits = filter_action_diagnostics(
        cbf, cfg, jnp.zeros((64, 64)), jnp.array([0.0, -1.0, 0.5]), pos,
        jnp.float32(0.0), jnp.float32(0.0), jnp.float32(0.0), jnp.float32(180.0),
        centers, radii, active, jnp.zeros(3), jnp.float32(1.0),
        jnp.array([0.0, 6_300.0, 0.0, 6_300.0]), surface,
    )
    assert bool(feasible) and int(bits) & CBF_REASON_URBAN
    assert float(safe[1]) > -1.0
    with pytest.raises(ValueError, match="validated urban obstacle surface"):
        make_policy_filter(cbf, cfg)


def test_physics_policy_filter_logs_reason_bits_under_jit():
    cfg = EnvConfig(
        n_blue=1,
        n_threat=2,
        n_threat_active=0,
        k_friend_obs=1,
        max_steps=2,
        weather=WeatherConfig(model=WEATHER_AR1, wind_enu_mps=((25.0, 0.0, 0.0),) * 2),
    )
    env = NaigosEnv(cfg)
    filt = make_policy_filter(CBFConfig(), cfg)
    _, trace = jax.jit(lambda key: env.rollout(
        key, lambda obs, action_key: jnp.zeros((1, 3)), action_filter=filt
    ))(jax.random.PRNGKey(5))
    assert trace["cbf_reason_bits"].shape == (2, 1)
    assert trace["cbf_reason_bits"].dtype == jnp.uint16
