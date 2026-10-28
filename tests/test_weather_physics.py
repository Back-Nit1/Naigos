"""Opt-in vehicle-weather physics and named stochastic stream."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from naigos.env import airframe, resource, weather
from naigos.env.config import (
    RESOURCE_WORLD_L1,
    WEATHER_AR1,
    DetectionConfig,
    EnvConfig,
    ResourceWorldConfig,
    TerrainConfig,
    WeatherConfig,
)
from naigos.env.flight_env import NaigosEnv
from naigos.rl.train import baseline_policies, make_weather_conservative_policy


def _weather(*, wind=(0.0, 0.0, 0.0), rho=1.0, sigma=0.0):
    return WeatherConfig(
        model=WEATHER_AR1,
        scenario_id="test",
        altitude_knots_m=(0.0, 12_000.0),
        wind_enu_mps=(wind, wind),
        density_ratio=(rho, rho),
        temperature_k=(288.15, 216.65),
        sensing_attenuation=(1.0, 1.0),
        turbulence_intensity_mps=(sigma, sigma),
        estimate_noise_scale=0.0,
    )


def _cfg(w):
    return EnvConfig(
        n_blue=2,
        n_threat=2,
        n_threat_active=0,
        max_steps=8,
        detection=DetectionConfig(los_samples=8),
        terrain=TerrainConfig(nx=16, ny=16, cell=1_000.0),
        weather=w,
    )


def _trim(st):
    return st._replace(weather=None)


def test_profile_interpolation_and_digest_are_stable():
    cfg = dataclasses.replace(
        _weather(),
        altitude_knots_m=(0.0, 1_000.0, 2_000.0),
        wind_enu_mps=((0.0, 0.0, 0.0), (20.0, -10.0, 2.0), (40.0, -20.0, 4.0)),
        density_ratio=(1.0, 0.9, 0.8),
        temperature_k=(290.0, 280.0, 270.0),
        sensing_attenuation=(1.0, 0.9, 0.8),
        turbulence_intensity_mps=(0.0, 5.0, 10.0),
    )
    p = weather.profile_at(cfg, jnp.array([500.0, 1_500.0]))
    np.testing.assert_allclose(p["wind_enu"], [[10, -5, 1], [30, -15, 3]])
    np.testing.assert_allclose(p["density_ratio"], [0.95, 0.85])
    assert weather.config_digest(cfg) == weather.config_digest(cfg)
    assert weather.config_digest(cfg) != weather.config_digest(dataclasses.replace(cfg, scenario_id="other"))


def test_neutral_weather_is_numerically_equivalent_to_legacy_transition():
    legacy = NaigosEnv(_cfg(WeatherConfig()))
    neutral = NaigosEnv(_cfg(_weather()))
    key = jax.random.PRNGKey(42)
    a, oa = legacy.reset(key)
    b, ob = neutral.reset(key)
    for x, y in zip(jax.tree.leaves(_trim(a)), jax.tree.leaves(_trim(b))):
        np.testing.assert_array_equal(x, y)
    np.testing.assert_array_equal(oa.ego, ob.ego[:, : oa.ego.shape[-1]])
    action = jnp.array([[0.2, -0.1, 0.4], [-0.3, 0.2, -0.2]])
    aa = legacy.step(a, action)
    bb = neutral.step(b, action)
    for x, y in zip(jax.tree.leaves(aa[0].air), jax.tree.leaves(bb[0].air)):
        np.testing.assert_array_equal(x, y)
    for x, y in zip(aa[2], bb[2]):
        np.testing.assert_array_equal(x, y)


def test_wind_changes_ground_track_not_air_relative_heading_or_speed():
    east = NaigosEnv(_cfg(_weather(wind=(20.0, 0.0, 0.0))))
    west = NaigosEnv(_cfg(_weather(wind=(-20.0, 0.0, 0.0))))
    cross = NaigosEnv(_cfg(_weather(wind=(0.0, 15.0, 0.0))))
    key = jax.random.PRNGKey(3)
    se = east.reset(key)[0]
    sw = west.reset(key)[0]
    sc = cross.reset(key)[0]
    action = jnp.zeros((2, 3))
    ne, *_ = east.step(se, action)
    nw, *_ = west.step(sw, action)
    nc, *_ = cross.step(sc, action)
    np.testing.assert_array_equal(ne.air.speed, nw.air.speed)
    np.testing.assert_array_equal(ne.air.psi, nw.air.psi)
    np.testing.assert_allclose(ne.air.pos[:, 0] - nw.air.pos[:, 0], 80.0, atol=2e-3)
    np.testing.assert_allclose(nc.air.pos[:, 1] - ne.air.pos[:, 1], 30.0, atol=2e-3)
    np.testing.assert_allclose(
        east.step(se, action)[4]["ground_velocity"],
        airframe.velocity(ne.air) + jnp.array([20.0, 0.0, 0.0]),
        atol=1e-6,
    )


def test_low_density_reduces_authority_and_increases_propulsive_demand():
    dense = NaigosEnv(_cfg(_weather(rho=1.0)))
    thin = NaigosEnv(_cfg(_weather(rho=0.55)))
    key = jax.random.PRNGKey(9)
    sd = dense.reset(key)[0]
    st = thin.reset(key)[0]
    action = jnp.tile(jnp.array([0.8, 0.8, 1.0]), (2, 1))
    nd, *_ = dense.step(sd, action)
    nt, *_ = thin.step(st, action)
    # Bank's short 0.5 s actuator can saturate its discrete lag at this dt;
    # climb and longitudinal channels retain the density-dependent authority.
    assert np.all(np.asarray(np.abs(nt.air.gamma)) < np.asarray(np.abs(nd.air.gamma)))
    assert np.all(np.asarray(nt.air.speed) < np.asarray(nd.air.speed))
    assert np.all(np.asarray(sd.air.fuel - nd.air.fuel) < np.asarray(st.air.fuel - nt.air.fuel))


def test_gust_replay_is_deterministic_bounded_and_jittable_under_vmap():
    cfg = _cfg(_weather(sigma=12.0))
    env = NaigosEnv(cfg)
    keys = jax.random.split(jax.random.PRNGKey(7), 4)

    def run(k):
        st, _ = env.reset(k)
        return env.step(st, jnp.zeros((2, 3)))[0].weather.gust_enu

    a = jax.jit(jax.vmap(run))(keys)
    b = jax.jit(jax.vmap(run))(keys)
    np.testing.assert_array_equal(a, b)
    assert np.isfinite(np.asarray(a)).all()
    assert np.abs(np.asarray(a)).max() <= 4.0 * 12.0 + 1e-5
    assert not np.array_equal(np.asarray(a[0]), np.asarray(a[1]))


def test_weather_stream_is_reserved_and_does_not_move_resource_draws():
    assert weather.WEATHER_RNG_STREAM_ID == resource.RNG_STREAM_IDS["weather"]
    base = _cfg(WeatherConfig()).replace(resource=ResourceWorldConfig(model=RESOURCE_WORLD_L1))
    wx = base.replace(weather=_weather(sigma=8.0))
    key = jax.random.PRNGKey(91)
    a = NaigosEnv(base).reset(key)[0]
    b = NaigosEnv(wx).reset(key)[0]
    np.testing.assert_array_equal(a.resource.initial_fuel, b.resource.initial_fuel)
    np.testing.assert_array_equal(a.resource.reserve_fuel, b.resource.reserve_fuel)
    np.testing.assert_array_equal(a.resource.burn_multiplier, b.resource.burn_multiplier)
    np.testing.assert_array_equal(a.threats.pos, b.threats.pos)


def test_weather_baseline_uses_only_local_actor_features_and_composes_with_resource_slice():
    cfg = _cfg(_weather(wind=(5.0, 12.0, 0.0), sigma=3.0)).replace(
        resource=ResourceWorldConfig(model=RESOURCE_WORLD_L1)
    )
    env = NaigosEnv(cfg)
    _, obs = env.reset(jax.random.PRNGKey(22))
    policies = baseline_policies(cfg)
    assert {"direct", "avoid_nap", "reserve_divert", "weather_conservative"} <= set(policies)
    for policy in policies.values():
        action = policy(obs, jax.random.PRNGKey(0))
        assert action.shape == (cfg.n_blue, 3) and np.isfinite(np.asarray(action)).all()

    policy = make_weather_conservative_policy(cfg)
    base = (14 if cfg.obs_edge_features else 10) + cfg.resource.observation_dim
    changed = obs._replace(ego=obs.ego.at[:, base + 1].add(0.5))
    original_action = policy(obs, jax.random.PRNGKey(0))
    changed_action = policy(changed, jax.random.PRNGKey(999))
    assert not np.array_equal(np.asarray(original_action[:, 0]), np.asarray(changed_action[:, 0]))
