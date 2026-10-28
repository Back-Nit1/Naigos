"""Deterministic, fixed-shape weather for the opt-in vehicle-weather model.

The turbulence implementation is a bounded, temporally correlated AR(1)
engineering surrogate.  It is not Dryden turbulence and is not CFD.  Its
configuration carries the vertical intensity profile and three scale lengths
needed to replace the recursion with a Dryden shaping filter in a future schema.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .config import WeatherConfig


WEATHER_RNG_STREAM_ID = 106
TURBULENCE_MODEL_ID = "bounded-ar1-gust-v1"


def _mix32(x):
    """Small stateless integer mixer with identical NumPy/JAX implementations."""
    x = jnp.asarray(x, dtype=jnp.uint32)
    x = (x ^ (x >> jnp.uint32(16))) * jnp.uint32(0x7FEB352D)
    x = (x ^ (x >> jnp.uint32(15))) * jnp.uint32(0x846CA68B)
    return x ^ (x >> jnp.uint32(16))


def counter_normal(stream_key: jax.Array, step_index, shape, substream: int = 0):
    """Deterministic unit-variance pseudo-normal noise from a uint32 counter.

    Six mixed uniforms minus their mean, scaled to unit variance. It is bounded
    and intentionally simple enough for the independent NumPy verifier to
    reproduce without importing JAX.
    """
    n = 1
    for dim in shape:
        n *= int(dim)
    idx = jnp.arange(n, dtype=jnp.uint32)
    base = (stream_key[0] ^ (stream_key[1] * jnp.uint32(0x9E3779B9))
            ^ jnp.asarray(step_index, jnp.uint32) * jnp.uint32(0x85EBCA6B)
            ^ jnp.uint32(substream * 0xC2B2AE35))
    draws = []
    for k in range(6):
        word = _mix32(base + idx + jnp.uint32((k + 1) * 0x27D4EB2D))
        draws.append((word.astype(jnp.float32) + 0.5) / jnp.float32(2**32))
    return ((jnp.stack(draws, 0).sum(0) - 3.0) * jnp.sqrt(2.0)).reshape(shape)


class WeatherState(NamedTuple):
    """Per-aircraft local truth and actor-visible estimates."""

    mean_wind_enu: jax.Array  # (B, 3)
    gust_enu: jax.Array  # (B, 3), correlated gust state
    density_ratio: jax.Array  # (B,)
    temperature_k: jax.Array  # (B,)
    sensing_attenuation: jax.Array  # (B,)
    turbulence_intensity: jax.Array  # (B,), stationary one-sigma m/s
    background_turbulence_intensity: jax.Array  # (B,)
    mechanical_turbulence_intensity: jax.Array  # (B,), evidence-grade urban surrogate
    estimated_wind_enu: jax.Array  # (B, 3), local/noisy actor estimate
    estimated_density_ratio: jax.Array  # (B,)
    stream_key: jax.Array  # (2,), named weather stream root


def config_digest(cfg: WeatherConfig) -> str:
    blob = json.dumps(dataclasses.asdict(cfg), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def _interp(cfg: WeatherConfig, altitude_m: jax.Array, values) -> jax.Array:
    return jnp.interp(
        altitude_m,
        jnp.asarray(cfg.altitude_knots_m, dtype=jnp.float32),
        jnp.asarray(values, dtype=jnp.float32),
    )


def profile_at(cfg: WeatherConfig, altitude_m: jax.Array):
    """Interpolate immutable profile truth at one or more altitudes."""
    wind = jnp.stack([_interp(cfg, altitude_m, [v[i] for v in cfg.wind_enu_mps]) for i in range(3)], -1)
    return {
        "wind_enu": wind,
        "density_ratio": _interp(cfg, altitude_m, cfg.density_ratio),
        "temperature_k": _interp(cfg, altitude_m, cfg.temperature_k),
        "sensing_attenuation": _interp(cfg, altitude_m, cfg.sensing_attenuation),
        "turbulence_intensity": _interp(cfg, altitude_m, cfg.turbulence_intensity_mps)
        * cfg.severity_scale,
    }


def _estimates(cfg: WeatherConfig, truth: dict, gust: jax.Array, key: jax.Array):
    sigma = truth["turbulence_intensity"]
    noise = jax.random.normal(key, gust.shape, dtype=gust.dtype)
    wind_sigma = cfg.estimate_noise_scale * jnp.maximum(sigma, 0.25)
    estimated_wind = truth["wind_enu"] + gust + noise * wind_sigma[:, None]
    density_noise = jax.random.normal(jax.random.fold_in(key, 1), sigma.shape, dtype=sigma.dtype)
    estimated_density = jnp.clip(
        truth["density_ratio"] * (1.0 + 0.02 * cfg.estimate_noise_scale * density_noise),
        0.2,
        1.5,
    )
    return estimated_wind, estimated_density


def initial_state(cfg: WeatherConfig, episode_key: jax.Array, altitude_m: jax.Array) -> WeatherState:
    root = jax.random.fold_in(episode_key, WEATHER_RNG_STREAM_ID)
    truth = profile_at(cfg, altitude_m)
    gust = jnp.zeros((altitude_m.shape[0], 3), dtype=jnp.float32)
    estimate, density_est = _estimates(cfg, truth, gust, jax.random.fold_in(root, 10))
    return WeatherState(
        mean_wind_enu=truth["wind_enu"],
        gust_enu=gust,
        density_ratio=truth["density_ratio"],
        temperature_k=truth["temperature_k"],
        sensing_attenuation=truth["sensing_attenuation"],
        turbulence_intensity=truth["turbulence_intensity"],
        background_turbulence_intensity=truth["turbulence_intensity"],
        mechanical_turbulence_intensity=jnp.zeros_like(truth["turbulence_intensity"]),
        estimated_wind_enu=estimate,
        estimated_density_ratio=density_est,
        stream_key=root,
    )


def step(
    cfg: WeatherConfig,
    state: WeatherState,
    altitude_m: jax.Array,
    true_airspeed_mps: jax.Array,
    dt: float,
    step_index: jax.Array,
    mechanical_intensity: jax.Array | None = None,
) -> WeatherState:
    """Advance the bounded AR(1) gust recursion exactly once per env step."""
    truth = profile_at(cfg, altitude_m)
    background = truth["turbulence_intensity"]
    mechanical = (jnp.zeros_like(background) if mechanical_intensity is None
                  else jnp.maximum(mechanical_intensity, 0.0))
    truth["turbulence_intensity"] = jnp.sqrt(background * background + mechanical * mechanical)
    scales = jnp.asarray(cfg.scale_length_m, dtype=jnp.float32)
    tau = scales[None, :] / jnp.maximum(true_airspeed_mps[:, None], 1.0)
    a = jnp.exp(-dt / jnp.maximum(tau, 1e-3))
    sigma = truth["turbulence_intensity"][:, None]
    key = jax.random.fold_in(state.stream_key, step_index.astype(jnp.uint32) + jnp.uint32(1))
    innovation = counter_normal(state.stream_key, step_index, state.gust_enu.shape, substream=0)
    gust = a * state.gust_enu + jnp.sqrt(jnp.maximum(1.0 - a * a, 0.0)) * sigma * innovation
    limit = cfg.gust_clip_sigma * sigma
    gust = jnp.where(sigma > 0.0, jnp.clip(gust, -limit, limit), 0.0)
    estimate, density_est = _estimates(cfg, truth, gust, jax.random.fold_in(key, 10))
    return WeatherState(
        mean_wind_enu=truth["wind_enu"],
        gust_enu=gust,
        density_ratio=truth["density_ratio"],
        temperature_k=truth["temperature_k"],
        sensing_attenuation=truth["sensing_attenuation"],
        turbulence_intensity=truth["turbulence_intensity"],
        background_turbulence_intensity=background,
        mechanical_turbulence_intensity=mechanical,
        estimated_wind_enu=estimate,
        estimated_density_ratio=density_est,
        stream_key=state.stream_key,
    )


def observation_features(cfg: WeatherConfig, state: WeatherState, psi: jax.Array) -> jax.Array:
    """Local estimates only; the vertical profile and future values stay privileged."""
    if not cfg.enabled:
        return jnp.zeros((psi.shape[0], 0), dtype=jnp.float32)
    c, s = jnp.cos(psi), jnp.sin(psi)
    w = state.estimated_wind_enu
    body = jnp.stack([w[:, 0] * c + w[:, 1] * s, -w[:, 0] * s + w[:, 1] * c, w[:, 2]], -1)
    uncertainty = cfg.estimate_noise_scale * jnp.maximum(state.turbulence_intensity, 0.25)
    return jnp.concatenate(
        [
            body / 100.0,
            (state.turbulence_intensity / 50.0)[:, None],
            state.estimated_density_ratio[:, None],
            (uncertainty / 50.0)[:, None],
        ],
        axis=-1,
    )
