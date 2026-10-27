"""L1 endurance and abstract recovery-gate model.

This module is deliberately independent of threat and reward code.  It owns
only the versioned resource state, deterministic reset sampling, observation
features, and terminal outcome labels.  ``baseline-v0`` receives neutral state
and the caller takes no resource branch, preserving the shipped environment.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from . import obs as obs_mod
from . import terrain as terrain_mod
from .airframe import AircraftState
from .config import N_DIVERSION_GATES, EnvConfig

OUTCOME_ONGOING = 0
OUTCOME_PRIMARY = 1
OUTCOME_DIVERTED = 2
OUTCOME_SHOTDOWN = 3
OUTCOME_TERRAIN = 4
OUTCOME_BOUNDS = 5
OUTCOME_FUEL = 6
OUTCOME_TIMEOUT = 7


# Episode-level stream IDs from docs/plans/resource-world.md. L1 consumes only
# "fuel" (its three latents are sub-folded 1..3 under that one stream); the
# rest are reserved so enabling a later resource never moves an earlier draw.
RNG_STREAM_IDS = {
    "fuel": 101,
    "sensor": 102,
    "comms": 103,
    "actuator": 104,
    "compute": 105,
    "weather": 106,
    "map": 107,
}


class ResourceState(NamedTuple):
    initial_fuel: jax.Array  # (B,) kg
    reserve_fuel: jax.Array  # (B,) kg
    burn_multiplier: jax.Array  # (B,)
    diversion_pos: jax.Array  # (B, 2, 3), abstract recovery gates
    diversion_active: jax.Array  # (B, 2) bool
    diverted: jax.Array  # (B,) bool, terminal safe recovery
    outcome: jax.Array  # (B,) uint8, OUTCOME_*


def nominal_burn_rate(cfg: EnvConfig, speed: jax.Array) -> jax.Array:
    """Straight-flight burn assumption used for onboard fuel-to-go estimates."""
    af = cfg.airframe
    throttle = jnp.clip((speed - af.v_stall) / (af.v_max - af.v_stall), 0.0, 1.0)
    return af.burn_base + af.burn_throttle * throttle


def _uniform(key, shape, bounds):
    lo, hi = bounds
    return jax.random.uniform(key, shape, minval=lo, maxval=hi)


def _gates(cfg: EnvConfig, hmap, start, objective, bounds):
    """Two route-relative recovery gates, clipped inside the current play area."""
    delta = objective[:, :2] - start[:, :2]
    length = jnp.linalg.norm(delta, axis=-1, keepdims=True)
    direction = delta / jnp.maximum(length, 1.0)
    normal = jnp.stack([-direction[:, 1], direction[:, 0]], axis=-1)
    short = jnp.minimum(bounds[1] - bounds[0], bounds[3] - bounds[2])
    lateral = cfg.resource.gate_lateral_frac * short
    near = start[:, :2] + 0.12 * delta + lateral * normal
    forward = start[:, :2] + 0.72 * delta - lateral * normal
    xy = jnp.stack([near, forward], axis=1)
    margin = jnp.maximum(cfg.resource.gate_arrival_radius, 100.0)
    xy = xy.at[..., 0].set(jnp.clip(xy[..., 0], bounds[0] + margin, bounds[1] - margin))
    xy = xy.at[..., 1].set(jnp.clip(xy[..., 1], bounds[2] + margin, bounds[3] - margin))
    z_ground = terrain_mod.sample_height(
        hmap, cfg.terrain, xy[..., 0].reshape(-1), xy[..., 1].reshape(-1)
    ).reshape(cfg.n_blue, N_DIVERSION_GATES)
    return jnp.concatenate([xy, (z_ground + 1_000.0)[..., None]], axis=-1)


def initial_state(cfg: EnvConfig, episode_key, hmap, start, objective, bounds) -> tuple[AircraftState, ResourceState]:
    """Return reset airframe/resource state without moving legacy RNG streams.

    Stable ``fold_in`` identifiers mean adding later resources cannot shift the
    fuel, reserve, or burn-multiplier draws.
    """
    B = cfg.n_blue
    if not cfg.resource.enabled:
        fuel = jnp.full((B,), cfg.airframe.fuel_init)
        rs = ResourceState(
            initial_fuel=fuel,
            reserve_fuel=jnp.zeros((B,)),
            burn_multiplier=jnp.ones((B,)),
            diversion_pos=jnp.zeros((B, N_DIVERSION_GATES, 3)),
            diversion_active=jnp.zeros((B, N_DIVERSION_GATES), dtype=bool),
            diverted=jnp.zeros((B,), dtype=bool),
            outcome=jnp.zeros((B,), dtype=jnp.uint8),
        )
    else:
        direct = jnp.linalg.norm(objective[:, :2] - start[:, :2], axis=-1)
        rate = nominal_burn_rate(cfg, jnp.full((B,), cfg.airframe.v_init))
        direct_fuel = direct / cfg.airframe.v_init * rate
        fuel_key = jax.random.fold_in(episode_key, RNG_STREAM_IDS["fuel"])
        ratio = _uniform(jax.random.fold_in(fuel_key, 1), (B,), cfg.resource.fuel_budget_ratio)
        reserve_ratio = _uniform(jax.random.fold_in(fuel_key, 2), (B,), cfg.resource.reserve_ratio)
        burn_multiplier = _uniform(jax.random.fold_in(fuel_key, 3), (B,), cfg.resource.burn_multiplier)
        fuel = direct_fuel * ratio
        rs = ResourceState(
            initial_fuel=fuel,
            reserve_fuel=direct_fuel * reserve_ratio,
            burn_multiplier=burn_multiplier,
            diversion_pos=_gates(cfg, hmap, start, objective, bounds),
            diversion_active=jnp.broadcast_to(
                jnp.asarray(cfg.resource.active_gate_mask, dtype=bool),
                (B, N_DIVERSION_GATES),
            ),
            diverted=jnp.zeros((B,), dtype=bool),
            outcome=jnp.zeros((B,), dtype=jnp.uint8),
        )
    air = AircraftState(
        pos=start,
        speed=jnp.full((B,), cfg.airframe.v_init),
        psi=jnp.arctan2((objective - start)[:, 1], (objective - start)[:, 0]),
        gamma=jnp.zeros((B,)),
        phi=jnp.zeros((B,)),
        fuel=fuel,
    )
    return air, rs


def fuel_to_go(cfg: EnvConfig, air: AircraftState, target: jax.Array, burn_estimate: jax.Array) -> jax.Array:
    distance = jnp.linalg.norm(target[..., :2] - air.pos[..., None, :2], axis=-1)
    speed = jnp.maximum(air.speed[..., None], cfg.airframe.v_stall)
    return distance / speed * nominal_burn_rate(cfg, speed) * burn_estimate[..., None]


def observation_features(cfg: EnvConfig, air: AircraftState, objective, rs: ResourceState) -> jax.Array:
    """Named L1 resource/gate facts available to the onboard actor only."""
    if not cfg.resource.enabled:
        return jnp.zeros((cfg.n_blue, 0), dtype=jnp.float32)
    # The true episode burn multiplier is privileged simulator state. Planning
    # margins use one declared nominal onboard estimate; realized fuel remains
    # observable and can reveal consumption over time without leaking truth at
    # reset.
    burn_estimate = jnp.full(
        (cfg.n_blue,), cfg.resource.onboard_burn_multiplier, dtype=air.fuel.dtype
    )
    primary_need = fuel_to_go(cfg, air, objective[:, None, :], burn_estimate)[:, 0]
    diversion_need = fuel_to_go(cfg, air, rs.diversion_pos, burn_estimate)
    scale = jnp.maximum(rs.initial_fuel, 1e-3)
    rate = nominal_burn_rate(cfg, air.speed) * burn_estimate
    resource = jnp.stack(
        [
            air.fuel / scale,
            rs.reserve_fuel / scale,
            jnp.clip(air.fuel / jnp.maximum(rate * cfg.dt * cfg.max_steps, 1e-3), 0.0, 4.0),
            (air.fuel - rs.reserve_fuel - primary_need) / scale,
            jnp.max(jnp.where(rs.diversion_active, air.fuel[:, None] - rs.reserve_fuel[:, None] - diversion_need, -1e9), axis=1) / scale,
            rate / jnp.maximum(cfg.airframe.burn_base + cfg.airframe.burn_throttle, 1e-3),
        ],
        axis=-1,
    )
    rel = rs.diversion_pos - air.pos[:, None, :]
    rel_e = obs_mod.to_ego_frame(rel, air.psi[:, None])
    rng = jnp.linalg.norm(rel[..., :2], axis=-1)
    bearing = jnp.arctan2(rel_e[..., 1], rel_e[..., 0])
    margin = (air.fuel[:, None] - rs.reserve_fuel[:, None] - diversion_need) / scale[:, None]
    gate = jnp.stack(
        [
            rel_e[..., 0] / obs_mod.POS_SCALE,
            rel_e[..., 1] / obs_mod.POS_SCALE,
            rng / obs_mod.RANGE_SCALE,
            jnp.sin(bearing),
            jnp.cos(bearing),
            margin,
            rs.diversion_active.astype(jnp.float32),
        ],
        axis=-1,
    )
    gate = gate * rs.diversion_active[..., None]
    return jnp.concatenate([resource, gate.reshape(cfg.n_blue, -1)], axis=-1)


# Simultaneous events within one step resolve by this order, highest first.
# Any loss beats any recovery (the env also refuses to grant arrival/diversion
# on a loss step); among losses a hostile kill is attributed first, then the
# physical terrain impact, then leaving the play box, then the fuel-exhaustion
# crossing (the aircraft still glides on that step). Primary beats diversion.
# Raw per-event flags stay in ``RewardTerms`` as evidence; metrics, the
# verifier and replay summaries count the exclusive label.
OUTCOME_PRECEDENCE = (
    OUTCOME_SHOTDOWN,
    OUTCOME_TERRAIN,
    OUTCOME_BOUNDS,
    OUTCOME_FUEL,
    OUTCOME_PRIMARY,
    OUTCOME_DIVERTED,
    OUTCOME_TIMEOUT,
)


def outcome_update(previous, primary, diverted, shot, terrain, bounds, dry, timeout):
    """Assign exactly one first terminal outcome following ``OUTCOME_PRECEDENCE``."""
    events = {
        OUTCOME_SHOTDOWN: shot,
        OUTCOME_TERRAIN: terrain,
        OUTCOME_BOUNDS: bounds,
        OUTCOME_FUEL: dry,
        OUTCOME_PRIMARY: primary,
        OUTCOME_DIVERTED: diverted,
        OUTCOME_TIMEOUT: timeout,
    }
    new = jnp.zeros_like(previous)
    for code in reversed(OUTCOME_PRECEDENCE):  # lowest priority first, so higher overwrites
        new = jnp.where(events[code], code, new)
    return jnp.where(previous == OUTCOME_ONGOING, new, previous).astype(jnp.uint8)
