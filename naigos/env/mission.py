"""Fixed-shape procedural mission state for opt-in mission environments."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .config import MISSION_FAMILIES, MissionConfig
from .obs import POS_SCALE, RANGE_SCALE


MISSION_RNG_STREAM_ID = 108
MISSION_SCHEMA_ID = "procedural-mission-state-v1"

OUTCOME_ONGOING = 0
OUTCOME_PRIMARY = 1
OUTCOME_RECOVERED = 2
OUTCOME_LOSS = 3
OUTCOME_TIMEOUT = 4


class MissionState(NamedTuple):
    stage_pos: jax.Array  # (B, S, 3), AMSL
    stage_radius: jax.Array  # (B, S, 2): horizontal, vertical half-height
    stage_mask: jax.Array  # (B, S) bool
    current_stage: jax.Array  # (B,) int32
    dwell_s: jax.Array  # (B,)
    deadline_step: jax.Array  # (B,) int32
    completed: jax.Array  # (B,) primary completion only
    recovered: jax.Array  # (B,) safe recovery/diversion only
    failed: jax.Array  # (B,) loss or timeout
    outcome: jax.Array  # (B,) uint8


def config_digest(cfg: MissionConfig) -> str:
    raw = json.dumps(dataclasses.asdict(cfg), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def _stage_count(cfg: MissionConfig) -> int:
    return {
        "transit": 1,
        "multi_leg": min(3, cfg.max_stages),
        "time_on_station": 1,
        "survey_corridor": min(4, cfg.max_stages),
        "recovery": 1,
    }[cfg.family]


def initial_state(
    cfg: MissionConfig,
    episode_key: jax.Array,
    start: jax.Array,
    objective: jax.Array,
    recovery_pos: jax.Array,
    recovery_active: jax.Array,
    dt: float,
) -> MissionState:
    """Construct anonymous stages without moving any existing reset RNG stream."""
    b = start.shape[0]
    smax = cfg.max_stages
    stage = jnp.broadcast_to(objective[:, None, :], (b, smax, 3))
    count = _stage_count(cfg)
    key = jax.random.fold_in(episode_key, MISSION_RNG_STREAM_ID)

    if cfg.family in ("multi_leg", "survey_corridor"):
        frac = jnp.linspace(1.0 / count, 1.0, count)
        delta = objective - start
        direction = delta[:, :2] / jnp.maximum(jnp.linalg.norm(delta[:, :2], axis=-1, keepdims=True), 1.0)
        normal = jnp.stack([-direction[:, 1], direction[:, 0]], -1)
        offsets = jax.random.uniform(key, (b, count), minval=-0.08, maxval=0.08)
        lateral_scale = jnp.linalg.norm(delta[:, :2], axis=-1, keepdims=True)
        xy = (start[:, None, :2] + frac[None, :, None] * delta[:, None, :2]
              + offsets[..., None] * lateral_scale[..., None] * normal[:, None, :])
        z = start[:, None, 2] + frac[None, :] * delta[:, None, 2]
        generated = jnp.concatenate([xy, z[..., None]], -1)
        stage = stage.at[:, :count, :].set(generated)
    elif cfg.family == "recovery":
        idx = jnp.argmax(recovery_active.astype(jnp.int32), axis=-1)
        target = recovery_pos[jnp.arange(b), idx]
        stage = stage.at[:, 0, :].set(target)

    mask = jnp.arange(smax)[None, :] < count
    mask = jnp.broadcast_to(mask, (b, smax))
    radius = jnp.zeros((b, smax, 2), dtype=jnp.float32)
    radius = radius.at[..., 0].set(cfg.stage_radius_m)
    radius = radius.at[..., 1].set(cfg.stage_vertical_half_m)
    deadline = jnp.full((b,), max(1, int(cfg.deadline_s / dt)), dtype=jnp.int32)
    return MissionState(
        stage_pos=stage,
        stage_radius=radius,
        stage_mask=mask,
        current_stage=jnp.zeros((b,), dtype=jnp.int32),
        dwell_s=jnp.zeros((b,), dtype=jnp.float32),
        deadline_step=deadline,
        completed=jnp.zeros((b,), dtype=bool),
        recovered=jnp.zeros((b,), dtype=bool),
        failed=jnp.zeros((b,), dtype=bool),
        outcome=jnp.zeros((b,), dtype=jnp.uint8),
    )


def current_target(state: MissionState) -> jax.Array:
    b = state.current_stage.shape[0]
    return state.stage_pos[jnp.arange(b), state.current_stage]


def advance(
    cfg: MissionConfig,
    state: MissionState,
    pos: jax.Array,
    dt: float,
    step_index: jax.Array,
    loss: jax.Array,
    recovery_event: jax.Array,
):
    """Advance stages and return ``(state, primary_now, recovered_now, timeout_now)``."""
    b = pos.shape[0]
    idx = state.current_stage
    target = state.stage_pos[jnp.arange(b), idx]
    radius = state.stage_radius[jnp.arange(b), idx]
    inside = ((jnp.linalg.norm(pos[:, :2] - target[:, :2], axis=-1) <= radius[:, 0])
              & (jnp.abs(pos[:, 2] - target[:, 2]) <= radius[:, 1]))
    active = ~state.completed & ~state.recovered & ~state.failed

    if cfg.family == "recovery":
        recovered_now = recovery_event & active & ~loss
        primary_now = jnp.zeros((b,), dtype=bool)
        next_idx = idx
        dwell = state.dwell_s
    elif cfg.family == "time_on_station":
        dwell = jnp.where(inside & active, state.dwell_s + dt, 0.0)
        primary_now = (dwell >= cfg.dwell_required_s) & active & ~loss
        recovered_now = jnp.zeros((b,), dtype=bool)
        next_idx = idx
    else:
        reached_stage = inside & active & ~loss
        last = idx >= (jnp.sum(state.stage_mask, axis=-1) - 1)
        primary_now = reached_stage & last
        next_idx = jnp.where(reached_stage & ~last, idx + 1, idx)
        dwell = jnp.zeros_like(state.dwell_s)
        recovered_now = jnp.zeros((b,), dtype=bool)

    completed = state.completed | primary_now
    recovered = state.recovered | recovered_now
    timeout_now = ((step_index >= state.deadline_step) & active & ~completed & ~recovered & ~loss)
    failed = state.failed | loss | timeout_now
    new_event = state.outcome == OUTCOME_ONGOING
    outcome = state.outcome
    outcome = jnp.where(new_event & timeout_now, OUTCOME_TIMEOUT, outcome)
    outcome = jnp.where(new_event & loss, OUTCOME_LOSS, outcome)
    outcome = jnp.where(new_event & recovered_now, OUTCOME_RECOVERED, outcome)
    outcome = jnp.where(new_event & primary_now, OUTCOME_PRIMARY, outcome).astype(jnp.uint8)
    return state._replace(
        current_stage=next_idx,
        dwell_s=dwell,
        completed=completed,
        recovered=recovered,
        failed=failed,
        outcome=outcome,
    ), primary_now, recovered_now, timeout_now


def observation_features(
    cfg: MissionConfig,
    state: MissionState,
    pos: jax.Array,
    psi: jax.Array,
    step_index: jax.Array,
    recovery_available: jax.Array,
) -> jax.Array:
    """Only current-stage ego-relative facts; future hidden stages never appear."""
    if not cfg.enabled:
        return jnp.zeros((pos.shape[0], 0), dtype=jnp.float32)
    target = current_target(state)
    rel = target - pos
    bearing = jnp.arctan2(rel[:, 1], rel[:, 0]) - psi
    rng = jnp.linalg.norm(rel[:, :2], axis=-1)
    dwell = jnp.clip(state.dwell_s / cfg.dwell_required_s, 0.0, 1.0)
    remaining = jnp.clip((state.deadline_step - step_index) / jnp.maximum(state.deadline_step, 1), 0.0, 1.0)
    valid = (~state.completed & ~state.recovered & ~state.failed).astype(jnp.float32)
    return jnp.stack(
        [
            jnp.sin(bearing),
            jnp.cos(bearing),
            jnp.clip(rng / RANGE_SCALE, 0.0, 4.0),
            jnp.clip(rel[:, 2] / POS_SCALE, -2.0, 2.0),
            dwell,
            remaining,
            valid,
            recovery_available.astype(jnp.float32),
        ],
        axis=-1,
    )
