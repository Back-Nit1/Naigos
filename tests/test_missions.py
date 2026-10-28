"""Procedural mission families, local observations, deadlines and recovery semantics."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from naigos.env import mission, resource
from naigos.env.config import (
    MISSION_V1,
    RESOURCE_WORLD_L1,
    DetectionConfig,
    EnvConfig,
    MissionConfig,
    ResourceWorldConfig,
    TerrainConfig,
)
from naigos.env.flight_env import NaigosEnv
from naigos.rl.train import rollout_metrics


def _cfg(family, **kw):
    values = dict(
        model=MISSION_V1,
        family=family,
        scenario_id=f"test/{family}",
        max_stages=4,
        dwell_required_s=6.0,
        deadline_s=20.0,
    )
    values.update(kw)
    return MissionConfig(**values)


def _state(cfg):
    start = jnp.array([[0.0, 0.0, 1_000.0], [0.0, 100.0, 1_000.0]])
    objective = jnp.array([[10_000.0, 0.0, 1_000.0], [10_000.0, 100.0, 1_000.0]])
    recovery = jnp.array([[[500.0, 500.0, 1_000.0], [900.0, 500.0, 1_000.0]]] * 2)
    active = jnp.array([[True, True], [True, False]])
    return mission.initial_state(cfg, jax.random.PRNGKey(4), start, objective, recovery, active, 2.0)


@pytest.mark.parametrize("family,count", [("transit", 1), ("multi_leg", 3), ("survey_corridor", 4)])
def test_ordered_stage_families_complete_only_after_the_last_stage(family, count):
    cfg = _cfg(family)
    st = _state(cfg)
    assert np.all(np.asarray(st.stage_mask.sum(axis=1)) == count)
    for i in range(count):
        pos = mission.current_target(st)
        st, completed, recovered, timeout = mission.advance(
            cfg, st, pos, 2.0, jnp.int32(i + 1), jnp.zeros(2, bool), jnp.zeros(2, bool)
        )
        assert not bool(np.any(recovered)) and not bool(np.any(timeout))
        assert bool(np.all(completed)) == (i == count - 1)
        assert np.all(np.asarray(st.current_stage) == min(i + 1, count - 1))
    assert bool(np.all(st.completed)) and np.all(np.asarray(st.outcome) == mission.OUTCOME_PRIMARY)


def test_time_on_station_requires_consecutive_dwell_and_resets_outside():
    cfg = _cfg("time_on_station")
    st = _state(cfg)
    inside = mission.current_target(st)
    outside = inside + jnp.array([10_000.0, 0.0, 0.0])
    st, done, *_ = mission.advance(cfg, st, inside, 2.0, jnp.int32(1), jnp.zeros(2, bool), jnp.zeros(2, bool))
    assert np.allclose(st.dwell_s, 2.0) and not bool(np.any(done))
    st, *_ = mission.advance(cfg, st, outside, 2.0, jnp.int32(2), jnp.zeros(2, bool), jnp.zeros(2, bool))
    assert np.allclose(st.dwell_s, 0.0)
    for step in (3, 4, 5):
        st, done, *_ = mission.advance(cfg, st, inside, 2.0, jnp.int32(step), jnp.zeros(2, bool), jnp.zeros(2, bool))
    assert bool(np.all(done)) and bool(np.all(st.completed))


def test_recovery_is_distinct_from_primary_completion():
    cfg = _cfg("recovery")
    st = _state(cfg)
    st, primary, recovered, timeout = mission.advance(
        cfg, st, mission.current_target(st), 2.0, jnp.int32(1),
        jnp.zeros(2, bool), jnp.ones(2, bool),
    )
    assert not bool(np.any(primary)) and bool(np.all(recovered)) and not bool(np.any(timeout))
    assert not bool(np.any(st.completed)) and bool(np.all(st.recovered))
    assert np.all(np.asarray(st.outcome) == mission.OUTCOME_RECOVERED)


def test_deadline_and_loss_are_exclusive_and_loss_has_precedence():
    cfg = _cfg("transit", deadline_s=4.0)
    st = _state(cfg)
    far = mission.current_target(st) + jnp.array([20_000.0, 0.0, 0.0])
    st, primary, recovered, timeout = mission.advance(
        cfg, st, far, 2.0, jnp.int32(2), jnp.array([True, False]), jnp.zeros(2, bool)
    )
    assert not bool(np.any(primary)) and not bool(np.any(recovered))
    assert bool(timeout[1]) and not bool(timeout[0])
    np.testing.assert_array_equal(st.outcome, [mission.OUTCOME_LOSS, mission.OUTCOME_TIMEOUT])


def test_actor_observation_contains_current_stage_only():
    cfg = _cfg("multi_leg")
    st = _state(cfg)
    pos = jnp.zeros((2, 3))
    psi = jnp.zeros(2)
    a = mission.observation_features(cfg, st, pos, psi, jnp.int32(0), jnp.ones(2, bool))
    edited = st._replace(stage_pos=st.stage_pos.at[:, 1:, :].add(1_000_000.0))
    b = mission.observation_features(cfg, edited, pos, psi, jnp.int32(0), jnp.ones(2, bool))
    np.testing.assert_array_equal(a, b)
    assert a.shape == (2, 8) and np.isfinite(np.asarray(a)).all()


def test_mission_stream_does_not_move_resource_or_threat_draws_and_env_jits():
    common = dict(
        n_blue=2,
        n_threat=3,
        n_threat_active=2,
        max_steps=8,
        detection=DetectionConfig(los_samples=8),
        terrain=TerrainConfig(nx=16, ny=16, cell=1_000.0),
        resource=ResourceWorldConfig(model=RESOURCE_WORLD_L1),
    )
    legacy = NaigosEnv(EnvConfig(**common))
    enabled = NaigosEnv(EnvConfig(**common, mission=_cfg("multi_leg")))
    key = jax.random.PRNGKey(19)
    a, _ = legacy.reset(key)
    b, ob = enabled.reset(key)
    assert mission.MISSION_RNG_STREAM_ID == resource.RNG_STREAM_IDS["mission"]
    np.testing.assert_array_equal(a.resource.initial_fuel, b.resource.initial_fuel)
    np.testing.assert_array_equal(a.threats.pos, b.threats.pos)
    assert ob.ego.shape[-1] == enabled.cfg.ego_dim
    out = jax.jit(enabled.step)(b, jnp.zeros((2, 3)))
    assert np.isfinite(np.asarray(out[0].air.pos)).all()


def test_recovery_family_fails_closed_without_endurance_resource():
    with pytest.raises(ValueError, match="requires the endurance"):
        NaigosEnv(EnvConfig(mission=_cfg("recovery")))


def test_mission_metrics_keep_timeout_completion_recovery_and_survival_distinct():
    cfg = EnvConfig(
        n_blue=1,
        n_threat=2,
        n_threat_active=0,
        k_friend_obs=1,
        max_steps=1,
        terrain=TerrainConfig(nx=16, ny=16, cell=1_000.0),
        detection=DetectionConfig(los_samples=8),
        resource=ResourceWorldConfig(model=RESOURCE_WORLD_L1),
        mission=_cfg("transit", deadline_s=2.0),
    )
    env = NaigosEnv(cfg)
    policy = lambda obs, key: jnp.zeros((1, 3))  # noqa: E731
    final, traj = jax.vmap(lambda key: env.rollout(key, policy))(
        jax.random.split(jax.random.PRNGKey(31), 2)
    )
    metrics = rollout_metrics(final, traj, 2, 1)
    assert metrics["timeout_rate"] == 1.0
    assert metrics["survival_rate"] == 0.0
    assert metrics["primary_mission_completion_rate"] == 0.0
    assert metrics["safe_diversion_recovery_rate"] == 0.0
    assert metrics["joint_mission_survival_rate"] == 0.0
    for name in (
        "building_collision_rate", "mission_duration_s", "route_length_m", "fuel_used_kg",
        "stall_envelope_rate", "max_gust_mps", "cbf_intervention_rate",
        "cbf_authority_exceedance_rate",
    ):
        assert np.isfinite(metrics[name])

    # Even a log claiming primary completion is not joint success after loss.
    tampered = final._replace(
        alive=jnp.zeros_like(final.alive),
        mission=final.mission._replace(completed=jnp.ones_like(final.mission.completed)),
        resource=final.resource._replace(
            outcome=jnp.full_like(final.resource.outcome, resource.OUTCOME_PRIMARY)
        ),
    )
    separated = rollout_metrics(tampered, traj, 2, 1)
    assert separated["primary_mission_completion_rate"] == 1.0
    assert separated["survival_rate"] == 0.0
    assert separated["joint_mission_survival_rate"] == 0.0
