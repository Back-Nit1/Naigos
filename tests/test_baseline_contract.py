"""baseline-v0 structural contract after the resource-world amendment.

Amended design (docs/plans/resource-world.md, "Fixed-shape JAX contract"):
``EnvState`` and ``RewardTerms`` are runtime-only pytrees and each gained
neutral fields (``resource``, ``diverted``). What baseline-v0 keeps byte-for-byte
is everything that is persisted or consumed by a trained actor: the ``Learner``
checkpoint pytree, the checkpoint blob keys the demo reads, the ego width, the
RNG streams, and the transition numerics. These tests are the proof.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from naigos.env.config import RESOURCE_WORLD_BASELINE, DetectionConfig, EnvConfig, TerrainConfig
from naigos.env.flight_env import EnvState, NaigosEnv, RewardTerms
from naigos.rl import checkpoint as ck
from naigos.rl.cbf_sweep import env_config_from_blob
from naigos.rl.ppo import Learner, PPOConfig, greedy_policy, init_learner
from naigos.rl.red_team import RedCurriculum
from naigos.rl.reward import RewardCurriculum, RewardWeights

SHIPPED = Path(__file__).resolve().parents[1] / "checkpoints" / "theatre_1000"
CFG = EnvConfig(
    n_blue=2, n_threat=4, n_threat_active=0, k_threat_obs=2, k_friend_obs=1, max_steps=6,
    detection=DetectionConfig(los_samples=8), terrain=TerrainConfig(nx=16, ny=16, cell=1_000.0),
)


def test_runtime_pytrees_only_gained_trailing_neutral_fields():
    assert EnvState._fields[:-2] == (
        "air", "threats", "lock", "dwell", "alive", "reached", "objective", "prev_dist",
        "hmap", "bounds", "t", "key", "route", "red_scales",
    ) and EnvState._fields[-2:] == ("resource", "weather")
    assert RewardTerms._fields[:5] == ("progress", "exposure", "lock_level", "envelope_dwell", "arrived")
    assert set(RewardTerms._fields) - {"diverted"} == set(RewardTerms._fields[:5]) | {
        "shotdown", "fuel_used", "g_excess", "terrain_violation", "ceiling_violation",
        "stall_violation", "bounds_violation", "out_of_fuel", "edge_proximity",
    }


def test_learner_checkpoint_pytree_is_unchanged_and_holds_no_env_state(tmp_path):
    assert Learner._fields == ("actor", "critic", "lam_raw", "lam_opt", "cost_accum", "episode_accum")
    ppo = PPOConfig(n_envs=1, n_steps=2, n_epochs=1, n_minibatches=1)
    key = jax.random.PRNGKey(0)
    learner = init_learner(key, CFG, ppo, NaigosEnv(CFG).reset(key)[1])
    path = ck.save(
        tmp_path / "ckpt_000001", learner=learner, key=key, iteration=1, red_level=0.0,
        shootdown_rate=0.0, survival_rate=1.0, survival_w=1.0, efficiency_w=0.0, history=[{"iter": 0}],
        env_cfg=CFG, ppo_cfg=ppo, red_curriculum=RedCurriculum(), reward_curriculum=RewardCurriculum(),
        reward_weights_base=RewardWeights(), wall_s=0.0,
    )
    blob = ck.load(path)
    assert not {"env_state", "state", "rollout", "resource"} & set(blob)
    assert blob["environment_model_id"] == RESOURCE_WORLD_BASELINE and blob["resource_bank"] is None
    ck.require_environment_compatible(blob, CFG, path)
    ck.require_actor_ego_dim(blob["actor"], CFG, path)
    restored = ck.restore_learner(init_learner(key, CFG, ppo, NaigosEnv(CFG).reset(key)[1]), blob)
    for a, b in zip(jax.tree.leaves(learner), jax.tree.leaves(restored)):
        np.testing.assert_array_equal(a, b)


@pytest.mark.skipif(not SHIPPED.exists(), reason="shipped checkpoint not present")
def test_shipped_baseline_checkpoint_runs_on_the_amended_runtime():
    blob = ck.load(SHIPPED)
    assert ck.environment_model_from_blob(blob) == RESOURCE_WORLD_BASELINE
    cfg = env_config_from_blob(blob, n_blue=2, n_threat=8, n_threat_active=4, max_steps=6)
    ck.require_environment_compatible(blob, cfg, SHIPPED)
    ck.require_actor_ego_dim(blob["actor"], cfg, SHIPPED)
    assert cfg.resource.model == RESOURCE_WORLD_BASELINE
    final, traj = NaigosEnv(cfg).rollout(jax.random.PRNGKey(0), greedy_policy(blob["actor"], cfg))
    assert bool(jnp.isfinite(final.air.pos).all()) and traj["pos"].shape[0] == cfg.max_steps


def test_baseline_transition_ignores_every_resource_latent():
    env = NaigosEnv(CFG)
    state, _ = env.reset(jax.random.PRNGKey(4))
    r = state.resource
    noisy = state._replace(resource=r._replace(
        burn_multiplier=r.burn_multiplier * 7.0, reserve_fuel=r.reserve_fuel + 1e9,
        initial_fuel=r.initial_fuel * 3.0, diversion_pos=state.air.pos[:, None, :] + jnp.zeros((2, 2, 3)),
        diversion_active=jnp.ones_like(r.diversion_active),
    ))
    action = jnp.array([[0.3, 0.1, 0.6]] * 2)
    clean_out = env.step(state, action)
    noisy_out = env.step(noisy, action)
    for a, b in zip(jax.tree.leaves(clean_out[0].air), jax.tree.leaves(noisy_out[0].air)):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(clean_out[1].ego, noisy_out[1].ego)
    assert clean_out[1].ego.shape[-1] == 10
    for a, b in zip(clean_out[2], noisy_out[2]):
        np.testing.assert_array_equal(a, b)
