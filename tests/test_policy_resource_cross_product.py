"""Policy family x resource world: every cell of the cross-product.

Policy identity (DeepSets@1 / GRU recurrent@1) and resource identity
(baseline-v0 / L1 endurance) are independent axes. Each cell must train, evaluate,
round-trip through a checkpoint, resume, trace every resource field, and refuse a
checkpoint whose identity on either axis differs.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from naigos.env.config import (
    RESOURCE_WORLD_BASELINE,
    RESOURCE_WORLD_L1,
    DetectionConfig,
    EnvConfig,
    ResourceWorldConfig,
    TerrainConfig,
)
from naigos.env.flight_env import NaigosEnv
from naigos.rl import checkpoint as ck
from naigos.rl import policies
from naigos.rl.ppo import PPOConfig, greedy_policy, init_learner, init_rollout, make_train
from naigos.rl.red_team import RedCurriculum
from naigos.rl.reward import RewardCurriculum, RewardWeights
from naigos.rl.train import TrainConfig, evaluate, run

L1 = ResourceWorldConfig(model=RESOURCE_WORLD_L1)
RESOURCES = {"baseline-v0": ResourceWorldConfig(), "l1-endurance": L1}
POLICIES = {
    "deepsets": PPOConfig(n_envs=2, n_steps=8, n_minibatches=2),
    "gru": PPOConfig(
        n_envs=2, n_steps=8, n_minibatches=2, policy_family="recurrent", recurrent_seq_len=4
    ),
}
CELLS = [(r, p) for r in RESOURCES for p in POLICIES]
CELL_IDS = [f"{r}+{p}" for r, p in CELLS]

TRACE_RESOURCE_FIELDS = (
    "fuel", "diverted", "outcome", "reserve_fuel", "initial_fuel", "burn_multiplier",
    "diversion_pos", "diversion_active", "objective", "bounds", "resource_observation",
)


def _env_cfg(resource):
    return EnvConfig(n_blue=2, n_threat=3, n_threat_active=3, resource=resource)


def _train_cfg(out, iterations):
    return TrainConfig(
        iterations=iterations, out_dir=str(out), seed=3, eval_every=2, eval_worlds=2,
        checkpoint_every=2, curriculum_every=2,
    )


@pytest.fixture(scope="module", params=CELLS, ids=CELL_IDS)
def cell(request, tmp_path_factory):
    """Fresh 2-iteration run, then an interruption-resume to 4 iterations."""
    resource_name, policy_name = request.param
    env_cfg, ppo = _env_cfg(RESOURCES[resource_name]), POLICIES[policy_name]
    root = tmp_path_factory.mktemp(f"{resource_name}-{policy_name}")
    learner, history = run(env_cfg, ppo, _train_cfg(root, 2))
    first = ck.latest_resumable(root)
    resumed, resumed_history = run(env_cfg, ppo, _train_cfg(root, 4), resume_from=first)
    return dict(
        env_cfg=env_cfg, ppo=ppo, root=root, learner=learner, history=history, first=first,
        resumed=resumed, resumed_history=resumed_history, resource=RESOURCES[resource_name],
        recurrent=policy_name == "gru",
    )


# --- fresh training, checkpoint round trip, resume ---------------------------


def test_fresh_training_is_finite_and_checkpointed(cell):
    assert cell["history"][-1]["iter"] == 2
    for row in cell["history"]:
        assert all(np.isfinite(v) for v in row.values() if isinstance(v, float))
    assert cell["first"].name == "ckpt_000002"


def test_checkpoint_carries_orthogonal_policy_and_resource_identity(cell):
    blob = ck.load(cell["first"])
    ident = ck.policy_identity(blob)
    assert ident["family"] == cell["ppo"].policy_family
    spec = policies.get_policy_spec(cell["env_cfg"], family=cell["ppo"].policy_family)
    assert ident["spec_sha256"] == spec.sha256
    assert ck.environment_model_from_blob(blob) == cell["resource"].model
    assert ck.resource_config_from_blob(blob) == cell["resource"]
    if cell["resource"].enabled:
        assert blob["observation_schema_id"] == ck.RESOURCE_OBSERVATION_SCHEMA
        assert blob["resource_bank"]["sha256"]
    else:
        assert blob["observation_schema_id"] == ck.BASELINE_OBSERVATION_SCHEMA
        assert blob["resource_bank"] is None
    assert blob["recovery"]["ppo"]["recurrent_seq_len"] == cell["ppo"].recurrent_seq_len


def test_checkpoint_round_trip_restores_the_learner_exactly(cell):
    blob = ck.load(cell["first"])
    env_cfg, ppo = cell["env_cfg"], cell["ppo"]
    env = NaigosEnv(env_cfg)
    _, obs = env.reset(jax.random.PRNGKey(99))
    template = init_learner(jax.random.PRNGKey(99), env_cfg, ppo, obs)
    spec = policies.get_policy_spec(env_cfg, family=ppo.policy_family)
    restored = ck.restore_learner(template, blob, expected_policy=spec)
    for a, b in zip(jax.tree.leaves(restored), jax.tree.leaves(cell["learner"])):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_resume_continues_history_and_advances_the_learner(cell):
    assert [r["iter"] for r in cell["resumed_history"]][-1] == 4
    prefix = [r["iter"] for r in cell["resumed_history"] if r["iter"] <= 2]
    assert prefix == [r["iter"] for r in cell["history"]]
    assert int(cell["resumed"].actor.step) > int(cell["learner"].actor.step)
    assert ck.latest_resumable(cell["root"]).name == "ckpt_000004"


# --- evaluation reconstructs both configurations -----------------------------


def test_evaluators_rebuild_policy_and_resource_from_the_checkpoint(cell):
    blob = ck.load(cell["first"])
    cfg = _env_cfg(ResourceWorldConfig()).replace(**ck.obs_config_from_blob(blob))
    assert cfg.resource == cell["resource"]
    ck.require_actor_ego_dim(blob["actor"], cfg, "cell")

    env = NaigosEnv(cfg)
    metrics = evaluate(
        env, blob["actor"], 2, jax.random.PRNGKey(0), **{
            "policy_family": ck.policy_identity(blob)["family"],
            "policy_version": ck.policy_identity(blob)["version"],
        },
    )
    assert all(np.isfinite(v) for v in metrics.values() if isinstance(v, float))

    sweep_cfg = __import__("naigos.rl.cbf_sweep", fromlist=["x"]).env_config_from_blob(blob)
    assert sweep_cfg.resource == cell["resource"]

    from naigos.pipeline.evaluation import load_actor_params

    loaded = load_actor_params(cell["first"])
    assert loaded.family == cell["ppo"].policy_family


def test_rollout_trace_carries_every_resource_field(cell):
    blob = ck.load(cell["first"])
    cfg = cell["env_cfg"]
    env = NaigosEnv(cfg)
    pol = greedy_policy(blob["actor"], cfg, **ck.policy_kwargs(blob))
    assert hasattr(pol, "initial_memory") == cell["recurrent"]
    _, traj = jax.jit(lambda k: env.rollout(k, pol, n_steps=5))(jax.random.PRNGKey(1))
    for name in TRACE_RESOURCE_FIELDS:
        assert name in traj, f"{name} missing from the {'GRU' if cell['recurrent'] else 'DeepSets'} trace"
    assert traj["resource_observation"].shape == (5, cfg.n_blue, cfg.resource.observation_dim)
    assert traj["fuel"].shape == (5, cfg.n_blue)


# --- refusal on any identity mismatch ----------------------------------------


def test_resume_refuses_every_mismatched_identity(cell):
    blob = ck.load(cell["first"])
    cfg, ppo, resource = cell["env_cfg"], cell["ppo"], cell["resource"]
    spec = policies.get_policy_spec(cfg, family=ppo.policy_family)
    ck.require_resume_compatible(blob, cfg, ppo, spec, "ok")  # the matching case passes

    other_family = "deepsets" if cell["recurrent"] else "recurrent"
    other_spec = policies.get_policy_spec(cfg, family=other_family)
    with pytest.raises(ck.CheckpointError, match="policy is"):
        ck.require_resume_compatible(blob, cfg, ppo, other_spec, "family")

    other_resource = L1 if not resource.enabled else ResourceWorldConfig()
    other_cfg = cfg.replace(resource=other_resource)
    other_policy = policies.get_policy_spec(other_cfg, family=ppo.policy_family)
    with pytest.raises(ck.CheckpointError, match="cross-schema"):
        ck.require_resume_compatible(blob, other_cfg, ppo, other_policy, "model")

    if cell["recurrent"]:
        with pytest.raises(ck.CheckpointError, match="recurrent_seq_len"):
            ck.require_resume_compatible(
                blob, cfg, dataclasses.replace(ppo, recurrent_seq_len=2), spec, "seq"
            )

    if resource.enabled:
        edited = dataclasses.replace(resource, gate_arrival_radius=resource.gate_arrival_radius + 1.0)
        with pytest.raises(ck.CheckpointError, match="resource_config"):
            ck.require_resume_compatible(blob, cfg.replace(resource=edited), ppo, spec, "cfg")
        bad_bank = {**blob, "resource_bank": {**blob["resource_bank"], "sha256": "0" * 64}}
        with pytest.raises(ck.CheckpointError, match="bank digest"):
            ck.require_resume_compatible(bad_bank, cfg, ppo, spec, "bank")


def test_legacy_checkpoint_is_baseline_deepsets_and_never_upgrades(cell):
    legacy = {
        k: v for k, v in ck.load(cell["first"]).items()
        if k not in ("policy", "environment_model_id", "observation_schema_id", "actor_schema_id",
                     "resource_config", "resource_bank")
    }
    legacy["schema"] = 4
    assert ck.policy_identity(legacy)["migrated_legacy_default"]
    assert ck.environment_model_from_blob(legacy) == RESOURCE_WORLD_BASELINE

    cfg, ppo = cell["env_cfg"], cell["ppo"]
    spec = policies.get_policy_spec(cfg, family=ppo.policy_family)
    wants_legacy = not cell["resource"].enabled and not cell["recurrent"]
    if wants_legacy:
        ck.require_resume_compatible(legacy, cfg, ppo, spec, "legacy")
    else:
        with pytest.raises(ck.CheckpointError):
            ck.require_resume_compatible(legacy, cfg, ppo, spec, "legacy")


# --- terminal semantics on the live collector --------------------------------

DIV_CFG = EnvConfig(
    n_blue=2, n_threat=2, n_threat_active=0, k_threat_obs=1, k_friend_obs=1, max_steps=40,
    detection=DetectionConfig(los_samples=8), terrain=TerrainConfig(nx=32, ny=32, cell=1_000.0),
    resource=ResourceWorldConfig(
        model=RESOURCE_WORLD_L1, fuel_budget_ratio=(5.0, 5.0), reserve_ratio=(0.1, 0.1),
        burn_multiplier=(1.0, 1.0), gate_arrival_radius=1_000.0,
    ),
)
DIV_PPO = {
    "deepsets": PPOConfig(n_envs=2, n_steps=4, n_epochs=1, n_minibatches=1),
    "gru": PPOConfig(
        n_envs=2, n_steps=4, n_epochs=1, n_minibatches=1, policy_family="recurrent",
        recurrent_seq_len=2,
    ),
}


def _diversion_setup(policy_name):
    ppo = DIV_PPO[policy_name]
    env = NaigosEnv(DIV_CFG, hmap=jnp.zeros((DIV_CFG.terrain.ny, DIV_CFG.terrain.nx)))
    k_init, k_roll, k_run = jax.random.split(jax.random.PRNGKey(13), 3)
    _, sample_obs = env.reset(k_init)
    learner = init_learner(k_init, DIV_CFG, ppo, sample_obs)
    train = make_train(env, ppo, RewardWeights())
    rollout = init_rollout(env, ppo, k_roll)
    state = rollout[0]
    # Aircraft zero sits on a recovery gate; aircraft one keeps each world open.
    gates = state.resource.diversion_pos.at[:, 0, 0].set(state.air.pos[:, 0])
    state = state._replace(resource=state.resource._replace(diversion_pos=gates))
    return ppo, learner, train, (state, *rollout[1:]), k_run


@pytest.mark.parametrize("policy_name", POLICIES)
def test_diverted_agent_leaves_live_samples_and_is_counted_once(policy_name):
    ppo, learner, train, rollout, key = _diversion_setup(policy_name)
    traj, *_ = jax.jit(train.collect)(learner, rollout, key)
    live, completed = np.asarray(traj["live"]), np.asarray(traj["completed"])
    np.testing.assert_array_equal(live[0, :, 0], True)
    np.testing.assert_array_equal(live[1:, :, 0], False)
    np.testing.assert_array_equal(completed[0, :, 0], 1.0)
    np.testing.assert_array_equal(completed[1:, :, 0], 0.0)
    np.testing.assert_array_equal(np.asarray(traj["r"])[1:, :, 0], 0.0)
    np.testing.assert_array_equal(np.asarray(traj["c"])[1:, :, 0], 0.0)
    np.testing.assert_array_equal(np.asarray(traj["reset"]), 0.0)
    updated, _, _ = jax.jit(train)(learner, rollout, key)
    assert float(updated.episode_accum) == ppo.n_envs


def test_gru_memory_resets_on_diversion_death_and_arrival():
    _, learner, train, rollout, key = _diversion_setup("gru")
    state = rollout[0]

    # Diversion: aircraft 0's pre-call memory is zero from the step after it diverts.
    traj, *_ = jax.jit(train.collect)(learner, rollout, key)
    mem = np.asarray(traj["memory"])  # (time, env, agent, hidden)
    np.testing.assert_array_equal(mem[1:, :, 0], 0.0)
    assert np.abs(mem[1:, :, 1]).max() > 0.0, "the live aircraft must still carry memory"

    # Death and arrival are terminal for memory too.
    for name, marked in (
        ("dead", state._replace(alive=state.alive.at[:, 0].set(False))),
        ("reached", state._replace(reached=state.reached.at[:, 0].set(True))),
    ):
        traj, *_ = jax.jit(train.collect)(learner, (marked, *rollout[1:]), key)
        mem = np.asarray(traj["memory"])
        np.testing.assert_array_equal(mem[1:, :, 0], 0.0, err_msg=name)
        assert np.abs(mem[1:, :, 1]).max() > 0.0


def test_gru_memory_resets_when_the_world_resets():
    # max_steps=3 ends the world inside the segment; the next call starts fresh.
    cfg = DIV_CFG.replace(max_steps=3)
    ppo = DIV_PPO["gru"]
    env = NaigosEnv(cfg, hmap=jnp.zeros((cfg.terrain.ny, cfg.terrain.nx)))
    k_init, k_roll, k_run = jax.random.split(jax.random.PRNGKey(5), 3)
    _, obs = env.reset(k_init)
    learner = init_learner(k_init, cfg, ppo, obs)
    train = make_train(env, ppo, RewardWeights())
    traj, *_ = jax.jit(train.collect)(learner, init_rollout(env, ppo, k_roll), k_run)
    np.testing.assert_array_equal(np.asarray(traj["reset"][2]), 1.0)
    np.testing.assert_array_equal(np.asarray(traj["memory"][3]), 0.0)


def test_gru_rollout_memory_resets_on_diversion():
    env = NaigosEnv(DIV_CFG, hmap=jnp.zeros((DIV_CFG.terrain.ny, DIV_CFG.terrain.nx)))
    state, obs = env.reset(jax.random.PRNGKey(2))
    gates = state.resource.diversion_pos.at[0, 0].set(state.air.pos[0])
    state = state._replace(resource=state.resource._replace(diversion_pos=gates))
    mem0 = jnp.zeros((DIV_CFG.n_blue, policies.RECURRENT_HIDDEN))

    seen = []

    def policy(o, k, memory):
        seen.append(memory)
        return jnp.zeros((DIV_CFG.n_blue, 3)), memory + 1.0

    policy.initial_memory = mem0
    # drive step() directly the way rollout() does: reset memory on agent_done
    memory = mem0
    for _ in range(3):
        _, memory = policy(obs, None, memory)
        state, obs, _, done, info = env.step(state, jnp.zeros((DIV_CFG.n_blue, 3)))
        memory = policies.reset_memory(memory, info["agent_done"] | done)
    assert bool(state.resource.diverted[0])
    np.testing.assert_array_equal(np.asarray(memory[0]), 0.0)
    assert float(memory[1, 0]) > 0.0


def test_baseline_recurrent_rollout_trace_matches_feedforward_trace_keys():
    cfg = DIV_CFG.replace(resource=ResourceWorldConfig())
    env = NaigosEnv(cfg, hmap=jnp.zeros((cfg.terrain.ny, cfg.terrain.nx)))
    key = jax.random.PRNGKey(0)
    ff = greedy_policy(
        init_learner(key, cfg, DIV_PPO["deepsets"], env.reset(key)[1]).actor.params, cfg
    )
    rec_ppo = DIV_PPO["gru"]
    rec = greedy_policy(
        init_learner(key, cfg, rec_ppo, env.reset(key)[1]).actor.params, cfg, family="recurrent"
    )
    _, t_ff = env.rollout(key, ff, n_steps=3)
    _, t_rec = env.rollout(key, rec, n_steps=3)
    assert set(t_ff) == set(t_rec)


# --- demo / replay / CBF -----------------------------------------------------


def test_demo_replay_and_cbf_run_every_cell_and_serialise_resource_fields(cell, tmp_path):
    from naigos.demo import replay

    blob = ck.load(cell["first"])
    cfg = _env_cfg(ResourceWorldConfig()).replace(**ck.obs_config_from_blob(blob))
    env = NaigosEnv(cfg)
    ident = ck.policy_identity(blob)
    results = replay.run(
        env, blob["actor"], 2, 0, use_cbf=True,
        policy_family=ident["family"], policy_version=ident["version"],
    )
    traj = results["trained"]["traj"]
    for name in TRACE_RESOURCE_FIELDS:
        assert name in traj
    assert np.isfinite(np.asarray(traj["pos"])).all()
    path = tmp_path / "demo.json"
    replay.to_json(results, cfg, path, env=env, world=0)
    assert path.exists()
