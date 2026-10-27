"""Trajectory recording from real Naigos collection paths.

Every policy family x resource world cell is rolled for real, written through
``naigos.trajectory.TrajectoryWriter`` and reopened through ``TrajectoryReader``.
Recording is opt-in: the default-off tests prove it changes nothing.
"""

from __future__ import annotations

import ast
import dataclasses
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from naigos.env.config import RESOURCE_WORLD_L1, EnvConfig, ResourceWorldConfig
from naigos.env.flight_env import NaigosEnv
from naigos.rl import checkpoint as ck
from naigos.rl import policies
from naigos.rl import trajectory_recording as trec
from naigos.rl.ppo import PPOConfig, greedy_policy, init_learner, init_rollout, make_train
from naigos.rl.red_team import RedCurriculum
from naigos.rl.reward import RewardCurriculum, RewardWeights
from naigos.rl.train import TrainConfig, evaluate, run
from naigos.trajectory import (
    ACTION_COMPONENTS,
    TERMINATION_CAUSES,
    ImmutableDatasetError,
    PartialDatasetError,
    TrajectoryReader,
    TrajectoryWriter,
    validate_dataset,
)

L1 = ResourceWorldConfig(model=RESOURCE_WORLD_L1)
RESOURCES = {"baseline-v0": ResourceWorldConfig(), "l1-endurance": L1}
POLICIES = {
    "deepsets": PPOConfig(n_envs=2, n_steps=8, n_minibatches=2),
    "gru": PPOConfig(n_envs=2, n_steps=8, n_minibatches=2, policy_family="recurrent", recurrent_seq_len=4),
}
CELLS = [(r, p) for r in RESOURCES for p in POLICIES]
IDS = [f"{r}+{p}" for r, p in CELLS]
N_WORLDS = 3
CAUSE = TERMINATION_CAUSES


def _cfg(resource):
    return EnvConfig(n_blue=2, n_threat=3, n_threat_active=3, resource=resource, max_steps=40)


def _rec(tmp, **kw):
    return trec.RecordingConfig(enabled=True, root=str(tmp), **kw)


def _rollout(env, pol, key, n=N_WORLDS, afilter=None, record=True):
    return jax.jit(jax.vmap(lambda k: env.rollout(k, pol, action_filter=afilter, record=record)))(
        jax.random.split(key, n)
    )


def _identity(cfg, ppo):
    return trec.policy_record("greedy-actor", cfg, family=ppo.policy_family, version=ppo.policy_version)


class _Cell:
    def __init__(self, resource_name, policy_name, tmp):
        self.resource_name, self.policy_name = resource_name, policy_name
        self.cfg, self.ppo = _cfg(RESOURCES[resource_name]), POLICIES[policy_name]
        self.recurrent = policy_name == "gru"
        self.env = NaigosEnv(self.cfg)
        _, obs = self.env.reset(jax.random.PRNGKey(0))
        self.learner = init_learner(jax.random.PRNGKey(1), self.cfg, self.ppo, obs)
        self.params = self.learner.actor.params
        self.pol = greedy_policy(self.params, self.cfg, family=self.ppo.policy_family)
        self.tmp = tmp
        from naigos.rl.cbf import CBFConfig, make_policy_filter

        self.cbf = make_policy_filter(CBFConfig(), self.cfg)
        self.key = jax.random.PRNGKey(2)
        self.identity = _identity(self.cfg, self.ppo)

    def record(self, label, *, afilter=None, chunk_steps=16, **kw):
        final, traj = _rollout(self.env, self.pol, self.key, afilter=afilter)
        ev = traj.pop("evidence")
        entry = trec.record_rollouts(
            _rec(self.tmp / label, chunk_steps=chunk_steps), self.env, ev, label=label, seed=2,
            policy=self.identity, **kw,
        )
        return final, traj, entry, TrajectoryReader(entry["path"])


@pytest.fixture(scope="module", params=CELLS, ids=IDS)
def cell(request, tmp_path_factory):
    resource_name, policy_name = request.param
    c = _Cell(resource_name, policy_name, tmp_path_factory.mktemp(f"{resource_name}-{policy_name}"))
    c.off = c.record("cbf-off")
    c.on = c.record("cbf-on", afilter=c.cbf)
    return c


def _per_world(arrays):
    eps = arrays["episode/id"]
    return [np.flatnonzero(eps == e) for e in np.unique(eps)]


# --- end to end: real rollout -> dataset -> reopen -> compare ------------------


def test_smoke_dataset_arrays_match_the_original_rollout(cell):
    final, traj, entry, reader = cell.off
    data = reader.read_all()
    assert entry["complete"] and entry["content_sha256"] == reader.report.content_sha256
    done = np.asarray(traj["done"])
    for w, rows in enumerate(_per_world(data)):
        keep = int(np.flatnonzero(done[w])[0]) + 1 if done[w].any() else done.shape[1]
        assert len(rows) == keep
        np.testing.assert_array_equal(data["action/executed"][rows], np.asarray(traj["action"])[w, :keep])
        np.testing.assert_array_equal(data["resource/fuel_after"][rows], np.asarray(traj["fuel"])[w, :keep])
        np.testing.assert_array_equal(data["status/alive"][rows], np.asarray(traj["alive"])[w, :keep])
        np.testing.assert_array_equal(data["status/reached"][rows], np.asarray(traj["reached"])[w, :keep])
        for name in ("progress", "exposure", "shotdown", "fuel_used", "out_of_fuel", "edge_proximity"):
            np.testing.assert_array_equal(
                data[f"reward/{name}"][rows], np.asarray(getattr(traj["terms"], name))[w, :keep]
            )


def test_trace_keeps_executed_alias_and_adds_requested(cell):
    _, traj, _, _ = cell.off
    np.testing.assert_array_equal(np.asarray(traj["action"]), np.asarray(traj["action_executed"]))
    assert traj["action_requested"].shape == traj["action"].shape


# --- requested vs executed, CBF ------------------------------------------------


def test_no_cbf_requested_equals_executed_and_feasible(cell):
    data = cell.off[3].read_all()
    np.testing.assert_array_equal(data["action/requested"], data["action/executed"])
    assert data["cbf/feasible"].all() and not data["cbf/intervention_l2"].any()


def test_cbf_on_records_requested_executed_and_exact_intervention(cell):
    final, traj, entry, reader = cell.on
    data = reader.read_all()
    done = np.asarray(traj["done"])
    for w, rows in enumerate(_per_world(data)):
        np.testing.assert_array_equal(data["action/executed"][rows], np.asarray(traj["action"])[w, : len(rows)])
        np.testing.assert_array_equal(
            data["cbf/feasible"][rows], np.asarray(traj["cbf_feasible"])[w, : len(rows)]
        )
    want = np.linalg.norm(
        data["action/executed"].astype(np.float64) - data["action/requested"].astype(np.float64), axis=-1
    )
    np.testing.assert_allclose(data["cbf/intervention_l2"], want, rtol=1e-5, atol=1e-6)
    assert np.abs(data["action/executed"]).max() <= 1.0 + 1e-6
    assert done.shape[0] == N_WORLDS


def test_a_modifying_filter_separates_requested_from_executed(cell):
    n = cell.cfg.n_blue
    halve = lambda state, obs, a: (0.5 * a, jnp.ones((n,), dtype=bool))  # noqa: E731
    final, traj, entry, reader = cell.record("halved", afilter=halve)
    data = reader.read_all()
    np.testing.assert_allclose(data["action/executed"], 0.5 * data["action/requested"], atol=1e-7)
    np.testing.assert_allclose(
        data["cbf/intervention_l2"], np.linalg.norm(0.5 * data["action/requested"], axis=-1), rtol=1e-5, atol=1e-6
    )
    # backward-compatible trace alias is the EXECUTED action
    np.testing.assert_array_equal(np.asarray(traj["action"]), np.asarray(traj["action_executed"]))
    assert not np.array_equal(np.asarray(traj["action_requested"]), np.asarray(traj["action"]))


# --- ordering, chunks ---------------------------------------------------------


def test_exact_episode_and_step_ordering(cell):
    data = cell.off[3].read_all()
    assert len(_per_world(data)) == N_WORLDS
    for w, rows in enumerate(_per_world(data)):
        assert (data["episode/id"][rows] == w).all()
        np.testing.assert_array_equal(data["step/index"][rows], np.arange(len(rows)))
        assert data["status/reset"][rows][0] and not data["status/reset"][rows][1:].any()


def test_bounded_multi_chunk_writing(cell):
    _, _, entry, reader = cell.off
    sizes = [len(c["episode/id"]) for c in reader.iter_chunks()]
    assert entry["chunks"] == len(sizes) > 1
    assert max(sizes) <= 16 and all(s == 16 for s in sizes[:-1])
    assert sum(sizes) == entry["records"]


def test_a_small_chunk_bound_is_respected(cell):
    _, _, entry, reader = cell.record("tiny-chunks", chunk_steps=5)
    assert all(len(c["episode/id"]) <= 5 for c in reader.iter_chunks())
    ref = cell.off[3].read_all()
    got = reader.read_all()
    np.testing.assert_array_equal(got["action/executed"], ref["action/executed"])  # chunking is storage only


# --- resource fields and outcomes --------------------------------------------


def test_resource_schema_fields_follow_the_world(cell):
    names = set(cell.off[3].metadata.schema.field_map)
    l1 = cell.resource_name == "l1-endurance"
    for name in ("resource/observation", "resource/reserve_fuel", "resource/diversion_pos", "privileged/burn_multiplier"):
        assert (name in names) == l1
    assert {"resource/outcome", "resource/diverted", "reward/diverted", "extension/route"} <= names


def test_l1_resource_values_and_privileged_latent_stay_out_of_actor_inputs(cell):
    if cell.resource_name != "l1-endurance":
        pytest.skip("L1 only")
    final, traj, _, reader = cell.off
    data = reader.read_all()
    base = 14 if cell.cfg.obs_edge_features else 10
    assert data["observation/ego"].shape[-1] == base
    assert data["resource/observation"].shape[-1] == cell.cfg.resource.observation_dim
    # the actor input is ego ++ resource block; check it against a fresh reset at step 0
    k0 = jax.random.split(jax.random.split(cell.key, N_WORLDS)[0])[0]
    _, obs0 = cell.env.reset(k0)
    rows = _per_world(data)[0]
    actor_in = np.concatenate([data["observation/ego"][rows[0]], data["resource/observation"][rows[0]]], axis=-1)
    np.testing.assert_allclose(actor_in, np.asarray(obs0.ego), atol=1e-6)
    # privileged: audited, namespaced, and not present inside any observation block
    spec = cell.off[3].metadata.schema.field_map["privileged/burn_multiplier"]
    assert "never an actor input" in spec.description
    burn = np.asarray(traj["burn_multiplier"])[:, 0]
    for w, rows in enumerate(_per_world(data)):
        np.testing.assert_array_equal(data["privileged/burn_multiplier"][rows][:, 0], np.full(len(rows), burn[w, 0]))
    np.testing.assert_array_equal(
        data["resource/outcome"], np.concatenate(
            [np.asarray(traj["outcome"])[w, : len(r)] for w, r in enumerate(_per_world(data))]
        ),
    )


def test_outcomes_and_termination_causes_agree(cell):
    data = cell.off[3].read_all()
    for rows in _per_world(data):
        final_outcome = data["resource/outcome"][rows][-1]
        ever = (data["status/termination_cause"][rows] != 0).any(axis=0)
        np.testing.assert_array_equal(ever, final_outcome != 0)
        # one nonzero cause per agent per episode (also enforced by the reader)
        assert ((data["status/termination_cause"][rows] != 0).sum(axis=0) <= 1).all()


def test_constraint_cost_matches_the_environment_cost_channel(cell):
    from naigos.rl.reward import constraint_cost

    final, traj, _, reader = cell.off
    data = reader.read_all()
    want = np.concatenate([
        np.asarray(constraint_cost(jax.tree.map(lambda x: x[w], traj["terms"])))[: len(r)]
        for w, r in enumerate(_per_world(data))
    ])
    np.testing.assert_array_equal(data["constraint_cost/aircraft_loss"], want)


def test_a_dive_terminates_once_and_frozen_steps_are_not_recorded(cell):
    dive = lambda obs, k: jnp.tile(jnp.array([0.0, -1.0, 0.6]), (cell.cfg.n_blue, 1))  # noqa: E731
    final, traj = _rollout(cell.env, dive, cell.key)
    ev = traj.pop("evidence")
    entry = trec.record_rollouts(
        _rec(cell.tmp / "dive"), cell.env, ev, label="dive", seed=2, policy=cell.identity
    )
    data = TrajectoryReader(entry["path"]).read_all()
    assert (data["status/termination_cause"] == CAUSE["terrain"]).any()
    for rows in _per_world(data):
        c = data["status/termination_cause"][rows]
        assert ((c == CAUSE["terrain"]).sum(axis=0) == 1).all()
        assert not (c[-1] == 0).all()  # the world ended on a terminal step, nothing frozen after it


# --- recurrent memory ---------------------------------------------------------


def test_recurrent_memory_is_carried_and_reset_only_for_gru(cell):
    schema = cell.off[3].metadata.schema.field_map
    if not cell.recurrent:
        assert not any(n.startswith("extension/memory") for n in schema)
        return
    data = cell.off[3].read_all()
    for rows in _per_world(data):
        before, after, reset = (data[f"extension/memory_{k}"][rows] for k in ("before", "after", "reset"))
        assert not before[0].any()  # fresh sortie
        np.testing.assert_array_equal(before[1:], after[:-1])  # carried unchanged between steps
        assert after[:-1][~reset[:-1]].any()  # a live GRU actually holds state
        assert reset[-1].any()  # the episode-ending step zeroed it
        np.testing.assert_array_equal(reset, ~after.any(axis=-1))


def test_requested_actions_reconstruct_from_checkpointed_params_and_recorded_inputs(cell, tmp_path):
    """Actor params + recorded observations (+ memory) reproduce `action/requested` exactly."""
    path = tmp_path / "ckpt_000001"
    ck.save(
        path, learner=cell.learner, key=jax.random.PRNGKey(0), iteration=1, red_level=0.0, shootdown_rate=1.0,
        survival_rate=0.0, survival_w=1.0, efficiency_w=0.0, history=[], env_cfg=cell.cfg, ppo_cfg=cell.ppo,
        red_curriculum=RedCurriculum(), reward_curriculum=RewardCurriculum(), reward_weights_base=RewardWeights(),
        wall_s=0.0,
    )
    blob = ck.load(path)
    identity = ck.policy_identity(blob)
    assert identity["spec_sha256"] == cell.identity.configuration["policy_spec_sha256"]
    assert trec.params_sha256(blob["actor"]) == trec.params_sha256(cell.params)
    actor = policies.create_actor(cell.cfg, family=identity["family"], version=identity["version"])
    apply = jax.jit(actor.apply)
    data = cell.on[3].read_all()
    rows = _per_world(data)[0]
    for t in rows[:12]:
        ego = np.concatenate([data["observation/ego"][t], data["resource/observation"][t]], -1) \
            if "resource/observation" in data else data["observation/ego"][t]
        args = (blob["actor"], ego, data["observation/threats"][t], data["observation/threat_mask"][t],
                data["observation/friends"][t], data["observation/friend_mask"][t])
        if cell.recurrent:
            mean, _, mem = apply(*args, data["extension/memory_before"][t])
        else:
            mean, _ = apply(*args)
        np.testing.assert_allclose(np.tanh(np.asarray(mean)), data["action/requested"][t], atol=1e-5)


# --- provenance, immutability, interruption ----------------------------------


def test_provenance_is_complete_and_immutable(cell):
    _, _, entry, reader = cell.off
    prov = reader.metadata.provenance
    assert prov.seed == 2 and prov.route and prov.scenario and prov.theatre == "synthetic-terrain"
    assert len(prov.code_commit) >= 7
    cfg = prov.environment_configuration
    assert cfg["env"]["resource"]["model"] == cell.cfg.resource.model
    assert cfg["observation_schema_id"] == ("L1EnduranceObservation-v1" if cell.cfg.resource.enabled else "Observation-v0")
    assert (cfg["resource_bank"] is not None) == cell.cfg.resource.enabled
    pol = prov.policy.configuration
    assert pol["policy_family"] == cell.ppo.policy_family and pol["environment_model_id"] == cell.cfg.resource.model
    assert len(pol["policy_spec_sha256"]) == 64
    root = Path(entry["path"])
    commit = json.loads((root / "metadata.commit.json").read_text())
    assert commit["sha256"] == entry["metadata_sha256"]
    before = (root / "metadata.json").read_bytes()
    with pytest.raises(ImmutableDatasetError):
        cell.record("cbf-off")  # same path again: never overwritten
    assert (root / "metadata.json").read_bytes() == before
    with pytest.raises(ImmutableDatasetError):
        TrajectoryWriter.resume(root)
    assert validate_dataset(root).content_sha256 == entry["content_sha256"]


def test_writer_interruption_leaves_a_partial_dataset_without_a_hash(cell, monkeypatch):
    original = TrajectoryWriter.append
    calls = []

    def flaky(self, arrays):
        calls.append(1)
        if len(calls) == 3:
            raise OSError("disk went away")
        return original(self, arrays)

    monkeypatch.setattr(TrajectoryWriter, "append", flaky)
    index = cell.tmp / "interrupted-index.json"
    with pytest.raises(OSError, match="disk went away"):
        cell.record("interrupted", chunk_steps=8, index_path=index)
    root = cell.tmp / "interrupted" / "interrupted"
    assert not (root / "manifest.json").exists()
    with pytest.raises(PartialDatasetError):
        TrajectoryReader(root)
    report = validate_dataset(root, allow_partial=True)
    assert not report.complete and report.content_sha256 is None and report.chunks == 2
    row = json.loads(index.read_text())["datasets"][0]
    assert row["complete"] is False and row["content_sha256"] is None and row["chunks"] == 2


def test_index_is_atomic_and_never_downgrades_a_finalized_dataset(cell, tmp_path):
    index = tmp_path / "idx.json"
    _, _, entry, _ = cell.record("indexed", index_path=index)
    doc = json.loads(index.read_text())
    assert doc["protocol"] == trec.INDEX_PROTOCOL and doc["datasets"][0]["content_sha256"] == entry["content_sha256"]
    assert not list(tmp_path.glob(".idx.json.*"))
    with pytest.raises(trec.RecordingError):
        trec.update_index(index, [{**entry, "complete": False, "content_sha256": None}])


# --- no new action dimensions / offensive controls -----------------------------


def test_schema_adds_no_action_dimension_or_offensive_control(cell):
    schema = cell.off[3].metadata.schema
    assert schema.dimension_map["action"] == 3
    actions = [f for f in schema.fields if f.name.startswith("action/")]
    assert [f.name for f in actions] == ["action/requested", "action/executed"]
    assert all(f.components == ACTION_COMPONENTS for f in actions)
    assert not [f.name for f in schema.fields if any(w in f.name for w in ("fire", "target", "engage", "weapon"))]


# --- default-off equivalence ----------------------------------------------------


def test_default_trace_has_no_evidence_and_matches_the_recorded_run(cell):
    final0, traj0 = _rollout(cell.env, cell.pol, cell.key, record=False)
    assert "evidence" not in traj0
    final1, traj1, _, _ = cell.off
    for name in ("pos", "fuel", "action", "alive", "reached", "done"):
        np.testing.assert_array_equal(np.asarray(traj0[name]), np.asarray(traj1[name]))
    jax.tree.map(np.testing.assert_array_equal, final0, final1)


def test_evaluate_metrics_are_identical_with_and_without_recording(cell, tmp_path):
    kw = dict(policy_family=cell.ppo.policy_family, policy_version=cell.ppo.policy_version)
    plain = evaluate(cell.env, cell.params, 3, cell.key, **kw)
    off = evaluate(cell.env, cell.params, 3, cell.key, recording=trec.RecordingConfig(), **kw)
    on = evaluate(cell.env, cell.params, 3, cell.key, recording=_rec(tmp_path / "ev"), record_index=tmp_path / "i.json", **kw)
    assert plain == off == on
    assert (tmp_path / "ev").exists()  # recording wrote its dataset
    assert len(list((tmp_path / "ev").iterdir())) == 1


def test_training_records_every_world_without_changing_the_run(cell, tmp_path):
    def go(sub, recording):
        return run(
            cell.cfg, cell.ppo,
            TrainConfig(iterations=2, out_dir=str(tmp_path / sub), seed=3, eval_every=2, eval_worlds=2,
                        checkpoint_every=2, curriculum_every=2, recording=recording),
        )

    base, hist0 = go("plain", None)
    off, hist1 = go("disabled", trec.RecordingConfig())
    rec, hist2 = go("recorded", trec.RecordingConfig(enabled=True, chunk_steps=6))
    for other in (off, rec):
        jax.tree.map(np.testing.assert_array_equal, base.actor.params, other.actor.params)
        jax.tree.map(np.testing.assert_array_equal, base.critic.params, other.critic.params)
    strip = lambda h: [{k: v for k, v in r.items() if k not in ("wall_s", "iter_s", "env_steps_per_s")} for r in h]  # noqa: E731
    assert strip(hist0) == strip(hist1) == strip(hist2)
    for sub in ("plain", "disabled"):
        assert not (tmp_path / sub / "trajectories").exists() and not (tmp_path / sub / trec.INDEX_FILENAME).exists()

    index = json.loads((tmp_path / "recorded" / trec.INDEX_FILENAME).read_text())
    assert [d["world"] for d in index["datasets"]] == [0, 1]
    for d in index["datasets"]:
        assert d["complete"] and d["records"] == 2 * cell.ppo.n_steps and len(d["content_sha256"]) == 64
        reader = TrajectoryReader(d["path"])
        assert reader.report.content_sha256 == d["content_sha256"]
        data = reader.read_all()
        assert (data["extension/collection_iteration"] == np.repeat([1, 2], cell.ppo.n_steps)).all()
        # PPO has no CBF: the sampled action was both requested and executed
        np.testing.assert_array_equal(data["action/requested"], data["action/executed"])
        assert all(len(c["episode/id"]) <= 6 for c in reader.iter_chunks())
        prov = reader.metadata.provenance
        assert prov.policy.kind == "online-learner" and prov.policy.configuration["policy_family"] == cell.ppo.policy_family
        if cell.recurrent:
            assert data["extension/memory_reset"].any() or not (data["status/termination_cause"] != 0).any()
    # shards of different worlds are different datasets with different scenarios
    scen = {TrajectoryReader(d["path"]).metadata.provenance.scenario for d in index["datasets"]}
    assert len(scen) == 2


def test_resume_starts_a_new_session_shard_and_a_crash_leaves_partials(cell, tmp_path, monkeypatch):
    out = tmp_path / "run"
    rcfg = trec.RecordingConfig(enabled=True, chunk_steps=8)

    def cfg(iterations):
        return TrainConfig(iterations=iterations, out_dir=str(out), seed=3, eval_every=2, eval_worlds=2,
                           checkpoint_every=2, curriculum_every=2, recording=rcfg)

    run(cell.cfg, cell.ppo, cfg(2))
    first = json.loads((out / trec.INDEX_FILENAME).read_text())["datasets"]
    assert {d["session"] for d in first} == {1}
    run(cell.cfg, cell.ppo, cfg(4), resume_from=ck.latest_resumable(out))
    rows = json.loads((out / trec.INDEX_FILENAME).read_text())["datasets"]
    assert {d["session"] for d in rows} == {1, 2} and all(d["complete"] for d in rows)
    for d in first:  # the first session's datasets were not touched
        assert next(r for r in rows if r["path"] == d["path"]) == d
    resumed = [TrajectoryReader(r["path"]).metadata.provenance for r in rows if r["session"] == 2]
    assert all(p.environment_configuration["collection"]["resumed_session"] for p in resumed)

    crash = tmp_path / "crash"
    original = trec.TrainingRecorder.ingest
    calls = []

    def boom(self, evidence, iteration):
        calls.append(iteration)
        if len(calls) == 2:
            raise RuntimeError("collector died")
        return original(self, evidence, iteration)

    monkeypatch.setattr(trec.TrainingRecorder, "ingest", boom)
    with pytest.raises(RuntimeError, match="collector died"):
        run(cell.cfg, cell.ppo, dataclasses.replace(cfg(4), out_dir=str(crash)))
    rows = json.loads((crash / trec.INDEX_FILENAME).read_text())["datasets"]
    assert rows and all(not r["complete"] and r["content_sha256"] is None for r in rows)
    for r in rows:
        assert not (Path(r["path"]) / "manifest.json").exists()


# --- terminal semantics on synthetic evidence ------------------------------------


def _terms(t=1, b=1, **on):
    names = ("arrived", "diverted", "shotdown", "terrain_violation", "bounds_violation", "out_of_fuel")
    return {n: np.full((t, b), 1.0 if n in on else 0.0, dtype=np.float32) for n in names}


def _ev(done=False, alive=True, reached=False, diverted=False, t=1, b=1):
    full = lambda v: np.full((t, b), v)  # noqa: E731
    return {"alive": full(alive), "reached": full(reached), "diverted": full(diverted),
            "done": np.full((t,), done)}


@pytest.mark.parametrize("raw,alive,reached,expected", [
    (dict(shotdown=1), False, False, "shot_down"),
    (dict(terrain_violation=1), False, False, "terrain"),
    (dict(bounds_violation=1), False, False, "out_of_bounds"),
    (dict(arrived=1), True, True, "reached"),
    (dict(shotdown=1, terrain_violation=1, bounds_violation=1), False, False, "terrain"),
    (dict(shotdown=1, bounds_violation=1), False, False, "out_of_bounds"),
    (dict(shotdown=1, arrived=1), False, True, "shot_down"),
    (dict(terrain_violation=1, arrived=1), False, True, "terrain"),
])
def test_simultaneous_raw_events_have_one_primary_cause(raw, alive, reached, expected):
    got = trec.termination_causes(_terms(**raw), _ev(alive=alive, reached=reached), resource_enabled=True)
    assert got.item() == CAUSE[expected]


def test_fuel_exhaustion_is_terminal_only_under_l1_and_loses_to_physical_losses():
    t, e = _terms(out_of_fuel=1), _ev(alive=False)
    assert trec.termination_causes(t, e, resource_enabled=True).item() == CAUSE["out_of_fuel"]
    assert trec.termination_causes(t, _ev(alive=True), resource_enabled=False).item() == 0  # baseline-v0: a cost
    both = _terms(out_of_fuel=1, shotdown=1)
    assert trec.termination_causes(both, e, resource_enabled=True).item() == CAUSE["shot_down"]


def test_diversion_timeout_and_frozen_steps():
    assert trec.termination_causes(_terms(diverted=1), _ev(diverted=True), True).item() == trec.CAUSE_DIVERTED
    active = _ev(done=True)
    assert trec.termination_causes(_terms(), active, True).item() == CAUSE["timeout"]
    assert trec.termination_causes(_terms(), _ev(done=False), True).item() == 0  # not the last step
    for frozen in (_ev(done=True, alive=False), _ev(done=True, reached=True), _ev(done=True, diverted=True)):
        assert trec.termination_causes(_terms(), frozen, True).item() == 0  # no repeated outcome
    # a terminal event on the timeout step wins over the timeout
    assert trec.termination_causes(_terms(shotdown=1), _ev(done=True, alive=False), True).item() == CAUSE["shot_down"]


def test_episode_layout_tracks_auto_resets_across_segments():
    done = np.array([0, 0, 1, 0, 0], dtype=bool)
    ep, step, reset, carry = trec.episode_layout(done, True, -1, 0)
    np.testing.assert_array_equal(ep, [0, 0, 0, 1, 1])
    np.testing.assert_array_equal(step, [0, 1, 2, 0, 1])
    np.testing.assert_array_equal(reset, [1, 0, 0, 1, 0])
    ep2, step2, reset2, carry2 = trec.episode_layout(np.array([0, 1, 0], dtype=bool), *(carry[1], carry[0], carry[2]))
    np.testing.assert_array_equal(ep2, [1, 1, 2])
    np.testing.assert_array_equal(step2, [2, 3, 0])
    np.testing.assert_array_equal(reset2, [0, 0, 1])
    assert carry2 == (2, False, 1)


# --- boundary hygiene -------------------------------------------------------------


def test_trajectory_package_stays_framework_neutral():
    allowed = {"numpy", "__future__", "dataclasses", "hashlib", "json", "math", "re", "os", "uuid", "datetime",
               "typing", "pathlib", "contextlib", "tempfile", "errno", "stat", "collections", "sys"}
    pkg = Path(__file__).resolve().parents[1] / "naigos" / "trajectory"
    for path in pkg.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            mods = [a.name for a in node.names] if isinstance(node, ast.Import) else \
                [node.module or ""] if isinstance(node, ast.ImportFrom) and not node.level else []
            for m in mods:
                assert m.split(".")[0] in allowed, f"{path.name} imports {m}"


def test_recording_config_is_off_by_default_and_validates():
    assert trec.RecordingConfig().enabled is False and not trec.is_recording(None)
    with pytest.raises(trec.RecordingError):
        trec.RecordingConfig(chunk_steps=0)
    with pytest.raises(trec.RecordingError):
        trec.RecordingConfig(enabled=True).resolved(None)
    assert TrainConfig().recording is None


def test_make_train_without_record_returns_three_values():
    cfg = _cfg(RESOURCES["baseline-v0"])
    env, ppo = NaigosEnv(cfg), POLICIES["deepsets"]
    _, obs = env.reset(jax.random.PRNGKey(0))
    learner = init_learner(jax.random.PRNGKey(1), cfg, ppo, obs)
    roll = init_rollout(env, ppo, jax.random.PRNGKey(2))
    assert len(make_train(env, ppo, RewardWeights())(learner, roll, jax.random.PRNGKey(3))) == 3
    out = make_train(env, ppo, RewardWeights(), record=True)(learner, roll, jax.random.PRNGKey(3))
    assert len(out) == 4 and out[3]["action_requested"].shape[:2] == (ppo.n_steps, ppo.n_envs)


# --- the other real-rollout entry points ------------------------------------------


def _small(resource_name, policy_name):
    c = _Cell(resource_name, policy_name, None)
    return c


def test_route_bank_writes_one_dataset_per_scenario_and_keeps_metrics(tmp_path):
    from naigos.env import config as env_config
    from naigos.rl.train import evaluate_route_bank

    c = _small("baseline-v0", "deepsets")
    rows = [r for r in env_config.route_bank() if r.split == "heldout"]
    plain = evaluate_route_bank(c.env, c.pol, splits=("heldout",))
    rec = evaluate_route_bank(
        c.env, c.pol, splits=("heldout",), recording=_rec(tmp_path, chunk_steps=64), record_policy=c.identity,
        record_index=tmp_path / "idx.json",
    )
    assert plain == rec
    index = json.loads((tmp_path / "idx.json").read_text())["datasets"]
    assert len(index) == len(rows) and all(d["complete"] for d in index)
    seeds = {TrajectoryReader(d["path"]).metadata.provenance.seed for d in index}
    assert seeds == {r.seed for r in rows}  # one seed / route per dataset, never mixed
    with pytest.raises(trec.RecordingError):
        evaluate_route_bank(c.env, c.pol, splits=("heldout",), recording=_rec(tmp_path / "x"))


def test_resource_bank_datasets_carry_each_scenarios_resource_config(tmp_path):
    from naigos.env.resource_bank import resource_bank
    from naigos.rl.train import evaluate_resource_bank

    c = _small("l1-endurance", "gru")
    rows = list(resource_bank("heldout"))
    plain = evaluate_resource_bank(c.env, c.pol, splits=("heldout",))
    rec = evaluate_resource_bank(
        c.env, c.pol, splits=("heldout",), recording=_rec(tmp_path), record_policy=c.identity,
        record_index=tmp_path / "idx.json",
    )
    assert plain["verified"] and rec["verified"] and plain["heldout"] == rec["heldout"]
    index = json.loads((tmp_path / "idx.json").read_text())["datasets"]
    assert len(index) == len(rows) and all(d["complete"] for d in index)
    by_seed = {r.seed: r for r in rows}
    for d in index:
        reader = TrajectoryReader(d["path"])
        prov = reader.metadata.provenance
        row = by_seed[prov.seed]
        assert prov.environment_configuration["env"]["resource"]["burn_multiplier"] == pytest.approx(
            [row.burn_multiplier] * 2
        )
        data = reader.read_all()
        np.testing.assert_allclose(data["privileged/burn_multiplier"], row.burn_multiplier, rtol=1e-6)
        assert prov.environment_configuration["resource_bank"]["bank_sha256"]


def test_replay_run_records_only_the_trained_policy(tmp_path):
    from naigos.demo import replay

    c = _small("baseline-v0", "deepsets")
    plain = replay.run(c.env, c.params, 2, 7)
    rec = replay.run(c.env, c.params, 2, 7, recording=_rec(tmp_path), record_index=tmp_path / "idx.json")
    assert {k: v["summary"] for k, v in plain.items()} == {k: v["summary"] for k, v in rec.items()}
    index = json.loads((tmp_path / "idx.json").read_text())["datasets"]
    assert [d["label"] for d in index] == ["replay-trained"] and index[0]["complete"]
    assert TrajectoryReader(index[0]["path"]).metadata.provenance.policy.artifact_sha256 == trec.params_sha256(c.params)


def test_pipeline_evaluation_records_the_candidate(tmp_path):
    from naigos.pipeline import evaluation as pe

    c = _small("baseline-v0", "deepsets")
    plan = {"worlds_per_seed": 2, "seeds": [11], "use_cbf": False, "verifier_episodes": 0, "training_seeds": []}
    loaded = pe.LoadedActor(params=c.params, family="deepsets", version="1")
    plain = pe.run_evaluation(c.env, plan, loaded)
    rec = pe.run_evaluation(c.env, plan, loaded, recording=_rec(tmp_path), record_index=tmp_path / "idx.json")
    assert plain["candidate"] == rec["candidate"] and plain["baselines"] == rec["baselines"]
    index = json.loads((tmp_path / "idx.json").read_text())["datasets"]
    assert [d["label"] for d in index] == ["pipeline-candidate"]


def test_cli_flags_are_opt_in():
    import argparse

    p = argparse.ArgumentParser()
    trec.add_cli_args(p)
    assert trec.config_from_args(p.parse_args([])) is None
    cfg = trec.config_from_args(p.parse_args(["--record-trajectories", "--record-chunk-steps", "32"]))
    assert cfg.enabled and cfg.root is None and cfg.chunk_steps == 32
    assert trec.config_from_args(p.parse_args(["--record-trajectories", "/x"])).root == "/x"
