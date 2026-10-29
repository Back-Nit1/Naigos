"""Fixed-intent, causal-data, and pinned-asset contracts for the Qwen planner."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from naigos.env.config import EnvConfig
from naigos.env.flight_env import NaigosEnv
from naigos.planner import INTENTS, INTENT_LABELS
from naigos.planner.assets import ModelAssetError, load_spec, verify_model_dir
from naigos.planner.dataset import (
    DATASET_SCHEMA,
    PlannerDatasetError,
    derive_behavior_labels,
    prepare_dataset,
)
from naigos.planner.intents import intent_for_label, route_waypoint
from naigos.planner.prompt import HISTORY_STEPS, PROMPT_SCHEMA, build_prompt, prompt_sha256
from naigos.rl import trajectory_recording as trec


def _frame(value: float = 0.0):
    ego = np.linspace(-0.5, 0.5, 10, dtype=np.float32) + value
    threats = np.zeros((6, 14), dtype=np.float32)
    threats[0, :9] = [0.2, -0.1, 0.0, 0.0, 0.0, 0.3, 0.4, 0.6, 0.7]
    threats[0, 10] = 1.0
    friends = np.zeros((3, 7), dtype=np.float32)
    friends[0, 5] = 0.2
    return {
        "ego": ego,
        "threats": threats,
        "threat_mask": np.array([True, False, False, False, False, False]),
        "friends": friends,
        "friend_mask": np.array([True, False, False]),
    }


def test_intent_vocabulary_is_exactly_nine_evasive_route_choices():
    assert INTENT_LABELS == tuple("ABCDEFGHI")
    assert len(INTENTS) == 9
    assert {(i.cross_track_m, i.altitude_reference, i.altitude_m) for i in INTENTS} == {
        (offset, ref, altitude)
        for offset in (15_000.0, 0.0, -15_000.0)
        for ref, altitude in (("agl", 200.0), ("agl", 1_000.0), ("amsl", 4_000.0))
    }
    with pytest.raises(ValueError):
        intent_for_label("turn left")


def test_route_adapter_clips_xy_and_resolves_agl_without_changing_primary():
    pos = np.array([50_000.0, 50_000.0, 2_000.0])
    primary = np.array([90_000.0, 50_000.0, 1_000.0])
    before = primary.copy()
    waypoint = route_waypoint(pos, primary, [0, 100_000, 0, 100_000], "A", target_ground_m=600.0)
    np.testing.assert_array_equal(primary, before)
    np.testing.assert_allclose(waypoint, [80_000.0, 65_000.0, 800.0])


def test_prompt_is_deterministic_causal_and_fixed_vocabulary():
    history = [_frame(i / 100.0) for i in range(HISTORY_STEPS)]
    prompt = build_prompt(history)
    assert prompt == build_prompt(history)
    assert PROMPT_SCHEMA in prompt and prompt.endswith("intent=")
    assert "Output exactly one label A-I" in prompt
    assert len(prompt_sha256(prompt)) == 64
    changed = list(history)
    changed[-1] = _frame(0.9)
    assert build_prompt(changed) != prompt
    with pytest.raises(ValueError):
        build_prompt(history[:-1])


def test_pinned_model_spec_and_hash_verifier(tmp_path):
    spec = load_spec()
    assert spec["repo_id"] == "Qwen/Qwen3-0.6B-Base"
    assert len(spec["revision"]) == 40 and len(spec["files"]) == 9
    for row in spec["files"]:
        (tmp_path / row["name"]).write_bytes(b"wrong")
    with pytest.raises(ModelAssetError, match="invalid local Qwen assets"):
        verify_model_dir(tmp_path)


def test_planner_import_does_not_import_torch_or_transformers():
    code = (
        "import sys; import naigos.planner; "
        "assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def _record_short_rollout(tmp_path: Path):
    cfg = EnvConfig(n_blue=2, n_threat=3, n_threat_active=3, max_steps=24)
    env = NaigosEnv(cfg)

    def policy(obs, key):
        del key
        return jnp.zeros((cfg.n_blue, cfg.action_dim), dtype=obs.ego.dtype)

    key = jax.random.PRNGKey(41)
    final, traj = jax.jit(jax.vmap(lambda k: env.rollout(k, policy, record=True)))(
        jax.random.split(key, 1)
    )
    del final
    entry = trec.record_rollouts(
        trec.RecordingConfig(enabled=True, root=str(tmp_path / "trajectories"), chunk_steps=7),
        env, traj["evidence"], label="planner-source", seed=41,
        policy=trec.policy_record("scripted", cfg, name="planner-test"),
    )
    return entry


def test_prepare_streams_only_labeled_causal_windows(tmp_path):
    entry = _record_short_rollout(tmp_path)
    labels = tmp_path / "labels.jsonl"
    rows = [
        {
            "content_sha256": entry["content_sha256"], "episode_id": 0,
            "step_index": 15, "agent": 0, "label": "A", "split": "train",
        },
        {
            "content_sha256": entry["content_sha256"], "episode_id": 0,
            "step_index": 18, "agent": 1, "label": "I", "split": "train",
        },
    ]
    labels.write_text("".join(json.dumps(row) + "\n" for row in rows))
    output = tmp_path / "planner.jsonl"
    manifest = prepare_dataset([entry["path"]], labels, output)
    assert manifest["records"] == 2 and manifest["split_counts"]["train"] == 2
    examples = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["label"] for row in examples] == ["A", "I"]
    assert all(row["schema"] == DATASET_SCHEMA for row in examples)
    assert all(row["prompt"].endswith("intent=") for row in examples)
    assert json.loads((tmp_path / "planner.jsonl.manifest.json").read_text())["sha256"] == manifest["sha256"]
    with pytest.raises(PlannerDatasetError, match="refusing to overwrite"):
        prepare_dataset([entry["path"]], labels, output)


def test_behavior_labeler_is_explicitly_weak_and_trace_anchored(tmp_path):
    entry = _record_short_rollout(tmp_path)
    labels = tmp_path / "weak-labels.jsonl"
    report = derive_behavior_labels(
        [entry["path"]], labels, split="train", stride=4, horizon=4,
        successful_only=False,
    )
    assert report["records"] > 0 and report["counterfactual_optimal"] is False
    rows = [json.loads(line) for line in labels.read_text().splitlines()]
    assert all(row["content_sha256"] == entry["content_sha256"] for row in rows)
    assert all(row["label_source"]["method"] == "behavior-imitation-v1" for row in rows)
    assert all(row["label"] in INTENT_LABELS for row in rows)
