"""Stream Naigos trajectory shards into labeled Qwen planner examples."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from ..trajectory import TrajectoryReader
from ..trajectory.schema import canonical_json_bytes
from .intents import INTENT_LABELS, intent_for_label
from .prompt import HISTORY_STEPS, PROMPT_SCHEMA, build_prompt, prompt_sha256

DATASET_SCHEMA = "naigos.qwen-planner-dataset/1"
SPLITS = ("train", "validation", "test")


class PlannerDatasetError(ValueError):
    pass


@dataclass(frozen=True)
class LabelAnchor:
    content_sha256: str
    episode_id: int
    step_index: int
    agent: int
    label: str
    split: str

    @property
    def key(self) -> tuple[str, int, int, int]:
        return self.content_sha256, self.episode_id, self.step_index, self.agent


def read_label_index(path: str | os.PathLike) -> dict[tuple[str, int, int, int], LabelAnchor]:
    anchors: dict[tuple[str, int, int, int], LabelAnchor] = {}
    with Path(path).open() as fh:
        for lineno, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                anchor = LabelAnchor(
                    content_sha256=str(row["content_sha256"]),
                    episode_id=int(row["episode_id"]), step_index=int(row["step_index"]),
                    agent=int(row["agent"]), label=str(row["label"]), split=str(row["split"]),
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise PlannerDatasetError(f"{path}:{lineno}: malformed label anchor: {exc}") from exc
            intent_for_label(anchor.label)
            if anchor.split not in SPLITS:
                raise PlannerDatasetError(f"{path}:{lineno}: split must be one of {SPLITS}")
            if anchor.key in anchors:
                raise PlannerDatasetError(f"{path}:{lineno}: duplicate label anchor {anchor.key}")
            anchors[anchor.key] = anchor
    if not anchors:
        raise PlannerDatasetError("label index is empty")
    return anchors


def _actor_frame(chunk: Mapping[str, np.ndarray], row: int, agent: int) -> dict[str, np.ndarray]:
    ego_parts = [np.asarray(chunk["observation/ego"][row, agent])]
    for name in (
        "resource/observation", "extension/weather_observation", "extension/mission_observation",
    ):
        if name in chunk:
            ego_parts.append(np.asarray(chunk[name][row, agent]))
    return {
        "ego": np.concatenate(ego_parts).astype(np.float32, copy=False),
        "threats": np.asarray(chunk["observation/threats"][row, agent]),
        "threat_mask": np.asarray(chunk["observation/threat_mask"][row, agent]),
        "friends": np.asarray(chunk["observation/friends"][row, agent]),
        "friend_mask": np.asarray(chunk["observation/friend_mask"][row, agent]),
    }


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def prepare_dataset(
    trajectory_roots: Iterable[str | os.PathLike],
    labels_path: str | os.PathLike,
    output_path: str | os.PathLike,
) -> dict:
    """Create deterministic JSONL from externally branch-labeled trace anchors.

    Labels are deliberately external. A trajectory alone records what one
    policy did; it cannot truthfully say which of nine counterfactual intents
    would have been best. The branch collector must produce the label index
    under common random numbers.
    """

    output = Path(output_path)
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    if output.exists() or manifest_path.exists():
        raise PlannerDatasetError(f"refusing to overwrite planner dataset {output}")
    anchors = read_label_index(labels_path)
    used: set[tuple[str, int, int, int]] = set()
    examples: list[dict] = []
    sources: list[dict] = []
    identity_split: dict[tuple[str, int, str], str] = {}

    for root in sorted(map(Path, trajectory_roots), key=lambda p: str(p)):
        reader = TrajectoryReader(root)
        content = reader.report.content_sha256
        relevant = [a for a in anchors.values() if a.content_sha256 == content]
        if not relevant:
            continue
        splits = {a.split for a in relevant}
        if len(splits) != 1:
            raise PlannerDatasetError(f"{root}: one immutable trajectory dataset cannot span splits {splits}")
        split = next(iter(splits))
        prov = reader.metadata.provenance
        identity = (prov.theatre, int(prov.seed), prov.scenario)
        previous = identity_split.setdefault(identity, split)
        if previous != split:
            raise PlannerDatasetError(f"scenario identity {identity} appears in both {previous} and {split}")
        n_agents = reader.metadata.schema.dimension_map["agent"]
        histories = [deque(maxlen=HISTORY_STEPS) for _ in range(n_agents)]
        current_episode: int | None = None
        source_count = 0
        for chunk in reader.iter_chunks():
            for row in range(len(chunk["episode/id"])):
                episode = int(chunk["episode/id"][row])
                step = int(chunk["step/index"][row])
                if current_episode != episode or bool(chunk["status/reset"][row]):
                    for history in histories:
                        history.clear()
                    current_episode = episode
                for agent in range(n_agents):
                    histories[agent].append(_actor_frame(chunk, row, agent))
                    key = (content, episode, step, agent)
                    anchor = anchors.get(key)
                    if anchor is None:
                        continue
                    if len(histories[agent]) != HISTORY_STEPS:
                        raise PlannerDatasetError(
                            f"label {key} has only {len(histories[agent])} causal observations; need {HISTORY_STEPS}"
                        )
                    prompt = build_prompt(tuple(histories[agent]))
                    examples.append({
                        "schema": DATASET_SCHEMA,
                        "split": anchor.split,
                        "prompt": prompt,
                        "label": anchor.label,
                        "prompt_sha256": prompt_sha256(prompt),
                        "source": {
                            "content_sha256": content, "episode_id": episode,
                            "step_index": step, "agent": agent,
                            "theatre": prov.theatre, "scenario": prov.scenario, "seed": int(prov.seed),
                        },
                    })
                    used.add(key)
                    source_count += 1
        sources.append({"path": str(root), "content_sha256": content, "examples": source_count})

    missing = sorted(set(anchors) - used)
    if missing:
        raise PlannerDatasetError(f"{len(missing)} label anchors were not found in the supplied trajectories; first={missing[0]}")
    if not examples:
        raise PlannerDatasetError("no planner examples were produced")
    body = b"".join(canonical_json_bytes(row) for row in examples)
    digest = hashlib.sha256(body).hexdigest()
    counts = {split: sum(row["split"] == split for row in examples) for split in SPLITS}
    manifest = {
        "schema": DATASET_SCHEMA,
        "prompt_schema": PROMPT_SCHEMA,
        "sha256": digest,
        "records": len(examples),
        "split_counts": counts,
        "sources": sources,
    }
    _write_atomic(output, body)
    _write_atomic(manifest_path, canonical_json_bytes(manifest))
    return manifest


def derive_behavior_labels(
    trajectory_roots: Iterable[str | os.PathLike],
    output_path: str | os.PathLike,
    *,
    split: str,
    stride: int = 16,
    horizon: int = 16,
    successful_only: bool = True,
) -> dict:
    """Derive weak high-level imitation labels from a policy's future behavior.

    This makes the pipeline runnable directly from traces, but these are *not*
    counterfactual optimal labels. Lateral intent comes from mean executed bank
    over the future horizon; altitude intent is the nearest of the three fixed
    altitude bands at the horizon endpoint. Use branch labels for the scientific
    planner comparison.
    """

    if split not in SPLITS:
        raise PlannerDatasetError(f"split must be one of {SPLITS}")
    if stride < 1 or horizon < 1:
        raise PlannerDatasetError("stride and horizon must be positive")
    output = Path(output_path)
    if output.exists():
        raise PlannerDatasetError(f"refusing to overwrite label index {output}")
    rows: list[dict] = []
    sources: list[dict] = []

    def label_episode(content: str, episode_rows: list[dict], ceiling_m: float) -> int:
        if not episode_rows:
            return 0
        actions = np.stack([row["action"] for row in episode_rows])
        ego = np.stack([row["ego"] for row in episode_rows])
        reached = np.stack([row["reached"] for row in episode_rows])
        n_agents = actions.shape[1]
        made = 0
        for agent in range(n_agents):
            if successful_only and not bool(np.any(reached[:, agent])):
                continue
            for anchor in range(HISTORY_STEPS - 1, len(episode_rows) - horizon, stride):
                future = actions[anchor + 1:anchor + 1 + horizon, agent]
                mean_bank = float(np.mean(future[:, 0]))
                lateral = 0 if mean_bank > 0.15 else (2 if mean_bank < -0.15 else 1)
                endpoint = ego[anchor + horizon, agent]
                agl_m = float(endpoint[2]) * 5_000.0
                amsl_m = float(endpoint[1]) * ceiling_m
                altitude = int(np.argmin([abs(agl_m - 200.0), abs(agl_m - 1_000.0), abs(amsl_m - 4_000.0)]))
                rows.append({
                    "content_sha256": content,
                    "episode_id": int(episode_rows[anchor]["episode"]),
                    "step_index": int(episode_rows[anchor]["step"]),
                    "agent": agent,
                    "label": INTENT_LABELS[lateral * 3 + altitude],
                    "split": split,
                    "label_source": {
                        "method": "behavior-imitation-v1", "horizon_steps": horizon,
                        "mean_executed_bank": mean_bank, "endpoint_agl_m": agl_m,
                        "endpoint_amsl_m": amsl_m,
                    },
                })
                made += 1
        return made

    for root in sorted(map(Path, trajectory_roots), key=lambda p: str(p)):
        reader = TrajectoryReader(root)
        content = reader.report.content_sha256
        env_cfg = reader.metadata.provenance.environment_configuration.get("env") or {}
        ceiling_m = float((env_cfg.get("airframe") or {}).get("ceiling", 12_000.0))
        current: int | None = None
        episode_rows: list[dict] = []
        count = 0
        for chunk in reader.iter_chunks():
            for i in range(len(chunk["episode/id"])):
                episode = int(chunk["episode/id"][i])
                if current is not None and episode != current:
                    count += label_episode(content, episode_rows, ceiling_m)
                    episode_rows = []
                current = episode
                episode_rows.append({
                    "episode": episode,
                    "step": int(chunk["step/index"][i]),
                    "action": np.asarray(chunk["action/executed"][i]),
                    "ego": np.asarray(chunk["observation/ego"][i]),
                    "reached": np.asarray(chunk["status/reached"][i]),
                })
        count += label_episode(content, episode_rows, ceiling_m)
        sources.append({"path": str(root), "content_sha256": content, "labels": count})
    if not rows:
        qualifier = " successful" if successful_only else ""
        raise PlannerDatasetError(f"no{qualifier} behavior windows were labelable")
    body = b"".join(canonical_json_bytes(row) for row in rows)
    _write_atomic(output, body)
    return {
        "schema": "naigos.qwen-planner-label-index/1", "method": "behavior-imitation-v1",
        "split": split, "records": len(rows), "sha256": hashlib.sha256(body).hexdigest(),
        "sources": sources, "counterfactual_optimal": False,
    }
