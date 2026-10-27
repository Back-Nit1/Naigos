"""JAX / environment adapter for the framework-neutral ``naigos.trajectory`` format.

``naigos.trajectory`` knows nothing about JAX or the simulator. This module is
the boundary: collectors (PPO training, evaluation, route/resource bank, the
pipeline evaluator, the demo replay) hand it raw per-step evidence taken at the
real actor -> optional CBF -> ``env.step`` boundary
(``NaigosEnv.evidence_frame``). It transfers that evidence to the host once,
converts it to canonical NumPy arrays and appends bounded chunks through
``TrajectoryWriter``. No file is touched inside ``jit``/``vmap``/``scan``.

Recording is explicitly opt-in (``RecordingConfig.enabled``). Disabled, nothing
here is called and the compiled programs are the ones that existed before.

Layout rules that keep the writer's strict episode-order contract:
  * one dataset never mixes theatre / route / scenario / seed / policy /
    environment / resource / code identity (those live once in its metadata);
  * a PPO batch interleaves worlds, so every world is its own shard and its
    records are appended in time order;
  * a resumed run, and a rollout re-initialised by a curriculum promotion, start
    new shards -- an in-flight episode is never pretended to continue.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import jax
import numpy as np

from ..env.config import EnvConfig
from ..trajectory import (
    TERMINATION_CAUSES,
    DatasetMetadata,
    FieldSpec,
    PolicyIdentity,
    Provenance,
    TrajectoryReader,
    TrajectoryWriter,
    canonical_json_bytes,
    default_trajectory_schema,
    validate_dataset,
)
from . import runmeta

INDEX_FILENAME = "trajectory_index.json"
INDEX_PROTOCOL = "naigos-trajectory-index-v1"
_SAFE = re.compile(r"[^A-Za-z0-9._-]+")
_PREFIX = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

# TERMINATION_CAUSES has no "diverted" code; a safe L1 diversion is reported as
# the schema's documented "other" and is fully described by `resource/outcome`.
CAUSE_DIVERTED = TERMINATION_CAUSES["other"]

_REWARD_TERMS = (
    "progress", "exposure", "lock_level", "envelope_dwell", "arrived", "shotdown", "fuel_used",
    "g_excess", "terrain_violation", "ceiling_violation", "stall_violation", "bounds_violation",
    "out_of_fuel", "edge_proximity",
)


class RecordingError(RuntimeError):
    """Recording was misconfigured or a collector violated the recording contract."""


@dataclasses.dataclass(frozen=True)
class RecordingConfig:
    """Opt-in trajectory recording. The default records nothing."""

    enabled: bool = False
    #: Output root holding one directory per dataset. ``None`` lets a caller with
    #: a run directory default it to ``<run>/trajectories``.
    root: str | None = None
    #: Records per committed chunk -- the bound on in-memory accumulation.
    chunk_steps: int = 512
    #: Optional dataset-id prefix.
    prefix: str = ""
    description: str = ""
    #: PPO collection only: record the first N interleaved worlds (None = all).
    max_worlds: int | None = None
    #: Provenance theatre label; derived from the environment when omitted.
    theatre: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.chunk_steps, bool) or not isinstance(self.chunk_steps, int) or self.chunk_steps < 1:
            raise RecordingError("chunk_steps must be a positive integer")
        if self.max_worlds is not None and (
            isinstance(self.max_worlds, bool) or not isinstance(self.max_worlds, int) or self.max_worlds < 1
        ):
            raise RecordingError("max_worlds must be a positive integer or None")
        if self.prefix and not _PREFIX.fullmatch(self.prefix):
            raise RecordingError(f"invalid dataset prefix {self.prefix!r}")

    @property
    def active(self) -> bool:
        return self.enabled

    def resolved(self, default_root: str | os.PathLike | None = None) -> "RecordingConfig":
        """Return a copy whose ``root`` is set, or refuse an enabled config with none."""
        if not self.enabled or self.root is not None:
            return self
        if default_root is None:
            raise RecordingError("recording is enabled but no output root was given")
        return dataclasses.replace(self, root=str(default_root))


def is_recording(cfg: RecordingConfig | None) -> bool:
    return cfg is not None and cfg.enabled


def add_cli_args(parser) -> None:
    """The shared opt-in flags. Without ``--record-trajectories`` nothing is recorded."""
    g = parser.add_argument_group("trajectory recording (opt-in, off by default)")
    g.add_argument("--record-trajectories", nargs="?", const="", default=None, metavar="DIR",
                   help="write trajectory datasets under DIR (no value: the run/output directory's trajectories/)")
    g.add_argument("--record-chunk-steps", type=int, default=512, help="records per committed chunk")
    g.add_argument("--record-prefix", default="", help="dataset id prefix")
    g.add_argument("--record-max-worlds", type=int, default=None, help="PPO: record only the first N worlds")


def config_from_args(args) -> RecordingConfig | None:
    """None unless the user asked; an empty ``--record-trajectories`` defers the root to the caller."""
    if args.record_trajectories is None:
        return None
    return RecordingConfig(
        enabled=True, root=args.record_trajectories or None, chunk_steps=args.record_chunk_steps,
        prefix=args.record_prefix, max_worlds=args.record_max_worlds,
    )


# --------------------------------------------------------------- provenance --


@lru_cache(maxsize=1)
def _code_state() -> tuple[str, bool]:
    """(commit, dirty) of the checkout this module lives in; unknown = all zeros + dirty."""
    info = runmeta.git_info(Path(__file__).resolve().parents[2])
    commit = info.get("commit")
    if not commit or not re.fullmatch(r"[0-9a-f]{7,64}", commit):
        return "0" * 40, True
    return commit, True if info.get("dirty") is None else bool(info["dirty"])


def key_words(key) -> list[int]:
    """The raw uint32 words of a legacy or typed PRNG key, for provenance."""
    try:
        data = jax.random.key_data(key)
    except Exception:
        data = key
    return [int(x) for x in np.asarray(data).reshape(-1)]


def params_sha256(params: Any) -> str:
    """SHA-256 over the dtype, shape and bytes of every parameter leaf, in tree order."""
    h = hashlib.sha256()
    for leaf in jax.tree.leaves(jax.device_get(params)):
        arr = np.ascontiguousarray(np.asarray(leaf))
        h.update(str(arr.dtype).encode() + str(arr.shape).encode() + arr.tobytes())
    return h.hexdigest()


def _hmap_label(hmap) -> str:
    if hmap is None:
        return "synthetic-terrain"
    arr = np.ascontiguousarray(np.asarray(hmap))
    return "dem-" + hashlib.sha256(str(arr.shape).encode() + arr.tobytes()).hexdigest()[:12]


def environment_record(cfg: EnvConfig, hmap=None, **collection: Any) -> dict:
    """Everything needed to rebuild the environment, as plain JSON.

    ``collection`` is whatever identifies this collection run (key, world count,
    session ...); it is part of the digest on purpose, so two shards that were
    collected differently never share an environment identity.
    """
    from . import checkpoint as ckpt

    record: dict[str, Any] = {
        "env": dataclasses.asdict(cfg),
        "terrain": _hmap_label(hmap),
        "observation_schema_id": ckpt.RESOURCE_OBSERVATION_SCHEMA if cfg.resource.enabled else ckpt.BASELINE_OBSERVATION_SCHEMA,
        "actor_schema_id": ckpt.RESOURCE_ACTOR_SCHEMA if cfg.resource.enabled else ckpt.BASELINE_ACTOR_SCHEMA,
        "resource_bank": None,
        "collection": collection,
    }
    if cfg.resource.enabled:
        from ..env.resource_bank import base_config_digest, resource_protocol

        record["resource_bank"] = resource_protocol(base_config_digest(cfg, hmap))
    return record


def policy_record(
    kind: str,
    cfg: EnvConfig,
    *,
    family: str | None = None,
    version: str | None = None,
    artifact_sha256: str | None = None,
    name: str | None = None,
    **extra: Any,
) -> PolicyIdentity:
    """Behaviour-policy identity: family/version/spec plus the resource axis.

    ``family=None`` is a scripted (non-learned) controller identified by ``name``.
    """
    from . import checkpoint as ckpt
    from . import policies

    configuration: dict[str, Any] = {
        "environment_model_id": cfg.resource.model,
        "resource_config": dataclasses.asdict(cfg.resource),
        "observation_schema_id": ckpt.RESOURCE_OBSERVATION_SCHEMA if cfg.resource.enabled else ckpt.BASELINE_OBSERVATION_SCHEMA,
        "actor_schema_id": ckpt.RESOURCE_ACTOR_SCHEMA if cfg.resource.enabled else ckpt.BASELINE_ACTOR_SCHEMA,
        **extra,
    }
    if cfg.resource.enabled:
        from ..env.resource_bank import resource_protocol

        configuration["resource_bank_sha256"] = resource_protocol()["bank_sha256"]
    if family is None:
        label = name or "scripted"
        configuration["scripted"] = label
        return PolicyIdentity(_SAFE.sub("-", f"scripted-{label}")[:128], kind, "numpy-jax", artifact_sha256, configuration)
    spec = policies.get_policy_spec(
        cfg, family=family or policies.DEFAULT_POLICY_FAMILY, version=version or policies.DEFAULT_POLICY_VERSION
    )
    configuration.update(
        policy_family=spec.family.value, policy_version=spec.version, policy_spec=spec.to_dict(),
        policy_spec_sha256=spec.sha256, recurrent=bool(spec.memory_requirements.recurrent),
    )
    return PolicyIdentity(
        _SAFE.sub("-", f"{spec.family.value}-{spec.version}-{kind}")[:128], kind, "jax-flax", artifact_sha256,
        configuration,
    )


# ------------------------------------------------------------------- schema --


def _extra_fields(cfg: EnvConfig, recurrent: bool, dims: Mapping[str, int]):
    ident = {"method": "identity"}

    def f(name, dtype, shape, unit, description, normalization=None):
        return FieldSpec(name, dtype, shape, unit, description, normalization or ident)

    fields = [
        f("reward/diverted", "float32", ("agent",), "1",
          "One on the transition an L1 recovery gate is reached (RewardTerms.diverted); zero in baseline-v0.", ident),
        f("resource/outcome", "uint8", ("agent",), "1",
          "The environment's exclusive first terminal outcome so far (0 ongoing, 1 primary, 2 diverted, 3 shot down, "
          "4 terrain, 5 bounds, 6 fuel, 7 timeout; env precedence shot>terrain>bounds>fuel>primary>diverted>timeout). "
          "Distinct from status/termination_cause, which follows the trajectory-schema precedence.",
          {"method": "categorical", "values": {
              "ongoing": 0, "primary": 1, "diverted": 2, "shot_down": 3, "terrain": 4, "out_of_bounds": 5,
              "out_of_fuel": 6, "timeout": 7}}),
        f("resource/diverted", "bool", ("agent",), "1", "Post-transition safe-diversion state (always false in baseline-v0).",
          ident),
        f("extension/route", "float32", ("route_feature",), "1",
          "Realized route of this episode: [bearing rad, route family id, legacy-placement flag] (EnvState.route).",
          ident),
        f("extension/collection_iteration", "uint32", (), "1",
          "Training iteration (or evaluation tag) that produced this record.", ident),
    ]
    if cfg.resource.enabled:
        fields += [
            f("resource/observation", "float32", ("agent", "resource_feature"), "1",
              "Pre-action L1 resource block the actor received appended after observation/ego "
              "(concatenate to reproduce the actor's ego input).", ident),
            f("resource/reserve_fuel", "float32", ("agent",), "kg", "Episode fuel reserve (absolute)."),
            f("resource/initial_fuel", "float32", ("agent",), "kg", "Episode initial fuel (absolute)."),
            f("resource/diversion_pos", "float32", ("agent", "gate", "xyz"), "m",
              "Abstract recovery-gate positions in world coordinates."),
            f("resource/diversion_active", "bool", ("agent", "gate"), "1", "Which recovery gates are enabled."),
            f("privileged/burn_multiplier", "float32", ("agent",), "1",
              "TRUE episode burn multiplier. Privileged simulator state for auditing only; never an actor input.", ident),
        ]
    if recurrent:
        fields += [
            f("extension/memory_before", "float32", ("agent", "memory_unit"), "1",
              "Recurrent memory carried INTO the action (zero at a fresh sortie).", ident),
            f("extension/memory_after", "float32", ("agent", "memory_unit"), "1",
              "Recurrent memory carried OUT of the action after lifecycle resets.", ident),
            f("extension/memory_reset", "bool", ("agent",), "1",
              "True where the lifecycle zeroed the carried memory (arrival, death, diversion, timeout, world reset).",
              ident),
        ]
    return fields


def build_schema(cfg: EnvConfig, ev: Mapping[str, Any]):
    """Schema for ``cfg``, with dimension sizes read from time-leading evidence ``ev``."""
    base = 14 if cfg.obs_edge_features else 10
    extra_dims = {"route_feature": 3}
    if cfg.resource.enabled:
        extra_dims.update(resource_feature=cfg.resource.observation_dim, gate=2, xyz=3)
    recurrent = "memory_before" in ev
    if recurrent:
        extra_dims["memory_unit"] = int(np.shape(ev["memory_before"])[-1])
    return default_trajectory_schema(
        n_agents=cfg.n_blue,
        ego_features=base,
        threat_slots=int(np.shape(ev["obs_threats"])[-2]),
        threat_features=int(np.shape(ev["obs_threats"])[-1]),
        friend_slots=int(np.shape(ev["obs_friends"])[-2]),
        friend_features=int(np.shape(ev["obs_friends"])[-1]),
        extra_dimensions=extra_dims,
        extra_fields=_extra_fields(cfg, recurrent, extra_dims),
    )


# --------------------------------------------------------------- conversion --


def termination_causes(terms: Mapping[str, np.ndarray], ev: Mapping[str, np.ndarray], resource_enabled: bool):
    """Exclusive per-agent terminal cause for each transition, ``(T, B)`` uint8.

    Schema precedence ``terrain > out_of_bounds > shot_down > reached`` over the
    raw events; then L1 fuel exhaustion (terminal only under L1: baseline-v0
    exhaustion repeats while alive and is a cost, not a termination), a safe
    diversion, and finally the timeout of an aircraft still active when the
    world's step budget ends. A frozen post-terminal step produces none of
    these, so no outcome is counted twice.
    """
    shape = terms["arrived"].shape
    cause = np.zeros(shape, dtype=np.uint8)
    for name, key in (("reached", "arrived"), ("shot_down", "shotdown"), ("out_of_bounds", "bounds_violation"),
                      ("terrain", "terrain_violation")):  # later = higher precedence
        cause = np.where(terms[key] > 0.5, TERMINATION_CAUSES[name], cause).astype(np.uint8)
    free = cause == 0
    if resource_enabled:
        cause = np.where(free & (terms["out_of_fuel"] > 0.5), TERMINATION_CAUSES["out_of_fuel"], cause)
        free = cause == 0
    cause = np.where(free & (terms["diverted"] > 0.5), CAUSE_DIVERTED, cause)
    free = cause == 0
    active = ev["alive"] & ~ev["reached"] & ~ev["diverted"]
    cause = np.where(free & active & ev["done"][..., None], TERMINATION_CAUSES["timeout"], cause)
    return cause.astype(np.uint8)


def episode_layout(done: np.ndarray, at_boundary: bool, last_id: int, next_step: int):
    """Episode id / step index / reset marker for one world's consecutive records.

    ``at_boundary`` says the first record starts a new episode (fresh rollout);
    otherwise it continues episode ``last_id`` at ``next_step``. A record starts a
    new episode exactly when the previous record's world was done (auto-reset).
    Returns ``(episode, step, reset)`` arrays plus the carry state.
    """
    t = len(done)
    starts = np.concatenate([[at_boundary], np.asarray(done[:-1], dtype=bool)])
    episode = last_id + np.cumsum(starts)
    idx = np.arange(t)
    anchor = np.maximum.accumulate(np.where(starts, idx, -1))
    step = np.where(anchor >= 0, idx - anchor, next_step + idx)
    return (
        episode.astype(np.uint64), step.astype(np.uint32), starts.astype(bool),
        (int(episode[-1]), bool(done[-1]), int(step[-1]) + 1),
    )


def to_chunk(
    cfg: EnvConfig,
    ev: Mapping[str, Any],
    *,
    episode: np.ndarray,
    step: np.ndarray,
    reset: np.ndarray,
    iteration: int,
) -> dict[str, np.ndarray]:
    """Canonical schema arrays for one world's time-leading evidence ``ev``."""
    base = 14 if cfg.obs_edge_features else 10
    terms = {k: np.asarray(v, dtype=np.float32) for k, v in ev["terms"].items()}
    t = len(episode)
    ev_np = {k: np.asarray(v) for k, v in ev.items() if k != "terms"}
    cause = termination_causes(terms, ev_np, cfg.resource.enabled)

    req = ev_np["action_requested"].astype(np.float32)
    exe = ev_np["action_executed"].astype(np.float32)
    ego = ev_np["obs_ego"].astype(np.float32)
    out: dict[str, np.ndarray] = {
        "episode/id": episode.astype(np.uint64),
        "step/index": step.astype(np.uint32),
        "observation/ego": ego[..., :base],
        "observation/threats": ev_np["obs_threats"].astype(np.float32),
        "observation/threat_mask": ev_np["obs_threat_mask"].astype(bool),
        "observation/friends": ev_np["obs_friends"].astype(np.float32),
        "observation/friend_mask": ev_np["obs_friend_mask"].astype(bool),
        "action/requested": req,
        "action/executed": exe,
        "cbf/feasible": ev_np["cbf_feasible"].astype(bool),
        "cbf/intervention_l2": np.linalg.norm(exe.astype(np.float64) - req.astype(np.float64), axis=-1).astype(np.float32),
        "constraint_cost/aircraft_loss": np.clip(
            terms["shotdown"] + terms["terrain_violation"] + terms["bounds_violation"] + terms["out_of_fuel"], 0, 1
        ).astype(np.float32),
        "status/alive": ev_np["alive"].astype(bool),
        "status/reached": ev_np["reached"].astype(bool),
        "status/reset": np.asarray(reset, dtype=bool),
        "status/termination_cause": cause,
        "resource/fuel_before": ev_np["fuel_before"].astype(np.float32),
        "resource/fuel_after": ev_np["fuel_after"].astype(np.float32),
        "reward/diverted": terms["diverted"],
        "resource/outcome": ev_np["outcome"].astype(np.uint8),
        "resource/diverted": ev_np["diverted"].astype(bool),
        "extension/route": ev_np["route"].astype(np.float32),
        "extension/collection_iteration": np.full((t,), iteration, dtype=np.uint32),
    }
    for name in _REWARD_TERMS:
        out[f"reward/{name}"] = terms[name]
    if cfg.resource.enabled:
        out.update({
            "resource/observation": ego[..., base:],
            "resource/reserve_fuel": ev_np["reserve_fuel"].astype(np.float32),
            "resource/initial_fuel": ev_np["initial_fuel"].astype(np.float32),
            "resource/diversion_pos": ev_np["diversion_pos"].astype(np.float32),
            "resource/diversion_active": ev_np["diversion_active"].astype(bool),
            "privileged/burn_multiplier": ev_np["burn_multiplier"].astype(np.float32),
        })
    if "memory_before" in ev_np:
        after = ev_np["memory_after"].astype(np.float32)
        out.update({
            "extension/memory_before": ev_np["memory_before"].astype(np.float32),
            "extension/memory_after": after,
            "extension/memory_reset": ~np.any(after != 0, axis=-1),
        })
    return out


# ------------------------------------------------------------------- shards --


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def dataset_name(rec: RecordingConfig, base: str) -> str:
    name = _SAFE.sub("-", rec.prefix + base).strip("-.")
    if not name:
        raise RecordingError(f"cannot derive a dataset id from {base!r}")
    return name[:128]


class Shard:
    """One dataset on disk, fed bounded chunks of exactly ``chunk_steps`` records."""

    def __init__(self, root: Path, name: str, metadata: DatasetMetadata, chunk_steps: int, info: dict):
        self.path = root / name
        self.info = info
        self.chunk_steps = chunk_steps
        self.metadata = metadata
        self.writer = TrajectoryWriter(self.path, metadata)  # refuses an existing path
        self._buf: list[dict[str, np.ndarray]] = []
        self._n = 0
        self.report = None

    def add(self, arrays: Mapping[str, np.ndarray]) -> None:
        self._buf.append(dict(arrays))
        self._n += len(arrays["episode/id"])
        while self._n >= self.chunk_steps:
            self._flush(self.chunk_steps)

    def _flush(self, n: int) -> None:
        names = self._buf[0].keys()
        merged = {k: np.concatenate([b[k] for b in self._buf], axis=0) for k in names}
        self.writer.append({k: v[:n] for k, v in merged.items()})
        rest = {k: v[n:] for k, v in merged.items()}
        self._n -= n
        self._buf = [rest] if self._n else []

    def finalize(self) -> dict:
        """Flush, finalize, then reopen through the reader and cross-check."""
        if self._n:
            self._flush(self._n)
        report = self.writer.finalize()
        reader = TrajectoryReader(self.path)
        records = sum(len(chunk["episode/id"]) for chunk in reader.iter_chunks())  # validates every chunk
        if records != report.records or reader.report.content_sha256 != report.content_sha256:
            raise RecordingError(f"{self.path}: reopened dataset disagrees with the finalized report")
        self.report = report
        return self.entry(complete=True)

    def entry(self, complete: bool) -> dict:
        """Index row. A partial dataset never carries a content hash."""
        if complete:
            report = self.report
        else:
            try:
                report = validate_dataset(self.path, allow_partial=True)
            except Exception:  # an unreadable partial is still listed, with no counts
                report = None
        commit = json.loads((self.path / "metadata.commit.json").read_text())
        prov = self.metadata.provenance
        return {
            **self.info,
            "name": self.path.name,
            "path": str(self.path),
            "complete": bool(complete),
            "records": None if report is None else int(report.records),
            "chunks": None if report is None else int(report.chunks),
            "metadata_sha256": commit["sha256"],
            "provenance_sha256": hashlib.sha256(canonical_json_bytes(prov.to_dict())).hexdigest(),
            "environment_sha256": prov.environment_sha256,
            "content_sha256": report.content_sha256 if complete else None,
        }


def new_shard(
    rec: RecordingConfig,
    root: Path,
    name: str,
    *,
    cfg: EnvConfig,
    ev: Mapping[str, Any],
    theatre: str,
    route: str,
    scenario: str,
    seed: int,
    environment: dict,
    policy: PolicyIdentity,
    info: dict,
    description: str = "",
) -> Shard:
    schema = build_schema(cfg, ev)
    commit, dirty = _code_state()
    provenance = Provenance(
        theatre=rec.theatre or theatre, route=route, scenario=scenario, seed=int(seed), code_commit=commit,
        code_dirty=dirty, environment_configuration=environment, policy=policy,
    )
    note = " | ".join(x for x in (rec.description, description) if x)
    metadata = DatasetMetadata(dataset_id=name, created_at=_utc_now(), schema=schema, provenance=provenance,
                               description=note)
    return Shard(root, name, metadata, rec.chunk_steps, info)


# -------------------------------------------------------------------- index --


def update_index(path: str | os.PathLike, entries: list[dict]) -> Path:
    """Merge ``entries`` (keyed by dataset path) into the index and replace it atomically."""
    path = Path(path)
    rows: dict[str, dict] = {}
    if path.exists():
        old = json.loads(path.read_text())
        if old.get("protocol") != INDEX_PROTOCOL:
            raise RecordingError(f"{path} is not a {INDEX_PROTOCOL} index")
        rows = {r["path"]: r for r in old["datasets"]}
    for e in entries:
        prev = rows.get(e["path"])
        if prev is not None and prev["complete"] and not e["complete"]:
            raise RecordingError(f"{e['path']} is finalized; refusing to downgrade it to partial")
        rows[e["path"]] = e
    doc = {"protocol": INDEX_PROTOCOL, "datasets": [rows[k] for k in sorted(rows)]}
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(json.dumps(doc, indent=2, sort_keys=True) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise
    return path


# ----------------------------------------------------- batch (eval) rollouts --


def route_label(route) -> str:
    if route is None:
        return "env-route-mode"
    r = np.asarray(route, dtype=np.float64)
    return f"bearing={np.rad2deg(r[0]):.4f}deg,family={int(r[1])}"


def record_rollouts(
    rec: RecordingConfig,
    env,
    evidence: Mapping[str, Any],
    *,
    label: str,
    seed: int,
    policy: PolicyIdentity,
    index_path: str | os.PathLike | None = None,
    route=None,
    iteration: int = 0,
    cfg: EnvConfig | None = None,
    collection: Mapping[str, Any] | None = None,
    scenario: str | None = None,
) -> dict:
    """Write a vmapped ``env.rollout(..., record=True)`` evidence tree as ONE dataset.

    ``evidence`` has a leading world axis. World ``w`` becomes episode ``w``,
    truncated after its first done step (later steps are frozen and carry no new
    outcome). Fails closed: on any error the partial dataset stays on disk, listed
    in the index as incomplete with no content hash, and the error propagates.
    """
    rec = rec.resolved(None if index_path is None else Path(index_path).parent)
    cfg = cfg or env.cfg
    root = Path(rec.root)
    root.mkdir(parents=True, exist_ok=True)
    host = jax.device_get(evidence)  # the single device->host transfer
    n_worlds = int(np.shape(host["done"])[0])
    name = dataset_name(rec, label)
    info = {"kind": "rollout", "label": label, "session": None, "segment": None, "world": None}
    shard: Shard | None = None
    index = Path(index_path) if index_path is not None else root / INDEX_FILENAME
    try:
        for w in range(n_worlds):
            ev_w = jax.tree.map(lambda a: np.asarray(a)[w], host)
            done = np.flatnonzero(ev_w["done"])
            keep = int(done[0]) + 1 if done.size else len(ev_w["done"])
            ev_w = jax.tree.map(lambda a: a[:keep], ev_w)
            episode, step, reset, _ = episode_layout(ev_w["done"], True, w - 1, 0)
            chunk = to_chunk(cfg, ev_w, episode=episode, step=step, reset=reset, iteration=iteration)
            if shard is None:
                shard = new_shard(
                    rec, root, name, cfg=cfg, ev=ev_w, theatre=_hmap_label(env.fixed_hmap),
                    route=route_label(route), scenario=scenario or label, seed=seed,
                    environment=environment_record(cfg, env.fixed_hmap, n_worlds=n_worlds, **(collection or {})),
                    policy=policy, info=info,
                )
            shard.add(chunk)
        if shard is None:
            raise RecordingError("no worlds to record")
        entry = shard.finalize()
    except BaseException:
        if shard is not None:
            update_index(index, [shard.entry(complete=False)])
        raise
    update_index(index, [entry])
    return entry


# ------------------------------------------------------ PPO training capture --


class _World:
    """Per-world episode bookkeeping across consecutive PPO segments."""

    def __init__(self):
        self.shard: Shard | None = None
        self.at_boundary = True
        self.last_id = -1
        self.next_step = 0


class TrainingRecorder:
    """Records the PPO collector's own rollouts, one shard per interleaved world.

    ``begin_segment`` must be called whenever the in-flight rollout is not the
    continuation of the previous one: at start, on resume (a new process), and
    when a curriculum promotion re-initialises the rollout. Each call finalizes
    the previous segment's shards and starts new ones lazily.
    """

    def __init__(
        self,
        rec: RecordingConfig,
        env,
        *,
        out_dir: str | os.PathLike,
        seed: int,
        n_envs: int,
        n_steps: int,
        policy: PolicyIdentity,
        resumed: bool,
    ):
        self.rec = rec.resolved(Path(out_dir) / "trajectories")
        self.env = env
        self.root = Path(self.rec.root)
        self.index = Path(out_dir) / INDEX_FILENAME
        self.seed, self.n_envs, self.n_steps, self.policy, self.resumed = seed, n_envs, n_steps, policy, resumed
        self.n_worlds = min(n_envs, self.rec.max_worlds or n_envs)
        self.root.mkdir(parents=True, exist_ok=True)
        self.session = self._next_session()
        self.segment = -1
        self._worlds: list[_World] = []
        self._reason = ""
        self._red_level = 0.0
        self.entries: list[dict] = []

    def _next_session(self) -> int:
        pat = re.compile(re.escape(self.rec.prefix) + r"train-s(\d+)g\d+-w\d+$")
        seen = [int(m.group(1)) for p in self.root.iterdir() if (m := pat.match(p.name))] if self.root.exists() else []
        return max(seen, default=0) + 1

    def begin_segment(self, red_level: float, reason: str) -> None:
        self._finalize_shards()
        self.segment += 1
        self._reason, self._red_level = reason, float(red_level)
        self._worlds = [_World() for _ in range(self.n_worlds)]

    def _finalize_shards(self) -> None:
        done = []
        for w in self._worlds:
            if w.shard is not None:
                done.append(w.shard.finalize())
                w.shard = None
        if done:
            self.entries += done
            update_index(self.index, done)

    def ingest(self, evidence, iteration: int) -> None:
        """Append one PPO segment's evidence (time-major ``(T, n_envs, ...)``)."""
        k = self.n_worlds
        host = jax.device_get(jax.tree.map(lambda a: a[:, :k], evidence))  # slice on device, transfer once
        for w, world in enumerate(self._worlds):
            ev_w = jax.tree.map(lambda a: np.asarray(a)[:, w], host)
            episode, step, reset, (world.last_id, world.at_boundary, world.next_step) = episode_layout(
                ev_w["done"], world.at_boundary, world.last_id, world.next_step
            )
            chunk = to_chunk(self.env.cfg, ev_w, episode=episode, step=step, reset=reset, iteration=iteration)
            if world.shard is None:
                world.shard = self._open(w, ev_w)
            world.shard.add(chunk)

    def _open(self, w: int, ev_w) -> Shard:
        name = dataset_name(self.rec, f"train-s{self.session:03d}g{self.segment:03d}-w{w:04d}")
        scenario = f"ppo-train/session-{self.session:03d}/segment-{self.segment:03d}/world-{w:04d}"
        env_rec = environment_record(
            self.env.cfg, self.env.fixed_hmap, kind="ppo-train", n_envs=self.n_envs, n_steps=self.n_steps,
            world=w, session=self.session, segment=self.segment, segment_reason=self._reason,
            resumed_session=self.resumed, red_level=self._red_level,
        )
        info = {"kind": "ppo-train", "label": "train", "session": self.session, "segment": self.segment, "world": w}
        return new_shard(
            self.rec, self.root, name, cfg=self.env.cfg, ev=ev_w, theatre=_hmap_label(self.env.fixed_hmap),
            route=f"route_mode={self.env.cfg.route_mode}", scenario=scenario, seed=self.seed, environment=env_rec,
            policy=self.policy, info=info,
            description="online PPO behaviour policy; a segment may end mid-episode (rollout re-initialised)",
        )

    def close(self, success: bool) -> list[dict]:
        """Finalize on success; otherwise leave every open shard partial (no manifest)."""
        if success:
            self._finalize_shards()
            return self.entries
        partial = [w.shard.entry(complete=False) for w in self._worlds if w.shard is not None]
        if partial:
            update_index(self.index, partial)
        return self.entries + partial
