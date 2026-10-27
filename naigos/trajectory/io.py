"""Chunked serialization and integrity validation for trajectory datasets."""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np

from .schema import (
    FORMAT_NAME,
    FORMAT_VERSION,
    TERMINATION_CAUSES,
    DatasetMetadata,
    SchemaError,
    TrajectorySchema,
    UnsupportedVersionError,
    canonical_json_bytes,
)

METADATA_FILE = "metadata.json"
METADATA_COMMIT_FILE = "metadata.commit.json"
MANIFEST_FILE = "manifest.json"
CHUNKS_DIR = "chunks"
_CHUNK_STEM_RE = re.compile(r"^[0-9]{8}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class DatasetIOError(RuntimeError):
    """Base error for trajectory serialization and validation."""


class IntegrityError(DatasetIOError):
    """Committed bytes disagree with their digest or declared schema."""


class PartialDatasetError(DatasetIOError):
    """A reader requiring a finalized dataset found an incomplete one."""


class ImmutableDatasetError(DatasetIOError):
    """An operation would replace committed metadata or append after finalize."""


@dataclass(frozen=True)
class ValidationReport:
    path: Path
    complete: bool
    records: int
    chunks: int
    # None for a partial dataset: a committed prefix is not an immutable
    # dataset identity because later chunks can still extend it.
    content_sha256: str | None
    partial_reasons: tuple[str, ...]


@dataclass
class _SequenceState:
    """Bounded per-episode state for streaming transition-order validation."""

    current_episode: int | None = None
    next_step: int = 0
    seen_episodes: set[int] = field(default_factory=set)
    terminated_agents: tuple[bool, ...] = ()

    def copy(self) -> "_SequenceState":
        return _SequenceState(
            current_episode=self.current_episode,
            next_step=self.next_step,
            seen_episodes=set(self.seen_episodes),
            terminated_agents=tuple(self.terminated_agents),
        )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path, block_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    try:
        with path.open("rb") as fh:
            for block in iter(lambda: fh.read(block_size), b""):
                h.update(block)
    except OSError as exc:
        raise IntegrityError(f"cannot read {path}: {exc}") from exc
    return h.hexdigest()


def _object_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise IntegrityError(f"duplicate JSON key {key!r}")
        out[key] = value
    return out


def _invalid_constant(value: str) -> None:
    raise IntegrityError(f"non-finite JSON constant {value!r}")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_object_no_duplicates,
            parse_constant=_invalid_constant,
        )
    except IntegrityError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise IntegrityError(f"cannot read valid JSON from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise IntegrityError(f"{path} must contain a JSON object")
    return value


def _fsync_directory(path: Path) -> None:
    """Best-effort durability for a just-created directory entry."""
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _write_once_bytes(path: Path, payload: bytes) -> None:
    """Publish bytes without ever replacing an existing path.

    The temporary and final hard link are in one directory.  A crash can leave
    a clearly named temporary, but a reader never observes a half-written final
    record.  Hard-link publication also makes a concurrent second writer lose
    rather than overwrite the winner.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    try:
        with temp.open("xb") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        try:
            os.link(temp, path)
        except FileExistsError as exc:
            raise ImmutableDatasetError(f"refusing to replace immutable {path}") from exc
        _fsync_directory(path.parent)
    finally:
        try:
            temp.unlink()
        except FileNotFoundError:
            pass


def _write_once_json(path: Path, value: Mapping[str, Any]) -> None:
    _write_once_bytes(path, canonical_json_bytes(value))


def _exact_keys(data: Mapping[str, Any], expected: set[str], where: str) -> None:
    missing = expected - set(data)
    unknown = set(data) - expected
    if missing or unknown:
        detail = []
        if missing:
            detail.append(f"missing {sorted(missing)}")
        if unknown:
            detail.append(f"unknown {sorted(unknown)}")
        raise IntegrityError(f"{where}: {', '.join(detail)}")


def _file_record(path: Path, relative: str) -> dict[str, Any]:
    return {
        "file": relative,
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _validate_file_record(record: Mapping[str, Any], path: Path, expected_relative: str, where: str) -> None:
    _exact_keys(record, {"file", "bytes", "sha256"}, where)
    if record["file"] != expected_relative:
        raise IntegrityError(f"{where}: expected file {expected_relative!r}, found {record['file']!r}")
    if isinstance(record["bytes"], bool) or not isinstance(record["bytes"], int) or record["bytes"] < 0:
        raise IntegrityError(f"{where}: bytes must be a non-negative integer")
    if not isinstance(record["sha256"], str) or not _SHA256_RE.fullmatch(record["sha256"]):
        raise IntegrityError(f"{where}: sha256 must be 64 lower-case hex characters")
    if not path.is_file():
        raise IntegrityError(f"{where}: missing file {path}")
    actual_bytes = path.stat().st_size
    if actual_bytes != record["bytes"]:
        raise IntegrityError(f"{where}: byte count mismatch: recorded {record['bytes']}, actual {actual_bytes}")
    actual_hash = _sha256_file(path)
    if actual_hash != record["sha256"]:
        raise IntegrityError(f"{where}: sha256 mismatch: recorded {record['sha256']}, actual {actual_hash}")


def load_metadata(root: str | os.PathLike[str]) -> DatasetMetadata:
    """Load and validate immutable metadata, including its creation-time hash."""
    root = Path(root)
    metadata_path = root / METADATA_FILE
    commit_path = root / METADATA_COMMIT_FILE
    if not metadata_path.is_file():
        raise PartialDatasetError(f"{root} has no {METADATA_FILE}; dataset creation did not commit metadata")
    if not commit_path.is_file():
        raise PartialDatasetError(f"{root} has no {METADATA_COMMIT_FILE}; metadata is not committed")
    record = _read_json(commit_path)
    _validate_file_record(record, metadata_path, METADATA_FILE, METADATA_COMMIT_FILE)
    try:
        return DatasetMetadata.from_dict(_read_json(metadata_path))
    except UnsupportedVersionError:
        # Version refusal is an expected compatibility result, not generic
        # byte corruption; callers need to distinguish it for migration.
        raise
    except SchemaError as exc:
        raise IntegrityError(f"invalid {METADATA_FILE}: {exc}") from exc


def _as_numpy_arrays(arrays: Mapping[str, Any]) -> dict[str, np.ndarray]:
    if not isinstance(arrays, Mapping):
        raise SchemaError("a chunk must be a mapping from schema field name to array")
    converted: dict[str, np.ndarray] = {}
    for name, value in arrays.items():
        if not isinstance(name, str):
            raise SchemaError("chunk field names must be strings")
        try:
            converted[name] = np.asarray(value)
        except Exception as exc:
            raise SchemaError(f"{name}: cannot convert value to a NumPy array: {exc}") from exc
    return converted


def validate_chunk_arrays(schema: TrajectorySchema, arrays: Mapping[str, Any]) -> int:
    """Validate one in-memory chunk and return its transition count.

    Dtypes are strict.  The writer does not silently narrow float64 or signed
    identifiers, which would make provenance-dependent differences invisible.
    """
    values = _as_numpy_arrays(arrays)
    expected_names = set(schema.field_map)
    missing = expected_names - set(values)
    unknown = set(values) - expected_names
    if missing or unknown:
        detail = []
        if missing:
            detail.append(f"missing fields {sorted(missing)}")
        if unknown:
            detail.append(f"unknown fields {sorted(unknown)}")
        raise SchemaError("chunk does not match schema: " + "; ".join(detail))

    records: int | None = None
    for field in schema.fields:
        value = values[field.name]
        if value.ndim < 1:
            raise SchemaError(f"{field.name}: array needs a leading record axis")
        if records is None:
            records = value.shape[0]
            if records <= 0:
                raise SchemaError("chunks must contain at least one record")
        elif value.shape[0] != records:
            raise SchemaError(
                f"{field.name}: record count {value.shape[0]} differs from the chunk's {records}"
            )
        expected_shape = (records, *schema.resolved_shape(field))
        if value.shape != expected_shape:
            raise SchemaError(f"{field.name}: expected shape {expected_shape}, found {value.shape}")
        # dtype.name is endian-independent.  Chunks are written little-endian,
        # and NumPy correctly exposes those arrays on a big-endian reader too.
        if value.dtype.name != field.dtype:
            raise SchemaError(f"{field.name}: expected dtype {field.dtype}, found {value.dtype.name}")
        if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
            raise SchemaError(f"{field.name}: NaN and infinity are not permitted")

    assert records is not None
    _validate_core_semantics(values)
    return records


def _validate_core_semantics(arrays: Mapping[str, np.ndarray]) -> None:
    requested = arrays["action/requested"]
    executed = arrays["action/executed"]
    for name, action in (("action/requested", requested), ("action/executed", executed)):
        if np.any(action < -1.0 - 1e-6) or np.any(action > 1.0 + 1e-6):
            raise SchemaError(f"{name}: normalized controls must stay in [-1,1]")
    expected_intervention = np.linalg.norm(executed.astype(np.float64) - requested.astype(np.float64), axis=-1)
    recorded_intervention = arrays["cbf/intervention_l2"]
    if np.any(recorded_intervention < 0) or not np.allclose(
        recorded_intervention, expected_intervention, rtol=1e-5, atol=1e-6
    ):
        raise SchemaError("cbf/intervention_l2 must equal ||action/executed-action/requested||_2")

    bounded_reward_fields = (
        "exposure", "lock_level", "envelope_dwell", "arrived", "shotdown",
        "terrain_violation", "ceiling_violation", "stall_violation",
        "bounds_violation", "out_of_fuel", "edge_proximity",
    )
    for short_name in bounded_reward_fields:
        value = arrays[f"reward/{short_name}"]
        if np.any(value < 0) or np.any(value > 1):
            raise SchemaError(f"reward/{short_name} must be in [0,1]")
    if np.any(arrays["reward/fuel_used"] < 0) or np.any(arrays["reward/g_excess"] < 0):
        raise SchemaError("reward/fuel_used and reward/g_excess cannot be negative")

    for values_name, mask_name in (
        ("observation/threats", "observation/threat_mask"),
        ("observation/friends", "observation/friend_mask"),
    ):
        padded = ~arrays[mask_name]
        if np.any(arrays[values_name][padded] != 0):
            raise SchemaError(f"{values_name}: masked padding slots must be exactly zero")

    causes = arrays["status/termination_cause"]
    known_causes = np.array(tuple(TERMINATION_CAUSES.values()), dtype=np.uint8)
    if not np.isin(causes, known_causes).all():
        bad = np.unique(causes[~np.isin(causes, known_causes)]).tolist()
        raise SchemaError(f"status/termination_cause contains unknown codes {bad}")
    reached_event = causes == TERMINATION_CAUSES["reached"]
    if np.any(reached_event & ~arrays["status/reached"]):
        raise SchemaError("a reached termination cause requires status/reached=true")
    loss_codes = np.array(
        [TERMINATION_CAUSES[name] for name in ("shot_down", "terrain", "out_of_bounds")], dtype=np.uint8
    )
    if np.any(np.isin(causes, loss_codes) & arrays["status/alive"]):
        raise SchemaError("shot-down, terrain, and out-of-bounds terminal events require status/alive=false")

    # A single primary cause remains unambiguous when several raw terminal
    # indicators occur on the same transition.  The raw reward fields retain
    # every event; this precedence matches the independent verifier, which does
    # not attribute a terrain/bounds loss to a firing solution as well.
    primary = np.zeros_like(causes)
    for cause_name, reward_name in (
        ("terrain", "reward/terrain_violation"),
        ("out_of_bounds", "reward/bounds_violation"),
        ("shot_down", "reward/shotdown"),
        ("reached", "reward/arrived"),
    ):
        event = arrays[reward_name] > 0.5
        primary = np.where((primary == 0) & event, TERMINATION_CAUSES[cause_name], primary)
    has_primary_event = primary != 0
    if np.any(has_primary_event & (causes != primary)):
        raise SchemaError(
            "status/termination_cause must use terrain > out_of_bounds > shot_down > reached precedence"
        )
    fuel_cause = causes == TERMINATION_CAUSES["out_of_fuel"]
    if np.any(fuel_cause & (arrays["reward/out_of_fuel"] <= 0.5)):
        raise SchemaError("an out_of_fuel termination cause requires reward/out_of_fuel=1")

    reset = arrays["status/reset"]
    if np.any(reset & (arrays["step/index"] != 0)):
        raise SchemaError("records marked status/reset=true must have step/index=0")

    fuel_before = arrays["resource/fuel_before"]
    fuel_after = arrays["resource/fuel_after"]
    if np.any(fuel_before < 0) or np.any(fuel_after < 0):
        raise SchemaError("fuel resource state cannot be negative")
    if np.any(fuel_after > fuel_before + 1e-4):
        raise SchemaError("resource/fuel_after cannot exceed fuel_before in schema version 1")
    if not np.allclose(fuel_before - fuel_after, arrays["reward/fuel_used"], rtol=1e-5, atol=1e-4):
        raise SchemaError("reward/fuel_used must equal resource/fuel_before-resource/fuel_after")

    cost = arrays["constraint_cost/aircraft_loss"]
    if np.any(cost < 0) or np.any(cost > 1):
        raise SchemaError("constraint_cost/aircraft_loss must be in [0,1]")
    expected_cost = np.clip(
        arrays["reward/shotdown"]
        + arrays["reward/terrain_violation"]
        + arrays["reward/bounds_violation"]
        + arrays["reward/out_of_fuel"],
        0,
        1,
    )
    if not np.array_equal(cost, expected_cost):
        raise SchemaError("constraint_cost/aircraft_loss does not match the terminal raw reward terms")


def _advance_sequence(
    state: _SequenceState, arrays: Mapping[str, np.ndarray], *, where: str
) -> _SequenceState:
    """Validate record order and return the advanced streaming state.

    Chunks are storage boundaries only.  Episode identity, consecutive step
    indices, reset markers, and one-shot terminal causes therefore carry across
    them.  The input state is copied so a failed writer append cannot advance
    the committed prefix's state.
    """

    next_state = state.copy()
    episodes = arrays["episode/id"]
    steps = arrays["step/index"]
    resets = arrays["status/reset"]
    causes = arrays["status/termination_cause"]
    no_cause = TERMINATION_CAUSES["none"]

    for row in range(len(episodes)):
        episode = int(episodes[row])
        step = int(steps[row])
        reset = bool(resets[row])

        if next_state.current_episode is None or episode != next_state.current_episode:
            if episode in next_state.seen_episodes:
                raise SchemaError(
                    f"{where} record {row}: episode {episode} resumes after a different episode"
                )
            if step != 0 or not reset:
                raise SchemaError(
                    f"{where} record {row}: new episode {episode} must start at step/index=0 "
                    "with status/reset=true"
                )
            next_state.current_episode = episode
            next_state.next_step = 0
            next_state.seen_episodes.add(episode)
            next_state.terminated_agents = (False,) * causes.shape[1]

        if step != next_state.next_step:
            raise SchemaError(
                f"{where} record {row}: episode {episode} expected step/index="
                f"{next_state.next_step}, found {step}"
            )
        if reset != (step == 0):
            expectation = "true" if step == 0 else "false"
            raise SchemaError(
                f"{where} record {row}: status/reset must be {expectation} at step/index={step}"
            )

        terminated = list(next_state.terminated_agents)
        for agent, cause in enumerate(causes[row]):
            has_cause = int(cause) != no_cause
            if has_cause and terminated[agent]:
                raise SchemaError(
                    f"{where} record {row}: episode {episode} agent {agent} has a repeated "
                    "terminal cause"
                )
            terminated[agent] = terminated[agent] or has_cause
        next_state.terminated_agents = tuple(terminated)
        next_state.next_step += 1

    return next_state


def _advance_sequence_or_integrity(
    state: _SequenceState,
    arrays: Mapping[str, np.ndarray],
    *,
    chunk_index: int,
) -> _SequenceState:
    try:
        return _advance_sequence(state, arrays, where=f"chunk {chunk_index}")
    except SchemaError as exc:
        raise IntegrityError(f"chunk {chunk_index}: invalid transition sequence: {exc}") from exc


def _replay_sequence(
    root: Path, schema: TrajectorySchema, descriptors: list[Mapping[str, Any]]
) -> _SequenceState:
    state = _SequenceState()
    for descriptor in descriptors:
        arrays = _load_chunk(root, schema, descriptor)
        state = _advance_sequence_or_integrity(
            state, arrays, chunk_index=descriptor["index"]
        )
    return state


def _little_endian_contiguous(value: np.ndarray) -> np.ndarray:
    dtype = value.dtype
    if dtype.itemsize > 1 and dtype.byteorder not in ("<", "|"):
        value = value.astype(dtype.newbyteorder("<"), copy=False)
    return np.ascontiguousarray(value)


def _chunk_descriptor(index: int, records: int, path: Path) -> dict[str, Any]:
    relative = f"{CHUNKS_DIR}/{index:08d}.npz"
    return {
        "index": index,
        "file": relative,
        "records": records,
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _validate_chunk_descriptor(data: Mapping[str, Any], expected_index: int) -> None:
    _exact_keys(data, {"index", "file", "records", "bytes", "sha256"}, f"chunk {expected_index}")
    expected_file = f"{CHUNKS_DIR}/{expected_index:08d}.npz"
    if data["index"] != expected_index or data["file"] != expected_file:
        raise IntegrityError(
            f"chunk descriptor {expected_index}: expected index/file {expected_index}/{expected_file}, "
            f"found {data.get('index')!r}/{data.get('file')!r}"
        )
    for key in ("records", "bytes"):
        if isinstance(data[key], bool) or not isinstance(data[key], int) or data[key] <= 0:
            raise IntegrityError(f"chunk descriptor {expected_index}: {key} must be a positive integer")
    if not isinstance(data["sha256"], str) or not _SHA256_RE.fullmatch(data["sha256"]):
        raise IntegrityError(f"chunk descriptor {expected_index}: invalid sha256")


def _load_chunk(root: Path, schema: TrajectorySchema, descriptor: Mapping[str, Any]) -> dict[str, np.ndarray]:
    index = descriptor["index"]
    path = root / descriptor["file"]
    _validate_file_record(
        {"file": descriptor["file"], "bytes": descriptor["bytes"], "sha256": descriptor["sha256"]},
        path,
        descriptor["file"],
        f"chunk {index}",
    )
    try:
        with np.load(path, allow_pickle=False) as archive:
            if len(archive.files) != len(set(archive.files)):
                raise IntegrityError(f"chunk {index}: duplicate NPZ members")
            arrays = {name: archive[name] for name in archive.files}
    except IntegrityError:
        raise
    except Exception as exc:
        raise IntegrityError(f"chunk {index}: cannot read NPZ payload: {exc}") from exc
    try:
        records = validate_chunk_arrays(schema, arrays)
    except SchemaError as exc:
        raise IntegrityError(f"chunk {index}: {exc}") from exc
    if records != descriptor["records"]:
        raise IntegrityError(
            f"chunk {index}: descriptor records {descriptor['records']} differs from arrays {records}"
        )
    return arrays


def _scan_chunks(root: Path) -> tuple[list[dict[str, Any]], tuple[str, ...]]:
    chunks_dir = root / CHUNKS_DIR
    if not chunks_dir.is_dir():
        return [], (f"{CHUNKS_DIR}/ directory is absent",)
    descriptors: list[dict[str, Any]] = []
    reasons: list[str] = []
    descriptor_paths = sorted(path for path in chunks_dir.glob("*.json") if _CHUNK_STEM_RE.fullmatch(path.stem))
    for expected_index, descriptor_path in enumerate(descriptor_paths):
        index = int(descriptor_path.stem)
        if index != expected_index:
            raise IntegrityError(
                f"chunk descriptors must be contiguous from zero; expected {expected_index:08d}, found {index:08d}"
            )
        descriptor = _read_json(descriptor_path)
        _validate_chunk_descriptor(descriptor, expected_index)
        descriptors.append(descriptor)

    committed_npz = {f"{d['index']:08d}.npz" for d in descriptors}
    orphaned = sorted(path.name for path in chunks_dir.glob("*.npz") if path.name not in committed_npz)
    temporaries = sorted(path.name for path in chunks_dir.iterdir() if ".tmp-" in path.name)
    unexpected = sorted(
        path.name
        for path in chunks_dir.iterdir()
        if path.name not in committed_npz
        and path not in descriptor_paths
        and path.name not in orphaned
        and path.name not in temporaries
    )
    if orphaned:
        reasons.append(f"uncommitted chunk payloads: {orphaned}")
    if temporaries:
        reasons.append(f"interrupted temporary files: {temporaries}")
    if unexpected:
        reasons.append(f"unexpected files in chunks/: {unexpected}")
    return descriptors, tuple(reasons)


def _content_identity(metadata_record: Mapping[str, Any], descriptors: list[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "metadata_sha256": metadata_record["sha256"],
        "chunks": [
            {
                "index": descriptor["index"],
                "records": descriptor["records"],
                "sha256": descriptor["sha256"],
            }
            for descriptor in descriptors
        ],
    }


def _content_hash(metadata_record: Mapping[str, Any], descriptors: list[Mapping[str, Any]]) -> str:
    return _sha256_bytes(canonical_json_bytes(_content_identity(metadata_record, descriptors)))


def _manifest(metadata_record: Mapping[str, Any], descriptors: list[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "state": "complete",
        "hash_algorithm": "sha256",
        "metadata": dict(metadata_record),
        "chunks": [dict(descriptor) for descriptor in descriptors],
        "records": sum(descriptor["records"] for descriptor in descriptors),
        "content_sha256": _content_hash(metadata_record, descriptors),
    }


def _validate_manifest(
    data: Mapping[str, Any], metadata_record: Mapping[str, Any], descriptors: list[Mapping[str, Any]]
) -> None:
    expected = _manifest(metadata_record, descriptors)
    _exact_keys(data, set(expected), MANIFEST_FILE)
    if data != expected:
        for key in expected:
            if data.get(key) != expected[key]:
                raise IntegrityError(f"{MANIFEST_FILE}: {key} does not match committed dataset content")
        raise IntegrityError(f"{MANIFEST_FILE} does not match committed dataset content")


def validate_dataset(
    root: str | os.PathLike[str], *, allow_partial: bool = False, verify_arrays: bool = True
) -> ValidationReport:
    """Validate structure, hashes, schema, and optionally every chunk array.

    ``allow_partial`` only relaxes finalization.  Corrupt committed chunks still
    fail: incomplete and corrupt are intentionally different states.
    """
    root = Path(root)
    metadata = load_metadata(root)
    metadata_record = _read_json(root / METADATA_COMMIT_FILE)
    descriptors, scan_reasons = _scan_chunks(root)
    sequence = _SequenceState()
    for descriptor in descriptors:
        path = root / descriptor["file"]
        _validate_file_record(
            {"file": descriptor["file"], "bytes": descriptor["bytes"], "sha256": descriptor["sha256"]},
            path,
            descriptor["file"],
            f"chunk {descriptor['index']}",
        )
        if verify_arrays:
            arrays = _load_chunk(root, metadata.schema, descriptor)
            sequence = _advance_sequence_or_integrity(
                sequence, arrays, chunk_index=descriptor["index"]
            )

    manifest_path = root / MANIFEST_FILE
    complete = manifest_path.is_file()
    reasons = list(scan_reasons)
    if not complete:
        reasons.insert(0, f"{MANIFEST_FILE} is absent")
    else:
        if reasons:
            raise IntegrityError(f"finalized dataset contains incomplete artifacts: {'; '.join(reasons)}")
        _validate_manifest(_read_json(manifest_path), metadata_record, descriptors)

    content_hash = _content_hash(metadata_record, descriptors)
    report = ValidationReport(
        path=root,
        complete=complete,
        records=sum(descriptor["records"] for descriptor in descriptors),
        chunks=len(descriptors),
        content_sha256=content_hash if complete else None,
        partial_reasons=tuple(reasons),
    )
    if not complete and not allow_partial:
        raise PartialDatasetError(
            f"{root} is an incomplete trajectory dataset: {'; '.join(report.partial_reasons)}"
        )
    return report


class TrajectoryWriter:
    """Append validated array chunks and finalize them into one immutable dataset."""

    def __init__(self, root: str | os.PathLike[str], metadata: DatasetMetadata):
        self.root = Path(root)
        if self.root.exists():
            raise ImmutableDatasetError(
                f"refusing to create trajectory dataset over existing path {self.root}; use resume() for a partial dataset"
            )
        self.root.mkdir(parents=True)
        (self.root / CHUNKS_DIR).mkdir()
        metadata_bytes = canonical_json_bytes(metadata.to_dict())
        _write_once_bytes(self.root / METADATA_FILE, metadata_bytes)
        metadata_record = {
            "file": METADATA_FILE,
            "bytes": len(metadata_bytes),
            "sha256": _sha256_bytes(metadata_bytes),
        }
        _write_once_json(self.root / METADATA_COMMIT_FILE, metadata_record)
        self.metadata = metadata
        self._next_index = 0
        self._finalized = False
        self._sequence = _SequenceState()

    @classmethod
    def resume(
        cls, root: str | os.PathLike[str], *, expected_metadata: DatasetMetadata | None = None
    ) -> "TrajectoryWriter":
        """Continue a clean partial dataset without changing its metadata."""
        root = Path(root)
        metadata = load_metadata(root)
        if expected_metadata is not None and canonical_json_bytes(metadata.to_dict()) != canonical_json_bytes(
            expected_metadata.to_dict()
        ):
            raise ImmutableDatasetError("partial dataset metadata differs from expected metadata")
        report = validate_dataset(root, allow_partial=True)
        if report.complete:
            raise ImmutableDatasetError(f"{root} is finalized and cannot be appended")
        blockers = tuple(reason for reason in report.partial_reasons if reason != f"{MANIFEST_FILE} is absent")
        if blockers:
            raise PartialDatasetError(
                f"cannot resume {root} until interrupted artifacts are reviewed: {'; '.join(blockers)}"
            )
        writer = cls.__new__(cls)
        writer.root = root
        writer.metadata = metadata
        writer._next_index = report.chunks
        writer._finalized = False
        descriptors, _ = _scan_chunks(root)
        writer._sequence = _replay_sequence(root, metadata.schema, descriptors)
        return writer

    def append(self, arrays: Mapping[str, Any]) -> dict[str, Any]:
        """Validate and atomically commit one bounded chunk."""
        if self._finalized or (self.root / MANIFEST_FILE).exists():
            raise ImmutableDatasetError("a finalized trajectory dataset cannot be appended")
        values = _as_numpy_arrays(arrays)
        records = validate_chunk_arrays(self.metadata.schema, values)
        index = self._next_index
        sequence = _advance_sequence(self._sequence, values, where=f"chunk {index}")
        chunks_dir = self.root / CHUNKS_DIR
        payload_path = chunks_dir / f"{index:08d}.npz"
        descriptor_path = chunks_dir / f"{index:08d}.json"
        if payload_path.exists() or descriptor_path.exists():
            raise ImmutableDatasetError(f"chunk {index} already exists; refusing to replace it")

        temp = chunks_dir / f".{index:08d}.npz.tmp-{uuid.uuid4().hex}"
        try:
            with temp.open("xb") as fh:
                ordered = {
                    field.name: _little_endian_contiguous(values[field.name])
                    for field in self.metadata.schema.fields
                }
                np.savez_compressed(fh, **ordered)
                fh.flush()
                os.fsync(fh.fileno())
            try:
                os.link(temp, payload_path)
            except FileExistsError as exc:
                raise ImmutableDatasetError(f"chunk {index} was concurrently committed") from exc
            _fsync_directory(chunks_dir)
        finally:
            try:
                temp.unlink()
            except FileNotFoundError:
                pass

        descriptor = _chunk_descriptor(index, records, payload_path)
        _write_once_json(descriptor_path, descriptor)
        self._next_index += 1
        self._sequence = sequence
        return descriptor

    def finalize(self) -> ValidationReport:
        """Write the final manifest last; its presence is the completion marker."""
        if self._finalized or (self.root / MANIFEST_FILE).exists():
            raise ImmutableDatasetError("trajectory dataset is already finalized")
        report = validate_dataset(self.root, allow_partial=True)
        blockers = tuple(reason for reason in report.partial_reasons if reason != f"{MANIFEST_FILE} is absent")
        if blockers:
            raise PartialDatasetError(f"cannot finalize while partial artifacts exist: {'; '.join(blockers)}")
        if report.chunks == 0:
            raise PartialDatasetError("cannot finalize a trajectory dataset with no chunks")
        metadata_record = _read_json(self.root / METADATA_COMMIT_FILE)
        descriptors, _ = _scan_chunks(self.root)
        _write_once_json(self.root / MANIFEST_FILE, _manifest(metadata_record, descriptors))
        self._finalized = True
        return validate_dataset(self.root)

    def __enter__(self) -> "TrajectoryWriter":
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if exc_type is None and not self._finalized:
            self.finalize()
        return False


class TrajectoryReader:
    """Streaming reader; ``iter_chunks`` never materializes the whole dataset."""

    def __init__(self, root: str | os.PathLike[str], *, allow_partial: bool = False):
        self.root = Path(root)
        self.metadata = load_metadata(self.root)
        # Verify hashes and structure now; arrays are validated lazily per chunk.
        self.report = validate_dataset(self.root, allow_partial=allow_partial, verify_arrays=False)
        self._descriptors, _ = _scan_chunks(self.root)

    def iter_chunks(self) -> Iterator[dict[str, np.ndarray]]:
        sequence = _SequenceState()
        for descriptor in self._descriptors:
            arrays = _load_chunk(self.root, self.metadata.schema, descriptor)
            sequence = _advance_sequence_or_integrity(
                sequence, arrays, chunk_index=descriptor["index"]
            )
            yield arrays

    def read_all(self) -> dict[str, np.ndarray]:
        """Explicit convenience for small datasets; streaming is the default."""
        chunks = list(self.iter_chunks())
        if not chunks:
            return {
                field.name: np.empty((0, *self.metadata.schema.resolved_shape(field)), dtype=field.dtype)
                for field in self.metadata.schema.fields
            }
        return {
            field.name: np.concatenate([chunk[field.name] for chunk in chunks], axis=0)
            for field in self.metadata.schema.fields
        }
