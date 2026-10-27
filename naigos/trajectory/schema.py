"""The declarative Naigos trajectory schema.

Only JSON-compatible metadata is defined here.  There is intentionally no JAX,
Flax, Optax, PyTorch, or environment import: a dataset can be inspected and
validated without installing a training stack.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import re
from datetime import datetime
from typing import Any, Mapping, Sequence

import numpy as np

FORMAT_NAME = "naigos.trajectory"
FORMAT_VERSION = "1.0.0"
SCHEMA_NAME = "naigos.autonomous-routing.transition"
SCHEMA_VERSION = "1.0.0"
SUPPORTED_FORMAT_VERSIONS = frozenset({FORMAT_VERSION})
SUPPORTED_SCHEMA_VERSIONS = frozenset({SCHEMA_VERSION})

# The action labels are a compatibility and safety boundary, not presentation
# text.  A dataset claiming this schema cannot silently add a fourth action.
ACTION_COMPONENTS = ("bank_cmd", "gamma_cmd", "throttle_cmd")

# Stored as uint8 rather than repeated strings in every transition.  Values are
# append-only within schema major version 1.
TERMINATION_CAUSES = {
    "none": 0,
    "reached": 1,
    "shot_down": 2,
    "terrain": 3,
    "out_of_bounds": 4,
    "out_of_fuel": 5,
    "timeout": 6,
    "external_reset": 7,
    "other": 255,
}

_FIELD_RE = re.compile(r"^[a-z][a-z0-9_]*(?:/[a-z][a-z0-9_]*)+$")
_DIM_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{7,64}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ALLOWED_DTYPES = frozenset(
    {
        "bool",
        "int8", "int16", "int32", "int64",
        "uint8", "uint16", "uint32", "uint64",
        "float16", "float32", "float64",
    }
)
_NORMALIZATION_METHODS = frozenset({"identity", "affine", "componentwise", "categorical"})


class SchemaError(ValueError):
    """Metadata or a schema is malformed or violates the trajectory contract."""


class UnsupportedVersionError(SchemaError):
    """The dataset uses a format or schema version this reader does not know."""


def _json_ready(value: Any, path: str = "$") -> Any:
    """Return a plain, finite JSON tree and reject ambiguous Python values."""
    if dataclasses.is_dataclass(value):
        value = dataclasses.asdict(value)
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise SchemaError(f"{path}: JSON object keys must be strings")
            out[key] = _json_ready(item, f"{path}.{key}")
        return out
    if isinstance(value, (tuple, list)):
        return [_json_ready(item, f"{path}[{i}]") for i, item in enumerate(value)]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SchemaError(f"{path}: NaN and infinity are not valid metadata")
        return value
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise SchemaError(f"{path}: {type(value).__name__} is not JSON-compatible")


def canonical_json_bytes(value: Any) -> bytes:
    """Canonical UTF-8 JSON used by every metadata and content digest."""
    ready = _json_ready(value)
    return (json.dumps(ready, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                       allow_nan=False) + "\n").encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _exact_keys(data: Mapping[str, Any], required: set[str], optional: set[str], where: str) -> None:
    missing = required - set(data)
    unknown = set(data) - required - optional
    if missing:
        raise SchemaError(f"{where}: missing keys {sorted(missing)}")
    if unknown:
        raise SchemaError(f"{where}: unknown keys {sorted(unknown)}")


@dataclasses.dataclass(frozen=True)
class FieldSpec:
    """One array stored in every chunk.

    ``shape`` excludes the leading transition/record axis.  Entries are either
    positive literal sizes or names resolved through ``TrajectorySchema``'s
    dimensions.  ``normalization`` describes the bytes actually stored, not a
    recommendation for a future model.
    """

    name: str
    dtype: str
    shape: tuple[str | int, ...]
    unit: str
    description: str
    normalization: Mapping[str, Any]
    components: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not _FIELD_RE.fullmatch(self.name):
            raise SchemaError(f"invalid field name {self.name!r}; use lower-case group/name paths")
        if self.dtype not in _ALLOWED_DTYPES or np.dtype(self.dtype).name != self.dtype:
            raise SchemaError(f"{self.name}: unsupported or non-canonical dtype {self.dtype!r}")
        for axis in self.shape:
            if isinstance(axis, bool) or not isinstance(axis, (str, int)):
                raise SchemaError(f"{self.name}: shape axes must be dimension names or positive integers")
            if isinstance(axis, str) and not _DIM_RE.fullmatch(axis):
                raise SchemaError(f"{self.name}: invalid dimension name {axis!r}")
            if isinstance(axis, int) and axis <= 0:
                raise SchemaError(f"{self.name}: literal dimensions must be positive")
        if not isinstance(self.unit, str) or not self.unit:
            raise SchemaError(f"{self.name}: unit is required (use '1' for dimensionless)")
        if not isinstance(self.description, str) or not self.description.strip():
            raise SchemaError(f"{self.name}: description is required")
        norm = _json_ready(self.normalization, f"field[{self.name}].normalization")
        if not isinstance(norm, dict) or norm.get("method") not in _NORMALIZATION_METHODS:
            raise SchemaError(
                f"{self.name}: normalization.method must be one of {sorted(_NORMALIZATION_METHODS)}"
            )
        method = norm["method"]
        if method == "componentwise":
            entries = norm.get("components")
            if not isinstance(entries, list) or not entries:
                raise SchemaError(f"{self.name}: componentwise normalization needs a non-empty components list")
            for i, entry in enumerate(entries):
                if not isinstance(entry, dict) or not all(
                    isinstance(entry.get(key), str) and entry[key] for key in ("name", "source_unit", "transform")
                ):
                    raise SchemaError(
                        f"{self.name}: normalization component {i} needs name, source_unit, and transform strings"
                    )
        elif method == "categorical":
            values = norm.get("values")
            if not isinstance(values, dict) or not values or any(
                not isinstance(key, str) or isinstance(value, bool) or not isinstance(value, int)
                for key, value in values.items()
            ):
                raise SchemaError(f"{self.name}: categorical normalization needs a string-to-integer values object")
            if len(set(values.values())) != len(values):
                raise SchemaError(f"{self.name}: categorical values must be unique")
        elif method == "affine":
            if "offset" not in norm or "scale" not in norm:
                raise SchemaError(f"{self.name}: affine normalization needs offset and scale")
        if any(not isinstance(c, str) or not c for c in self.components):
            raise SchemaError(f"{self.name}: component labels must be non-empty strings")

    def to_dict(self) -> dict[str, Any]:
        out = {
            "name": self.name,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "unit": self.unit,
            "description": self.description,
            "normalization": _json_ready(self.normalization),
        }
        if self.components:
            out["components"] = list(self.components)
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FieldSpec":
        if not isinstance(data, Mapping):
            raise SchemaError("field specification must be an object")
        _exact_keys(
            data,
            {"name", "dtype", "shape", "unit", "description", "normalization"},
            {"components"},
            "field specification",
        )
        if not isinstance(data["shape"], list):
            raise SchemaError(f"field {data['name']!r}: shape must be a list")
        components = data.get("components", [])
        if not isinstance(components, list):
            raise SchemaError(f"field {data['name']!r}: components must be a list")
        return cls(
            name=data["name"],
            dtype=data["dtype"],
            shape=tuple(data["shape"]),
            unit=data["unit"],
            description=data["description"],
            normalization=data["normalization"],
            components=tuple(components),
        )


@dataclasses.dataclass(frozen=True)
class TrajectorySchema:
    name: str
    version: str
    dimensions: tuple[tuple[str, int], ...]
    fields: tuple[FieldSpec, ...]

    def __post_init__(self) -> None:
        # Dimension order has no semantic meaning.  Normalize it so a metadata
        # round trip through canonical JSON preserves typed-object equality.
        object.__setattr__(self, "dimensions", tuple(sorted(self.dimensions)))
        if self.name != SCHEMA_NAME:
            raise SchemaError(f"unsupported schema name {self.name!r}")
        if self.version not in SUPPORTED_SCHEMA_VERSIONS:
            raise UnsupportedVersionError(
                f"unsupported trajectory schema version {self.version!r}; supported: {sorted(SUPPORTED_SCHEMA_VERSIONS)}"
            )
        dimensions = dict(self.dimensions)
        if len(dimensions) != len(self.dimensions):
            raise SchemaError("schema dimension names must be unique")
        for name, size in self.dimensions:
            if not _DIM_RE.fullmatch(name) or isinstance(size, bool) or not isinstance(size, int) or size <= 0:
                raise SchemaError(f"invalid dimension {name!r}={size!r}")
        field_map = {field.name: field for field in self.fields}
        if len(field_map) != len(self.fields):
            raise SchemaError("schema field names must be unique")
        for field in self.fields:
            missing = [axis for axis in field.shape if isinstance(axis, str) and axis not in dimensions]
            if missing:
                raise SchemaError(f"{field.name}: unknown dimensions {missing}")
            if field.components:
                if not field.shape:
                    raise SchemaError(f"{field.name}: scalar fields cannot declare components")
                final = field.shape[-1]
                final_size = dimensions[final] if isinstance(final, str) else final
                if len(field.components) != final_size:
                    raise SchemaError(
                        f"{field.name}: {len(field.components)} labels for final axis of size {final_size}"
                    )
            normalization = _json_ready(field.normalization)
            if normalization["method"] == "componentwise":
                if not field.shape:
                    raise SchemaError(f"{field.name}: scalar fields cannot use componentwise normalization")
                final = field.shape[-1]
                final_size = dimensions[final] if isinstance(final, str) else final
                count = len(normalization["components"])
                if count != final_size:
                    raise SchemaError(
                        f"{field.name}: {count} normalization components for final axis of size {final_size}"
                    )
        self._validate_naigos_contract(field_map, dimensions)

    @staticmethod
    def _validate_naigos_contract(fields: Mapping[str, FieldSpec], dimensions: Mapping[str, int]) -> None:
        required = {
            "episode/id", "step/index",
            "observation/ego", "observation/threats", "observation/threat_mask",
            "observation/friends", "observation/friend_mask",
            "action/requested", "action/executed",
            "cbf/feasible", "cbf/intervention_l2",
            "status/alive", "status/reached", "status/reset", "status/termination_cause",
            "resource/fuel_before", "resource/fuel_after",
            "constraint_cost/aircraft_loss",
        }
        missing = required - set(fields)
        if missing:
            raise SchemaError(f"schema is missing required fields {sorted(missing)}")
        if not any(name.startswith("reward/") for name in fields):
            raise SchemaError("schema must contain raw reward/* fields")
        if not any(name.startswith("constraint_cost/") for name in fields):
            raise SchemaError("schema must contain constraint_cost/* fields separately from rewards")
        if dimensions.get("action") != 3:
            raise SchemaError("the Naigos action dimension must be exactly 3")
        for name in ("action/requested", "action/executed"):
            spec = fields[name]
            if spec.shape != ("agent", "action") or spec.dtype != "float32":
                raise SchemaError(f"{name}: expected float32 [agent, action]")
            if spec.components != ACTION_COMPONENTS:
                raise SchemaError(f"{name}: action labels must be exactly {ACTION_COMPONENTS}")
        expected = {
            "observation/threat_mask": (("agent", "threat_slot"), "bool"),
            "observation/friend_mask": (("agent", "friend_slot"), "bool"),
            "cbf/feasible": (("agent",), "bool"),
            "status/alive": (("agent",), "bool"),
            "status/reached": (("agent",), "bool"),
            "status/termination_cause": (("agent",), "uint8"),
            "status/reset": ((), "bool"),
        }
        for name, (shape, dtype) in expected.items():
            if fields[name].shape != shape or fields[name].dtype != dtype:
                raise SchemaError(f"{name}: expected {dtype} {list(shape)}")

    @property
    def dimension_map(self) -> dict[str, int]:
        return dict(self.dimensions)

    @property
    def field_map(self) -> dict[str, FieldSpec]:
        return {field.name: field for field in self.fields}

    def resolved_shape(self, field: FieldSpec | str) -> tuple[int, ...]:
        spec = self.field_map[field] if isinstance(field, str) else field
        dims = self.dimension_map
        return tuple(dims[axis] if isinstance(axis, str) else axis for axis in spec.shape)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "dimensions": {name: size for name, size in self.dimensions},
            "fields": [field.to_dict() for field in self.fields],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TrajectorySchema":
        if not isinstance(data, Mapping):
            raise SchemaError("schema must be an object")
        _exact_keys(data, {"name", "version", "dimensions", "fields"}, set(), "schema")
        if not isinstance(data["dimensions"], Mapping) or not isinstance(data["fields"], list):
            raise SchemaError("schema dimensions must be an object and fields must be a list")
        return cls(
            name=data["name"],
            version=data["version"],
            dimensions=tuple((name, size) for name, size in data["dimensions"].items()),
            fields=tuple(FieldSpec.from_dict(field) for field in data["fields"]),
        )


@dataclasses.dataclass(frozen=True)
class PolicyIdentity:
    """Identity of the behavior policy that requested the logged actions."""

    policy_id: str
    kind: str
    framework: str
    artifact_sha256: str | None = None
    configuration: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if not _ID_RE.fullmatch(self.policy_id):
            raise SchemaError(f"invalid policy id {self.policy_id!r}")
        if not self.kind or not self.framework:
            raise SchemaError("policy kind and framework are required")
        if self.artifact_sha256 is not None and not _SHA256_RE.fullmatch(self.artifact_sha256):
            raise SchemaError("policy artifact_sha256 must be 64 lower-case hex characters")
        _json_ready(self.configuration, "policy.configuration")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.policy_id,
            "kind": self.kind,
            "framework": self.framework,
            "artifact_sha256": self.artifact_sha256,
            "configuration": _json_ready(self.configuration),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PolicyIdentity":
        if not isinstance(data, Mapping):
            raise SchemaError("policy identity must be an object")
        _exact_keys(data, {"id", "kind", "framework", "artifact_sha256", "configuration"}, set(), "policy")
        return cls(data["id"], data["kind"], data["framework"], data["artifact_sha256"], data["configuration"])


@dataclasses.dataclass(frozen=True)
class Provenance:
    theatre: str
    route: str
    scenario: str
    seed: int
    code_commit: str
    code_dirty: bool
    environment_configuration: Mapping[str, Any]
    policy: PolicyIdentity

    def __post_init__(self) -> None:
        for name, value in (("theatre", self.theatre), ("route", self.route), ("scenario", self.scenario)):
            if not isinstance(value, str) or not value.strip():
                raise SchemaError(f"provenance {name} is required")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or not (0 <= self.seed < 2**64):
            raise SchemaError("provenance seed must be an unsigned 64-bit integer")
        if not _COMMIT_RE.fullmatch(self.code_commit):
            raise SchemaError("code_commit must be a 7-64 character lower-case hexadecimal revision")
        if not isinstance(self.code_dirty, bool):
            raise SchemaError("code_dirty must be true or false")
        _json_ready(self.environment_configuration, "environment.configuration")
        if not isinstance(self.policy, PolicyIdentity):
            raise SchemaError("policy must be a PolicyIdentity")

    @property
    def environment_sha256(self) -> str:
        return _digest(self.environment_configuration)

    def to_dict(self) -> dict[str, Any]:
        config = _json_ready(self.environment_configuration)
        return {
            "theatre": self.theatre,
            "route": self.route,
            "scenario": self.scenario,
            "seed": self.seed,
            "code": {"commit": self.code_commit, "dirty": self.code_dirty},
            "environment": {"configuration": config, "sha256": _digest(config)},
            "policy": self.policy.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Provenance":
        if not isinstance(data, Mapping):
            raise SchemaError("provenance must be an object")
        _exact_keys(
            data, {"theatre", "route", "scenario", "seed", "code", "environment", "policy"}, set(), "provenance"
        )
        code, env = data["code"], data["environment"]
        if not isinstance(code, Mapping) or not isinstance(env, Mapping):
            raise SchemaError("provenance code and environment must be objects")
        _exact_keys(code, {"commit", "dirty"}, set(), "provenance.code")
        _exact_keys(env, {"configuration", "sha256"}, set(), "provenance.environment")
        actual = _digest(env["configuration"])
        if env["sha256"] != actual:
            raise SchemaError(f"environment configuration digest mismatch: recorded {env['sha256']!r}, actual {actual}")
        return cls(
            theatre=data["theatre"], route=data["route"], scenario=data["scenario"], seed=data["seed"],
            code_commit=code["commit"], code_dirty=code["dirty"],
            environment_configuration=env["configuration"], policy=PolicyIdentity.from_dict(data["policy"]),
        )


@dataclasses.dataclass(frozen=True)
class DatasetMetadata:
    dataset_id: str
    created_at: str
    schema: TrajectorySchema
    provenance: Provenance
    description: str = ""

    def __post_init__(self) -> None:
        if not _ID_RE.fullmatch(self.dataset_id):
            raise SchemaError(f"invalid dataset id {self.dataset_id!r}")
        if not isinstance(self.created_at, str):
            raise SchemaError("created_at must be an RFC 3339 UTC string")
        try:
            stamp = datetime.fromisoformat(self.created_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise SchemaError("created_at must be an RFC 3339 timestamp") from exc
        if not self.created_at.endswith("Z") or stamp.utcoffset() is None or stamp.utcoffset().total_seconds() != 0:
            raise SchemaError("created_at must be expressed in UTC with a trailing Z")
        if not isinstance(self.schema, TrajectorySchema) or not isinstance(self.provenance, Provenance):
            raise SchemaError("schema and provenance must use their typed metadata objects")
        if not isinstance(self.description, str):
            raise SchemaError("description must be a string")

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": FORMAT_NAME,
            "format_version": FORMAT_VERSION,
            "dataset_id": self.dataset_id,
            "created_at": self.created_at,
            "description": self.description,
            "encoding": {
                "container": "npz",
                "compression": "deflate",
                "record_axis": 0,
                "byte_order": "little_endian",
            },
            "schema": self.schema.to_dict(),
            "provenance": self.provenance.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DatasetMetadata":
        if not isinstance(data, Mapping):
            raise SchemaError("metadata must be an object")
        _exact_keys(
            data,
            {"format", "format_version", "dataset_id", "created_at", "description", "encoding", "schema", "provenance"},
            set(),
            "metadata",
        )
        if data["format"] != FORMAT_NAME:
            raise UnsupportedVersionError(f"unsupported dataset format {data['format']!r}")
        if data["format_version"] not in SUPPORTED_FORMAT_VERSIONS:
            raise UnsupportedVersionError(
                f"unsupported trajectory format version {data['format_version']!r}; "
                f"supported: {sorted(SUPPORTED_FORMAT_VERSIONS)}"
            )
        expected_encoding = {
            "container": "npz", "compression": "deflate", "record_axis": 0, "byte_order": "little_endian"
        }
        if data["encoding"] != expected_encoding:
            raise SchemaError(f"unsupported encoding {data['encoding']!r}")
        return cls(
            dataset_id=data["dataset_id"],
            created_at=data["created_at"],
            description=data["description"],
            schema=TrajectorySchema.from_dict(data["schema"]),
            provenance=Provenance.from_dict(data["provenance"]),
        )


def _identity(**extra: Any) -> dict[str, Any]:
    return {"method": "identity", **extra}


def _componentwise(components: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {"method": "componentwise", "components": [_json_ready(c) for c in components]}


def _field(
    name: str,
    dtype: str,
    shape: tuple[str | int, ...],
    unit: str,
    description: str,
    normalization: Mapping[str, Any] | None = None,
    components: Sequence[str] = (),
) -> FieldSpec:
    return FieldSpec(name, dtype, shape, unit, description, normalization or _identity(), tuple(components))


def _ego_components(ego_features: int) -> tuple[dict[str, Any], ...]:
    base: list[dict[str, Any]] = [
        {"name": "speed_fraction", "source_unit": "m/s", "transform": "(speed-v_stall)/(v_max-v_stall)"},
        {"name": "altitude_fraction", "source_unit": "m AMSL", "transform": "altitude/ceiling"},
        {"name": "agl_fraction", "source_unit": "m AGL", "transform": "clip(agl/5000,-1,2)"},
        {"name": "fuel_fraction", "source_unit": "kg", "transform": "fuel/fuel_init"},
        {"name": "heading_error_sin", "source_unit": "rad", "transform": "sin(error)"},
        {"name": "heading_error_cos", "source_unit": "rad", "transform": "cos(error)"},
        {"name": "flight_path_angle_fraction", "source_unit": "rad", "transform": "gamma/gamma_max"},
        {"name": "bank_fraction", "source_unit": "rad", "transform": "phi/acos(1/n_max)"},
        {"name": "objective_range_fraction", "source_unit": "m", "transform": "clip(range/90000,0,4)"},
        {"name": "maximum_track_quality", "source_unit": "1", "transform": "identity"},
    ]
    if ego_features == 14:
        for direction in ("forward", "left", "right", "back"):
            base.append(
                {"name": f"edge_distance_{direction}", "source_unit": "m", "transform": "clip(distance/50000,0,2)"}
            )
    elif ego_features != 10:
        raise SchemaError("Naigos schema 1.0 supports the current 10- or 14-component ego observation")
    return tuple(base)


def default_trajectory_schema(
    *,
    n_agents: int,
    ego_features: int,
    threat_slots: int,
    threat_features: int,
    friend_slots: int,
    friend_features: int,
    extra_dimensions: Mapping[str, int] | None = None,
    extra_fields: Sequence[FieldSpec] = (),
) -> TrajectorySchema:
    """Build schema v1 for the current Naigos observation/action contract.

    Optional privileged state and future resources are added through
    ``extra_fields`` (for example ``privileged/global_state`` or
    ``resource/thermal_margin``), with their dimensions in ``extra_dimensions``.
    Once present in a schema they are required in every chunk, avoiding a
    per-chunk optionality ambiguity.
    """
    dims = {
        "agent": n_agents,
        "action": 3,
        "ego_feature": ego_features,
        "threat_slot": threat_slots,
        "threat_feature": threat_features,
        "friend_slot": friend_slots,
        "friend_feature": friend_features,
    }
    for name, size in (extra_dimensions or {}).items():
        if name in dims:
            raise SchemaError(f"extra dimension {name!r} collides with a core dimension")
        dims[name] = size

    threat_base = [
        ("relative_forward", "m", "value/50000"), ("relative_left", "m", "value/50000"),
        ("relative_up", "m", "value/50000"), ("velocity_forward", "m/s", "value/300"),
        ("velocity_left", "m/s", "value/300"), ("slant_range", "m", "value/90000"),
        ("lethal_range", "m", "value/90000"), ("detection_probability", "1", "identity"),
        ("track_quality", "1", "identity"),
    ]
    if threat_features < len(threat_base):
        raise SchemaError("threat_features is smaller than the nine fixed Naigos threat features")
    threat_components = [
        {"name": name, "source_unit": unit, "transform": transform} for name, unit, transform in threat_base
    ] + [
        {"name": f"kind_one_hot_{i}", "source_unit": "1", "transform": "identity"}
        for i in range(threat_features - len(threat_base))
    ]
    friend_base = (
        ("relative_forward", "m", "value/50000"), ("relative_left", "m", "value/50000"),
        ("relative_up", "m", "value/50000"), ("velocity_forward", "m/s", "value/300"),
        ("velocity_left", "m/s", "value/300"), ("range", "m", "value/90000"),
        ("alive", "1", "identity"),
    )
    if friend_features != len(friend_base):
        raise SchemaError("Naigos schema 1.0 expects the current seven-component friend observation")

    fields: list[FieldSpec] = [
        _field("episode/id", "uint64", (), "1", "Stable episode identifier within this dataset."),
        _field("step/index", "uint32", (), "1", "Zero-based action step within the episode."),
        _field(
            "observation/ego", "float32", ("agent", "ego_feature"), "1",
            "Pre-action local ego observation exactly as consumed by the actor.",
            _componentwise(_ego_components(ego_features)),
        ),
        _field(
            "observation/threats", "float32", ("agent", "threat_slot", "threat_feature"), "1",
            "Pre-action padded local threat observations in the ego body frame.",
            _componentwise(threat_components),
        ),
        _field(
            "observation/threat_mask", "bool", ("agent", "threat_slot"), "1",
            "True exactly where a padded threat observation slot is valid.",
        ),
        _field(
            "observation/friends", "float32", ("agent", "friend_slot", "friend_feature"), "1",
            "Pre-action padded local friendly-aircraft observations in the ego body frame.",
            _componentwise(
                [{"name": n, "source_unit": u, "transform": t} for n, u, t in friend_base]
            ),
        ),
        _field(
            "observation/friend_mask", "bool", ("agent", "friend_slot"), "1",
            "True exactly where a padded friendly observation slot is valid.",
        ),
        _field(
            "action/requested", "float32", ("agent", "action"), "1",
            "Action requested by the behavior policy before the CBF filter; bounded to [-1,1].",
            _identity(stored_range=[-1.0, 1.0], physical_mapping="environment_configuration.airframe"),
            ACTION_COMPONENTS,
        ),
        _field(
            "action/executed", "float32", ("agent", "action"), "1",
            "Action executed by the environment after the CBF filter; bounded to [-1,1].",
            _identity(stored_range=[-1.0, 1.0], physical_mapping="environment_configuration.airframe"),
            ACTION_COMPONENTS,
        ),
        _field(
            "cbf/feasible", "bool", ("agent",), "1",
            "Whether the CBF half-space intersection was feasible for this agent and step.",
        ),
        _field(
            "cbf/intervention_l2", "float32", ("agent",), "1",
            "Euclidean norm of executed minus requested normalized action.",
        ),
    ]

    reward_fields = (
        ("progress", "m", "Signed horizontal distance closed on the objective this step."),
        ("exposure", "1", "Maximum detection probability against this aircraft."),
        ("lock_level", "1", "Maximum track quality against this aircraft."),
        ("envelope_dwell", "1", "Indicator for a tracked lethal-envelope dwell."),
        ("arrived", "1", "One on the transition that first reaches the objective."),
        ("shotdown", "1", "One on the transition where the sortie is lost to a threat."),
        ("fuel_used", "kg", "Fuel consumed during this transition."),
        ("g_excess", "1", "Coordinated-turn load factor above one g."),
        ("terrain_violation", "1", "Terrain-floor violation indicator."),
        ("ceiling_violation", "1", "Service-ceiling contact indicator."),
        ("stall_violation", "1", "Stall-speed contact indicator."),
        ("bounds_violation", "1", "Play-area boundary violation indicator."),
        ("out_of_fuel", "1", "Fuel-exhaustion indicator."),
        ("edge_proximity", "1", "Soft play-area edge proximity in [0,1]."),
    )
    fields.extend(_field(f"reward/{name}", "float32", ("agent",), unit, desc) for name, unit, desc in reward_fields)
    fields.extend(
        [
            _field(
                "constraint_cost/aircraft_loss", "float32", ("agent",), "1",
                "CMDP aircraft-loss cost kept separate from raw reward terms; baseline-v0 fuel "
                "exhaustion may repeat while resource-world configurations may make it terminal.",
            ),
            _field("status/alive", "bool", ("agent",), "1", "Post-transition aircraft survival state."),
            _field("status/reached", "bool", ("agent",), "1", "Post-transition objective-reached state."),
            _field(
                "status/reset", "bool", (), "1",
                "True when the pre-action observation in this record came directly from an environment reset.",
            ),
            _field(
                "status/termination_cause", "uint8", ("agent",), "1",
                "Terminal event on this transition; zero for no new terminal event.",
                {"method": "categorical", "values": TERMINATION_CAUSES},
            ),
            _field(
                "resource/fuel_before", "float32", ("agent",), "kg",
                "Fuel state in the pre-action observation.",
            ),
            _field(
                "resource/fuel_after", "float32", ("agent",), "kg",
                "Fuel state after the executed action and environment transition.",
            ),
        ]
    )

    core_names = {field.name for field in fields}
    for field in extra_fields:
        if field.name in core_names:
            raise SchemaError(f"extra field {field.name!r} collides with a core field")
        if not (field.name.startswith("privileged/") or field.name.startswith("resource/")
                or field.name.startswith("reward/") or field.name.startswith("constraint_cost/")
                or field.name.startswith("extension/")):
            raise SchemaError(
                f"extra field {field.name!r} must use privileged/, resource/, reward/, "
                "constraint_cost/, or extension/ namespace"
            )
        fields.append(field)
        core_names.add(field.name)

    return TrajectorySchema(
        name=SCHEMA_NAME,
        version=SCHEMA_VERSION,
        dimensions=tuple(dims.items()),
        fields=tuple(fields),
    )
