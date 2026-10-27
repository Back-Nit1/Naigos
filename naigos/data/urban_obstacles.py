"""Physics-grade urban obstacles: the versioned, immutable artifact the env may collide with.

Contract: ``docs/contracts/urban-obstacles-v1.md``. Read it before changing anything here.

This module is framework-neutral on purpose: numpy and the standard library only. It imports no
JAX, Flax, RL, viewer or provider code, makes no network call, and never reads presentation data
(``data_cache/visual/``). A consumer converts the arrays with ``jnp.asarray`` itself.

Three rules the loader enforces and nothing else may relax:

  * No-data is NaN under ``coverage == 0``, never zero. A cell with no coverage is *unknown*; it
    is not open air and it is not a zero-height building.
  * Every byte is hashed. A partial, edited or re-labelled artifact is refused, not repaired.
  * Loading the same bytes yields the same typed object.

Grid convention (same as ``naigos.data.enu.ENUGrid``): arrays are ``(ny, nx)``, row 0 is the
SOUTH row, column 0 is the WEST column, ``+x`` east, ``+y`` north. ``grid_origin_enu_m`` is the
south-west corner of cell ``(0, 0)`` in the environment's ENU frame, whose own origin is the
absolute UTM point ``env_origin_utm_m``. Cell ``(r, c)`` covers
``[x0 + c*cell, x0 + (c+1)*cell) x [y0 + r*cell, y0 + (r+1)*cell)``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

SCHEMA_ID = "naigos.urban_obstacles"
SCHEMA_VERSION = 1
#: Namespace under the research cache root. Deliberately not ``visual/``.
CACHE_NAMESPACE = "urban_physics"
VERTICAL_DATUM = "AMSL"

PAYLOAD_FILE = "payload.bin"
METADATA_FILE = "metadata.json"
DESCRIPTOR_FILE = "descriptor.json"
COMPLETE_FILE = "COMPLETE"

#: Fixed payload order. ``uncertainty_m`` is last and present only when the metadata says so.
_ARRAYS = (
    ("ground_amsl_m", "<f4"),
    ("top_amsl_m", "<f4"),
    ("height_agl_m", "<f4"),
    ("coverage", "|u1"),
    ("building", "|u1"),
    ("uncertainty_m", "<f4"),
)
_FLOAT_FIELDS = ("ground_amsl_m", "top_amsl_m", "height_agl_m", "uncertainty_m")

#: A cell whose recorded height disagrees with ``top - ground`` by more than this is corrupt.
HEIGHT_CONSISTENCY_TOL_M = 1e-3
#: Hard ceiling on one artifact, so a malformed header cannot ask for an unbounded array.
MAX_CELLS = 16_000_000
MAX_HEIGHT_M = 1_000.0

NODATA_SEMANTICS = (
    "coverage==0 cells are UNKNOWN: ground, top, height and uncertainty are NaN there and the "
    "cell is not a zero-height building. coverage==1 cells are finite; building==0 means "
    "surveyed open (top==ground), building==1 means an obstacle (height>0). Consumers must "
    "choose their own explicit policy for unknown cells."
)

_SHA = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^([0-9a-f]{40}|unknown)$")


class UrbanObstacleError(ValueError):
    """The artifact or its metadata violates the contract. Never caught and defaulted."""


class UrbanObstacleUnavailable(UrbanObstacleError):
    """The artifact is absent or incomplete. A theatre with urban_physics must fail on this."""


# --- typed metadata ---------------------------------------------------------------------


@dataclass(frozen=True)
class UrbanSource:
    """One cited input. Capture, publication and retrieval are three different dates."""

    key: str
    name: str
    product_id: str
    role: str  # "ground" | "structures" | "footprints"
    licence: str
    licence_url: str
    citation: str
    url: str
    captured: str  # acquisition/survey date or range of the underlying data, "unknown" if so
    published: str  # publisher's release/update date, "unknown" if so
    retrieved: str  # when this repository fetched it
    sha256: str  # of the exact cached input bytes
    redistribution: str  # "repository" (derived data committed) | "cache_only" (never committed)

    def __post_init__(self):
        for f in ("key", "name", "product_id", "role", "licence", "citation", "url",
                  "captured", "published", "retrieved", "redistribution"):
            if not isinstance(getattr(self, f), str) or not getattr(self, f).strip():
                raise UrbanObstacleError(f"source {self.key!r}: {f} must be a non-empty string")
        if not _SHA.match(self.sha256):
            raise UrbanObstacleError(f"source {self.key!r}: sha256 must be 64 lowercase hex")
        if self.redistribution not in ("repository", "cache_only"):
            raise UrbanObstacleError(f"source {self.key!r}: bad redistribution {self.redistribution!r}")

    def as_dict(self) -> dict[str, Any]:
        return {f: getattr(self, f) for f in _SOURCE_FIELDS}


_SOURCE_FIELDS = ("key", "name", "product_id", "role", "licence", "licence_url", "citation",
                  "url", "captured", "published", "retrieved", "sha256", "redistribution")


@dataclass(frozen=True)
class BuildInfo:
    code_commit: str  # 40-hex git commit of the builder, or "unknown"
    dirty: bool  # the tree had uncommitted changes when it was built
    built_at: str  # ISO-8601 UTC, passed in so a rebuild can be byte-identical

    def __post_init__(self):
        if not _COMMIT.match(self.code_commit):
            raise UrbanObstacleError(f"code_commit {self.code_commit!r} is not 40-hex or 'unknown'")
        if not isinstance(self.dirty, bool):
            raise UrbanObstacleError("dirty must be a bool")
        if not self.built_at:
            raise UrbanObstacleError("built_at is required")


@dataclass(frozen=True)
class Exclusion:
    """A protected-zone box whose cells were set to no-data. Name and box only."""

    name: str
    bbox_wgs84: tuple[float, float, float, float]
    cells: int  # number of grid cells set to no-data by this zone


@dataclass(frozen=True)
class UrbanObstacleMeta:
    aoi_name: str
    aoi_fingerprint: str
    aoi_bounds_wgs84: tuple[float, float, float, float]  # west, south, east, north
    physics_bounds_wgs84: tuple[float, float, float, float]  # the documented urban physics box
    grid_corners_wgs84: tuple[tuple[float, float], ...]  # (lon, lat) SW, SE, NE, NW of the grid
    utm_epsg: int
    env_origin_utm_m: tuple[float, float]  # absolute UTM (E, N) of the env ENU origin
    grid_origin_enu_m: tuple[float, float]  # ENU (x, y) of the SW corner of cell (0, 0)
    cell_m: float
    nx: int
    ny: int
    sources: tuple[UrbanSource, ...]
    build: BuildInfo
    uncertainty_present: bool
    uncertainty_kind: str  # what uncertainty_m means, or "none"
    exclusions: tuple[Exclusion, ...] = ()
    notes: str = ""

    @property
    def n_cells(self) -> int:
        return self.nx * self.ny

    def as_dict(self) -> dict[str, Any]:
        """The canonical, hash-input form. Field order is irrelevant (keys are sorted)."""
        return {
            "schema": SCHEMA_ID, "schema_version": SCHEMA_VERSION,
            "aoi_name": self.aoi_name, "aoi_fingerprint": self.aoi_fingerprint,
            "aoi_bounds_wgs84": list(self.aoi_bounds_wgs84),
            "physics_bounds_wgs84": list(self.physics_bounds_wgs84),
            "grid_corners_wgs84": [list(c) for c in self.grid_corners_wgs84],
            "utm_epsg": self.utm_epsg,
            "env_origin_utm_m": list(self.env_origin_utm_m),
            "grid_origin_enu_m": list(self.grid_origin_enu_m),
            "cell_m": self.cell_m, "nx": self.nx, "ny": self.ny,
            "vertical_datum": VERTICAL_DATUM,
            "axes": "arrays (ny, nx); row 0 = south, col 0 = west; +x east, +y north",
            "nodata": NODATA_SEMANTICS,
            "sources": [s.as_dict() for s in self.sources],
            "build": {"code_commit": self.build.code_commit, "dirty": self.build.dirty,
                      "built_at": self.build.built_at},
            "uncertainty": {"present": self.uncertainty_present, "kind": self.uncertainty_kind},
            "exclusions": [{"name": e.name, "bbox_wgs84": list(e.bbox_wgs84), "cells": e.cells}
                           for e in self.exclusions],
            "notes": self.notes,
            "payload": _payload_layout(self.nx, self.ny, self.uncertainty_present),
        }


@dataclass(frozen=True)
class ArtifactHashes:
    metadata_sha256: str
    payload_sha256: str
    content_sha256: str

    @property
    def content_id(self) -> str:
        return self.content_sha256[:16]


@dataclass(frozen=True, eq=False)
class UrbanObstacleGrid:
    """The immutable artifact. Arrays are read-only ``(ny, nx)`` NumPy views."""

    meta: UrbanObstacleMeta
    ground_amsl_m: np.ndarray  # float32, bare earth, NaN where coverage == 0
    top_amsl_m: np.ndarray  # float32, obstacle top, NaN where coverage == 0
    height_agl_m: np.ndarray  # float32, top - ground, NaN where coverage == 0
    coverage: np.ndarray  # uint8 0/1
    building: np.ndarray  # uint8 0/1, subset of coverage
    uncertainty_m: np.ndarray | None  # float32 1-sigma-style, NaN where coverage == 0
    hashes: ArtifactHashes

    @property
    def shape(self) -> tuple[int, int]:
        return (self.meta.ny, self.meta.nx)

    @property
    def coverage_fraction(self) -> float:
        return float(self.coverage.mean())

    @property
    def building_cells(self) -> int:
        return int(self.building.sum())

    def top_with_nodata(self, no_data_top_amsl_m: float) -> np.ndarray:
        """``top_amsl_m`` with unknown cells replaced by an explicit, caller-chosen value.

        There is no default: choosing -inf (open) or +inf (blocked) is a safety decision the
        consumer must make visibly. The result is a fresh writable float32 array.
        """
        if math.isnan(no_data_top_amsl_m):
            raise UrbanObstacleError("no_data_top_amsl_m must be a number, not NaN")
        return np.where(self.coverage == 1, self.top_amsl_m, np.float32(no_data_top_amsl_m)
                        ).astype(np.float32)

    def origin_in_env(self, env_origin_utm_m: Sequence[float]) -> tuple[float, float]:
        """ENU ``(x, y)`` of cell (0, 0)'s SW corner in an env whose ENU origin is this UTM point.

        The env origin depends on the env's grid shape and cell size, so a consumer whose origin
        differs from the one recorded at build time re-expresses the grid here. Exact: both
        frames are the same UTM zone, so it is a translation.
        """
        m = self.meta
        return (m.env_origin_utm_m[0] + m.grid_origin_enu_m[0] - float(env_origin_utm_m[0]),
                m.env_origin_utm_m[1] + m.grid_origin_enu_m[1] - float(env_origin_utm_m[1]))

    def evidence(self) -> dict[str, Any]:
        """What a viewer or `/scene` shows about this artifact. No paths, no secrets."""
        m = self.meta
        return {
            "schema": SCHEMA_ID, "schema_version": SCHEMA_VERSION,
            "content_sha256": self.hashes.content_sha256,
            "metadata_sha256": self.hashes.metadata_sha256,
            "payload_sha256": self.hashes.payload_sha256,
            "aoi": m.aoi_name, "aoi_fingerprint": m.aoi_fingerprint,
            "cell_m": m.cell_m, "nx": m.nx, "ny": m.ny,
            "coverage_fraction": round(self.coverage_fraction, 6),
            "building_cells": self.building_cells,
            "max_height_agl_m": float(np.nanmax(self.height_agl_m)) if self.coverage.any() else 0.0,
            "sources": [{"key": s.key, "product_id": s.product_id, "licence": s.licence,
                         "captured": s.captured, "published": s.published}
                        for s in m.sources],
            "code_commit": m.build.code_commit, "dirty": m.build.dirty,
        }


@dataclass(frozen=True)
class ExpectedFrame:
    """What the caller (the theatre) requires the artifact to be georeferenced to."""

    aoi_name: str
    aoi_fingerprint: str
    utm_epsg: int
    aoi_bounds_wgs84: tuple[float, float, float, float]
    #: When set, the artifact must have been built against exactly this env ENU origin.
    env_origin_utm_m: tuple[float, float] | None = None
    #: When set, the artifact must have been built from exactly this content hash.
    content_sha256: str | None = None


# --- hashing and layout -----------------------------------------------------------------


def canonical_json(obj: Any) -> bytes:
    """Deterministic bytes: sorted keys, no whitespace, ASCII-only, no NaN/Infinity."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("ascii")


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _payload_layout(nx: int, ny: int, with_uncertainty: bool) -> list[dict[str, Any]]:
    layout, offset = [], 0
    for name, dt in _ARRAYS:
        if name == "uncertainty_m" and not with_uncertainty:
            continue
        nbytes = nx * ny * np.dtype(dt).itemsize
        layout.append({"name": name, "dtype": dt, "shape": [ny, nx], "offset": offset,
                       "nbytes": nbytes})
        offset += nbytes
    return layout


def _content_sha(metadata_sha: str, payload_sha: str) -> str:
    return _sha(f"{SCHEMA_ID}/{SCHEMA_VERSION}\n{metadata_sha}\n{payload_sha}\n".encode("ascii"))


def _payload_bytes(arrays: Mapping[str, np.ndarray], with_uncertainty: bool) -> bytes:
    parts = []
    for name, dt in _ARRAYS:
        if name == "uncertainty_m" and not with_uncertainty:
            continue
        parts.append(np.ascontiguousarray(arrays[name], dtype=dt).tobytes(order="C"))
    return b"".join(parts)


# --- validation -------------------------------------------------------------------------


def _bounds(name: str, b: Sequence[float]) -> tuple[float, float, float, float]:
    try:
        w, s, e, n = (float(v) for v in b)
    except (TypeError, ValueError):
        raise UrbanObstacleError(f"{name}: expected 4 numbers") from None
    if not all(map(math.isfinite, (w, s, e, n))) or not (w < e and s < n):
        raise UrbanObstacleError(f"{name}: malformed bounds {b!r}")
    if not (-180 <= w and e <= 180 and -90 <= s and n <= 90):
        raise UrbanObstacleError(f"{name}: not WGS84 degrees: {b!r}")
    return (w, s, e, n)


def _inside(inner: Sequence[float], outer: Sequence[float]) -> bool:
    return (inner[0] >= outer[0] and inner[1] >= outer[1]
            and inner[2] <= outer[2] and inner[3] <= outer[3])


def _validate_meta(m: UrbanObstacleMeta) -> None:
    if not m.aoi_name or not re.match(r"^[a-z0-9_]+$", m.aoi_name):
        raise UrbanObstacleError(f"bad aoi_name {m.aoi_name!r}")
    if not re.match(r"^[0-9a-f]{10}$", m.aoi_fingerprint):
        raise UrbanObstacleError(f"bad aoi_fingerprint {m.aoi_fingerprint!r}")
    aoi = _bounds("aoi_bounds_wgs84", m.aoi_bounds_wgs84)
    phys = _bounds("physics_bounds_wgs84", m.physics_bounds_wgs84)
    if not _inside(phys, aoi):
        raise UrbanObstacleError("physics_bounds_wgs84 is not inside aoi_bounds_wgs84")
    if len(m.grid_corners_wgs84) != 4:
        raise UrbanObstacleError("grid_corners_wgs84 must hold 4 (lon, lat) corners")
    # The grid must lie inside the documented physics box, to a cell of slack for projection.
    slack = m.cell_m / 90_000.0
    for lon, lat in m.grid_corners_wgs84:
        if not (phys[0] - slack <= lon <= phys[2] + slack and phys[1] - slack <= lat <= phys[3] + slack):
            raise UrbanObstacleError("grid corner lies outside physics_bounds_wgs84")
    if not (32601 <= m.utm_epsg <= 32660 or 32701 <= m.utm_epsg <= 32760):
        raise UrbanObstacleError(f"utm_epsg {m.utm_epsg} is not a WGS84 UTM zone")
    for name, v in (("env_origin_utm_m", m.env_origin_utm_m), ("grid_origin_enu_m", m.grid_origin_enu_m)):
        if len(v) != 2 or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in v):
            raise UrbanObstacleError(f"{name} must be two finite numbers")
    if not (isinstance(m.cell_m, (int, float)) and math.isfinite(m.cell_m) and 0.5 <= m.cell_m <= 100.0):
        raise UrbanObstacleError(f"cell_m {m.cell_m!r} outside [0.5, 100] m")
    if not (isinstance(m.nx, int) and isinstance(m.ny, int) and m.nx >= 2 and m.ny >= 2
            and m.nx * m.ny <= MAX_CELLS):
        raise UrbanObstacleError(f"malformed dimensions nx={m.nx!r} ny={m.ny!r}")
    if not m.sources:
        raise UrbanObstacleError("an artifact with no cited sources is refused")
    if len({s.key for s in m.sources}) != len(m.sources):
        raise UrbanObstacleError("duplicate source keys")
    if m.uncertainty_present == (m.uncertainty_kind == "none"):
        raise UrbanObstacleError("uncertainty_kind must be 'none' iff uncertainty is absent")
    for e in m.exclusions:
        _bounds(f"exclusion {e.name!r}", e.bbox_wgs84)
        if e.cells < 0:
            raise UrbanObstacleError(f"exclusion {e.name!r}: negative cell count")


def _validate_arrays(m: UrbanObstacleMeta, a: Mapping[str, np.ndarray | None]) -> None:
    shape = (m.ny, m.nx)
    want = {n: dt for n, dt in _ARRAYS}
    names = [n for n, _ in _ARRAYS if n != "uncertainty_m" or m.uncertainty_present]
    if not m.uncertainty_present and a.get("uncertainty_m") is not None:
        raise UrbanObstacleError("uncertainty_m given but metadata says it is absent")
    for n in names:
        arr = a.get(n)
        if not isinstance(arr, np.ndarray) or arr.shape != shape:
            raise UrbanObstacleError(f"{n}: expected ndarray of shape {shape}, got "
                                     f"{getattr(arr, 'shape', type(arr).__name__)}")
        if arr.dtype != np.dtype(want[n]):
            raise UrbanObstacleError(f"{n}: dtype {arr.dtype} != {np.dtype(want[n])}")
    cov, bld = a["coverage"], a["building"]
    if not (np.isin(cov, (0, 1)).all() and np.isin(bld, (0, 1)).all()):
        raise UrbanObstacleError("coverage and building must be 0/1")
    if (bld.astype(bool) & ~cov.astype(bool)).any():
        raise UrbanObstacleError("building cell without coverage")
    covered = cov.astype(bool)
    if not covered.any():
        raise UrbanObstacleError("artifact has no covered cell")
    for n in [n for n in _FLOAT_FIELDS if n in names]:
        arr = a[n]
        if np.isinf(arr).any():
            raise UrbanObstacleError(f"{n}: infinite value")
        if not np.isfinite(arr[covered]).all():
            raise UrbanObstacleError(f"{n}: non-finite value in a covered cell")
        if not np.isnan(arr[~covered]).all():
            raise UrbanObstacleError(f"{n}: no-data cells must be NaN, found a number")
    g, t, h = (a[n][covered].astype(np.float64) for n in ("ground_amsl_m", "top_amsl_m", "height_agl_m"))
    if (t < g).any():
        raise UrbanObstacleError("obstacle top below ground")
    if (h < 0).any() or (h > MAX_HEIGHT_M).any():
        raise UrbanObstacleError(f"height_agl_m outside [0, {MAX_HEIGHT_M}] m")
    if (np.abs((t - g) - h) > HEIGHT_CONSISTENCY_TOL_M).any():
        raise UrbanObstacleError("height_agl_m disagrees with top - ground")
    isb = bld[covered].astype(bool)
    if (h[isb] <= 0).any():
        raise UrbanObstacleError("building cell with non-positive height")
    if (h[~isb] > HEIGHT_CONSISTENCY_TOL_M).any():
        raise UrbanObstacleError("non-building covered cell carries height")
    if m.uncertainty_present and (a["uncertainty_m"][covered] < 0).any():
        raise UrbanObstacleError("negative uncertainty")
    if sum(e.cells for e in m.exclusions) > int((~covered).sum()):
        raise UrbanObstacleError("exclusions claim more no-data cells than exist")


def _frozen(arr: np.ndarray) -> np.ndarray:
    out = np.array(arr, copy=True, order="C")
    out.setflags(write=False)
    return out


def build_urban_obstacle_grid(meta: UrbanObstacleMeta, *, ground_amsl_m: np.ndarray,
                              top_amsl_m: np.ndarray, coverage: np.ndarray,
                              building: np.ndarray,
                              uncertainty_m: np.ndarray | None = None) -> UrbanObstacleGrid:
    """Validate and freeze an artifact in memory, computing every hash.

    Inputs are float arrays with NaN under ``coverage == 0``. ``height_agl_m`` is derived as
    ``top - ground`` in float32 so the stored value is exactly reproducible.
    """
    _validate_meta(meta)
    cov = np.asarray(coverage).astype("|u1")
    g = np.asarray(ground_amsl_m).astype("<f4")
    t = np.asarray(top_amsl_m).astype("<f4")
    with np.errstate(invalid="ignore"):
        h = (t.astype("<f8") - g.astype("<f8")).astype("<f4")
    arrays: dict[str, np.ndarray | None] = {
        "ground_amsl_m": g, "top_amsl_m": t, "height_agl_m": h, "coverage": cov,
        "building": np.asarray(building).astype("|u1"),
        "uncertainty_m": None if uncertainty_m is None else np.asarray(uncertainty_m).astype("<f4"),
    }
    _validate_arrays(meta, arrays)
    meta_sha = _sha(canonical_json(meta.as_dict()))
    payload_sha = _sha(_payload_bytes(arrays, meta.uncertainty_present))
    hashes = ArtifactHashes(meta_sha, payload_sha, _content_sha(meta_sha, payload_sha))
    return UrbanObstacleGrid(
        meta=meta, **{k: (None if v is None else _frozen(v)) for k, v in arrays.items()},
        hashes=hashes)


# --- on-disk artifact -------------------------------------------------------------------


def artifact_dir(cache_root: str | os.PathLike, aoi_name: str, content_sha256: str) -> Path:
    """``<cache_root>/urban_physics/<aoi>/<content-id>/``; never under ``visual/``."""
    return Path(cache_root) / CACHE_NAMESPACE / aoi_name / content_sha256[:16]


def write_urban_obstacles(grid: UrbanObstacleGrid, cache_root: str | os.PathLike) -> Path:
    """Publish payload, then metadata, then descriptor, then the completion marker, then reopen.

    A reader that finds no ``COMPLETE`` refuses the directory, so an interrupted write is never
    mistaken for an artifact. Re-publishing identical content is a no-op; an incomplete
    directory for the same content id is replaced.
    """
    m = grid.meta
    dest = artifact_dir(cache_root, m.aoi_name, grid.hashes.content_sha256)
    if (dest / COMPLETE_FILE).exists():
        reopened = load_urban_obstacles(dest)
        if reopened.hashes != grid.hashes:
            raise UrbanObstacleError(f"{dest} is complete but holds different content")
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    for stale in dest.iterdir():
        stale.unlink()  # only ever an unfinished write: COMPLETE is absent
    arrays = {n: getattr(grid, n) for n, _ in _ARRAYS}
    (dest / PAYLOAD_FILE).write_bytes(_payload_bytes(arrays, m.uncertainty_present))
    (dest / METADATA_FILE).write_bytes(canonical_json(m.as_dict()))
    descriptor = {
        "schema": SCHEMA_ID, "schema_version": SCHEMA_VERSION,
        "metadata_sha256": grid.hashes.metadata_sha256,
        "payload_sha256": grid.hashes.payload_sha256,
        "content_sha256": grid.hashes.content_sha256,
        "files": {PAYLOAD_FILE: _layout_bytes(m), METADATA_FILE: (dest / METADATA_FILE).stat().st_size},
    }
    (dest / DESCRIPTOR_FILE).write_bytes(canonical_json(descriptor))
    (dest / COMPLETE_FILE).write_text(grid.hashes.content_sha256 + "\n")
    reopened = load_urban_obstacles(dest)
    if reopened.hashes != grid.hashes:  # pragma: no cover - would mean a writer bug
        raise UrbanObstacleError("reopened artifact does not match what was written")
    return dest


def _layout_bytes(m: UrbanObstacleMeta) -> int:
    return sum(item["nbytes"] for item in _payload_layout(m.nx, m.ny, m.uncertainty_present))


def _meta_from_dict(d: Mapping[str, Any]) -> UrbanObstacleMeta:
    try:
        if d["schema"] != SCHEMA_ID:
            raise UrbanObstacleError(f"schema {d['schema']!r} is not {SCHEMA_ID!r}")
        if d["schema_version"] != SCHEMA_VERSION:
            raise UrbanObstacleError(
                f"unknown schema_version {d['schema_version']!r}; this loader reads {SCHEMA_VERSION}")
        if d["vertical_datum"] != VERTICAL_DATUM:
            raise UrbanObstacleError(f"vertical_datum {d['vertical_datum']!r} != {VERTICAL_DATUM}")
        if d["nodata"] != NODATA_SEMANTICS:
            raise UrbanObstacleError("no-data semantics differ from this schema version")
        b = d["build"]
        meta = UrbanObstacleMeta(
            aoi_name=d["aoi_name"], aoi_fingerprint=d["aoi_fingerprint"],
            aoi_bounds_wgs84=tuple(d["aoi_bounds_wgs84"]),
            physics_bounds_wgs84=tuple(d["physics_bounds_wgs84"]),
            grid_corners_wgs84=tuple(tuple(c) for c in d["grid_corners_wgs84"]),
            utm_epsg=d["utm_epsg"], env_origin_utm_m=tuple(d["env_origin_utm_m"]),
            grid_origin_enu_m=tuple(d["grid_origin_enu_m"]), cell_m=d["cell_m"],
            nx=d["nx"], ny=d["ny"],
            sources=tuple(UrbanSource(**s) for s in d["sources"]),
            build=BuildInfo(b["code_commit"], b["dirty"], b["built_at"]),
            uncertainty_present=d["uncertainty"]["present"], uncertainty_kind=d["uncertainty"]["kind"],
            exclusions=tuple(Exclusion(e["name"], tuple(e["bbox_wgs84"]), e["cells"])
                             for e in d["exclusions"]),
            notes=d["notes"],
        )
    except (KeyError, TypeError) as exc:
        raise UrbanObstacleError(f"malformed metadata: {exc!r}") from None
    if d["payload"] != _payload_layout(meta.nx, meta.ny, meta.uncertainty_present):
        raise UrbanObstacleError("payload layout does not match nx, ny and dtype table")
    return meta


def load_urban_obstacles(path: str | os.PathLike,
                         expected: ExpectedFrame | None = None) -> UrbanObstacleGrid:
    """Read and fully verify one artifact directory. Strict; there is no lenient mode.

    ``expected`` ties the artifact to a theatre. Pass it whenever a theatre is loading: without
    it only the artifact's internal consistency is proven, not that it belongs to that AOI.
    """
    d = Path(path)
    if not d.is_dir():
        raise UrbanObstacleUnavailable(f"{d}: no such artifact directory")
    marker = d / COMPLETE_FILE
    missing = [f for f in (PAYLOAD_FILE, METADATA_FILE, DESCRIPTOR_FILE) if not (d / f).is_file()]
    if not marker.is_file() or missing:
        raise UrbanObstacleUnavailable(
            f"{d}: incomplete artifact (missing {missing or []}{'' if marker.is_file() else ' COMPLETE'})")
    names = {p.name for p in d.iterdir()}
    if names != {PAYLOAD_FILE, METADATA_FILE, DESCRIPTOR_FILE, COMPLETE_FILE}:
        raise UrbanObstacleError(f"{d}: unexpected files {sorted(names)}")

    meta_bytes = (d / METADATA_FILE).read_bytes()
    try:
        desc = json.loads((d / DESCRIPTOR_FILE).read_bytes())
        meta_doc = json.loads(meta_bytes)
    except ValueError as exc:
        raise UrbanObstacleError(f"{d}: unreadable JSON: {exc}") from None
    if not isinstance(desc, dict) or not isinstance(meta_doc, dict):
        raise UrbanObstacleError(f"{d}: descriptor and metadata must be JSON objects")
    if desc.get("schema") != SCHEMA_ID or desc.get("schema_version") != SCHEMA_VERSION:
        raise UrbanObstacleError(f"{d}: unknown descriptor schema {desc.get('schema')!r} "
                                 f"v{desc.get('schema_version')!r}")
    if meta_bytes != canonical_json(meta_doc):
        raise UrbanObstacleError(f"{d}: metadata.json is not in canonical form")
    meta_sha = _sha(meta_bytes)
    if meta_sha != desc.get("metadata_sha256"):
        raise UrbanObstacleError(f"{d}: metadata hash mismatch")
    payload = (d / PAYLOAD_FILE).read_bytes()
    payload_sha = _sha(payload)
    if payload_sha != desc.get("payload_sha256"):
        raise UrbanObstacleError(f"{d}: payload hash mismatch")
    content = _content_sha(meta_sha, payload_sha)
    if content != desc.get("content_sha256") or marker.read_text().strip() != content:
        raise UrbanObstacleError(f"{d}: content hash or completion marker mismatch")
    if d.name != content[:16]:
        raise UrbanObstacleError(f"{d}: directory name is not the content id {content[:16]}")

    meta = _meta_from_dict(meta_doc)
    if d.parent.name != meta.aoi_name:
        raise UrbanObstacleError(f"{d}: stored under {d.parent.name!r} but built for {meta.aoi_name!r}")
    if len(payload) != _layout_bytes(meta):
        raise UrbanObstacleError(f"{d}: payload is {len(payload)} bytes, expected {_layout_bytes(meta)}")
    arrays: dict[str, np.ndarray | None] = {"uncertainty_m": None}
    for item in _payload_layout(meta.nx, meta.ny, meta.uncertainty_present):
        raw = payload[item["offset"]: item["offset"] + item["nbytes"]]
        arrays[item["name"]] = np.frombuffer(raw, dtype=item["dtype"]).reshape(item["shape"])
    _validate_meta(meta)
    _validate_arrays(meta, arrays)
    if expected is not None:
        _check_expected(meta, content, expected)
    return UrbanObstacleGrid(
        meta=meta, **{k: (None if v is None else _frozen(v)) for k, v in arrays.items()},
        hashes=ArtifactHashes(meta_sha, payload_sha, content))


def _check_expected(meta: UrbanObstacleMeta, content: str, exp: ExpectedFrame) -> None:
    problems = []
    if meta.aoi_name != exp.aoi_name:
        problems.append(f"aoi {meta.aoi_name!r} != {exp.aoi_name!r}")
    if meta.aoi_fingerprint != exp.aoi_fingerprint:
        problems.append(f"aoi fingerprint {meta.aoi_fingerprint} != {exp.aoi_fingerprint}")
    if meta.utm_epsg != exp.utm_epsg:
        problems.append(f"utm_epsg {meta.utm_epsg} != {exp.utm_epsg}")
    if exp.env_origin_utm_m is not None and any(
            abs(a - b) > 1e-6 for a, b in zip(meta.env_origin_utm_m, exp.env_origin_utm_m)):
        problems.append(f"env origin {meta.env_origin_utm_m} != {exp.env_origin_utm_m}")
    if any(abs(a - b) > 1e-9 for a, b in zip(meta.aoi_bounds_wgs84, exp.aoi_bounds_wgs84)):
        problems.append("aoi bounds differ")
    if exp.content_sha256 is not None and exp.content_sha256 != content:
        problems.append("content_sha256 differs from the cited one")
    if problems:
        raise UrbanObstacleError("artifact does not belong to this theatre: " + "; ".join(problems))
