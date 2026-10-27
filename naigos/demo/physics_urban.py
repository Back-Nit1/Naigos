"""The viewer's side of the physics-grade urban obstacles (docs/contracts/urban-obstacles-v1.md).

``--visual physics`` is evidence-grade, so when a theatre declares ``urban_physics`` the viewer
draws the SAME validated artifact the theatre cites (``naigos.data.theatre``) -- the surface the
simulation's structure layer is made of -- and nothing else. It never extrudes OSM footprints and
calls them physics, and it never turns a missing artifact into an empty city: the state is
``unavailable`` and the HUD says so.

Three states, always reported in ``/scene`` as ``physics_urban``:

  not_declared   the theatre has no ``urban_physics`` block; there are no physical urban obstacles
  available      a validated artifact; ``/physics_urban`` serves it to the physics page
  unavailable    declared but missing or corrupt; the reason names the fix. The env fails closed
                 on the same condition (``Theatre.urban_obstacles``); the viewer only reports it.

The payload carries no filesystem path and no credential: schema, hashes, resolution, coverage and
the arrays the page needs to rebuild the surface. Presentation geometry never enters it.
"""

from __future__ import annotations

import base64
import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..data import urban_obstacles as uo

PAGE_SCHEMA = "naigos.physics_urban.page/1"
FLAG_COVERED, FLAG_BUILDING = 1, 2
NODATA_DM = -32768
EVIDENCE_NOTE = (
    "The simulation's own structure surface: the validated artifact the theatre cites, drawn "
    "cell for cell. Unknown cells are not drawn and are not open air. Presentation buildings "
    "(OSM, provider tiles) are a different layer and are never this."
)
UNAVAILABLE_NOTE = ("physics urban obstacles unavailable: no validated artifact is attached, so no "
                    "urban structure is physical here. Presentation buildings, if drawn, are not.")


@dataclass(frozen=True)
class PhysicsUrban:
    state: str  # not_declared | available | unavailable
    reason: str | None = None
    grid: uo.UrbanObstacleGrid | None = None

    @property
    def available(self) -> bool:
        return self.state == "available"

    def scene_block(self) -> dict[str, Any]:
        """The `/scene` ``physics_urban`` block. Evidence only: no paths, no secrets."""
        out: dict[str, Any] = {"state": self.state, "evidence_grade": self.available,
                               "geometry_source": "physics_obstacle_artifact" if self.available else None,
                               "reason": self.reason}
        if self.grid is not None:
            out["evidence"] = self.grid.evidence()
            out["note"] = EVIDENCE_NOTE
        elif self.state == "unavailable":
            out["note"] = UNAVAILABLE_NOTE
        return out

    def page_payload(self) -> dict[str, Any] | None:
        """What `/physics_urban` serves and a physics export embeds; None unless available."""
        g = self.grid
        if g is None:
            return None
        m = g.meta
        covered = g.coverage == 1
        ground = np.where(covered, np.rint(g.ground_amsl_m * 10.0), NODATA_DM).astype("<i2")
        top = np.where(covered, np.rint(g.top_amsl_m * 10.0), NODATA_DM).astype("<i2")
        flags = (g.coverage * FLAG_COVERED + g.building * FLAG_BUILDING).astype("|u1")
        b64 = lambda a: base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")  # noqa: E731
        return {
            "schema": PAGE_SCHEMA, "artifact_schema": uo.SCHEMA_ID, "artifact_version": uo.SCHEMA_VERSION,
            "evidence": g.evidence(), "nx": m.nx, "ny": m.ny, "cell_m": m.cell_m,
            "corners_wgs84": [list(c) for c in m.grid_corners_wgs84],  # SW, SE, NE, NW as [lon, lat]
            "units": "decimetres AMSL, int16 little-endian, -32768 = unknown; rows south to north",
            "ground_dm": b64(ground), "top_dm": b64(top), "flags": b64(flags),
            "flag_bits": {"covered": FLAG_COVERED, "building": FLAG_BUILDING},
            "note": EVIDENCE_NOTE,
        }


def _no_paths(exc: Exception) -> str:
    """An error message with every filesystem path removed: a scene never discloses where files live."""
    return re.sub(r"(?:/[\w.\-@+~ ]+){2,}", "<path>", str(exc))


def load(aoi_name: str | None, notes: dict | None = None) -> PhysicsUrban:
    """Resolve the theatre's artifact for display. Never raises for a missing or corrupt artifact.

    ``notes["urban_obstacles"]`` -- an artifact already attached to the environment by the env
    bridge -- wins; otherwise the artifact the theatre cites is loaded, with the same strict
    verification the env uses. A theatre that does not declare ``urban_physics`` is
    ``not_declared`` (and never gets presentation geometry in its place).
    """
    attached = (notes or {}).get("urban_obstacles")
    if attached is not None:
        return PhysicsUrban("available", None, attached)
    if not aoi_name:
        return PhysicsUrban("not_declared", "no theatre named")
    from ..data.theatre import load_urban_obstacles_for
    from ..research.aoi import get_aoi

    try:
        declared = get_aoi(aoi_name).urban_physics is not None
    except KeyError:
        return PhysicsUrban("not_declared", f"unknown theatre {aoi_name!r}")
    if not declared:
        return PhysicsUrban("not_declared", "this theatre declares no urban_physics block")
    try:
        return PhysicsUrban("available", None, load_urban_obstacles_for(aoi_name))
    except uo.UrbanObstacleError as exc:
        return PhysicsUrban(
            "unavailable",
            f"{_no_paths(exc)}. Rebuild it with: uv run naigos-research --aoi {aoi_name}")
    except (FileNotFoundError, KeyError, ValueError) as exc:
        return PhysicsUrban(
            "unavailable",
            f"data.urban_obstacles is missing or unreadable ({type(exc).__name__}). "
            f"Rebuild it with: uv run naigos-research --aoi {aoi_name}")
