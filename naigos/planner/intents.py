"""The complete and deliberately small Qwen planner action vocabulary."""

from __future__ import annotations

import dataclasses
from typing import Mapping

import numpy as np


@dataclasses.dataclass(frozen=True)
class PlannerIntent:
    label: str
    cross_track_m: float
    altitude_reference: str  # "agl" | "amsl"
    altitude_m: float

    @property
    def description(self) -> str:
        side = {15_000.0: "left_15km", 0.0: "center", -15_000.0: "right_15km"}[self.cross_track_m]
        return f"{side},{self.altitude_reference}_{self.altitude_m / 1000.0:.1f}km"


# Append-only: labels are persisted in planner datasets and adapter artifacts.
INTENTS: tuple[PlannerIntent, ...] = tuple(
    PlannerIntent(chr(ord("A") + i), offset, reference, altitude)
    for i, (offset, reference, altitude) in enumerate(
        (offset, reference, altitude)
        # Positive cross-track is left of the direction of travel.
        for offset in (15_000.0, 0.0, -15_000.0)
        for reference, altitude in (("agl", 200.0), ("agl", 1_000.0), ("amsl", 4_000.0))
    )
)
INTENT_LABELS: tuple[str, ...] = tuple(intent.label for intent in INTENTS)
_BY_LABEL: Mapping[str, PlannerIntent] = {intent.label: intent for intent in INTENTS}


def intent_for_label(label: str) -> PlannerIntent:
    """Resolve a strict single-label decision; free text is never executable."""

    if label not in _BY_LABEL:
        raise ValueError(f"planner intent must be exactly one of {INTENT_LABELS}, got {label!r}")
    return _BY_LABEL[label]


def route_waypoint(
    position_xyz,
    primary_objective_xyz,
    bounds_xyxy,
    intent: PlannerIntent | str,
    *,
    target_ground_m: float,
    lookahead_m: float = 30_000.0,
) -> np.ndarray:
    """Map an intent to a clipped intermediate routing objective.

    This is the deterministic adapter seam for a future online cache consumer.
    It changes only the objective shown to the low-level actor.  Arrival,
    rewards, and mission completion continue to use the environment's primary
    objective.  Terrain at the proposed waypoint must be supplied by the caller
    so this framework-neutral module never imports the simulator.
    """

    intent = intent_for_label(intent) if isinstance(intent, str) else intent
    pos = np.asarray(position_xyz, dtype=np.float64)
    objective = np.asarray(primary_objective_xyz, dtype=np.float64)
    bounds = np.asarray(bounds_xyxy, dtype=np.float64)
    if pos.shape != (3,) or objective.shape != (3,) or bounds.shape != (4,):
        raise ValueError("position/objective must be xyz triples and bounds must be [xmin,xmax,ymin,ymax]")
    along = objective[:2] - pos[:2]
    distance = float(np.linalg.norm(along))
    if not np.isfinite(distance) or distance <= 1e-6:
        raise ValueError("cannot construct a route intent at the primary objective")
    along /= distance
    left = np.array([-along[1], along[0]])
    xy = pos[:2] + along * min(lookahead_m, distance) + left * intent.cross_track_m
    xy[0] = np.clip(xy[0], bounds[0], bounds[1])
    xy[1] = np.clip(xy[1], bounds[2], bounds[3])
    z = target_ground_m + intent.altitude_m if intent.altitude_reference == "agl" else intent.altitude_m
    return np.array([xy[0], xy[1], z], dtype=np.float32)
