"""Frozen evaluation scenarios for resource-world L1.

The records are plain data: each one fixes route geometry, every resource
parameter, active recovery gates, and the master seed that drives terrain,
placement, threats, and resource-local random keys.  The canonical digest makes
an edited bank a new protocol rather than a silent change to an old result.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass

from .config import EnvConfig, RESOURCE_WORLD_L1, ROUTE_FAMILY_NAMES, ResourceWorldConfig

RESOURCE_BANK_VERSION = "resource-bank-v1"
RESOURCE_BANK_SEED_BASE = {"train_reference": 730_000, "heldout": 740_000}
RESOURCE_BANK_SEED_SPAN = 1_000
# Filled from the canonical JSON emitted by ``resource_bank_digest``. Changing
# any row requires publishing a new version and pin, never updating this value.
RESOURCE_BANK_V1_SHA256 = "12c909dc9aecdd683b8a95c25570472385ef72ab29fe3525ab36c334001531de"


@dataclass(frozen=True)
class ResourceScenario:
    scenario_id: str
    split: str
    combination: str
    route_family: str
    bearing_deg: float
    seed: int
    fuel_budget_ratio: float
    reserve_ratio: float
    burn_multiplier: float
    active_gate_mask: tuple[bool, bool]

    @property
    def family_id(self) -> int:
        return ROUTE_FAMILY_NAMES.index(self.route_family)


def _build_resource_bank_v1() -> tuple[ResourceScenario, ...]:
    # Paired combinations make train-reference and held-out summaries directly
    # comparable while keeping all master seeds and exact parameter tuples
    # disjoint. Values are generic research assumptions, not platform data.
    specs = {
        "train_reference": (
            ("nominal", "west_east", 0.0, 1.55, 0.15, 1.00, (True, True)),
            ("marginal_fuel", "west_east", 0.0, 0.75, 0.12, 1.00, (True, True)),
            ("burn_multiplier", "south_north", 90.0, 1.45, 0.16, 1.35, (True, True)),
            ("farther_gate_only", "east_west", 180.0, 1.30, 0.14, 1.05, (False, True)),
            ("threat_detour", "southwest_northeast", 45.0, 1.20, 0.15, 1.10, (True, True)),
        ),
        "heldout": (
            ("nominal", "north_south", 270.0, 1.60, 0.18, 0.95, (True, True)),
            ("marginal_fuel", "north_south", 270.0, 0.70, 0.10, 1.05, (True, True)),
            ("burn_multiplier", "northeast_southwest", 225.0, 1.50, 0.17, 1.40, (True, True)),
            ("farther_gate_only", "west_east", 4.0, 1.28, 0.13, 1.08, (False, True)),
            ("threat_detour", "southeast_northwest", 135.0, 1.18, 0.14, 1.12, (True, True)),
        ),
    }
    rows: list[ResourceScenario] = []
    for split, entries in specs.items():
        for index, entry in enumerate(entries):
            combination, family, bearing, fuel, reserve, burn, active = entry
            rows.append(ResourceScenario(
                scenario_id=f"{RESOURCE_BANK_VERSION}/{split}/{index:03d}-{combination}",
                split=split,
                combination=combination,
                route_family=family,
                bearing_deg=bearing,
                seed=RESOURCE_BANK_SEED_BASE[split] + index,
                fuel_budget_ratio=fuel,
                reserve_ratio=reserve,
                burn_multiplier=burn,
                active_gate_mask=active,
            ))
    return tuple(rows)


def resource_bank_digest(rows) -> str:
    payload = json.dumps(
        [dataclasses.asdict(row) for row in rows], sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def resource_bank(split: str | None = None, version: str = RESOURCE_BANK_VERSION):
    if version != RESOURCE_BANK_VERSION:
        raise ValueError(f"unknown resource bank {version!r}; expected {RESOURCE_BANK_VERSION!r}")
    rows = _build_resource_bank_v1()
    digest = resource_bank_digest(rows)
    if digest != RESOURCE_BANK_V1_SHA256:
        raise RuntimeError(
            f"{RESOURCE_BANK_VERSION} hashes to {digest}, not the pinned "
            f"{RESOURCE_BANK_V1_SHA256}; publish edits under a new bank version"
        )
    if split is None:
        return rows
    if split not in RESOURCE_BANK_SEED_BASE:
        raise ValueError(f"unknown resource-bank split {split!r}")
    return tuple(row for row in rows if row.split == split)


def config_for_scenario(cfg: EnvConfig, scenario: ResourceScenario) -> EnvConfig:
    """Apply all realized resource parameters from one immutable record."""
    old = cfg.resource
    return cfg.replace(resource=ResourceWorldConfig(
        model=RESOURCE_WORLD_L1,
        fuel_budget_ratio=(scenario.fuel_budget_ratio, scenario.fuel_budget_ratio),
        reserve_ratio=(scenario.reserve_ratio, scenario.reserve_ratio),
        burn_multiplier=(scenario.burn_multiplier, scenario.burn_multiplier),
        gate_lateral_frac=old.gate_lateral_frac,
        gate_arrival_radius=old.gate_arrival_radius,
        active_gate_mask=scenario.active_gate_mask,
    ))


class ResourceBankOverlap(ValueError):
    pass


def assert_resource_bank_disjoint(training_seeds) -> None:
    """Refuse training seeds inside either reserved resource-bank range."""
    seeds = {int(seed) for seed in training_seeds}
    bad = sorted(
        seed
        for seed in seeds
        if any(
            base <= seed < base + RESOURCE_BANK_SEED_SPAN
            for base in RESOURCE_BANK_SEED_BASE.values()
        )
    )
    if bad:
        raise ResourceBankOverlap(
            f"training seeds {bad} overlap resource-bank reserved ranges "
            f"{RESOURCE_BANK_SEED_BASE} (+{RESOURCE_BANK_SEED_SPAN})"
        )


def resource_protocol() -> dict:
    return {
        "environment_model_id": RESOURCE_WORLD_L1,
        "observation_schema_id": "ResourceObservation-v1",
        "bank": RESOURCE_BANK_VERSION,
        "bank_sha256": RESOURCE_BANK_V1_SHA256,
        "seed_ranges": {
            name: [base, base + RESOURCE_BANK_SEED_SPAN]
            for name, base in RESOURCE_BANK_SEED_BASE.items()
        },
        "assumption": "generic research fuel ratios and abstract recovery gates; not platform data",
    }
