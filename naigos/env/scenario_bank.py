"""Immutable, seed-disjoint NYC/LA weather + mission evaluation bank."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass

from .config import (
    MISSION_V1,
    RESOURCE_WORLD_L1,
    URBAN_OBSTACLE_V1,
    WEATHER_AR1,
    EnvConfig,
    MissionConfig,
    ResourceWorldConfig,
    UrbanPhysicsConfig,
    WeatherConfig,
)


SCENARIO_BANK_VERSION = "city-weather-mission-bank-v1"
SCENARIO_BANK_V1_SHA256 = "039529a9f7af1982cf2752515e09807942a6582175d76fca98be72814e8a3453"
SCENARIO_SEED_BASE = {"train_reference": 760_000, "heldout": 770_000}
SCENARIO_SEED_SPAN = 1_000


@dataclass(frozen=True)
class CityScenario:
    scenario_id: str
    split: str
    theatre: str
    case: str
    master_seed: int
    wind_enu_mps: tuple[float, float, float]
    density_ratio: float
    turbulence_mps: float
    mission_family: str
    marginal_resource: bool = False
    requires_urban: bool = False
    note: str = "notional frozen research scenario; not current conditions"


def _build_bank() -> tuple[CityScenario, ...]:
    specs = {
        "nyc_harbor": (
            ("calm", (0.0, 0.0, 0.0), 1.00, 0.0, "transit", False, False),
            ("harbor_crosswind", (2.0, 18.0, 0.0), 0.98, 2.0, "multi_leg", False, False),
            ("strong_building_wake", (18.0, 4.0, 0.0), 0.96, 6.0, "time_on_station", False, True),
            ("low_density_high_demand", (5.0, -8.0, 0.0), 0.82, 3.0, "survey_corridor", False, False),
            ("marginal_fuel_headwind", (-25.0, 0.0, 0.0), 0.95, 3.0, "recovery", True, False),
        ),
        "los_angeles_basin": (
            ("calm", (0.0, 0.0, 0.0), 1.00, 0.0, "transit", False, False),
            ("coastal_onshore", (10.0, 12.0, 0.0), 0.96, 3.0, "multi_leg", False, False),
            ("basin_crosswind", (-12.0, 20.0, 0.0), 0.92, 4.0, "survey_corridor", False, False),
            ("mountain_wave_rotor_surrogate", (16.0, 2.0, 2.0), 0.88, 10.0, "time_on_station", False, False),
            ("marginal_fuel_headwind", (-28.0, -3.0, 0.0), 0.90, 4.0, "recovery", True, False),
            ("reduced_authority_crosswind", (4.0, 24.0, 0.0), 0.72, 7.0, "multi_leg", False, True),
        ),
    }
    rows = []
    counts = {k: 0 for k in SCENARIO_SEED_BASE}
    for theatre, cases in specs.items():
        for i, (case, wind, rho, turb, mission, marginal, urban) in enumerate(cases):
            # Calm/reference cases are training-geometry controls; all stressed
            # cases are held out and draw from a separate reserved range.
            split = "train_reference" if i == 0 else "heldout"
            index = counts[split]
            counts[split] += 1
            rows.append(CityScenario(
                scenario_id=f"{SCENARIO_BANK_VERSION}/{theatre}/{case}",
                split=split,
                theatre=theatre,
                case=case,
                master_seed=SCENARIO_SEED_BASE[split] + index,
                wind_enu_mps=wind,
                density_ratio=rho,
                turbulence_mps=turb,
                mission_family=mission,
                marginal_resource=marginal,
                requires_urban=urban,
            ))
    return tuple(rows)


def scenario_bank_digest(rows) -> str:
    raw = json.dumps([dataclasses.asdict(r) for r in rows], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def scenario_bank(theatre: str | None = None, split: str | None = None):
    rows = _build_bank()
    got = scenario_bank_digest(rows)
    if got != SCENARIO_BANK_V1_SHA256:
        raise RuntimeError(f"{SCENARIO_BANK_VERSION} hashes to {got}, not pinned {SCENARIO_BANK_V1_SHA256}")
    if theatre is not None:
        rows = tuple(r for r in rows if r.theatre == theatre)
    if split is not None:
        if split not in SCENARIO_SEED_BASE:
            raise ValueError(f"unknown scenario split {split!r}")
        rows = tuple(r for r in rows if r.split == split)
    return rows


def config_for_scenario(base: EnvConfig, row: CityScenario) -> EnvConfig:
    weather = WeatherConfig(
        model=WEATHER_AR1,
        scenario_id=row.scenario_id,
        altitude_knots_m=(0.0, 3_000.0, 12_000.0),
        wind_enu_mps=(row.wind_enu_mps, row.wind_enu_mps, row.wind_enu_mps),
        density_ratio=(row.density_ratio, row.density_ratio, row.density_ratio),
        temperature_k=(288.15, 268.65, 216.65),
        sensing_attenuation=(1.0, 0.98, 0.95),
        turbulence_intensity_mps=(row.turbulence_mps,) * 3,
        severity_scale=1.0,
    )
    resource = ResourceWorldConfig(
        model=RESOURCE_WORLD_L1,
        fuel_budget_ratio=(0.72, 0.72) if row.marginal_resource else (1.35, 1.35),
        reserve_ratio=(0.12, 0.12),
        burn_multiplier=(1.08, 1.08),
    )
    mission = MissionConfig(
        model=MISSION_V1,
        family=row.mission_family,
        scenario_id=row.scenario_id,
        deadline_s=900.0,
    )
    urban = (UrbanPhysicsConfig(model=URBAN_OBSTACLE_V1, los_surface="terrain_plus_urban")
             if row.requires_urban else base.urban)
    return base.replace(weather=weather, resource=resource, mission=mission, urban=urban)


def assert_scenario_seeds_disjoint(training_seeds) -> None:
    bad = sorted(
        int(seed) for seed in training_seeds
        if any(base <= int(seed) < base + SCENARIO_SEED_SPAN for base in SCENARIO_SEED_BASE.values())
    )
    if bad:
        raise ValueError(f"training seeds {bad} overlap {SCENARIO_BANK_VERSION} reserved ranges")


def scenario_protocol() -> dict:
    return {
        "bank": SCENARIO_BANK_VERSION,
        "bank_sha256": SCENARIO_BANK_V1_SHA256,
        "seed_stream_version": "named-streams-v1",
        "seed_ranges": {k: [v, v + SCENARIO_SEED_SPAN] for k, v in SCENARIO_SEED_BASE.items()},
        "assumption": "notional frozen city-transfer scenarios; not current weather or operational data",
    }
