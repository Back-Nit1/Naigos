#!/usr/bin/env python
"""Benchmark an offline, hash-validated urban obstacle artifact.

Example using the committed NYC physics fixture (no network):

  uv run python scripts/bench_urban_physics.py \
    tests/fixtures/urban_physics_root/urban_physics/nyc_harbor/c68d8582bf27f84f
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax.numpy as jnp  # noqa: E402

from naigos.bench.urban_physics import measure  # noqa: E402
from naigos.data.urban_obstacles import load_urban_obstacles  # noqa: E402
from naigos.env.config import (  # noqa: E402
    MISSION_V1, RESOURCE_WORLD_L1, URBAN_OBSTACLE_V1, WEATHER_AR1,
    DetectionConfig, EnvConfig, MissionConfig, ResourceWorldConfig,
    TerrainConfig, UrbanPhysicsConfig, WeatherConfig,
)
from naigos.env.flight_env import NaigosEnv  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("artifact", type=Path, help="completed naigos.urban_obstacles/1 directory")
    ap.add_argument("--worlds", type=int, default=8)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)

    grid = load_urban_obstacles(args.artifact)
    cfg = EnvConfig(
        n_blue=2, n_threat=4, n_threat_active=2, max_steps=max(64, args.steps + 1),
        terrain=TerrainConfig(nx=128, ny=128, cell=500.0),
        detection=DetectionConfig(los_samples=32),
        resource=ResourceWorldConfig(model=RESOURCE_WORLD_L1),
        weather=WeatherConfig(model=WEATHER_AR1, scenario_id="offline-benchmark",
                              turbulence_intensity_mps=(3.0, 3.0)),
        mission=MissionConfig(model=MISSION_V1, family="multi_leg", scenario_id="offline-benchmark"),
        urban=UrbanPhysicsConfig(model=URBAN_OBSTACLE_V1,
                                 los_surface="terrain_plus_urban", collision_samples=128),
    )
    env = NaigosEnv(
        cfg, hmap=jnp.zeros((cfg.terrain.ny, cfg.terrain.nx)), urban_obstacles=grid,
        env_origin_utm_m=grid.meta.env_origin_utm_m,
    )
    control_cfg = dataclasses.replace(cfg, urban=UrbanPhysicsConfig())
    control = NaigosEnv(control_cfg, hmap=jnp.zeros((cfg.terrain.ny, cfg.terrain.nx)))
    urban_report = measure(env, n_worlds=args.worlds, n_steps=args.steps,
                           repeats=args.repeats, seed=args.seed)
    control_report = measure(control, n_worlds=args.worlds, n_steps=args.steps,
                             repeats=args.repeats, seed=args.seed)
    report = {
        "schema": "naigos.bench.urban_physics_comparison/1",
        "vehicle_weather_control": control_report,
        "urban_physics": urban_report,
        "steady_state_overhead_ratio": (
            control_report["timing"]["world_steps_per_s"]
            / urban_report["timing"]["world_steps_per_s"]
        ),
        "artifact": {
        "schema": "naigos.urban_obstacles/1",
        "aoi": grid.meta.aoi_name,
        "sha256": grid.hashes.content_sha256,
        "shape": list(grid.shape),
        "cell_m": grid.meta.cell_m,
        },
    }
    payload = json.dumps(report, indent=2) + "\n"
    print(payload, end="")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload)
    return 0 if urban_report["budget"]["passed"] and control_report["budget"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
