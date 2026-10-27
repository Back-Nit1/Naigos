"""Required policy/resource/weather/mission/theatre construction matrix."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _theatre_fixtures import offline_theatre
from naigos.env.config import RESOURCE_WORLD_L1, ResourceWorldConfig
from naigos.env.flight_env import NaigosEnv
from naigos.env.scenario_bank import config_for_scenario, scenario_bank
from naigos.env.theatre_bridge import env_from_theatre
from naigos.rl.policies import get_policy_spec


CELLS = (
    ("deepsets", "baseline", None, "legacy", "nyc_harbor"),
    ("recurrent", "baseline", None, "legacy", "los_angeles_basin"),
    ("deepsets", "endurance", "harbor_crosswind", "multi_leg", "nyc_harbor"),
    ("recurrent", "endurance", "coastal_onshore", "multi_leg", "los_angeles_basin"),
    ("recurrent", "endurance", "strong_building_wake", "recovery", "nyc_harbor"),
    ("recurrent", "endurance", "reduced_authority_crosswind", "multi_leg", "los_angeles_basin"),
)


@pytest.mark.parametrize("family,resource,case,mission,theatre", CELLS)
def test_required_physics_matrix_constructs_steps_and_has_finite_policy_contract(
    family, resource, case, mission, theatre
):
    with offline_theatre(theatre):
        cfg, hmap, notes = env_from_theatre(aoi=theatre, n_blue=1, n_threat=2, cell_m=1500.0)
    if resource == "endurance":
        cfg = cfg.replace(resource=ResourceWorldConfig(model=RESOURCE_WORLD_L1))
    if case is not None:
        row = next(row for row in scenario_bank(theatre) if row.case == case)
        cfg = config_for_scenario(cfg, row)
    if mission != "legacy" and cfg.mission.family != mission:
        cfg = cfg.replace(mission=dataclasses.replace(cfg.mission, family=mission))
    kwargs = {}
    if cfg.urban.enabled:
        kwargs = {"urban_obstacles": notes["urban_obstacles"],
                  "env_origin_utm_m": notes["env_origin_utm_m"]}
    env = NaigosEnv(cfg, hmap=hmap, **kwargs)
    state, obs = env.reset(jax.random.PRNGKey(4))
    state, obs, _, _, _ = env.step(state, jnp.zeros((1, 3)))
    spec = get_policy_spec(cfg, family=family)
    assert spec.family == family
    assert np.isfinite(np.asarray(state.air.pos)).all()
    assert np.isfinite(np.asarray(obs.ego)).all()


def test_matrix_covers_both_theatres_and_required_axes():
    assert {row[4] for row in CELLS} == {"nyc_harbor", "los_angeles_basin"}
    assert {row[0] for row in CELLS} == {"deepsets", "recurrent"}
    assert any(row[2] == "strong_building_wake" and row[3] == "recovery" for row in CELLS)
