"""Exclusive L1 outcomes: precedence, metric masks, and baseline() accounting."""

from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from naigos.env import resource as R
from naigos.env.config import DetectionConfig, EnvConfig, TerrainConfig
from naigos.env.flight_env import NaigosEnv
from naigos.env.resource_bank import config_for_scenario, resource_bank
from naigos.rl.train import baseline, rollout_metrics


def _outcome(**flags):
    names = ("primary", "diverted", "shot", "terrain", "building", "bounds", "dry", "timeout")
    args = [jnp.array([bool(flags.get(n, False))]) for n in names]
    return int(R.outcome_update(jnp.zeros((1,), jnp.uint8), *args)[0])


@pytest.mark.parametrize(
    "flags, expected",
    [
        ({"shot": 1, "terrain": 1, "bounds": 1, "dry": 1}, R.OUTCOME_SHOTDOWN),
        ({"shot": 1, "building": 1, "terrain": 1}, R.OUTCOME_SHOTDOWN),
        ({"building": 1, "terrain": 1, "bounds": 1}, R.OUTCOME_BUILDING),
        ({"terrain": 1, "bounds": 1, "dry": 1}, R.OUTCOME_TERRAIN),
        ({"bounds": 1, "dry": 1}, R.OUTCOME_BOUNDS),
        ({"dry": 1, "timeout": 1}, R.OUTCOME_FUEL),
        ({"primary": 1, "diverted": 1}, R.OUTCOME_PRIMARY),
        ({"diverted": 1, "timeout": 1}, R.OUTCOME_DIVERTED),
        ({"timeout": 1}, R.OUTCOME_TIMEOUT),
        ({}, R.OUTCOME_ONGOING),
    ],
)
def test_simultaneous_events_resolve_by_documented_precedence(flags, expected):
    assert _outcome(**flags) == expected


def test_first_outcome_is_never_overwritten():
    prev = jnp.array([R.OUTCOME_DIVERTED], dtype=jnp.uint8)
    out = R.outcome_update(prev, *[jnp.array([True])] * 8)
    assert int(out[0]) == R.OUTCOME_DIVERTED


def _fake(outcome, *, exposure, alt, feasible, diverted, reached, shot=0.0, terrain=0.0):
    """One world, one aircraft, S steps; every array is (W=1, S, B=1)."""
    S = len(exposure)
    col = lambda x: np.asarray(x, dtype=np.float32).reshape(1, S, 1)  # noqa: E731
    flag = lambda x: np.asarray(x, dtype=bool).reshape(1, S, 1)  # noqa: E731
    zeros = col([0.0] * S)
    terms = SimpleNamespace(
        exposure=col(exposure), lock_level=zeros,
        shotdown=col([shot] * S), terrain_violation=col([terrain] * S),
        bounds_violation=zeros, out_of_fuel=zeros,
    )
    traj = {
        "alive": flag([True] * S), "reached": flag(reached), "diverted": flag(diverted),
        "alt_agl": col(alt), "cbf_feasible": flag(feasible), "terms": terms,
    }
    final = SimpleNamespace(
        alive=np.array([[True]]), reached=np.array([[reached[-1]]]),
        air=SimpleNamespace(fuel=np.array([[100.0]])),
        resource=SimpleNamespace(
            diversion_active=np.ones((1, 1, 2), dtype=bool), diverted=np.array([[diverted[-1]]]),
            reserve_fuel=np.array([[10.0]]), outcome=np.array([[outcome]], dtype=np.uint8),
        ),
    )
    return final, traj


def test_post_diversion_samples_are_excluded_from_every_live_mean():
    final, traj = _fake(
        R.OUTCOME_DIVERTED,
        exposure=[0.5, 0.5, 0.0, 0.0], alt=[100, 200, 999, 999],
        feasible=[False, True, False, False],
        diverted=[False, True, True, True], reached=[False] * 4,
    )
    m = rollout_metrics(final, traj, 1, 1)
    assert m["mean_exposure"] == pytest.approx(0.5)  # 1.0 over two in-flight samples
    assert m["mean_agl_live"] == pytest.approx(150.0)
    assert m["cbf_infeasible_rate"] == pytest.approx(0.5)
    assert (m["survival_rate"], m["diversion_rate"], m["recovery_rate"]) == (0.0, 1.0, 1.0)
    assert m["timeout_rate"] == 0.0 and m["fuel_loss_rate"] == 0.0


def test_exclusive_rates_keep_raw_evidence_for_simultaneous_losses():
    final, traj = _fake(
        R.OUTCOME_SHOTDOWN, exposure=[0.1, 0.1], alt=[50, 50], feasible=[True, True],
        diverted=[False, False], reached=[False, False], shot=1.0, terrain=1.0,
    )
    final.alive[:] = False
    m = rollout_metrics(final, traj, 1, 1)
    assert (m["shootdown_rate"], m["terrain_rate"]) == (1.0, 0.0)
    assert m["raw_terrain_rate"] > 0 and m["raw_shootdown_rate"] > 0
    assert m["shootdown_rate"] + m["terrain_rate"] + m["bounds_rate"] + m["fuel_loss_rate"] <= 1.0


def _small_l1(combination):
    base = EnvConfig(
        n_blue=1, n_threat=2, n_threat_active=0, k_threat_obs=1, k_friend_obs=0, max_steps=200,
        terrain=TerrainConfig(nx=32, ny=32, cell=1_000.0), detection=DetectionConfig(los_samples=8),
    )
    row = next(r for r in resource_bank("train_reference") if r.combination == combination)
    return config_for_scenario(base, row)


def test_l1_baseline_separates_diversion_and_timeout_from_ordinary_survival():
    import jax

    out = baseline(NaigosEnv(_small_l1("marginal_fuel")), 2, jax.random.PRNGKey(0))
    assert {"direct", "avoid_nap", "reserve_divert"} <= {k.split("_survival_rate")[0] for k in out if k.endswith("_survival_rate")}
    for name in ("direct", "avoid_nap", "reserve_divert"):
        completed = out[f"{name}_objective_rate"]
        assert out[f"{name}_survival_rate"] == completed
        assert out[f"{name}_recovery_rate"] == pytest.approx(completed + out[f"{name}_diversion_rate"])
    assert out["reserve_divert_diversion_rate"] > 0.0
    assert out["reserve_divert_survival_rate"] < out["reserve_divert_recovery_rate"]


def test_baseline_v0_baseline_keys_and_policies_are_unchanged():
    import jax

    cfg = EnvConfig(
        n_blue=1, n_threat=2, n_threat_active=0, k_threat_obs=1, k_friend_obs=0, max_steps=8,
        terrain=TerrainConfig(nx=32, ny=32, cell=1_000.0), detection=DetectionConfig(los_samples=8),
    )
    out = baseline(NaigosEnv(cfg), 2, jax.random.PRNGKey(0))
    assert {k.split("_")[0] for k in out} == {"direct", "avoid"}
    assert "direct_diversion_rate" not in out and "direct_timeout_rate" not in out
