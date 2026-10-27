"""Demo/replay summaries keep L1 outcomes apart."""

from __future__ import annotations

import jax

from naigos.demo import replay
from naigos.demo.live import Counters
from naigos.env.config import DetectionConfig, EnvConfig, TerrainConfig
from naigos.env.flight_env import NaigosEnv
from naigos.env.resource_bank import config_for_scenario, resource_bank
from naigos.rl.train import make_reserve_divert_policy


def test_replay_summary_and_table_report_each_l1_outcome_separately():
    base = EnvConfig(
        n_blue=1, n_threat=2, n_threat_active=0, k_threat_obs=1, k_friend_obs=0, max_steps=200,
        terrain=TerrainConfig(nx=32, ny=32, cell=1_000.0), detection=DetectionConfig(los_samples=8),
    )
    row = next(r for r in resource_bank("train_reference") if r.combination == "marginal_fuel")
    cfg = config_for_scenario(base, row)
    env = NaigosEnv(cfg)
    pol = make_reserve_divert_policy(cfg)
    keys = jax.random.split(jax.random.PRNGKey(0), 2)
    final, traj = jax.jit(jax.vmap(lambda k: env.rollout(k, pol)))(keys)
    s = replay._summary(final, traj, cfg)
    assert s["safe_diversion_rate"] > 0.0
    assert s["survival_rate"] == s["primary_completion_rate"]  # a diversion is not ordinary survival
    assert s["resource_loss_rate"] == 0.0 and s["timeout_rate"] == 0.0
    table = replay._table({"trained": {"summary": s}})
    for label in ("primary completion", "safe diversion", "timeout (live)", "resource loss (fuel)"):
        assert label in table


def test_legacy_summary_has_no_l1_rows():
    s = {"survival_rate": 1.0, "objective_rate": 1.0, "shootdowns": 0, "terrain_losses": 0,
         "bounds_losses": 0, "mean_detection_prob": 0.0, "mean_track_quality": 0.0, "mean_min_agl_m": 1.0}
    assert "safe diversion" not in replay._table({"trained": {"summary": s}})


def test_live_counters_track_diversions_as_finished_but_not_reached():
    c = Counters()
    c.record(reached=1, diverted=2)
    d = c.as_dict()
    assert d["safe_diversions"] == 2 and d["objectives_reached"] == 1
    assert d["success_rate"] == round(1 / 3, 3)
