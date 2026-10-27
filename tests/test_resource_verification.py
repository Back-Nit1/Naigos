"""L1 verifier: reward/CMDP cross-check, latent constancy, sampled ranges."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from naigos.env.config import (
    RESOURCE_WORLD_L1,
    DetectionConfig,
    EnvConfig,
    ResourceWorldConfig,
    TerrainConfig,
)
from naigos.env.flight_env import NaigosEnv
from naigos.rl import reward, verifier

CFG = EnvConfig(
    n_blue=2,
    n_threat=2,
    n_threat_active=0,
    k_threat_obs=1,
    k_friend_obs=0,
    max_steps=8,
    detection=DetectionConfig(los_samples=8),
    terrain=TerrainConfig(nx=32, ny=32, cell=1_000.0),
    resource=ResourceWorldConfig(
        model=RESOURCE_WORLD_L1,
        fuel_budget_ratio=(0.001, 0.002),
        reserve_ratio=(0.1, 0.2),
        burn_multiplier=(1.0, 1.2),
    ),
)
W = reward.RewardWeights()


def _episode():
    env = NaigosEnv(CFG)
    key = jax.random.PRNGKey(3)
    start = np.asarray(env.reset(key)[0].air.pos)
    _, traj = env.rollout(key, lambda _o, _k: jnp.array([[0.2, 0.0, 0.7]] * 2))
    trace = {n: np.asarray(v) for n, v in traj.items() if n != "terms"}
    trace["terms"] = type(traj["terms"])(*[np.asarray(x) for x in traj["terms"]])
    rew = np.asarray(reward.compute(traj["terms"], W))
    cost = np.asarray(reward.constraint_cost(traj["terms"]))
    return trace, start, rew, cost


def _verify(trace, start, rew, cost):
    return verifier.verify_resource_trace(
        CFG, trace, start=start, weights=dataclasses.asdict(W), reward=rew, cost=cost
    )


def test_clean_trace_passes_every_cross_check_and_cost_is_one_per_loss():
    trace, start, rew, cost = _episode()
    report = _verify(trace, start, rew, cost)
    assert report.ok, report.mismatches
    assert report.total_cost == cost.sum() == 2.0  # both aircraft run dry exactly once


def test_tampered_reward_or_cost_is_caught():
    trace, start, rew, cost = _episode()
    assert any("reward" in m for m in _verify(trace, start, rew + 0.5, cost).mismatches)
    bad_cost = cost.copy()
    bad_cost[0, 0] += 1.0
    assert any("CMDP cost" in m for m in _verify(trace, start, rew, bad_cost).mismatches)


def test_duplicate_terminal_payment_is_caught():
    trace, start, rew, cost = _episode()
    terms = trace["terms"]
    out_of_fuel = terms.out_of_fuel.copy()
    step = int(np.argmax(out_of_fuel[:, 0]))
    out_of_fuel[min(step + 1, len(out_of_fuel) - 1), 0] = 1.0
    bad = {**trace, "terms": terms._replace(out_of_fuel=out_of_fuel)}
    assert any("exactly once" in m or "more than once" in m
               for m in _verify(bad, start, rew, cost).mismatches)


def test_latent_and_gate_constancy_is_enforced():
    trace, start, rew, cost = _episode()
    for name in ("burn_multiplier", "reserve_fuel", "diversion_pos", "objective"):
        arr = trace[name].astype(np.float64).copy()
        arr[-1] += 1.0
        problems = verifier._latent_problems(CFG, {**trace, name: arr}, start)
        assert any(name in p and "changes" in p for p in problems), name
    active = trace["diversion_active"].copy()
    active[-1, 0, 0] = ~active[-1, 0, 0]
    assert any("diversion_active" in p for p in verifier._latent_problems(CFG, {**trace, "diversion_active": active}, start))


def test_sampled_latents_stay_in_configured_ranges_and_cover_them():
    cfg = CFG.replace(n_blue=1)
    env = NaigosEnv(cfg)
    keys = jax.random.split(jax.random.PRNGKey(11), 256)
    states = jax.jit(jax.vmap(lambda k: env.reset(k)[0]))(keys)
    r = states.resource
    direct = np.linalg.norm(np.asarray(states.objective - states.air.pos)[..., :2], axis=-1)
    af = cfg.airframe
    rate = af.burn_base + af.burn_throttle * np.clip((af.v_init - af.v_stall) / (af.v_max - af.v_stall), 0, 1)
    direct_fuel = direct / af.v_init * rate
    draws = {
        "fuel_budget_ratio": np.asarray(r.initial_fuel) / direct_fuel,
        "reserve_ratio": np.asarray(r.reserve_fuel) / direct_fuel,
        "burn_multiplier": np.asarray(r.burn_multiplier),
    }
    for name, values in draws.items():
        lo, hi = getattr(cfg.resource, name)
        assert values.min() >= lo * 0.999 and values.max() <= hi * 1.001, name
        assert values.min() < lo + 0.2 * (hi - lo) and values.max() > hi - 0.2 * (hi - lo), name
    # and the trace-level range check rejects a latent forced outside its range
    trace, start, _, _ = _episode()
    bad = {**trace, "burn_multiplier": np.full_like(trace["burn_multiplier"], 2.0)}
    assert any("burn_multiplier" in p and "outside" in p for p in verifier._latent_problems(CFG, bad, start))
