"""Primary, safe-divert, and loss are distinct L1 outcomes."""

from __future__ import annotations

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
from naigos.env.flight_env import NaigosEnv, RewardTerms
from naigos.env.resource import OUTCOME_DIVERTED, OUTCOME_PRIMARY
from naigos.rl.reward import RewardWeights, compute


CFG = EnvConfig(
    n_blue=1,
    n_threat=2,
    n_threat_active=0,
    k_threat_obs=1,
    k_friend_obs=0,
    max_steps=8,
    detection=DetectionConfig(los_samples=8),
    terrain=TerrainConfig(nx=32, ny=32, cell=1_000.0),
    resource=ResourceWorldConfig(model=RESOURCE_WORLD_L1, gate_arrival_radius=1_000.0),
)


def _terms(**values):
    fields = {name: jnp.zeros(1) for name in RewardTerms._fields}
    fields.update({name: jnp.asarray([value]) for name, value in values.items()})
    return RewardTerms(**fields)


def test_both_abstract_gates_are_inside_the_play_area():
    state, _ = NaigosEnv(CFG).reset(jax.random.PRNGKey(1))
    gates = np.asarray(state.resource.diversion_pos[0])
    x0, x1, y0, y1 = np.asarray(state.bounds)
    assert gates.shape == (2, 3)
    assert np.all((gates[:, 0] >= x0) & (gates[:, 0] <= x1))
    assert np.all((gates[:, 1] >= y0) & (gates[:, 1] <= y1))
    assert bool(state.resource.diversion_active.all())


def test_reaching_a_gate_is_terminal_safe_diversion_not_primary_completion():
    env = NaigosEnv(CFG)
    state, _ = env.reset(jax.random.PRNGKey(2))
    gates = state.resource.diversion_pos.at[0, 0].set(state.air.pos[0])
    state = state._replace(resource=state.resource._replace(diversion_pos=gates))
    after, _, terms, done, info = env.step(state, jnp.zeros((1, 3)))
    assert bool(done)
    assert bool(after.resource.diverted[0])
    assert bool(after.alive[0])
    assert not bool(after.reached[0])
    assert float(terms.diverted[0]) == 1.0
    assert float(terms.arrived[0]) == 0.0
    assert int(after.resource.outcome[0]) == OUTCOME_DIVERTED
    assert int(info["n_diverted"]) == 1

    frozen, _, again, *_ = env.step(after, jnp.ones((1, 3)))
    assert float(again.diverted[0]) == 0.0
    np.testing.assert_array_equal(frozen.air.pos, after.air.pos)


def test_primary_arrival_takes_task_credit_without_diversion_credit():
    env = NaigosEnv(CFG)
    state, _ = env.reset(jax.random.PRNGKey(4))
    state = state._replace(objective=state.air.pos)
    after, _, terms, done, _ = env.step(state, jnp.zeros((1, 3)))
    assert bool(done)
    assert bool(after.reached[0])
    assert not bool(after.resource.diverted[0])
    assert float(terms.arrived[0]) == 1.0
    assert float(terms.diverted[0]) == 0.0
    assert int(after.resource.outcome[0]) == OUTCOME_PRIMARY


def test_reward_orders_primary_above_divert_above_timeout_and_loss():
    weights = RewardWeights()
    primary = float(compute(_terms(arrived=1.0), weights)[0])
    divert = float(compute(_terms(diverted=1.0), weights)[0])
    timeout = float(compute(_terms(), weights)[0])
    loss = float(compute(_terms(out_of_fuel=1.0), weights)[0])
    assert primary > divert > timeout > loss
