"""L1 fuel integration and terminal semantics."""

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
from naigos.env.flight_env import NaigosEnv
from naigos.env.resource import OUTCOME_FUEL
from naigos.rl.reward import constraint_cost


CFG = EnvConfig(
    n_blue=1,
    n_threat=2,
    n_threat_active=0,
    k_threat_obs=1,
    k_friend_obs=0,
    max_steps=8,
    detection=DetectionConfig(los_samples=8),
    terrain=TerrainConfig(nx=32, ny=32, cell=1_000.0),
    resource=ResourceWorldConfig(model=RESOURCE_WORLD_L1),
)


def test_initial_fuel_and_reserve_scale_with_direct_route_fuel_to_go():
    fixed = CFG.replace(
        resource=ResourceWorldConfig(
            model=RESOURCE_WORLD_L1,
            fuel_budget_ratio=(1.5, 1.5),
            reserve_ratio=(0.2, 0.2),
            burn_multiplier=(1.0, 1.0),
        )
    )
    state, _ = NaigosEnv(fixed).reset(jax.random.PRNGKey(0))
    distance = jnp.linalg.norm((state.objective - state.air.pos)[:, :2], axis=-1)
    throttle = (fixed.airframe.v_init - fixed.airframe.v_stall) / (
        fixed.airframe.v_max - fixed.airframe.v_stall
    )
    direct = distance / fixed.airframe.v_init * (
        fixed.airframe.burn_base + fixed.airframe.burn_throttle * throttle
    )
    np.testing.assert_allclose(state.resource.initial_fuel, direct * 1.5, rtol=2e-6)
    np.testing.assert_allclose(state.resource.reserve_fuel, direct * 0.2, rtol=2e-6)


def test_burn_multiplier_scales_the_existing_burn_equation():
    one = CFG.replace(resource=ResourceWorldConfig(model=RESOURCE_WORLD_L1, burn_multiplier=(1.0, 1.0)))
    two = CFG.replace(resource=ResourceWorldConfig(model=RESOURCE_WORLD_L1, burn_multiplier=(2.0, 2.0)))
    key = jax.random.PRNGKey(3)
    s1, _ = NaigosEnv(one).reset(key)
    s2, _ = NaigosEnv(two).reset(key)
    action = jnp.array([[0.0, 0.0, 0.5]])
    n1, *_ = NaigosEnv(one).step(s1, action)
    n2, *_ = NaigosEnv(two).step(s2, action)
    used1 = s1.air.fuel - n1.air.fuel
    used2 = s2.air.fuel - n2.air.fuel
    np.testing.assert_allclose(used2, used1 * 2.0, rtol=2e-5)


def test_first_fuel_exhaustion_is_one_terminal_loss_and_one_cmdp_cost():
    env = NaigosEnv(CFG)
    state, _ = env.reset(jax.random.PRNGKey(9))
    state = state._replace(air=state.air._replace(fuel=jnp.array([0.05])))
    action = jnp.zeros((1, 3))

    dead, _, first, done, info = env.step(state, action)
    assert bool(done)
    assert not bool(dead.alive[0])
    assert float(first.out_of_fuel[0]) == 1.0
    assert float(constraint_cost(first)[0]) == 1.0
    assert int(dead.resource.outcome[0]) == OUTCOME_FUEL
    assert bool(info["agent_done"][0])

    frozen, _, second, _, _ = env.step(dead, action)
    assert float(second.out_of_fuel[0]) == 0.0
    assert float(constraint_cost(second)[0]) == 0.0
    np.testing.assert_array_equal(frozen.air.pos, dead.air.pos)
    assert int(frozen.resource.outcome[0]) == OUTCOME_FUEL
