"""Point-mass 3D flight dynamics with a setpoint action interface.

Following the Nomos choice of *setpoints, not raw torques*: the policy commands
a bank angle, a flight-path angle and a throttle; first-order actuator lags and
hard envelope limits turn those into achievable motion. That keeps the action
space small and bounded while the airframe stays physically plausible.

State (per aircraft):
    pos    (3,)  local ENU metres, z is AMSL
    speed  ()    true airspeed, m/s
    psi    ()    heading, rad (0 = +x, CCW positive)
    gamma  ()    flight-path angle, rad
    phi    ()    bank angle, rad
    fuel   ()    kg

The coordinated-turn relation `psi_dot = g tan(phi) / V` is what couples the
g-limit to the achievable turn rate: at high speed the same bank buys you less
heading rate, so evasive turns cost energy. That coupling is the whole reason a
2D bicycle model would not do.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .config import G0, AirframeConfig


class AircraftState(NamedTuple):
    pos: jax.Array  # (..., 3)
    speed: jax.Array  # (...,)
    psi: jax.Array  # (...,)
    gamma: jax.Array  # (...,)
    phi: jax.Array  # (...,)
    fuel: jax.Array  # (...,)


def max_bank(cfg: AirframeConfig) -> float:
    """Bank angle at the load-factor limit: n = 1/cos(phi)."""
    return float(jnp.arccos(1.0 / cfg.n_max))


def load_factor(phi: jax.Array) -> jax.Array:
    """Load factor for a coordinated level turn at bank `phi`."""
    return 1.0 / jnp.maximum(jnp.cos(phi), 1e-3)


def decode_action(cfg: AirframeConfig, action: jax.Array, speed: jax.Array):
    """Map a bounded action in [-1, 1]^3 to physical setpoints.

    action[0] -> bank command, scaled by the g-limit
    action[1] -> flight-path-angle command, scaled by BOTH gamma_max and the
                 rate-of-climb limit (which bites at low speed)
    action[2] -> speed command between stall and v_max
    """
    a = jnp.clip(action, -1.0, 1.0)
    phi_lim = jnp.arccos(1.0 / cfg.n_max)
    phi_cmd = a[..., 0] * phi_lim

    gamma_from_roc = jnp.arcsin(jnp.clip(cfg.roc_max / jnp.maximum(speed, cfg.v_stall), -1.0, 1.0))
    gamma_lim = jnp.minimum(cfg.gamma_max, gamma_from_roc)
    gamma_cmd = a[..., 1] * gamma_lim

    speed_cmd = cfg.v_stall + 0.5 * (a[..., 2] + 1.0) * (cfg.v_max - cfg.v_stall)
    return phi_cmd, gamma_cmd, speed_cmd


def step(
    cfg: AirframeConfig,
    st: AircraftState,
    action: jax.Array,
    dt: float,
    burn_multiplier: jax.Array | float = 1.0,
) -> AircraftState:
    """Integrate one step. Semi-implicit Euler; dt of ~1 s is stable at these lags.

    ``burn_multiplier`` is the L1 resource-world's episode-local uncertainty.
    Its default is exactly one so all baseline callers retain the original
    transition without a second fuel calculation in the environment.
    """
    phi_cmd, gamma_cmd, speed_cmd = decode_action(cfg, action, st.speed)

    # first-order actuator lags
    phi = st.phi + (phi_cmd - st.phi) * jnp.minimum(dt / cfg.tau_bank, 1.0)
    gamma = st.gamma + (gamma_cmd - st.gamma) * jnp.minimum(dt / cfg.tau_gamma, 1.0)
    speed = st.speed + (speed_cmd - st.speed) * jnp.minimum(dt / cfg.tau_speed, 1.0)

    # envelope clamps (violations are recorded by the verifier, not silently ok)
    speed = jnp.clip(speed, cfg.v_stall, cfg.v_max)
    gamma = jnp.clip(gamma, -cfg.gamma_max, cfg.gamma_max)

    # coordinated turn
    psi = st.psi + dt * (G0 * jnp.tan(phi) / jnp.maximum(speed, cfg.v_stall))
    psi = jnp.arctan2(jnp.sin(psi), jnp.cos(psi))

    cg = jnp.cos(gamma)
    vel = jnp.stack([speed * cg * jnp.cos(psi), speed * cg * jnp.sin(psi), speed * jnp.sin(gamma)], axis=-1)
    pos = st.pos + dt * vel
    pos = pos.at[..., 2].set(jnp.minimum(pos[..., 2], cfg.ceiling))

    throttle = (speed - cfg.v_stall) / (cfg.v_max - cfg.v_stall)
    n = load_factor(phi)
    burn = cfg.burn_base + cfg.burn_throttle * throttle + cfg.burn_maneuver * jnp.abs(n - 1.0)
    fuel = jnp.maximum(st.fuel - dt * burn * burn_multiplier, 0.0)

    # out of fuel: the airframe glides, it does not hover
    dry = (fuel <= 0.0).astype(jnp.float32)
    speed = speed * (1 - dry) + jnp.maximum(speed - 5.0 * dt, cfg.v_stall) * dry

    return AircraftState(pos=pos, speed=speed, psi=psi, gamma=gamma, phi=phi, fuel=fuel)


def step_atmosphere(
    cfg: AirframeConfig,
    st: AircraftState,
    action: jax.Array,
    dt: float,
    mean_wind_enu: jax.Array,
    gust_enu: jax.Array,
    density_ratio: jax.Array,
    burn_multiplier: jax.Array | float = 1.0,
) -> AircraftState:
    """Opt-in weather transition; legacy ``step`` remains unchanged.

    Air-relative heading and speed produce the aerodynamic velocity.  Only
    position sees mean wind plus gust.  Density affects three documented
    quantities: actuator/longitudinal authority scales with ``sqrt(rho/rho0)``,
    the true-airspeed stall floor scales inversely with it (constant equivalent
    stall speed), and propulsive fuel demand scales inversely with density.
    All ratios are bounded because this is an engineering training model, not a
    full engine deck.
    """
    rho = jnp.clip(density_ratio, 0.35, 1.25)
    authority = jnp.clip(jnp.sqrt(rho), 0.55, 1.10)
    phi_cmd, gamma_cmd, speed_cmd = decode_action(cfg, action, st.speed)
    gamma_cmd = gamma_cmd * authority

    phi = st.phi + (phi_cmd - st.phi) * jnp.minimum(dt * authority / cfg.tau_bank, 1.0)
    gamma = st.gamma + (gamma_cmd - st.gamma) * jnp.minimum(dt * authority / cfg.tau_gamma, 1.0)
    speed = st.speed + (speed_cmd - st.speed) * jnp.minimum(dt * authority / cfg.tau_speed, 1.0)

    true_stall = cfg.v_stall / jnp.maximum(jnp.sqrt(rho), 0.5)
    speed = jnp.clip(speed, true_stall, cfg.v_max)
    gamma = jnp.clip(gamma, -cfg.gamma_max, cfg.gamma_max)
    psi = st.psi + dt * (G0 * jnp.tan(phi) / jnp.maximum(speed, true_stall))
    psi = jnp.arctan2(jnp.sin(psi), jnp.cos(psi))

    cg = jnp.cos(gamma)
    air_vel = jnp.stack(
        [speed * cg * jnp.cos(psi), speed * cg * jnp.sin(psi), speed * jnp.sin(gamma)], axis=-1
    )
    pos = st.pos + dt * (air_vel + mean_wind_enu + gust_enu)
    pos = pos.at[..., 2].set(jnp.minimum(pos[..., 2], cfg.ceiling))

    throttle = jnp.clip((speed - true_stall) / jnp.maximum(cfg.v_max - true_stall, 1.0), 0.0, 1.0)
    n = load_factor(phi)
    propulsive_demand = jnp.clip(1.0 / rho, 0.8, 2.0)
    burn = (
        cfg.burn_base
        + cfg.burn_throttle * throttle * propulsive_demand
        + cfg.burn_maneuver * jnp.abs(n - 1.0) / authority
    )
    fuel = jnp.maximum(st.fuel - dt * burn * burn_multiplier, 0.0)
    dry = (fuel <= 0.0).astype(jnp.float32)
    speed = speed * (1 - dry) + jnp.maximum(speed - 5.0 * dt, true_stall) * dry
    return AircraftState(pos=pos, speed=speed, psi=psi, gamma=gamma, phi=phi, fuel=fuel)


def velocity(st: AircraftState) -> jax.Array:
    """Air-relative velocity vector (..., 3)."""
    cg = jnp.cos(st.gamma)
    return jnp.stack(
        [st.speed * cg * jnp.cos(st.psi), st.speed * cg * jnp.sin(st.psi), st.speed * jnp.sin(st.gamma)],
        axis=-1,
    )


def ground_velocity(st: AircraftState, mean_wind_enu: jax.Array, gust_enu: jax.Array) -> jax.Array:
    """Inertial velocity for the weather model."""
    return velocity(st) + mean_wind_enu + gust_enu
