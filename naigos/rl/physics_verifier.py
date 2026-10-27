"""Independent NumPy verifier for weather, vehicle, mission and urban evidence.

No function in this module imports or calls the JAX transition.  The formulas
are intentionally restated so a defect in the training implementation can be
detected instead of agreed with by construction.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from ..env.config import EnvConfig


def _mix32(x):
    x = np.asarray(x, dtype=np.uint32)
    with np.errstate(over="ignore"):
        x = (x ^ (x >> np.uint32(16))) * np.uint32(0x7FEB352D)
        x = (x ^ (x >> np.uint32(15))) * np.uint32(0x846CA68B)
    return x ^ (x >> np.uint32(16))


def counter_normal(stream_key, step_index: int, shape, substream: int = 0):
    n = int(np.prod(shape))
    idx = np.arange(n, dtype=np.uint32)
    key = np.asarray(stream_key, dtype=np.uint32)
    with np.errstate(over="ignore"):
        base = (key[0] ^ key[1] * np.uint32(0x9E3779B9)
                ^ np.uint32(step_index) * np.uint32(0x85EBCA6B)
                ^ np.uint32(substream * 0xC2B2AE35))
        draws = [
            (_mix32(base + idx + np.uint32((k + 1) * 0x27D4EB2D)).astype(np.float32) + 0.5)
            / np.float32(2**32)
            for k in range(6)
        ]
    return ((np.stack(draws).sum(0) - 3.0) * np.sqrt(np.float32(2.0))).reshape(shape)


def _interp(altitude, knots, values):
    return np.interp(np.asarray(altitude, dtype=np.float64), knots, values)


def weather_profile(cfg, altitude):
    w = cfg.weather
    wind = np.stack([
        _interp(altitude, w.altitude_knots_m, [v[i] for v in w.wind_enu_mps]) for i in range(3)
    ], axis=-1)
    return (
        wind,
        _interp(altitude, w.altitude_knots_m, w.density_ratio),
        _interp(altitude, w.altitude_knots_m, w.turbulence_intensity_mps) * w.severity_scale,
    )


def _terrain_height(hmap, cfg, x, y):
    t = cfg.terrain
    fx = np.clip(np.asarray(x) / t.cell, 0.0, t.nx - 1.0)
    fy = np.clip(np.asarray(y) / t.cell, 0.0, t.ny - 1.0)
    x0, y0 = np.floor(fx).astype(int), np.floor(fy).astype(int)
    x1, y1 = np.minimum(x0 + 1, t.nx - 1), np.minimum(y0 + 1, t.ny - 1)
    tx, ty = fx - x0, fy - y0
    a = hmap[y0, x0] * (1 - tx) + hmap[y0, x1] * tx
    b = hmap[y1, x0] * (1 - tx) + hmap[y1, x1] * tx
    return a * (1 - ty) + b * ty


def _urban_sample(grid, env_origin_utm_m, hmap, cfg, x, y):
    x0, y0 = grid.origin_in_env(env_origin_utm_m)
    c = grid.meta.cell_m
    ix = np.floor((np.asarray(x) - x0) / c).astype(int)
    iy = np.floor((np.asarray(y) - y0) / c).astype(int)
    inside = (ix >= 0) & (ix < grid.meta.nx) & (iy >= 0) & (iy < grid.meta.ny)
    sx, sy = np.clip(ix, 0, grid.meta.nx - 1), np.clip(iy, 0, grid.meta.ny - 1)
    covered = inside & (grid.coverage[sy, sx] == 1)
    building = covered & (grid.building[sy, sx] == 1)
    terrain = _terrain_height(hmap, cfg, x, y)
    top = np.where(covered, grid.top_amsl_m[sy, sx], terrain)
    return top, building, covered


def _swept_building(grid, env_origin, hmap, cfg, p0, p1):
    s = np.linspace(0.0, 1.0, cfg.urban.collision_samples + 1)
    p = p0[:, None, :] + s[None, :, None] * (p1 - p0)[:, None, :]
    top, building, _ = _urban_sample(grid, env_origin, hmap, cfg, p[..., 0], p[..., 1])
    return np.any(building & (p[..., 2] - cfg.urban.capsule_radius_m <= top), axis=-1)


@dataclasses.dataclass
class PhysicsVerificationReport:
    steps: int
    max_gust_error: float
    max_velocity_error: float
    max_vehicle_error: float
    building_collisions: int
    mission_completions: int
    mismatches: list[str]

    @property
    def ok(self):
        return not self.mismatches


def verify_physics_trace(
    cfg: EnvConfig,
    hmap,
    traj,
    *,
    urban_grid=None,
    env_origin_utm_m=None,
    expected_urban_digest: str | None = None,
    tol: float = 4e-3,
) -> PhysicsVerificationReport:
    """Verify the explicit physics evidence emitted by ``NaigosEnv.rollout``."""
    steps = len(np.asarray(traj["pos"]))
    mismatches: list[str] = []
    max_gust = max_vel = max_vehicle = 0.0

    if cfg.urban.enabled:
        if urban_grid is None or env_origin_utm_m is None:
            mismatches.append("urban model has no verifier artifact/origin")
        elif expected_urban_digest and urban_grid.hashes.content_sha256 != expected_urban_digest:
            mismatches.append("urban artifact digest mismatch")

    for t in range(steps):
        pos = np.asarray(traj["pos"][t], dtype=np.float64)
        p0 = np.asarray(traj["pos_before"][t], dtype=np.float64)
        speed = np.asarray(traj["speed"][t], dtype=np.float64)
        psi = np.asarray(traj["psi"][t], dtype=np.float64)
        gamma = np.asarray(traj["gamma"][t], dtype=np.float64)

        if cfg.weather.enabled:
            alt = np.asarray(traj["weather_altitude_input"][t], dtype=np.float64)
            tas0 = np.asarray(traj["weather_airspeed_input"][t], dtype=np.float64)
            wind, rho, background = weather_profile(cfg, alt)
            mechanical = np.asarray(traj["mechanical_turbulence"][t], dtype=np.float64)
            sigma = np.sqrt(background**2 + mechanical**2)
            scales = np.asarray(cfg.weather.scale_length_m, dtype=np.float64)
            a = np.exp(-cfg.dt / np.maximum(scales[None, :] / np.maximum(tas0[:, None], 1.0), 1e-3))
            key = np.asarray(traj["weather_stream_key"][t], dtype=np.uint32)
            step_index = int(np.asarray(traj["step_index_before"][t]))
            noise = counter_normal(key, step_index, np.asarray(traj["gust_before"][t]).shape)
            want_gust = a * np.asarray(traj["gust_before"][t]) + np.sqrt(np.maximum(1 - a * a, 0)) * sigma[:, None] * noise
            want_gust = np.where(sigma[:, None] > 0, np.clip(
                want_gust, -cfg.weather.gust_clip_sigma * sigma[:, None],
                cfg.weather.gust_clip_sigma * sigma[:, None]), 0.0)
            ge = np.max(np.abs(want_gust - np.asarray(traj["gust_enu"][t])))
            max_gust = max(max_gust, float(ge))
            if ge > tol:
                mismatches.append(f"t={t}: gust recursion mismatch {ge:.3e}")
            if np.max(np.abs(wind - np.asarray(traj["mean_wind_enu"][t]))) > tol:
                mismatches.append(f"t={t}: wind-profile interpolation mismatch")
            if np.max(np.abs(rho - np.asarray(traj["density_ratio"][t]))) > tol:
                mismatches.append(f"t={t}: density interpolation mismatch")
        else:
            wind = np.zeros_like(pos)
            want_gust = np.zeros_like(pos)

        cg = np.cos(gamma)
        air = np.stack([speed * cg * np.cos(psi), speed * cg * np.sin(psi), speed * np.sin(gamma)], -1)
        ground = air + wind + want_gust
        ve = max(
            np.max(np.abs(air - np.asarray(traj["air_velocity"][t]))),
            np.max(np.abs(ground - np.asarray(traj["ground_velocity"][t]))),
        )
        max_vel = max(max_vel, float(ve))
        if ve > tol:
            mismatches.append(f"t={t}: air/ground velocity mismatch {ve:.3e}")

        if cfg.weather.enabled:
            # Independent density-scaled vehicle/fuel integration. Skip agents
            # whose post-terminal state is frozen by the environment.
            af = cfg.airframe
            p = p0.copy()
            v = np.asarray(traj["speed_before"][t], dtype=np.float64).copy()
            heading = np.asarray(traj["psi_before"][t], dtype=np.float64).copy()
            path = np.asarray(traj["gamma_before"][t], dtype=np.float64).copy()
            bank = np.asarray(traj["phi_before"][t], dtype=np.float64).copy()
            fuel = np.asarray(traj["fuel_before_transition"][t], dtype=np.float64).copy()
            action = np.clip(np.asarray(traj["action_executed"][t], dtype=np.float64), -1.0, 1.0)
            multiplier = np.asarray(traj["burn_multiplier"][t], dtype=np.float64)
            authority = np.clip(np.sqrt(np.clip(rho, 0.35, 1.25)), 0.55, 1.10)
            h = cfg.dt / cfg.substeps
            for _ in range(cfg.substeps):
                phi_cmd = action[:, 0] * np.arccos(1.0 / af.n_max)
                gamma_lim = np.minimum(
                    af.gamma_max,
                    np.arcsin(np.clip(af.roc_max / np.maximum(v, af.v_stall), -1.0, 1.0)),
                )
                gamma_cmd = action[:, 1] * gamma_lim * authority
                speed_cmd = af.v_stall + 0.5 * (action[:, 2] + 1.0) * (af.v_max - af.v_stall)
                bank += (phi_cmd - bank) * np.minimum(h * authority / af.tau_bank, 1.0)
                path += (gamma_cmd - path) * np.minimum(h * authority / af.tau_gamma, 1.0)
                v += (speed_cmd - v) * np.minimum(h * authority / af.tau_speed, 1.0)
                true_stall = af.v_stall / np.maximum(np.sqrt(np.clip(rho, 0.35, 1.25)), 0.5)
                v = np.clip(v, true_stall, af.v_max)
                path = np.clip(path, -af.gamma_max, af.gamma_max)
                heading += h * (9.80665 * np.tan(bank) / np.maximum(v, true_stall))
                heading = np.arctan2(np.sin(heading), np.cos(heading))
                cg2 = np.cos(path)
                av = np.stack([v * cg2 * np.cos(heading), v * cg2 * np.sin(heading), v * np.sin(path)], -1)
                p += h * (av + wind + want_gust)
                p[:, 2] = np.minimum(p[:, 2], af.ceiling)
                throttle = np.clip((v - true_stall) / np.maximum(af.v_max - true_stall, 1.0), 0.0, 1.0)
                load = 1.0 / np.maximum(np.cos(bank), 1e-3)
                demand = np.clip(1.0 / np.clip(rho, 0.35, 1.25), 0.8, 2.0)
                burn = (af.burn_base + af.burn_throttle * throttle * demand
                        + af.burn_maneuver * np.abs(load - 1.0) / authority)
                fuel = np.maximum(fuel - h * burn * multiplier, 0.0)
                v = np.where(fuel <= 0.0, np.maximum(v - 5.0 * h, true_stall), v)
            frozen = np.all(np.abs(pos - p0) < 1e-8, axis=-1)
            errs = np.stack([
                np.max(np.abs(p - pos), axis=-1), np.abs(v - speed),
                np.abs(path - gamma), np.abs(bank - np.asarray(traj["phi"][t])),
                np.abs(fuel - np.asarray(traj["fuel"][t])),
            ], axis=-1)
            err = float(np.max(np.where(frozen[:, None], 0.0, errs)))
            max_vehicle = max(max_vehicle, err)
            if err > 20 * tol:
                mismatches.append(f"t={t}: density-scaled vehicle/fuel mismatch {err:.3e}")

        if cfg.urban.enabled and urban_grid is not None and env_origin_utm_m is not None:
            top, _, _ = _urban_sample(urban_grid, env_origin_utm_m, hmap, cfg, pos[:, 0], pos[:, 1])
            ce = np.max(np.abs((pos[:, 2] - top) - np.asarray(traj["obstacle_clearance"][t])))
            if ce > tol:
                mismatches.append(f"t={t}: obstacle clearance mismatch {ce:.3e}")
            hit = _swept_building(urban_grid, env_origin_utm_m, hmap, cfg, p0, pos)
            if not np.array_equal(hit, np.asarray(traj["building_collision"][t], dtype=bool)):
                mismatches.append(f"t={t}: swept building collision mismatch")

        # Fixed-shape mission invariants and explicit deadline/outcome checks.
        if cfg.mission.enabled:
            stage = np.asarray(traj["mission_stage"][t])
            before = np.asarray(traj["mission_stage_before_transition"][t])
            mask = np.asarray(traj["mission_stage_mask"][t])
            if np.any(stage < before) or np.any(stage - before > 1):
                mismatches.append(f"t={t}: mission stage progression is not ordered")
            if np.any(stage >= mask.sum(axis=-1)):
                mismatches.append(f"t={t}: mission stage indexes padding")
            deadline = np.asarray(traj["mission_deadline_step"][t])
            failed = np.asarray(traj["mission_failed"][t], dtype=bool)
            completed = np.asarray(traj["mission_completed"][t], dtype=bool)
            recovered = np.asarray(traj["mission_recovered"][t], dtype=bool)
            if np.any(completed & recovered):
                mismatches.append(f"t={t}: primary completion and recovery both set")
            expired = int(np.asarray(traj["step_index_before"][t])) + 1 >= deadline
            if np.any(expired & ~completed & ~recovered & ~failed):
                mismatches.append(f"t={t}: missed mission deadline without terminal state")

    return PhysicsVerificationReport(
        steps=steps,
        max_gust_error=max_gust,
        max_velocity_error=max_vel,
        max_vehicle_error=max_vehicle,
        building_collisions=int(np.asarray(traj["building_collision"]).sum()),
        mission_completions=int(np.asarray(traj["mission_completed"]).any(axis=0).sum()),
        mismatches=mismatches,
    )
