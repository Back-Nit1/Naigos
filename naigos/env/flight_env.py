"""The Naigos environment: vectorized JAX 3D contested-airspace flight.

Contract (copied from Nomos): pure `reset` / `step` over **one** world, both
`jit`-able and `vmap`-able over a batch of worlds. Every array is fixed-size and
padded; nothing is data-dependent in shape; no Python control flow on traced
values.

    env = NaigosEnv(cfg)
    state, obs = env.reset(key)
    state, obs, terms, done, info = env.step(state, action)

`terms` is a `RewardTerms` of *raw, unweighted* quantities. Weighting lives in
`naigos.rl.reward` so the curriculum can re-weight without touching the physics,
and so `naigos.rl.verifier` can recheck the constraint channel independently.

BLUE HAS NO WEAPON. `action` is (n_blue, 3) = (bank, flight-path angle,
throttle). See config.BLUE_ACTION_NAMES and tests/test_invariant.py.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from . import airframe as af_mod
from . import detection as det_mod
from . import mission as mission_mod
from . import obs as obs_mod
from . import resource as resource_mod
from . import terrain as terrain_mod
from . import threats as threats_mod
from . import urban as urban_mod
from . import weather as weather_mod
from .airframe import AircraftState
from .config import (
    ROUTE_FAMILY_NAMES,
    TRAIN_ROUTE_CENTERS_DEG,
    EnvConfig,
    resource_world_problems,
    mission_problems,
    route_protocol_problems,
    weather_problems,
    urban_problems,
)
from .threats import RedScales, ThreatState, red_scales_from_cfg

# EnvState.route for the legacy placement: due east, family west_east, legacy flag.
_LEGACY_ROUTE = (0.0, float(ROUTE_FAMILY_NAMES.index("west_east")), 1.0)


class EnvState(NamedTuple):
    air: AircraftState
    threats: ThreatState
    lock: jax.Array  # (T, B) track quality in [0, 1]
    dwell: jax.Array  # (T, B) seconds of continuous lock
    alive: jax.Array  # (B,) bool
    reached: jax.Array  # (B,) bool
    objective: jax.Array  # (B, 3)
    prev_dist: jax.Array  # (B,) horizontal range to objective at the last step
    hmap: jax.Array  # (ny, nx) DEM in metres AMSL
    bounds: jax.Array  # (4,) x_min, x_max, y_min, y_max -- this world's play area
    t: jax.Array  # () int32 step counter
    key: jax.Array
    # (3,) float32: route bearing (rad, direction of travel, psi convention),
    # family index into config.ROUTE_FAMILY_NAMES, and 1.0 if the episode used
    # the legacy west-to-east placement. Read only by respawn, which re-tasks
    # along the same route, and by per-family evaluation. Nothing in step,
    # reward or observation depends on it.
    route: jax.Array
    # The red curriculum's annealed knobs, traced rather than read off the
    # static `cfg` (next-steps.md G-4) -- see `naigos.env.threats.RedScales`.
    # Carried for the life of the episode and passed on at auto-reset, so a
    # promotion changes what a *new* episode spawns with, without retracing
    # anything that was already compiled for a different level.
    red_scales: RedScales
    # Always present so the pytree is fixed-shape. baseline-v0 carries neutral
    # values and never enters the resource transition branch.
    resource: resource_mod.ResourceState
    # Fixed-shape neutral state in legacy models; active only when the explicit
    # vehicle-weather environment identity is selected.
    weather: weather_mod.WeatherState
    mission: mission_mod.MissionState


class RewardTerms(NamedTuple):
    """Raw, unweighted per-agent quantities. Signs are applied in reward.py."""

    progress: jax.Array  # (B,) metres closed on the objective this step
    exposure: jax.Array  # (B,) max detection probability against this aircraft
    lock_level: jax.Array  # (B,) max track quality against this aircraft
    envelope_dwell: jax.Array  # (B,) 1 if inside any lethal envelope
    arrived: jax.Array  # (B,) 1 on the step the objective is reached
    diverted: jax.Array  # (B,) 1 on the step an L1 recovery gate is reached
    shotdown: jax.Array  # (B,) 1 on the step the aircraft is killed
    fuel_used: jax.Array  # (B,) kg burned this step
    g_excess: jax.Array  # (B,) load factor above 1
    terrain_violation: jax.Array  # (B,) 1 if below the AGL floor (or underground)
    ceiling_violation: jax.Array  # (B,)
    stall_violation: jax.Array  # (B,)
    bounds_violation: jax.Array  # (B,)
    out_of_fuel: jax.Array  # (B,)
    edge_proximity: jax.Array  # (B,) 0 inside, ramps to 1 at the map boundary
    building_collision: jax.Array  # (B,) evidence-grade structure impact


class NaigosEnv:
    def __init__(
        self,
        cfg: EnvConfig,
        hmap: jax.Array | None = None,
        red_policy=None,
        urban_obstacles=None,
        env_origin_utm_m=None,
    ):
        """`hmap` pins a fixed DEM (the real-data path). If None, each reset
        generates synthetic terrain -- useful for tests and for domain
        randomisation, but the headline result should be run on a real DEM.
        """
        problems = (route_protocol_problems(cfg) + resource_world_problems(cfg.resource)
                    + weather_problems(cfg.weather) + mission_problems(cfg.mission, cfg.resource)
                    + urban_problems(cfg.urban))
        if problems:
            raise ValueError("invalid route configuration:\n" + "\n".join(f"  - {p}" for p in problems))
        self.cfg = cfg
        self.fixed_hmap = hmap
        if cfg.urban.enabled:
            if urban_obstacles is None:
                raise ValueError("urban obstacle physics requires an explicitly validated artifact")
            self.urban_surface = urban_mod.from_artifact(urban_obstacles, env_origin_utm_m)
            max_wind = max(sum(x * x for x in w) ** 0.5 for w in cfg.weather.wind_enu_mps)
            max_turb = max(cfg.weather.turbulence_intensity_mps) * cfg.weather.gust_clip_sigma
            spacing = cfg.dt * (cfg.airframe.v_max + max_wind + max_turb) / cfg.urban.collision_samples
            if spacing > self.urban_surface.cell_m:
                raise ValueError(
                    f"urban collision sample spacing {spacing:.2f} m exceeds artifact cell "
                    f"{self.urban_surface.cell_m:.2f} m; increase collision_samples"
                )
        else:
            # Passing a presentation/viewer payload while disabled cannot alter
            # simulation state: it is intentionally ignored.
            self.urban_surface = None
        if red_policy is None:
            from ..rl.red_team import scripted_red

            red_policy = scripted_red
        self.red_policy = red_policy

    # ---------------------------------------------------------------- reset --
    def reset(self, key: jax.Array, red_scales: RedScales | None = None) -> tuple[EnvState, obs_mod.Observation]:
        """A fresh episode, with the route drawn as `cfg.route_mode` says.

        `red_scales` overrides `cfg`'s own curriculum fields for this episode
        (next-steps.md G-4); omitted, a fresh episode starts at `cfg`'s levels,
        exactly as before G-4. The training loop's auto-reset (`naigos.rl.ppo.
        make_train`) passes the ending episode's own `state.red_scales`, so an
        annealed level survives a mid-rollout respawn.
        """
        return self._reset(key, None, red_scales)

    def reset_route(
        self, key: jax.Array, route: jax.Array, red_scales: RedScales | None = None,
    ) -> tuple[EnvState, obs_mod.Observation]:
        """A fresh episode along a GIVEN route: `route` is (2,) [bearing rad, family id].

        This is how the frozen route bank is played (`naigos.rl.train.
        evaluate_route_bank`). It ignores `route_mode`: the bearing comes from the
        caller, everything else -- terrain, box, offsets, threats -- from `key`.
        jit/vmap-safe; `route` may be traced.
        """
        route = jnp.asarray(route, dtype=jnp.float32)
        bearing = jnp.arctan2(jnp.sin(route[0]), jnp.cos(route[0]))  # same (-pi, pi] as drawn routes
        return self._reset(key, jnp.stack([bearing, route[1], jnp.float32(0.0)]), red_scales)

    def _reset(
        self, key: jax.Array, route: jax.Array | None, red_scales: RedScales | None = None,
    ) -> tuple[EnvState, obs_mod.Observation]:
        cfg = self.cfg
        if red_scales is None:
            red_scales = red_scales_from_cfg(cfg)
        k_map, k_start, k_obj, k_thr, k_next = jax.random.split(key, 5)

        hmap = self.fixed_hmap if self.fixed_hmap is not None else terrain_mod.synthetic_terrain(k_map, cfg.terrain)

        if cfg.map_randomize:
            # its own split, so the flag-off key stream is untouched
            k_next, k_box = jax.random.split(k_next)
            bounds = self._sample_bounds(k_box)
        else:
            bounds = jnp.array([0.0, cfg.terrain.extent_x, 0.0, cfg.terrain.extent_y], dtype=jnp.float32)
        x0, x1, y0, y1 = self._box(bounds)
        w, h = x1 - x0, y1 - y0
        B = cfg.n_blue

        if route is None and cfg.route_mode == "legacy":
            # start on one edge, objective on the far edge: the direct line crosses
            # the whole threat field, so the naive baseline is genuinely exposed.
            lateral = y0 + jnp.linspace(0.25, 0.75, B) * h
            jitter = jax.random.uniform(k_start, (B,), minval=-0.04, maxval=0.04) * h
            inset = cfg.spawn_inset_frac
            start_xy = jnp.stack(
                [
                    jnp.full((B,), x0 + inset * w),
                    jnp.clip(lateral + jitter, y0 + 1.5 * inset * h, y0 + (1 - 1.5 * inset) * h),
                ],
                axis=-1,
            )

            obj_lat = y0 + jax.random.uniform(k_obj, (B,), minval=0.25, maxval=0.75) * h
            obj_xy = jnp.stack([jnp.full((B,), x0 + (1.0 - inset) * w), obj_lat], axis=-1)
            route = jnp.array(_LEGACY_ROUTE, dtype=jnp.float32)
        else:
            if route is None:
                # its own split again: neither the legacy nor the map_randomize
                # stream moves when the route is drawn
                k_next, k_route = jax.random.split(k_next)
                route = self._sample_route(k_route)
            start_xy, obj_xy = self._route_endpoints(route[0], (x0, x1, y0, y1), k_start, k_obj)

        g_start = terrain_mod.sample_height(hmap, cfg.terrain, start_xy[:, 0], start_xy[:, 1])
        g_obj = terrain_mod.sample_height(hmap, cfg.terrain, obj_xy[:, 0], obj_xy[:, 1])
        start = jnp.concatenate([start_xy, (g_start + 2_500.0)[:, None]], axis=-1)
        objective = jnp.concatenate([obj_xy, (g_obj + 1_000.0)[:, None]], axis=-1)

        corridor = jnp.stack([jnp.mean(start, axis=0), jnp.mean(objective, axis=0)], axis=0)
        tstate = threats_mod.spawn(k_thr, cfg, hmap, corridor, red_scales)

        to_obj = objective - start
        air, resource = resource_mod.initial_state(cfg, key, hmap, start, objective, bounds)
        weather = weather_mod.initial_state(cfg.weather, key, start[:, 2])
        mission = mission_mod.initial_state(
            cfg.mission, key, start, objective, resource.diversion_pos,
            resource.diversion_active, cfg.dt,
        )
        if cfg.mission.enabled:
            objective = mission_mod.current_target(mission)
            to_obj = objective - start

        state = EnvState(
            air=air,
            threats=tstate,
            lock=jnp.zeros((cfg.n_threat, B)),
            dwell=jnp.zeros((cfg.n_threat, B)),
            alive=jnp.ones((B,), dtype=bool),
            reached=jnp.zeros((B,), dtype=bool),
            objective=objective,
            prev_dist=jnp.linalg.norm(to_obj[:, :2], axis=-1),
            hmap=hmap,
            bounds=bounds,
            t=jnp.int32(0),
            key=k_next,
            route=route,
            red_scales=red_scales,
            resource=resource,
            weather=weather,
            mission=mission,
        )
        return state, self.observe(state)

    def _sample_route(self, key: jax.Array) -> jax.Array:
        """One training route: a family from `route_train_families`, uniformly,
        and a bearing uniform inside that family's +/- jitter sector.

        Only training centres can come out of here -- the config validator has
        already refused any family or jitter that would reach held-out geometry.
        """
        cfg = self.cfg
        k_fam, k_jit = jax.random.split(key)
        fams = cfg.route_train_families
        centres = jnp.deg2rad(jnp.array([TRAIN_ROUTE_CENTERS_DEG[f] for f in fams], dtype=jnp.float32))
        ids = jnp.array([ROUTE_FAMILY_NAMES.index(f) for f in fams], dtype=jnp.float32)
        i = jax.random.randint(k_fam, (), 0, len(fams))
        j = jnp.deg2rad(cfg.route_bearing_jitter_deg)
        bearing = centres[i] + jax.random.uniform(k_jit, (), minval=-j, maxval=j)
        return jnp.stack([jnp.arctan2(jnp.sin(bearing), jnp.cos(bearing)), ids[i], jnp.float32(0.0)])

    def _route_endpoints(self, bearing: jax.Array, box, k_start: jax.Array, k_obj: jax.Array):
        """Start and objective positions (B, 2) for a route flying along `bearing`.

        Geometry, in the "inset box" -- the play box shrunk by spawn_inset_frac on
        EVERY side, so both ends clear the boundary ramp (edge_margin_frac is
        smaller) whatever the direction:

          * each aircraft's start lies on the line through the inset box's centre
            offset cross-track by s_i, at the point where that line ENTERS the
            box; its objective on a line offset by o_i, where it LEAVES. For due
            east that is the legacy picture: west inset edge to east inset edge.
          * |s_i|, |o_i| <= route_lateral_frac * m, m the inset half short side,
            so every line passes through the box's inscribed disk and each end is
            at least m*sqrt(1 - f^2) from the centre along track (the length
            bound in config.route_min_length_guarantee).
          * the along-track reach is capped at the inset box's half LONG side, so a
            diagonal is no longer than a cardinal route along the long axis; the
            cap never binds on a cardinal route.

        Starts are spread across the track like the legacy starts (evenly plus
        jitter); objectives are drawn independently, also like legacy.
        """
        cfg = self.cfg
        B = cfg.n_blue
        x0, x1, y0, y1 = box
        inset = cfg.spawn_inset_frac
        sx0, sx1 = x0 + inset * (x1 - x0), x1 - inset * (x1 - x0)
        sy0, sy1 = y0 + inset * (y1 - y0), y1 - inset * (y1 - y0)
        cx, cy = 0.5 * (sx0 + sx1), 0.5 * (sy0 + sy1)
        a, b = 0.5 * (sx1 - sx0), 0.5 * (sy1 - sy0)
        m = jnp.minimum(a, b)
        reach = jnp.maximum(a, b)

        dx, dy = jnp.cos(bearing), jnp.sin(bearing)
        nx, ny = -dy, dx
        lim = cfg.route_lateral_frac * m

        spread = jnp.linspace(-0.8, 0.8, B) if B > 1 else jnp.zeros((1,))
        s = jnp.clip(spread + jax.random.uniform(k_start, (B,), minval=-0.1, maxval=0.1), -1.0, 1.0) * lim
        o = jax.random.uniform(k_obj, (B,), minval=-1.0, maxval=1.0) * lim

        def chord(off):
            # slab test of the line (c + off*n) + t*d against the inset box
            px, py = cx + off * nx, cy + off * ny
            inv_x = 1.0 / jnp.where(jnp.abs(dx) < 1e-9, 1e-9, dx)
            inv_y = 1.0 / jnp.where(jnp.abs(dy) < 1e-9, 1e-9, dy)
            tx0, tx1 = (sx0 - px) * inv_x, (sx1 - px) * inv_x
            ty0, ty1 = (sy0 - py) * inv_y, (sy1 - py) * inv_y
            t_in = jnp.maximum(jnp.minimum(tx0, tx1), jnp.minimum(ty0, ty1))
            t_out = jnp.minimum(jnp.maximum(tx0, tx1), jnp.maximum(ty0, ty1))
            return px, py, t_in, t_out

        spx, spy, t_in, _ = chord(s)
        opx, opy, _, t_out = chord(o)
        t_s = jnp.maximum(t_in, -reach)
        t_o = jnp.minimum(t_out, reach)
        start_xy = jnp.stack([spx + t_s * dx, spy + t_s * dy], axis=-1)
        obj_xy = jnp.stack([opx + t_o * dx, opy + t_o * dy], axis=-1)
        return start_xy, obj_xy

    def _sample_bounds(self, key: jax.Array) -> jax.Array:
        """Draw a random rectangular play area inside the fixed DEM grid.

        The grid is a static shape, so map-size variety has to live inside it:
        width uniform in [map_min_extent_m, extent_x], height = width * aspect
        clipped to [map_min_extent_m, extent_y], placed at a random offset that
        keeps the whole box on the grid.
        """
        cfg = self.cfg
        ex, ey = cfg.terrain.extent_x, cfg.terrain.extent_y
        k_w, k_a, k_x, k_y = jax.random.split(key, 4)
        w = jax.random.uniform(k_w, (), minval=min(cfg.map_min_extent_m, ex), maxval=ex)
        lo, hi = cfg.map_aspect_range
        aspect = jax.random.uniform(k_a, (), minval=lo, maxval=hi)
        h = jnp.clip(w * aspect, min(cfg.map_min_extent_m, ey), ey)
        x0 = jax.random.uniform(k_x, ()) * (ex - w)
        y0 = jax.random.uniform(k_y, ()) * (ey - h)
        return jnp.stack([x0, x0 + w, y0, y0 + h]).astype(jnp.float32)

    def _box(self, bounds: jax.Array):
        """(x_min, x_max, y_min, y_max) of the play area.

        With map_randomize off the box IS the grid, so it comes back as the same
        Python constants the env used before bounds existed: the default path
        stays bit-identical instead of picking up float32 rounding.
        """
        if not self.cfg.map_randomize:
            return 0.0, self.cfg.terrain.extent_x, 0.0, self.cfg.terrain.extent_y
        return bounds[0], bounds[1], bounds[2], bounds[3]

    # -------------------------------------------------------------- observe --
    def _geometry(self, state: EnvState):
        cfg = self.cfg
        tparams = threats_mod.per_threat_params(cfg, state.threats.kind, state.red_scales)
        det = det_mod.detection_probability(
            state.hmap,
            cfg.terrain,
            cfg.detection,
            state.air.pos,
            state.air.psi,
            state.threats.pos,
            tparams,
            state.threats.active,
            visibility_fn=self._urban_visibility(state.hmap) if cfg.urban.enabled else None,
        )
        if cfg.weather.enabled:
            det["pd"] = det["pd"] * state.weather.sensing_attenuation[None, :]
        # a dead or arrived aircraft stops generating returns
        live = (state.alive & ~state.reached & ~state.resource.diverted).astype(jnp.float32)
        det["pd"] = det["pd"] * live[None, :]
        return tparams, det

    def _urban_visibility(self, hmap):
        return lambda p_from, p_to: urban_mod.visibility(
            self.urban_surface, hmap, self.cfg.terrain, self.cfg.detection, p_from, p_to
        )

    def observe(self, state: EnvState) -> obs_mod.Observation:
        cfg = self.cfg
        tparams, det = self._geometry(state)
        ob = obs_mod.build(
            cfg,
            state.air.pos,
            state.air.psi,
            state.air.gamma,
            state.air.phi,
            state.air.speed,
            state.air.fuel,
            af_mod.velocity(state.air),
            state.alive,
            state.objective,
            det["alt_agl"],
            state.threats,
            tparams,
            det["pd"],
            state.lock,
            det["slant"],
            det["vis"],
            bounds=state.bounds,
        )
        if cfg.resource.enabled:
            extra = resource_mod.observation_features(cfg, state.air, state.objective, state.resource)
            ob = ob._replace(ego=jnp.concatenate([ob.ego, extra], axis=-1))
        if cfg.weather.enabled:
            extra = weather_mod.observation_features(cfg.weather, state.weather, state.air.psi)
            ob = ob._replace(ego=jnp.concatenate([ob.ego, extra], axis=-1))
        if cfg.mission.enabled:
            recovery_available = jnp.any(state.resource.diversion_active, axis=-1)
            extra = mission_mod.observation_features(
                cfg.mission, state.mission, state.air.pos, state.air.psi, state.t,
                recovery_available,
            )
            ob = ob._replace(ego=jnp.concatenate([ob.ego, extra], axis=-1))
        return ob

    # ----------------------------------------------------------------- step --
    def step(self, state: EnvState, action: jax.Array):
        """One env step. `action` is (n_blue, 3) in [-1, 1]."""
        cfg = self.cfg
        dt = cfg.dt
        key, k_kill, k_red = jax.random.split(state.key, 3)

        frozen = (~state.alive) | state.reached | state.resource.diverted
        if cfg.mission.enabled:
            frozen = frozen | state.mission.completed | state.mission.recovered | state.mission.failed
        fuel_before = state.air.fuel

        # sub-step the airframe: the decision rate (dt) is coarser than the rate
        # at which a 7 g turn actually needs integrating.
        h = dt / cfg.substeps
        air_next = state.air
        if cfg.weather.enabled:
            mechanical = (
                urban_mod.mechanical_wake_intensity(
                    self.urban_surface, state.hmap, cfg.terrain,
                    state.air.pos, state.weather.mean_wind_enu,
                )
                if cfg.urban.enabled
                else jnp.zeros((cfg.n_blue,), dtype=jnp.float32)
            )
            weather_next = weather_mod.step(
                cfg.weather, state.weather, state.air.pos[:, 2], state.air.speed, dt, state.t,
                mechanical,
            )
        else:
            weather_next = state.weather
        for _ in range(cfg.substeps):
            if cfg.weather.enabled:
                burn_multiplier = (state.resource.burn_multiplier
                                   if cfg.resource.enabled else 1.0)
                air_next = af_mod.step_atmosphere(
                    cfg.airframe, air_next, action, h,
                    weather_next.mean_wind_enu, weather_next.gust_enu,
                    weather_next.density_ratio, burn_multiplier,
                )
            elif cfg.resource.enabled:
                air_next = af_mod.step(
                    cfg.airframe, air_next, action, h, state.resource.burn_multiplier
                )
            else:
                air_next = af_mod.step(cfg.airframe, air_next, action, h)
        air = jax.tree.map(lambda new, old: jnp.where(_bcast(frozen, new), old, new), air_next, state.air)
        weather = weather_next._replace(
            mean_wind_enu=jnp.where(frozen[:, None], state.weather.mean_wind_enu, weather_next.mean_wind_enu),
            gust_enu=jnp.where(frozen[:, None], state.weather.gust_enu, weather_next.gust_enu),
            density_ratio=jnp.where(frozen, state.weather.density_ratio, weather_next.density_ratio),
            temperature_k=jnp.where(frozen, state.weather.temperature_k, weather_next.temperature_k),
            sensing_attenuation=jnp.where(
                frozen, state.weather.sensing_attenuation, weather_next.sensing_attenuation
            ),
            turbulence_intensity=jnp.where(
                frozen, state.weather.turbulence_intensity, weather_next.turbulence_intensity
            ),
            background_turbulence_intensity=jnp.where(
                frozen, state.weather.background_turbulence_intensity,
                weather_next.background_turbulence_intensity,
            ),
            mechanical_turbulence_intensity=jnp.where(
                frozen, state.weather.mechanical_turbulence_intensity,
                weather_next.mechanical_turbulence_intensity,
            ),
            estimated_wind_enu=jnp.where(
                frozen[:, None], state.weather.estimated_wind_enu, weather_next.estimated_wind_enu
            ),
            estimated_density_ratio=jnp.where(
                frozen, state.weather.estimated_density_ratio, weather_next.estimated_density_ratio
            ),
        )

        # --- red team moves, using the picture it had at the start of the step -
        psi_cmd, speed_cmd = self.red_policy(
            k_red, cfg, state.threats, state.air.pos, af_mod.velocity(state.air),
            state.alive & ~state.reached & ~state.resource.diverted,
            state.lock, state.red_scales,
        )
        # vertical pursuit runs off the SAME start-of-step track picture the red
        # policy just used, so the two channels cannot disagree about which blue
        # is being chased -- or about whether one has been seen at all.
        vz_cmd = threats_mod.vertical_command(
            cfg, state.threats, state.air.pos,
            state.alive & ~state.reached & ~state.resource.diverted, state.lock, state.red_scales
        )
        tstate = threats_mod.step(
            cfg, state.threats, state.hmap, psi_cmd, speed_cmd, vz_cmd, state.red_scales
        )

        # --- detection against the new geometry -------------------------------
        tparams = threats_mod.per_threat_params(cfg, tstate.kind, state.red_scales)
        det = det_mod.detection_probability(
            state.hmap, cfg.terrain, cfg.detection, air.pos, air.psi, tstate.pos, tparams, tstate.active,
            visibility_fn=self._urban_visibility(state.hmap) if cfg.urban.enabled else None,
        )
        if cfg.weather.enabled:
            det["pd"] = det["pd"] * weather.sensing_attenuation[None, :]
        live = (state.alive & ~state.reached & ~state.resource.diverted).astype(jnp.float32)
        pd = det["pd"] * live[None, :]

        lock = det_mod.update_track(state.lock, pd, tparams, dt)
        locked = lock >= cfg.detection.lock_threshold
        dwell = jnp.where(locked, state.dwell + dt, 0.0)

        in_env, firing, hazard = det_mod.engagement(
            cfg.detection, lock, dwell, det["slant"], det["alt_agl"], air.pos[:, 2], tparams, dt
        )
        hazard = hazard * live

        # --- outcomes ----------------------------------------------------------
        shot = ((jax.random.uniform(k_kill, (cfg.n_blue,)) < hazard) & state.alive
                & ~state.reached & ~state.resource.diverted)

        ground = terrain_mod.sample_height(state.hmap, cfg.terrain, air.pos[:, 0], air.pos[:, 1])
        alt_agl = air.pos[:, 2] - ground
        terrain_hit = (alt_agl < cfg.airframe.floor_agl) & ~frozen
        building_hit = jnp.zeros((cfg.n_blue,), dtype=bool)
        obstacle_clearance = alt_agl
        if cfg.urban.enabled:
            terrain_hit, building_hit, _, _ = urban_mod.swept_collisions(
                self.urban_surface, state.hmap, cfg.terrain, cfg.urban,
                state.air.pos, air.pos, cfg.airframe.floor_agl,
            )
            terrain_hit = terrain_hit & ~frozen
            building_hit = building_hit & ~frozen
            obstacle_clearance = urban_mod.clearance(
                self.urban_surface, state.hmap, cfg.terrain, air.pos
            )[0]
        ceiling_hit = (air.pos[:, 2] >= cfg.airframe.ceiling - 1e-3) & ~frozen
        stall = (air.speed <= cfg.airframe.v_stall + 1e-3) & ~frozen
        # out of bounds means out of THIS world's play area, not off the DEM grid
        x0, x1, y0, y1 = self._box(state.bounds)
        oob = (
            (air.pos[:, 0] < x0)
            | (air.pos[:, 0] > x1)
            | (air.pos[:, 1] < y0)
            | (air.pos[:, 1] > y1)
        ) & ~frozen
        if cfg.resource.enabled:
            dry = (fuel_before > 0.0) & (air.fuel <= 0.0) & ~frozen
        else:
            dry = (air.fuel <= 0.0) & ~frozen

        # Soft boundary ramp. The terminal bounds penalty is a cliff with no
        # gradient: by the time it fires the aircraft is already gone, so the
        # policy learns nothing about *approaching* the edge. MEASURED: without
        # this, out-of-bounds losses climbed to 30% as the policy learned to
        # dodge threats by leaving the map.
        # Extent-relative so it stays narrower than the spawn inset on any map
        # size. See EnvConfig.edge_margin; with map_randomize it scales with the box.
        margin = cfg.edge_margin_frac * jnp.minimum(x1 - x0, y1 - y0) if cfg.map_randomize else cfg.edge_margin
        dx_edge = jnp.minimum(air.pos[:, 0] - x0, x1 - air.pos[:, 0])
        dy_edge = jnp.minimum(air.pos[:, 1] - y0, y1 - air.pos[:, 1])
        edge = jnp.clip(1.0 - jnp.minimum(dx_edge, dy_edge) / margin, 0.0, 1.0)

        dist = jnp.linalg.norm((state.objective - air.pos)[:, :2], axis=-1)
        arrive = (
            (dist <= cfg.objective_radius)
            & state.alive
            & ~state.reached
            & ~state.resource.diverted
        )
        if cfg.resource.enabled:
            # A simultaneous physical loss is not a successful recovery. Keep
            # the legacy ordering untouched, but make L1's exclusive outcome
            # labels agree with the reached/diverted state flags.
            arrive = arrive & ~shot & ~terrain_hit & ~building_hit & ~oob & ~dry

        if cfg.resource.enabled:
            gate_dist = jnp.linalg.norm(
                state.resource.diversion_pos[..., :2] - air.pos[:, None, :2], axis=-1
            )
            at_gate = jnp.any(
                (gate_dist <= cfg.resource.gate_arrival_radius) & state.resource.diversion_active,
                axis=-1,
            )
            diverted_now = (at_gate & ~arrive & ~shot & ~terrain_hit & ~building_hit & ~oob & ~dry
                             & state.alive & ~state.reached & ~state.resource.diverted)
            if cfg.mission.enabled and cfg.mission.family == "recovery":
                diverted_now = (at_gate & ~shot & ~terrain_hit & ~building_hit & ~oob & ~dry
                                 & state.alive & ~state.resource.diverted)
        else:
            diverted_now = jnp.zeros((cfg.n_blue,), dtype=bool)

        mission_timeout = jnp.zeros((cfg.n_blue,), dtype=bool)
        mission = state.mission
        if cfg.mission.enabled:
            loss = shot | terrain_hit | building_hit | oob | (dry if cfg.resource.enabled else False)
            mission, primary_now, recovered_now, mission_timeout = mission_mod.advance(
                cfg.mission, state.mission, air.pos, dt, state.t + 1, loss, diverted_now,
            )
            arrive = primary_now
            diverted_now = diverted_now | recovered_now

        alive = state.alive & ~shot & ~terrain_hit & ~building_hit & ~oob
        if cfg.resource.enabled:
            alive = alive & ~dry
        reached = state.reached | arrive

        terms = RewardTerms(
            progress=jnp.where(frozen, 0.0, state.prev_dist - dist),
            exposure=jnp.max(pd, axis=0),
            lock_level=jnp.max(lock, axis=0),
            envelope_dwell=jnp.max(in_env * (lock > 0.05), axis=0) * live,
            arrived=arrive.astype(jnp.float32),
            diverted=diverted_now.astype(jnp.float32),
            shotdown=shot.astype(jnp.float32),
            fuel_used=jnp.where(frozen, 0.0, fuel_before - air.fuel),
            g_excess=jnp.where(frozen, 0.0, af_mod.load_factor(air.phi) - 1.0),
            terrain_violation=terrain_hit.astype(jnp.float32),
            ceiling_violation=ceiling_hit.astype(jnp.float32),
            stall_violation=stall.astype(jnp.float32),
            bounds_violation=oob.astype(jnp.float32),
            out_of_fuel=dry.astype(jnp.float32),
            edge_proximity=jnp.where(frozen, 0.0, edge),
            building_collision=building_hit.astype(jnp.float32),
        )

        t = state.t + 1
        diverted = state.resource.diverted | diverted_now
        timeout = ((jnp.full((cfg.n_blue,), t >= cfg.max_steps) | mission_timeout)
                   & alive & ~reached & ~diverted)
        outcome = resource_mod.outcome_update(
            state.resource.outcome, arrive, diverted_now, shot, terrain_hit, building_hit, oob,
            dry if cfg.resource.enabled else jnp.zeros_like(dry), timeout,
        )
        rstate = state.resource._replace(diverted=diverted, outcome=outcome)
        objective = mission_mod.current_target(mission) if cfg.mission.enabled else state.objective
        next_dist = jnp.linalg.norm((objective - air.pos)[:, :2], axis=-1)
        new_state = EnvState(
            air=air,
            threats=tstate,
            lock=lock,
            dwell=dwell,
            alive=alive,
            reached=reached,
            objective=objective,
            prev_dist=jnp.where(frozen, state.prev_dist, next_dist),
            hmap=state.hmap,
            bounds=state.bounds,
            t=t,
            key=key,
            route=state.route,
            red_scales=state.red_scales,
            resource=rstate,
            weather=weather,
            mission=mission,
        )

        agent_done = (~alive) | reached | diverted
        if cfg.mission.enabled:
            agent_done = agent_done | mission.failed | mission.completed | mission.recovered
        episode_done = jnp.all(agent_done) | (t >= cfg.max_steps)

        info = {
            "hazard": hazard,
            "firing": firing,
            "alt_agl": alt_agl,
            "dist_to_objective": dist,
            "agent_done": agent_done,
            "n_alive": jnp.sum(alive),
            "n_reached": jnp.sum(reached),
            "n_diverted": jnp.sum(diverted),
            "outcome": outcome,
            "mission_outcome": mission.outcome,
            "mean_wind_enu": weather.mean_wind_enu,
            "gust_enu": weather.gust_enu,
            "density_ratio": weather.density_ratio,
            "background_turbulence": weather.background_turbulence_intensity,
            "mechanical_turbulence": weather.mechanical_turbulence_intensity,
            "obstacle_clearance": obstacle_clearance,
            "building_collision": building_hit,
            "air_velocity": af_mod.velocity(air),
            "ground_velocity": af_mod.ground_velocity(
                air, weather.mean_wind_enu, weather.gust_enu
            ) if cfg.weather.enabled else af_mod.velocity(air),
            "pos_before": state.air.pos,
            "speed_before": state.air.speed,
            "psi_before": state.air.psi,
            "gamma_before": state.air.gamma,
            "phi_before": state.air.phi,
            "fuel_before": state.air.fuel,
            "weather_altitude_input": state.air.pos[:, 2],
            "weather_airspeed_input": state.air.speed,
            "gust_before": state.weather.gust_enu,
            "weather_stream_key": state.weather.stream_key,
            "step_index_before": state.t,
            "mission_stage_before": state.mission.current_stage,
            "mission_dwell_before": state.mission.dwell_s,
        }
        return new_state, self.observe(new_state), terms, episode_done, info

    # ------------------------------------------------------------- respawn --
    def respawn(self, state: EnvState, key: jax.Array, mask: jax.Array) -> EnvState:
        """Re-task the masked aircraft: fresh start, fresh objective, cleared track.

        prompt.md s2 offers a choice between a finite cohort that freezes and
        continuous respawn. Training uses the finite cohort, because an episode
        return has to mean "one sortie"; this is the continuous-tasking variant,
        used by the live viewer so the theatre never empties out.

        Crucially it also clears each respawned aircraft's COLUMN of the lock and
        dwell matrices. Leaving them would hand a brand-new aircraft the track
        history of the one that just died at the far end of the map -- it would
        be shot down seconds after spawning, for reasons invisible in the trace.
        """
        cfg = self.cfg
        k_lat, k_obj, k_next = jax.random.split(key, 3)
        B = cfg.n_blue
        # re-task inside this world's own play area
        x0, x1, y0, y1 = self._box(state.bounds)
        w, h = x1 - x0, y1 - y0
        inset = cfg.spawn_inset_frac

        # Both placements are computed and the episode's own flag picks one, so a
        # world keeps its route however it was reset: legacy, drawn, or from the
        # route bank. `where` only selects, so legacy worlds are bit-identical.
        lat = y0 + jax.random.uniform(k_lat, (B,), minval=1.5 * inset, maxval=1 - 1.5 * inset) * h
        start_legacy = jnp.stack([jnp.full((B,), x0 + inset * w), lat], axis=-1)
        obj_lat = y0 + jax.random.uniform(k_obj, (B,), minval=0.25, maxval=0.75) * h
        obj_legacy = jnp.stack([jnp.full((B,), x0 + (1.0 - inset) * w), obj_lat], axis=-1)
        # same bearing the episode began with: the threat field was built around
        # that corridor, so a re-task along a new bearing would fly a route the
        # threats were never placed against.
        start_route, obj_route = self._route_endpoints(state.route[0], (x0, x1, y0, y1), k_lat, k_obj)
        legacy = state.route[2] > 0.5
        start_xy = jnp.where(legacy, start_legacy, start_route)
        obj_xy = jnp.where(legacy, obj_legacy, obj_route)

        g_s = terrain_mod.sample_height(state.hmap, cfg.terrain, start_xy[:, 0], start_xy[:, 1])
        g_o = terrain_mod.sample_height(state.hmap, cfg.terrain, obj_xy[:, 0], obj_xy[:, 1])
        start = jnp.concatenate([start_xy, (g_s + 2_500.0)[:, None]], axis=-1)
        objective = jnp.concatenate([obj_xy, (g_o + 1_000.0)[:, None]], axis=-1)
        to_obj = objective - start

        fresh_air, fresh_resource = resource_mod.initial_state(
            cfg, key, state.hmap, start, objective, state.bounds
        )
        fresh_weather = weather_mod.initial_state(cfg.weather, key, start[:, 2])
        fresh_mission = mission_mod.initial_state(
            cfg.mission, key, start, objective, fresh_resource.diversion_pos,
            fresh_resource.diversion_active, cfg.dt,
        )
        m = mask[:, None]
        air = jax.tree.map(
            lambda fresh, old: jnp.where(_bcast(mask, fresh), fresh, old),
            fresh_air,
            state.air,
        )
        resource = jax.tree.map(
            lambda fresh, old: jnp.where(_bcast(mask, fresh), fresh, old),
            fresh_resource,
            state.resource,
        )
        weather = state.weather._replace(
            mean_wind_enu=jnp.where(m, fresh_weather.mean_wind_enu, state.weather.mean_wind_enu),
            gust_enu=jnp.where(m, fresh_weather.gust_enu, state.weather.gust_enu),
            density_ratio=jnp.where(mask, fresh_weather.density_ratio, state.weather.density_ratio),
            temperature_k=jnp.where(mask, fresh_weather.temperature_k, state.weather.temperature_k),
            sensing_attenuation=jnp.where(mask, fresh_weather.sensing_attenuation, state.weather.sensing_attenuation),
            turbulence_intensity=jnp.where(mask, fresh_weather.turbulence_intensity, state.weather.turbulence_intensity),
            background_turbulence_intensity=jnp.where(
                mask, fresh_weather.background_turbulence_intensity,
                state.weather.background_turbulence_intensity,
            ),
            mechanical_turbulence_intensity=jnp.where(
                mask, fresh_weather.mechanical_turbulence_intensity,
                state.weather.mechanical_turbulence_intensity,
            ),
            estimated_wind_enu=jnp.where(m, fresh_weather.estimated_wind_enu, state.weather.estimated_wind_enu),
            estimated_density_ratio=jnp.where(mask, fresh_weather.estimated_density_ratio, state.weather.estimated_density_ratio),
            stream_key=fresh_weather.stream_key,
        )
        mission = jax.tree.map(
            lambda fresh, old: jnp.where(_bcast(mask, fresh), fresh, old),
            fresh_mission,
            state.mission,
        )
        if cfg.mission.enabled:
            objective = mission_mod.current_target(mission)
            to_obj = objective - start
        clear = ~mask[None, :]
        return state._replace(
            air=air,
            resource=resource,
            weather=weather,
            mission=mission,
            objective=jnp.where(m, objective, state.objective),
            prev_dist=jnp.where(mask, jnp.linalg.norm(to_obj[:, :2], axis=-1), state.prev_dist),
            alive=state.alive | mask,
            reached=state.reached & ~mask,
            lock=state.lock * clear,
            dwell=state.dwell * clear,
            key=k_next,
        )

    def reroll_threats(self, state: EnvState, key: jax.Array) -> EnvState:
        """Draw a fresh threat field over the same terrain, clearing all tracks.

        MEASURED NEED: `respawn` re-tasks aircraft but leaves the threat layout
        fixed, so a long live session measures one draw rather than the policy.
        A benign draw showed a 100% success rate against 55% over 24 sampled
        layouts -- the counters looked like a result and were an artefact of one
        map.
        """
        k_thr, k_next = jax.random.split(key)
        # the corridor is drawn from positions and objectives inside state.bounds,
        # and the play area itself is kept. Threats may land outside the box on
        # purpose: an emitter past the border still sees in.
        corridor = jnp.stack([jnp.mean(state.air.pos, axis=0), jnp.mean(state.objective, axis=0)], axis=0)
        tstate = threats_mod.spawn(k_thr, self.cfg, state.hmap, corridor, state.red_scales)
        return state._replace(
            threats=tstate,
            lock=jnp.zeros_like(state.lock),
            dwell=jnp.zeros_like(state.dwell),
            key=k_next,
        )

    # ------------------------------------------------------------- rollouts --
    def evidence_frame(
        self, state, obs, requested, executed, feasible, state2, terms, done, info,
        memory=None, memory_after=None,
    ):
        """Raw evidence for one transition, for trajectory recording.

        Everything is read from values the step already computed (no extra env
        step). `obs`/`state` are the PRE-action pair, `state2` the post-step state
        before any auto-reset. `memory`/`memory_after` are the recurrent carry
        into and out of the action, `None` for feed-forward policies. Only
        built when a caller asks for it, so default compiled programs are
        unchanged.
        """
        res = state2.resource
        ev = {
            "obs_ego": obs.ego, "obs_threats": obs.threats, "obs_threat_mask": obs.threat_mask,
            "obs_friends": obs.friends, "obs_friend_mask": obs.friend_mask,
            "action_requested": requested, "action_executed": executed, "cbf_feasible": feasible,
            "terms": terms._asdict(),
            "fuel_before": state.air.fuel, "fuel_after": state2.air.fuel,
            "alive": state2.alive, "reached": state2.reached, "diverted": res.diverted,
            "outcome": res.outcome, "agent_done": info["agent_done"], "done": done,
            "route": state.route,
            "reserve_fuel": res.reserve_fuel, "initial_fuel": res.initial_fuel,
            "burn_multiplier": res.burn_multiplier,
            "diversion_pos": res.diversion_pos, "diversion_active": res.diversion_active,
            "mean_wind_enu": state2.weather.mean_wind_enu,
            "gust_enu": state2.weather.gust_enu,
            "density_ratio": state2.weather.density_ratio,
            "background_turbulence": state2.weather.background_turbulence_intensity,
            "mechanical_turbulence": state2.weather.mechanical_turbulence_intensity,
            "air_velocity": af_mod.velocity(state2.air),
            "ground_velocity": info["ground_velocity"],
            "mission_stage_before": state.mission.current_stage,
            "mission_stage_after": state2.mission.current_stage,
            "mission_dwell_before": state.mission.dwell_s,
            "mission_dwell_after": state2.mission.dwell_s,
            "mission_outcome": state2.mission.outcome,
            "mission_stage_pos": state2.mission.stage_pos,
            "mission_stage_mask": state2.mission.stage_mask,
            "mission_deadline_step": state2.mission.deadline_step,
            "mission_completed": state2.mission.completed,
            "mission_recovered": state2.mission.recovered,
            "mission_failed": state2.mission.failed,
            "obstacle_clearance": info["obstacle_clearance"],
            "building_collision": info["building_collision"],
            "pos_before": info["pos_before"],
            "speed_before": info["speed_before"],
            "psi_before": info["psi_before"],
            "gamma_before": info["gamma_before"],
            "phi_before": info["phi_before"],
            "fuel_before_transition": info["fuel_before"],
            "weather_altitude_input": info["weather_altitude_input"],
            "weather_airspeed_input": info["weather_airspeed_input"],
            "gust_before": info["gust_before"],
            "weather_stream_key": info["weather_stream_key"],
            "step_index_before": info["step_index_before"],
            "mission_stage_before_transition": info["mission_stage_before"],
            "mission_dwell_before_transition": info["mission_dwell_before"],
        }
        if memory is not None:
            ev["memory_before"], ev["memory_after"] = memory, memory_after
        return ev

    def _trace_frame(self, state2, obs2, action, feasible, terms, done, info, requested=None):
        """One rollout trace row. Shared by the stateless and recurrent scans so
        both carry every L1 resource field.

        `action` is what the environment executed (after the optional CBF);
        `requested` is what the policy asked for. `action` stays as a
        backward-compatible alias of `action_executed`.
        """
        return {
            "pos": state2.air.pos,
            "psi": state2.air.psi,
            # attitude state the airframe integrated, logged as-is so a
            # replay can draw bank and climb instead of inferring them
            "gamma": state2.air.gamma,
            "phi": state2.air.phi,
            "speed": state2.air.speed,
            "fuel": state2.air.fuel,
            "alt_agl": info["alt_agl"],
            "alive": state2.alive,
            "reached": state2.reached,
            "diverted": state2.resource.diverted,
            "outcome": state2.resource.outcome,
            "reserve_fuel": state2.resource.reserve_fuel,
            "initial_fuel": state2.resource.initial_fuel,
            "burn_multiplier": state2.resource.burn_multiplier,
            "diversion_pos": state2.resource.diversion_pos,
            "diversion_active": state2.resource.diversion_active,
            "objective": state2.objective,
            "bounds": state2.bounds,
            "resource_observation": obs2.ego[:,
                (14 if self.cfg.obs_edge_features else 10):
                (14 if self.cfg.obs_edge_features else 10) + self.cfg.resource.observation_dim]
            if self.cfg.resource.enabled
            else jnp.zeros((self.cfg.n_blue, 0), dtype=obs2.ego.dtype),
            "action": action,
            "action_executed": action,
            "action_requested": action if requested is None else requested,
            "cbf_feasible": feasible,
            "terms": terms,
            "done": done,
            "threat_pos": state2.threats.pos,
            "threat_psi": state2.threats.psi,
            "lock": state2.lock,
            # (T, B) engagement firing solutions this step -- which threat a
            # logged shootdown can be attributed to
            "firing": info["firing"],
            "mean_wind_enu": info["mean_wind_enu"],
            "gust_enu": info["gust_enu"],
            "density_ratio": info["density_ratio"],
            "background_turbulence": info["background_turbulence"],
            "mechanical_turbulence": info["mechanical_turbulence"],
            "air_velocity": info["air_velocity"],
            "ground_velocity": info["ground_velocity"],
            "mission_stage": state2.mission.current_stage,
            "mission_dwell_s": state2.mission.dwell_s,
            "mission_completed": state2.mission.completed,
            "mission_failed": state2.mission.failed,
            "mission_outcome": state2.mission.outcome,
            "obstacle_clearance": info["obstacle_clearance"],
            "building_collision": info["building_collision"],
            "pos_before": info["pos_before"],
            "speed_before": info["speed_before"],
            "psi_before": info["psi_before"],
            "gamma_before": info["gamma_before"],
            "phi_before": info["phi_before"],
            "weather_altitude_input": info["weather_altitude_input"],
            "weather_airspeed_input": info["weather_airspeed_input"],
            "gust_before": info["gust_before"],
            "weather_stream_key": info["weather_stream_key"],
            "step_index_before": info["step_index_before"],
            "mission_stage_pos": state2.mission.stage_pos,
            "mission_stage_mask": state2.mission.stage_mask,
            "mission_deadline_step": state2.mission.deadline_step,
            "mission_recovered": state2.mission.recovered,
            "fuel_before_transition": info["fuel_before"],
            "mission_stage_before_transition": info["mission_stage_before"],
            "mission_dwell_before_transition": info["mission_dwell_before"],
        }

    def rollout(self, key, policy, n_steps: int | None = None, action_filter=None, route=None, record: bool = False):
        """Scan a whole episode. `policy(obs, key) -> (n_blue, 3)`.

        `route`, if given, is a (2,) [bearing rad, family id] and the episode
        starts from `reset_route` instead of `reset` -- how the route bank runs.

        `action_filter(state, obs, action) -> (action, feasible)` is the optional
        runtime safety backstop (see `naigos.rl.cbf.make_policy_filter`). It sits
        between the policy and the env exactly as `a_exec = filter(a_policy,
        state)` describes, and its per-step feasibility flags are logged so the
        infeasibility rate can be reported rather than hidden.

        `record=True` adds `traj["evidence"]` (see `evidence_frame`): the
        pre-action observation, requested action and recurrent memory, taken
        from the same scan -- no extra step. Off by default, and then the
        compiled program is exactly the one without it.

        Returns `(final_state, traj)` where traj is a pytree stacked on a leading
        time axis -- the trace the verifier and the demo replay consume.
        """
        n_steps = n_steps or self.cfg.max_steps

        # Recurrent policies advertise their fresh memory explicitly. Keep a
        # separate scan signature so the long-standing stateless policy path
        # below remains byte-for-byte compatible for demos and baselines.
        initial_memory = getattr(policy, "initial_memory", None)
        if initial_memory is not None:
            def recurrent_body(carry, _):
                state, obs, memory, k = carry
                k, ka = jax.random.split(k)
                requested, memory_next = policy(obs, ka, memory)
                action = requested
                if action_filter is None:
                    feasible = jnp.ones((self.cfg.n_blue,), dtype=bool)
                else:
                    action, feasible = action_filter(state, obs, action)
                state2, obs2, terms, done, info = self.step(state, action)
                memory_next = jnp.where(
                    (info["agent_done"] | done)[..., None],
                    jnp.zeros_like(memory_next), memory_next,
                )
                out = self._trace_frame(state2, obs2, action, feasible, terms, done, info, requested)
                if record:
                    out["evidence"] = self.evidence_frame(
                        state, obs, requested, action, feasible, state2, terms, done, info,
                        memory, memory_next,
                    )
                return (state2, obs2, memory_next, k), out

            k0, k1 = jax.random.split(key)
            state, obs = self.reset(k0) if route is None else self.reset_route(k0, route)
            (final, _, _, _), traj = jax.lax.scan(
                recurrent_body, (state, obs, initial_memory, k1), None,
                length=n_steps,
            )
            return final, traj

        def body(carry, _):
            state, obs, k = carry
            k, ka = jax.random.split(k)
            requested = policy(obs, ka)
            action = requested
            if action_filter is None:
                feasible = jnp.ones((self.cfg.n_blue,), dtype=bool)
            else:
                action, feasible = action_filter(state, obs, action)
            state2, obs2, terms, done, info = self.step(state, action)
            out = self._trace_frame(state2, obs2, action, feasible, terms, done, info, requested)
            if record:
                out["evidence"] = self.evidence_frame(
                    state, obs, requested, action, feasible, state2, terms, done, info
                )
            return (state2, obs2, k), out

        k0, k1 = jax.random.split(key)
        state, obs = self.reset(k0) if route is None else self.reset_route(k0, route)
        (final, _, _), traj = jax.lax.scan(body, (state, obs, k1), None, length=n_steps)
        return final, traj


def _bcast(mask: jax.Array, like: jax.Array) -> jax.Array:
    """Broadcast a (B,) boolean over a (B, ...) leaf."""
    return mask.reshape(mask.shape + (1,) * (like.ndim - 1))
