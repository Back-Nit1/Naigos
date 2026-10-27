"""JAX consumer for validated ``naigos.urban_obstacles/1`` artifacts.

File I/O and hash verification happen in ``naigos.data.urban_obstacles`` before
this module is called.  Construction converts each NumPy array exactly once;
all reset/step/LOS work below is pure array computation. Unknown and outside
coverage explicitly fall back to the bare-earth terrain surface.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from .config import DetectionConfig, TerrainConfig, UrbanPhysicsConfig
from . import terrain as terrain_mod


@dataclass(frozen=True)
class UrbanSurface:
    top_amsl_m: jax.Array
    coverage: jax.Array
    building: jax.Array
    origin_x_m: float
    origin_y_m: float
    cell_m: float
    nx: int
    ny: int
    content_sha256: str
    schema: str = "naigos.urban_obstacles/1"


def from_artifact(grid, env_origin_utm_m) -> UrbanSurface:
    """Convert one already-strictly-validated artifact to immutable JAX arrays."""
    if env_origin_utm_m is None:
        raise ValueError("urban physics requires the environment's absolute UTM origin")
    x0, y0 = grid.origin_in_env(env_origin_utm_m)
    return UrbanSurface(
        top_amsl_m=jnp.asarray(grid.top_with_nodata(0.0), dtype=jnp.float32),
        coverage=jnp.asarray(grid.coverage, dtype=bool),
        building=jnp.asarray(grid.building, dtype=bool),
        origin_x_m=float(x0),
        origin_y_m=float(y0),
        cell_m=float(grid.meta.cell_m),
        nx=int(grid.meta.nx),
        ny=int(grid.meta.ny),
        content_sha256=grid.hashes.content_sha256,
    )


def sample(surface: UrbanSurface, hmap, tcfg: TerrainConfig, x, y):
    """Return combined top, building bit and coverage using containing cells."""
    fx = jnp.floor((x - surface.origin_x_m) / surface.cell_m).astype(jnp.int32)
    fy = jnp.floor((y - surface.origin_y_m) / surface.cell_m).astype(jnp.int32)
    inside = (fx >= 0) & (fx < surface.nx) & (fy >= 0) & (fy < surface.ny)
    ix = jnp.clip(fx, 0, surface.nx - 1)
    iy = jnp.clip(fy, 0, surface.ny - 1)
    covered = inside & surface.coverage[iy, ix]
    building = covered & surface.building[iy, ix]
    terrain = terrain_mod.sample_height(hmap, tcfg, x, y)
    top = jnp.where(covered, surface.top_amsl_m[iy, ix], terrain)
    return top, building, covered


def clearance(surface: UrbanSurface, hmap, tcfg: TerrainConfig, pos):
    top, building, covered = sample(surface, hmap, tcfg, pos[..., 0], pos[..., 1])
    return pos[..., 2] - top, building, covered


def swept_collisions(
    surface: UrbanSurface,
    hmap,
    tcfg: TerrainConfig,
    ucfg: UrbanPhysicsConfig,
    p0,
    p1,
    terrain_floor_m: float,
):
    """Conservative fixed-count swept capsule test across a transition.

    The broad phase is the raster's containing-cell lookup.  The configured
    count must keep spacing below the evidence grid's cell width for the chosen
    decision interval; construction refuses configurations that cannot do so.
    """
    s = jnp.linspace(0.0, 1.0, ucfg.collision_samples + 1, dtype=jnp.float32)
    points = p0[:, None, :] + s[None, :, None] * (p1 - p0)[:, None, :]
    top, building, _ = sample(surface, hmap, tcfg, points[..., 0], points[..., 1])
    ground = terrain_mod.sample_height(hmap, tcfg, points[..., 0], points[..., 1])
    building_hit_samples = building & (points[..., 2] - ucfg.capsule_radius_m <= top)
    terrain_hit_samples = points[..., 2] - ucfg.capsule_radius_m < ground + terrain_floor_m
    building_hit = jnp.any(building_hit_samples, axis=-1)
    terrain_hit = jnp.any(terrain_hit_samples & ~building_hit_samples, axis=-1)
    first_building = jnp.argmax(building_hit_samples, axis=-1)
    first_terrain = jnp.argmax(terrain_hit_samples & ~building_hit_samples, axis=-1)
    return terrain_hit, building_hit, first_terrain, first_building


def los_clearance(
    surface: UrbanSurface,
    hmap,
    tcfg: TerrainConfig,
    dcfg: DetectionConfig,
    p_from,
    p_to,
):
    s = (jnp.arange(dcfg.los_samples, dtype=jnp.float32) + 0.5) / dcfg.los_samples
    s = s.reshape((1,) * (p_from.ndim - 1) + (dcfg.los_samples,))
    seg = p_to - p_from
    px = p_from[..., None, 0] + s * seg[..., None, 0]
    py = p_from[..., None, 1] + s * seg[..., None, 1]
    pz = p_from[..., None, 2] + s * seg[..., None, 2]
    top, _, _ = sample(surface, hmap, tcfg, px, py)
    d_horiz = jnp.sqrt(jnp.sum(seg[..., :2] ** 2, axis=-1))[..., None]
    drop = (s * d_horiz) * ((1.0 - s) * d_horiz) / (2.0 * dcfg.earth_radius_eff)
    return jnp.min((pz - drop) - top, axis=-1)


def visibility(surface, hmap, tcfg, dcfg, p_from, p_to):
    c = los_clearance(surface, hmap, tcfg, dcfg, p_from, p_to)
    return jax.nn.sigmoid(c / dcfg.los_clearance_scale)


def mechanical_wake_intensity(surface: UrbanSurface, hmap, tcfg: TerrainConfig, pos, mean_wind_enu):
    """Deterministic building-wake surrogate, returning local gust sigma (m/s).

    Four broad-phase samples look upwind along the incoming flow. A structure's
    height, downwind distance, wind speed, and height above its roof determine a
    bounded decaying intensity. This is an engineering roughness/wake surrogate,
    not CFD; the interface can later be replaced by a hashed directional field.
    """
    wind_xy = mean_wind_enu[..., :2]
    speed = jnp.linalg.norm(wind_xy, axis=-1)
    direction = wind_xy / jnp.maximum(speed[:, None], 1e-3)
    multiples = jnp.asarray([1.0, 2.0, 4.0, 8.0], dtype=jnp.float32)
    dist = multiples[None, :] * surface.cell_m
    upstream = pos[:, None, :2] - dist[..., None] * direction[:, None, :]
    top, building, _ = sample(surface, hmap, tcfg, upstream[..., 0], upstream[..., 1])
    ground = terrain_mod.sample_height(hmap, tcfg, upstream[..., 0], upstream[..., 1])
    height = jnp.maximum(top - ground, 1.0)
    above = jnp.maximum(pos[:, None, 2] - top, 0.0)
    decay = jnp.exp(-dist / jnp.maximum(10.0 * height, surface.cell_m))
    vertical = jnp.exp(-above / jnp.maximum(height, 1.0))
    sigma = 0.22 * speed[:, None] * decay * vertical * building.astype(jnp.float32)
    return jnp.clip(jnp.max(sigma, axis=-1), 0.0, 25.0)
