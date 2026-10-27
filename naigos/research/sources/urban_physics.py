"""Build the physics-grade urban obstacle artifact (docs/contracts/urban-obstacles-v1.md).

Pipeline, in the order the contract requires:

  1. ground   USGS 3DEP bare-earth DEM, resampled to the obstacle grid.
  2. points   USGS 3DEP lidar (Entwine Point Tiles on AWS): ASPRS class 6 (building) and 17
              (bridge deck) are structures; class 2 and 9 (ground, water) are the local ground
              the structure heights are measured against. Cached as one derived subset.
  3. nyc only the city's Building Footprints (footprint + roof height), reconciled with lidar.
  4. clip     strictly to the documented ``urban_physics`` box; the grid lies inside it.
  5. exclude  every ``extraction`` protected zone becomes no-data (never zero height).
  6. publish  payload, metadata, descriptor, completion marker; reopen; verify.

Everything crosses the allowlist (`cache.fetch`/`check_url`), is hashed, and a second run reads
the cache with no network. Presentation data (OSM, provider tiles) is never read here.
"""

from __future__ import annotations

import concurrent.futures as cf
import datetime as dt
import io
import json
import math
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .. import cache
from ..allowlist import check_url
from ..aoi import AOI
from ...data import urban_obstacles as uo

EPT_BASE = "https://usgs-lidar-public.s3.amazonaws.com"
TNM_PRODUCTS = "https://tnmaccess.nationalmap.gov/api/v1/products"
NYC_FOOTPRINTS = "https://data.cityofnewyork.us/resource/5zhs-2jue.json"
NYC_FOOTPRINTS_VIEW = "https://data.cityofnewyork.us/api/views/5zhs-2jue.json"

#: Per theatre: the EPT collection that covers its physics box and how deep to read it.
#: ``max_depth`` bounds the download: nominal point spacing halves per level.
LIDAR = {
    "nyc_harbor": {"collection": "NY_NewYorkCity", "max_depth": 7},
    "los_angeles_basin": {"collection": "CA_LosAngeles_1_B23", "max_depth": 10},
}
STRUCTURE_CLASSES = (6, 17)  # building, bridge deck
GROUND_CLASSES = (2, 9)  # ground, water
MIN_STRUCTURE_POINTS = 2  # returns in a cell before it can be a structure
MIN_STRUCTURE_HEIGHT_M = 3.0  # below this a cell is a vehicle/shed/noise, not an obstacle
COVERAGE_RADIUS_M = 30.0  # a cell is surveyed if a lidar return lies this close
WATER_ELEVATION_M = 2.0  # DEM at/below this inside the collection counts as surveyed open water
MAX_GROUND_DISAGREEMENT_M = 5.0  # median(lidar ground - DEM) beyond this means a datum/unit error
FEET_TO_M = 0.3048


# --- grid ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class GridSpec:
    epsg: int
    env_origin_utm: tuple[float, float]
    x0: float  # ENU of cell (0,0) SW corner
    y0: float
    cell: float
    nx: int
    ny: int

    @property
    def sw_utm(self) -> tuple[float, float]:
        return (self.env_origin_utm[0] + self.x0, self.env_origin_utm[1] + self.y0)

    def centers_utm(self) -> tuple[np.ndarray, np.ndarray]:
        e = self.sw_utm[0] + (np.arange(self.nx) + 0.5) * self.cell
        n = self.sw_utm[1] + (np.arange(self.ny) + 0.5) * self.cell
        return np.meshgrid(e, n)  # (ny, nx), row 0 = south

    def corners_wgs84(self) -> tuple[tuple[float, float], ...]:
        from pyproj import Transformer

        back = Transformer.from_crs(f"EPSG:{self.epsg}", "EPSG:4326", always_xy=True)
        e0, n0 = self.sw_utm
        e1, n1 = e0 + self.nx * self.cell, n0 + self.ny * self.cell
        return tuple(tuple(round(float(v), 7) for v in back.transform(e, n))
                     for e, n in ((e0, n0), (e1, n0), (e1, n1), (e0, n1)))


def reference_env_origin(aoi: AOI) -> tuple[float, float]:
    """The ENU origin of the theatre's reference env: ``env_from_theatre`` at 1500 m, centred."""
    from ...data.enu import natural_grid_shape, real_terrain
    from ...data.terrain import TerrainGrid
    from ...data.theatre import _dem_path

    grid = TerrainGrid.from_npz(_dem_path(aoi.name))
    nx, ny = natural_grid_shape(1500.0, grid)
    g = real_terrain(nx=nx, ny=ny, cell_m=1500.0, grid=grid)
    return (g.origin_easting_m, g.origin_northing_m)


def grid_spec(aoi: AOI, env_origin: tuple[float, float] | None = None) -> GridSpec:
    """The largest lattice-aligned grid inside ``aoi.urban_physics.bounds`` (UTM inner rectangle)."""
    from pyproj import Transformer

    from ..sources.terrain import utm_epsg

    spec = aoi.urban_physics
    if spec is None:
        raise ValueError(f"{aoi.name} has no urban_physics block")
    epsg = utm_epsg(*aoi.center)
    env = env_origin if env_origin is not None else reference_env_origin(aoi)
    tf = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    w, s, e, n = spec.bounds
    (esw, nsw), (ese, nse), (ene, nne), (enw, nnw) = (tf.transform(*p) for p in ((w, s), (e, s), (e, n), (w, n)))
    min_e, max_e = max(esw, enw) - env[0], min(ese, ene) - env[0]
    min_n, max_n = max(nsw, nse) - env[1], min(nnw, nne) - env[1]
    c = spec.cell_m
    x0, y0 = math.ceil(min_e / c) * c, math.ceil(min_n / c) * c
    nx, ny = int((max_e - x0) // c), int((max_n - y0) // c)
    return GridSpec(epsg, (float(env[0]), float(env[1])), float(x0), float(y0), float(c), nx, ny)


# --- ground -------------------------------------------------------------------------------


def fetch_ground(aoi: AOI, gs: GridSpec, force: bool = False) -> cache.Artifact:
    """3DEP bare earth over the physics box (plus a margin), as a UTM GeoTIFF in the cache."""
    spec = aoi.urban_physics
    w, s, e, n = spec.bounds
    pad = 0.002
    bbox = (w - pad, s - pad, e + pad, n + pad)

    def build() -> bytes:
        import py3dep
        import rioxarray  # noqa: F401

        check_url("https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer", "usgs_3dep")
        dem = py3dep.get_dem(bbox, int(spec.cell_m))
        dem = dem.rio.reproject(f"EPSG:{gs.epsg}", resolution=spec.cell_m)
        dem = dem.rio.write_nodata(np.float32("nan"), encoded=False)
        buf = io.BytesIO()
        dem.astype("float32").rio.to_raster(buf, driver="GTiff", compress="deflate")
        return buf.getvalue()

    return cache.produce(
        key=f"urban_ground/{aoi.name}/{aoi.fingerprint}/{int(spec.cell_m)}m", source_key="usgs_3dep",
        url="https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer",
        rel_path=f"urban_src/{aoi.name}_{aoi.fingerprint}_ground_{int(spec.cell_m)}m_utm.tif",
        builder=build, force=force,
        note=f"USGS 3DEP bare-earth DEM, best available source resampled to {spec.cell_m:g} m, UTM.")


def sample_ground(art: cache.Artifact, gs: GridSpec) -> np.ndarray:
    from ...data.terrain import TerrainGrid

    g = TerrainGrid.from_geotiff(art.abs_path)
    e, n = gs.centers_utm()
    return g.elevation(e, n).astype(np.float32)


# --- lidar --------------------------------------------------------------------------------


def _fetch_json(key: str, url: str, rel: str) -> dict[str, Any]:
    art = cache.fetch(key, "usgs_3dep_lidar", url, rel,
                      validate=lambda b: b.lstrip()[:1] in (b"{", b"["), timeout=120)
    return json.loads(art.abs_path.read_bytes())


def ept_nodes(collection: str, root_bounds: list[float], bbox_3857: tuple[float, float, float, float],
              max_depth: int) -> list[str]:
    """Node keys with points that intersect ``bbox_3857`` down to ``max_depth`` (hierarchy walk)."""
    minx, miny, _, maxx, maxy, _ = root_bounds
    size = maxx - minx

    def hits(key: str) -> bool:
        d, x, y, _z = (int(v) for v in key.split("-"))
        w = size / (1 << d)
        nx0, ny0 = minx + x * w, miny + y * w
        return not (nx0 > bbox_3857[2] or nx0 + w < bbox_3857[0] or ny0 > bbox_3857[3] or ny0 + w < bbox_3857[1])

    out, todo, seen = [], ["0-0-0-0"], set()
    while todo:
        sub = todo.pop()
        if sub in seen:
            continue
        seen.add(sub)
        h = _fetch_json(f"lidar/{collection}/hier/{sub}", f"{EPT_BASE}/{collection}/ept-hierarchy/{sub}.json",
                        f"lidar/{collection}/ept-hierarchy/{sub}.json")
        for key, count in h.items():
            if int(key.split("-")[0]) > max_depth or not hits(key):
                continue
            if count == -1:
                todo.append(key)
            elif count > 0:
                out.append(key)
    return sorted(set(out))


def _read_node(collection: str, key: str, gs: GridSpec, box_utm: tuple[float, float, float, float]):
    import laspy
    import requests
    from pyproj import Transformer

    url = f"{EPT_BASE}/{collection}/ept-data/{key}.laz"
    check_url(url, "usgs_3dep_lidar")
    last = None
    for attempt in range(4):
        try:
            r = requests.get(url, timeout=120, headers={"User-Agent": cache.USER_AGENT})
            r.raise_for_status()
            break
        except Exception as exc:  # network retry; the failure is re-raised below if persistent
            last = exc
            time.sleep(1.5 * (attempt + 1))
    else:
        raise RuntimeError(f"{url}: {last}")
    sha = cache.sha256_bytes(r.content)
    las = laspy.read(io.BytesIO(r.content))
    cls = np.asarray(las.classification)
    keep = np.isin(cls, STRUCTURE_CLASSES + GROUND_CLASSES)
    tf = Transformer.from_crs("EPSG:3857", f"EPSG:{gs.epsg}", always_xy=True)
    e, n = tf.transform(np.asarray(las.x), np.asarray(las.y))
    inside = keep & (e >= box_utm[0]) & (e <= box_utm[2]) & (n >= box_utm[1]) & (n <= box_utm[3])
    gps = np.asarray(las.gps_time)
    return (key, sha, len(r.content), int(len(cls)), e[inside], n[inside],
            np.asarray(las.z)[inside].astype(np.float32), cls[inside].astype(np.uint8),
            float(gps.min()) if len(gps) else math.nan, float(gps.max()) if len(gps) else math.nan)


def _gps_date(t: float) -> str | None:
    """Adjusted GPS time (seconds since 1980-01-06, minus 1e9) to an ISO date; None if not that."""
    if not math.isfinite(t) or t < 1e6:
        return None
    return (dt.datetime(1980, 1, 6) + dt.timedelta(seconds=t + 1e9)).date().isoformat()


def fetch_points(aoi: AOI, gs: GridSpec, force: bool = False) -> cache.Artifact:
    """Download the lidar nodes over the grid once and cache the structure/ground subset."""
    cfg = LIDAR[aoi.name]
    coll, depth = cfg["collection"], cfg["max_depth"]
    ept = _fetch_json(f"lidar/{coll}/ept", f"{EPT_BASE}/{coll}/ept.json", f"lidar/{coll}/ept.json")

    def build() -> bytes:
        from pyproj import Transformer

        e0, n0 = gs.sw_utm
        box = (e0 - 50.0, n0 - 50.0, e0 + gs.nx * gs.cell + 50.0, n0 + gs.ny * gs.cell + 50.0)
        tf = Transformer.from_crs(f"EPSG:{gs.epsg}", "EPSG:3857", always_xy=True)
        xs, ys = zip(*(tf.transform(e, n) for e in (box[0], box[2]) for n in (box[1], box[3])))
        keys = ept_nodes(coll, ept["bounds"], (min(xs), min(ys), max(xs), max(ys)), depth)
        if not keys:
            raise RuntimeError(f"no lidar nodes of {coll} intersect the {aoi.name} physics box")
        with cf.ThreadPoolExecutor(max_workers=8) as ex:
            parts = list(ex.map(lambda k: _read_node(coll, k, gs, box), keys))
        e = np.concatenate([p[4] for p in parts]); n = np.concatenate([p[5] for p in parts])
        z = np.concatenate([p[6] for p in parts]); c = np.concatenate([p[7] for p in parts])
        t0 = np.nanmin([p[8] for p in parts]); t1 = np.nanmax([p[9] for p in parts])
        manifest = {
            "collection": coll, "max_depth": depth, "nodes": [[p[0], p[1], p[2], p[3]] for p in parts],
            "points_read": int(sum(p[3] for p in parts)), "points_kept": int(len(z)),
            "gps_time_min": float(t0), "gps_time_max": float(t1),
            "captured_min": _gps_date(float(t0)), "captured_max": _gps_date(float(t1)),
        }
        buf = io.BytesIO()
        np.savez(buf, e=e, n=n, z=z, c=c, manifest=np.frombuffer(json.dumps(manifest).encode(), np.uint8))
        return buf.getvalue()

    return cache.produce(
        key=f"urban_lidar/{aoi.name}/{aoi.fingerprint}/{int(gs.cell)}m", source_key="usgs_3dep_lidar",
        url=f"{EPT_BASE}/{coll}/ept.json", rel_path=f"urban_src/{aoi.name}_{aoi.fingerprint}_lidar_subset.npz",
        builder=build, force=force,
        note=f"USGS 3DEP lidar ({coll}, EPT depth <= {depth}), classes 2/6/9/17 inside the physics box.")


def lidar_products(aoi: AOI, bbox: tuple[float, float, float, float], force: bool = False) -> list[dict]:
    """TNM Access product records (title, publication date) overlapping the physics box."""
    w, s, e, n = bbox
    url = (f"{TNM_PRODUCTS}?datasets=Lidar%20Point%20Cloud%20(LPC)&bbox={w},{s},{e},{n}&max=100")
    art = cache.fetch(f"urban_lidar_products/{aoi.name}/{aoi.fingerprint}", "usgs_3dep_lidar", url,
                      f"urban_src/{aoi.name}_{aoi.fingerprint}_tnm_products.json", force=force,
                      validate=lambda b: b.lstrip()[:1] == b"{")
    items = json.loads(art.abs_path.read_bytes()).get("items", [])
    return [{"title": i["title"], "publicationDate": i.get("publicationDate")} for i in items]


# --- rasterisation ------------------------------------------------------------------------


def _cell_index(e: np.ndarray, n: np.ndarray, gs: GridSpec) -> np.ndarray:
    col = np.floor((e - gs.sw_utm[0]) / gs.cell).astype(np.int64)
    row = np.floor((n - gs.sw_utm[1]) / gs.cell).astype(np.int64)
    ok = (col >= 0) & (col < gs.nx) & (row >= 0) & (row < gs.ny)
    return np.where(ok, row * gs.nx + col, -1)


def _reduce(idx: np.ndarray, vals: np.ndarray, shape: tuple[int, int], op: str) -> np.ndarray:
    """Per-cell max / mean / count of ``vals``; NaN (count 0) where empty."""
    out = np.full(shape[0] * shape[1], np.nan, np.float64)
    ok = idx >= 0
    idx, vals = idx[ok], vals[ok].astype(np.float64)
    if idx.size == 0:
        return out.reshape(shape)
    order = np.argsort(idx, kind="stable")
    idx, vals = idx[order], vals[order]
    cells, start = np.unique(idx, return_index=True)
    if op == "max":
        out[cells] = np.maximum.reduceat(vals, start)
    elif op == "mean":
        out[cells] = np.add.reduceat(vals, start) / np.diff(np.append(start, idx.size))
    else:
        out[cells] = np.diff(np.append(start, idx.size))
    return out.reshape(shape)


def _dilate(mask: np.ndarray, r: int) -> np.ndarray:
    out = mask.copy()
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dx * dx + dy * dy <= r * r:
                sh = np.zeros_like(mask)
                ys = slice(max(dy, 0), mask.shape[0] + min(dy, 0)); yd = slice(max(-dy, 0), mask.shape[0] + min(-dy, 0))
                xs = slice(max(dx, 0), mask.shape[1] + min(dx, 0)); xd = slice(max(-dx, 0), mask.shape[1] + min(-dx, 0))
                sh[yd, xd] = mask[ys, xs]
                out |= sh
    return out


def _fill_nearest(a: np.ndarray) -> np.ndarray:
    """Fill NaNs from the nearest valid neighbour by repeated 8-neighbour propagation."""
    out = a.copy()
    for _ in range(max(a.shape)):
        gap = np.isnan(out)
        if not gap.any() or gap.all():
            break
        pad = np.pad(out, 1, constant_values=np.nan)
        nb = np.stack([pad[1 + dy: 1 + dy + a.shape[0], 1 + dx: 1 + dx + a.shape[1]]
                       for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy or dx)])
        with np.errstate(all="ignore"):
            fill = np.nanmean(nb, axis=0)
        out = np.where(gap, fill, out)
    return out


def lidar_rasters(points: dict[str, np.ndarray], gs: GridSpec) -> dict[str, np.ndarray]:
    shape = (gs.ny, gs.nx)
    idx = _cell_index(points["e"], points["n"], gs)
    st = np.isin(points["c"], STRUCTURE_CLASSES)
    gr = np.isin(points["c"], GROUND_CLASSES)
    return {
        "top": _reduce(idx[st], points["z"][st], shape, "max"),
        "count": _reduce(idx[st], points["z"][st], shape, "count"),
        "ground": _reduce(idx[gr], points["z"][gr], shape, "mean"),
        "any": _reduce(idx, points["z"], shape, "count"),
    }


def nyc_footprints(aoi: AOI, gs: GridSpec, force: bool = False) -> tuple[cache.Artifact, list[dict]]:
    """Footprint polygons + roof height inside the grid box, from NYC Open Data (cached JSON)."""
    from pyproj import Transformer

    back = Transformer.from_crs(f"EPSG:{gs.epsg}", "EPSG:4326", always_xy=True)
    ws = [back.transform(*p) for p in ((gs.sw_utm[0], gs.sw_utm[1]),
                                       (gs.sw_utm[0] + gs.nx * gs.cell, gs.sw_utm[1] + gs.ny * gs.cell))]
    w, s, e, n = ws[0][0] - 0.001, ws[0][1] - 0.001, ws[1][0] + 0.001, ws[1][1] + 0.001
    sel = "the_geom,height_roof,ground_elevation,last_edited_date,last_status_type,geom_source"
    url = (f"{NYC_FOOTPRINTS}?$select={sel}&$where=within_box(the_geom,{s},{w},{n},{e})&$limit=200000")
    art = cache.fetch(f"nyc_footprints/{aoi.name}/{aoi.fingerprint}", "nyc_building_footprints", url,
                      f"urban_src/{aoi.name}_{aoi.fingerprint}_nyc_footprints.json", force=force,
                      validate=lambda b: b.lstrip()[:1] == b"[", timeout=300)
    return art, json.loads(art.abs_path.read_bytes())


def rasterise_footprints(rows: list[dict], gs: GridSpec) -> tuple[np.ndarray, np.ndarray]:
    """Cells touched by a footprint, with the tallest roof height (m) per cell; NaN if none."""
    from pyproj import Transformer
    from rasterio import features
    from rasterio.transform import from_origin
    from shapely.geometry import shape
    from shapely.ops import transform as shp_transform

    tf = Transformer.from_crs("EPSG:4326", f"EPSG:{gs.epsg}", always_xy=True)
    shapes = []
    for r in rows:
        try:
            h = float(r["height_roof"]) * FEET_TO_M
        except (KeyError, TypeError, ValueError):
            continue
        if r.get("last_status_type") not in (None, "Constructed") or not math.isfinite(h) or h <= 0:
            continue
        shapes.append((shp_transform(tf.transform, shape(r["the_geom"])), h))
    shapes.sort(key=lambda t: t[1])  # tallest last, so it wins a shared cell
    h = np.full((gs.ny, gs.nx), np.nan, np.float32)
    if shapes:
        # rasterio rows run north to south; flip afterwards to the contract's south-first order
        tr = from_origin(gs.sw_utm[0], gs.sw_utm[1] + gs.ny * gs.cell, gs.cell, gs.cell)
        ras = features.rasterize(shapes, out_shape=(gs.ny, gs.nx), transform=tr, fill=np.nan,
                                 all_touched=True, dtype="float32")
        h = ras[::-1]
    return np.isfinite(h), h


# --- assembly -----------------------------------------------------------------------------


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=cache.REPO_ROOT, capture_output=True, text=True,
                          check=False).stdout.strip()


def build_info() -> uo.BuildInfo:
    commit = _git("rev-parse", "HEAD")
    dirty = bool(_git("status", "--porcelain", "--untracked-files=no"))
    return uo.BuildInfo(commit if len(commit) == 40 else "unknown", dirty,
                        time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))


def _source(key: str, name: str, product_id: str, role: str, src_key: str, art: cache.Artifact,
            captured: str, published: str, redistribution: str = "repository") -> uo.UrbanSource:
    from ..allowlist import ALLOWLIST

    s = ALLOWLIST[src_key]
    return uo.UrbanSource(key=key, name=name, product_id=product_id, role=role, licence=s.license,
                          licence_url=s.license_url, citation=s.citation, url=art.url, captured=captured,
                          published=published, retrieved=art.fetched_at, sha256=art.sha256,
                          redistribution=redistribution)


def exclusion_mask(aoi: AOI, gs: GridSpec) -> list[tuple[Any, np.ndarray]]:
    from pyproj import Transformer

    e, n = gs.centers_utm()
    lon, lat = Transformer.from_crs(f"EPSG:{gs.epsg}", "EPSG:4326", always_xy=True).transform(e, n)
    out = []
    for z in aoi.protected_zones:
        if "extraction" in z.policies:
            w, s, ea, no = z.buffered()
            out.append((z, (lon >= w) & (lon <= ea) & (lat >= s) & (lat <= no)))
    return out


def build_artifact(aoi: AOI, cache_root: Path | None = None, force: bool = False,
                   build: uo.BuildInfo | None = None) -> tuple[uo.UrbanObstacleGrid, Path, dict[str, Any], list[cache.Artifact]]:
    """Run the pipeline and publish the artifact. Returns (grid, directory, evidence, inputs)."""
    gs = grid_spec(aoi)
    ground_art = fetch_ground(aoi, gs, force)
    dem = sample_ground(ground_art, gs)
    pts_art = fetch_points(aoi, gs, force)
    with np.load(pts_art.abs_path) as z:
        points = {k: z[k] for k in ("e", "n", "z", "c")}
        pm = json.loads(bytes(z["manifest"]).decode())
    ras = lidar_rasters(points, gs)

    # Datum/unit check: lidar ground must agree with the DEM, or heights are measured on a wrong base.
    both = np.isfinite(ras["ground"]) & np.isfinite(dem)
    if both.sum() < 100:
        raise RuntimeError("too few cells with both lidar ground and DEM to verify the vertical datum")
    ground_offset = float(np.median(ras["ground"][both] - dem[both]))
    if abs(ground_offset) > MAX_GROUND_DISAGREEMENT_M:
        raise RuntimeError(f"lidar ground differs from the DEM by a median {ground_offset:.2f} m; "
                           "refusing to build on a datum/unit mismatch")

    base = _fill_nearest(ras["ground"])
    lid_h = ras["top"] - base
    lid_mask = (ras["count"] >= MIN_STRUCTURE_POINTS) & (lid_h >= MIN_STRUCTURE_HEIGHT_M)
    height = np.where(lid_mask, lid_h, np.nan)
    unc_cell = np.where(lid_mask, np.where(ras["count"] >= 5, 1.5, 3.0), np.nan)
    sources_extra: list[uo.UrbanSource] = []
    inputs = [ground_art, pts_art]
    stats: dict[str, Any] = {"lidar_ground_minus_dem_median_m": round(ground_offset, 3),
                             "lidar_points_kept": pm["points_kept"], "lidar_nodes": len(pm["nodes"])}

    fp_art = None
    if aoi.name == "nyc_harbor":
        fp_art, rows = nyc_footprints(aoi, gs, force)
        inputs.append(fp_art)
        fp_mask, fp_h = rasterise_footprints(rows, gs)
        both_m = fp_mask & lid_mask
        stats["footprint_records"] = len(rows)
        stats["cells_footprint_only"] = int((fp_mask & ~lid_mask).sum())
        stats["cells_lidar_only"] = int((lid_mask & ~fp_mask).sum())
        stats["cells_both"] = int(both_m.sum())
        if both_m.any():
            stats["footprint_vs_lidar_height_median_diff_m"] = round(float(np.median(fp_h[both_m] - lid_h[both_m])), 2)
            stats["footprint_vs_lidar_height_median_abs_diff_m"] = round(float(np.median(np.abs(fp_h[both_m] - lid_h[both_m]))), 2)
        only_fp = fp_mask & ~lid_mask & (fp_h >= MIN_STRUCTURE_HEIGHT_M)
        height = np.where(only_fp, fp_h, height)
        unc_cell = np.where(only_fp, 5.0, unc_cell)
        lid_mask = lid_mask | only_fp
        edited = sorted(r["last_edited_date"][:10] for r in rows if r.get("last_edited_date"))
        view = json.loads(cache.fetch(f"nyc_footprints_view/{aoi.name}", "nyc_building_footprints",
                                      NYC_FOOTPRINTS_VIEW, f"urban_src/nyc_footprints_view.json",
                                      validate=lambda b: b.lstrip()[:1] == b"{").abs_path.read_bytes())
        pub = dt.datetime.fromtimestamp(view.get("rowsUpdatedAt", 0), dt.timezone.utc).date().isoformat()
        sources_extra.append(_source(
            "nyc_building_footprints", "NYC Building Footprints", "NYC Open Data 5zhs-2jue", "footprints",
            "nyc_building_footprints", fp_art,
            f"feature last-edited {edited[0]} .. {edited[-1]} (heights from city lidar/photogrammetry)" if edited else "unknown",
            pub))

    # coverage: surveyed where a lidar return lies nearby, or open water inside the collection
    near = _dilate(np.isfinite(ras["any"]), max(1, int(round(COVERAGE_RADIUS_M / gs.cell))))
    covered = np.isfinite(dem) & (near | (dem <= WATER_ELEVATION_M))
    # protected-zone exclusions -> no-data, recorded per zone
    exclusions = []
    already = np.zeros_like(covered)
    for z, m in exclusion_mask(aoi, gs):
        newly = m & covered & ~already
        exclusions.append(uo.Exclusion(z.name, tuple(float(v) for v in (z.west, z.south, z.east, z.north)), int(newly.sum())))
        covered &= ~m
        already |= m
    is_b = covered & lid_mask & np.isfinite(height)
    ground = np.where(covered, dem, np.nan).astype(np.float32)
    top = np.where(is_b, dem + np.nan_to_num(height), ground).astype(np.float32)
    unc = np.where(covered, np.where(is_b, np.nan_to_num(unc_cell, nan=3.0), 1.0), np.nan).astype(np.float32)
    exclusions = [e for e in exclusions if e.cells > 0]
    if not covered.any():
        raise RuntimeError("no covered cell")

    products = lidar_products(aoi, aoi.urban_physics.bounds, force)
    cap = (pm["captured_min"] if pm["captured_min"] == pm["captured_max"]
           else f"{pm['captured_min']} .. {pm['captured_max']}") if pm["captured_min"] else "unknown (no adjusted GPS time)"
    pub = max((p["publicationDate"] for p in products if p.get("publicationDate")), default="unknown")
    titles = "; ".join(sorted({p["title"] for p in products})) or "none listed by TNM for this box"
    sources = [
        _source("usgs_3dep_dem", "USGS 3DEP bare-earth DEM", f"3DEP best-available, {aoi.urban_physics.cell_m:g} m", "ground",
                "usgs_3dep", ground_art, "varies by source tile (3DEP seamless mosaic)", "unknown (service mosaic)"),
        _source("usgs_3dep_lidar", "USGS 3DEP lidar point cloud (EPT)", f"{LIDAR[aoi.name]['collection']}; TNM: {titles}",
                "structures", "usgs_3dep_lidar", pts_art, cap, pub),
        *sources_extra,
    ]
    meta = uo.UrbanObstacleMeta(
        aoi_name=aoi.name, aoi_fingerprint=aoi.fingerprint, aoi_bounds_wgs84=aoi.bbox,
        physics_bounds_wgs84=aoi.urban_physics.bounds, grid_corners_wgs84=gs.corners_wgs84(),
        utm_epsg=gs.epsg, env_origin_utm_m=gs.env_origin_utm, grid_origin_enu_m=(gs.x0, gs.y0),
        cell_m=gs.cell, nx=gs.nx, ny=gs.ny, sources=tuple(sources), build=build or build_info(),
        uncertainty_present=True,
        uncertainty_kind="heuristic height uncertainty, metres: 1.0 open ground; 1.5/3.0 lidar structure by return count; 5.0 footprint-only",
        exclusions=tuple(exclusions),
        notes=(f"max-per-cell structure top over a {gs.cell:g} m cell (conservative: any structure in the cell "
               f"blocks it); lidar nominal spacing at EPT depth {LIDAR[aoi.name]['max_depth']}; structure heights "
               "are lower bounds for spires and masts; classification is the provider's"))
    grid = uo.build_urban_obstacle_grid(meta, ground_amsl_m=ground, top_amsl_m=top, coverage=covered.astype(np.uint8),
                                        building=is_b.astype(np.uint8), uncertainty_m=unc)
    root = cache_root if cache_root is not None else cache.cache_dir()
    path = uo.write_urban_obstacles(grid, root)
    stats.update(grid.evidence())
    stats["excluded_cells"] = sum(e.cells for e in exclusions)
    return grid, path, stats, inputs
