"""los_angeles_basin: a coastal basin under two ranges, with a dated, physics-grade Downtown structure surface.

Offline, from the committed snapshot, fixtures and obstacle artifact; nothing here reads the
network or the research cache. The contract every urban_physics theatre shares lives in
tests/test_theatre_urban_physics.py; this file holds what is particular to los_angeles_basin and the
simulation checks (environment, procedural threats, generic hazards, the zero-shot checkpoint).
"""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _theatre_fixtures import offline_theatre, viewer_sampler

from naigos.data import urban_obstacles as uo
from naigos.demo import cities, presentation, scenario
from naigos.research import cache
from naigos.research.aoi import get_aoi

AOI = "los_angeles_basin"
REPO = Path(__file__).resolve().parents[1]
SNAP = REPO / "components" / "aoi" / AOI


def comp(cid: str) -> dict:
    return json.loads((SNAP / f"{cid}.json").read_text())


def artifact():
    return uo.load_urban_obstacles(
        REPO / "tests" / "fixtures" / "urban_physics_root" / comp("data.urban_obstacles")["parameters"]["artifact_dir"])


# --- the AOI and its georeference -------------------------------------------------------------


def test_documented_bounds_and_extent():
    a = get_aoi(AOI)
    assert a.bbox == (-118.62, 33.8, -118.0, 34.3)
    ew, ns = a.span_km()
    assert (55.0, 59.0, 53.0, 57.0)[0] < ew < (55.0, 59.0, 53.0, 57.0)[1] and (55.0, 59.0, 53.0, 57.0)[2] < ns < (55.0, 59.0, 53.0, 57.0)[3]
    assert a.country == "US" and a.dem_source == "usgs_3dep" and a.exclude_military_airfields
    env = comp("env.aoi")["parameters"]
    assert env["fingerprint"] == a.fingerprint == "89ce98e489" and env["bbox_wgs84"] == list(a.bbox)
    assert env["utm_epsg"] == 32611
    assert env["urban_physics"] == {"bounds_wgs84": list(a.urban_physics.bounds), "cell_m": 10.0}


def test_the_documented_urban_boxes_are_separate_and_nested():
    a = get_aoi(AOI)
    c = cities.get_city(AOI)
    pw, ps, pe, pn = a.urban_physics.bounds
    vw, vs, ve, vn = c.urban_bounds
    assert a.west < vw < ve < a.east and a.south < vs < vn < a.north
    assert vw <= pw < pe <= ve and vs <= ps < pn <= vn, "the physics box sits inside the visual box"
    assert (vn - vs) * (ve - vw) > (pn - ps) * (pe - pw), "the two boxes are documented separately"


def test_the_dem_lands_where_it_should():
    m = viewer_sampler(AOI).m
    lo, hi = ((-5.0, 5.0), (1500.0, 2000.0))
    assert abs(m["west"] - get_aoi(AOI).west) < 0.02 and abs(m["east"] - get_aoi(AOI).east) < 0.02
    assert lo[0] < m["min_m"] < lo[1] and hi[0] < m["max_m"] < hi[1]
    s = viewer_sampler(AOI)
    for (lon, lat), (a, b) in [((-118.2437, 34.0522), (60.0, 120.0)), ((-118.06, 34.22), (700.0, 1900.0))]:
        assert a < s(lon, lat) < b, (lon, lat)


def test_the_weather_model_ground_height_agrees_with_the_dem():
    site = comp("data.atmosphere")["evidence"]["site"]
    h = viewer_sampler(AOI)(site["lon"], site["lat"])
    assert abs(h - site["model_elevation_m"]) < 60.0


def test_a_wrong_utm_zone_would_be_caught():
    from naigos.data.geodetic import GeoRef

    meta = viewer_sampler(AOI).m
    g = GeoRef(**meta["georef"])
    bad = GeoRef(**{**meta["georef"], "utm_epsg": meta["georef"]["utm_epsg"] + 1})
    assert g.utm_epsg == 32611
    assert abs(float(g.to_wgs84(0.0, 0.0)[0]) - float(bad.to_wgs84(0.0, 0.0)[0])) > 1.0


def test_the_obstacle_surface_has_the_expected_physical_scale():
    g = artifact()
    h = g.height_agl_m[g.building == 1]
    lo, hi = (280.0, 340.0)
    assert lo < float(h.max()) < hi, "the tallest cell is a plausible tower, not noise"
    assert (4.0, 20.0)[0] < float(np.median(h)) < (4.0, 20.0)[1]
    ground = g.ground_amsl_m[g.coverage == 1]
    assert (60.0, 130.0)[0] < float(np.median(ground)) < (60.0, 130.0)[1]


# --- offline rebuild --------------------------------------------------------------------------


def test_a_second_research_run_uses_the_cache(tmp_path, monkeypatch):
    """Every network entry point the urban builder has returns the cached artifact, never fetches."""
    import requests

    from naigos.research.roots import research_roots
    from naigos.research.sources import terrain, urban_physics as up

    a = get_aoi(AOI)

    def no_network(*args, **kw):
        raise AssertionError("a cached rerun made a network call")

    monkeypatch.setattr(requests, "get", no_network)
    monkeypatch.setattr(requests, "head", no_network)
    monkeypatch.setattr(terrain, "BUILDERS", {k: (no_network, u, lbl) for k, (_, u, lbl) in terrain.BUILDERS.items()})
    with research_roots(cache_dir=tmp_path):
        gs = up.GridSpec(32611, (0.0, 0.0), 0.0, 0.0, 10.0, 4, 4)
        seeded = {
            f"urban_ground/{AOI}/{a.fingerprint}/10m": ("usgs_3dep", "urban_src/g.tif"),
            f"urban_lidar_rasters/{AOI}/{a.fingerprint}/10m": ("usgs_3dep_lidar", "urban_src/l.npz"),
            f"lidar/{up.LIDAR[AOI]['collection']}/ept": ("usgs_3dep_lidar", "lidar/ept.json"),
        }
        for key, (src, rel) in seeded.items():
            body = b'{"cached": true}' if rel.endswith(".json") else b"cached"
            (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
            (tmp_path / rel).write_bytes(body)
            cache.record(key, src, "https://example.invalid", rel, body)
        assert up.fetch_ground(a, gs).path == "urban_src/g.tif"
        assert up.fetch_lidar(a, gs).path == "urban_src/l.npz"


def test_the_builder_only_talks_to_allowlisted_hosts():
    from naigos.research.sources import urban_physics as up

    from urllib.parse import urlparse

    for url, key in ((up.EPT_BASE + "/x", "usgs_3dep_lidar"), (up.TNM_PRODUCTS, "usgs_3dep_lidar"),
                     (up.NYC_FOOTPRINTS, "nyc_building_footprints")):
        from naigos.research.allowlist import check_url

        check_url(url, key)
    with pytest.raises(Exception):
        from naigos.research.allowlist import check_url

        check_url("https://example.com/ept.json", "usgs_3dep_lidar")


# --- the simulation ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def theatre():
    from naigos.env.theatre_bridge import env_from_theatre

    with offline_theatre(AOI):
        return env_from_theatre(aoi=AOI, n_blue=4, n_threat=10, cell_m=1500.0)


def test_the_env_builds_and_steps_finite(theatre):
    from naigos.env.flight_env import NaigosEnv

    cfg, hmap, notes = theatre
    assert notes["theatre"] == AOI and bool(jnp.isfinite(hmap).all())
    env = NaigosEnv(cfg, hmap=hmap)
    st, obs = env.reset(jax.random.PRNGKey(0))
    act = jnp.zeros((cfg.n_blue, cfg.action_dim))
    for _ in range(3):
        st, obs, *_ = env.step(st, act)
    for leaf in jax.tree_util.tree_leaves(st):
        if jnp.issubdtype(leaf.dtype, jnp.floating):
            assert bool(jnp.isfinite(leaf).all())


def test_threats_are_the_generic_classes_and_blue_is_evasive_only(theatre):
    from naigos.env.config import BLUE_ACTION_NAMES

    cfg, _, _ = theatre
    tehran = json.loads((REPO / "components" / "aoi" / "tehran_basin" / "model.detection.json")
                        .read_text())["parameters"]["threat_classes"]
    ours = comp("model.detection")["parameters"]["threat_classes"]
    assert set(ours) == set(tehran)
    for k in ours:
        assert ours[k]["radar"]["design_range_km"] == tehran[k]["radar"]["design_range_km"]
    assert cfg.action_dim == 3 and BLUE_ACTION_NAMES == ("bank_cmd", "gamma_cmd", "throttle_cmd")


def test_layouts_are_procedural_and_differ_by_seed(theatre):
    from naigos.env.flight_env import NaigosEnv

    cfg, hmap, _ = theatre
    env = NaigosEnv(cfg, hmap=hmap)
    seen = []
    for seed in range(4):
        st, _ = env.reset(jax.random.PRNGKey(seed))
        seen.append(np.asarray(st.threats.pos).copy())
    assert not np.allclose(seen[0], seen[1]) and not np.allclose(seen[1], seen[2])


def test_layouts_stay_inside_the_aoi(theatre):
    from naigos.data.geodetic import GeoRef
    from naigos.env.flight_env import NaigosEnv

    cfg, hmap, notes = theatre
    env = NaigosEnv(cfg, hmap=hmap)
    g = GeoRef(**notes["georef"])
    a = get_aoi(AOI)
    for seed in range(4):
        st, _ = env.reset(jax.random.PRNGKey(seed))
        for arr in (np.asarray(st.threats.pos), np.asarray(st.air.pos), np.asarray(st.objective)):
            lon, lat = g.to_wgs84(arr[:, 0], arr[:, 1])
            assert (lon > a.west).all() and (lon < a.east).all() and (lat > a.south).all() and (lat < a.north).all()


# --- presentation -----------------------------------------------------------------------------


def test_the_declared_cameras_exist_and_stay_in_the_safe_region():
    c = cities.get_city(AOI)
    assert set(c.presets) == {"urban_overview", "valley_overview", "street_canyon"} and c.default_preset == "urban_overview"
    out = cities.city_presets(c, viewer_sampler(AOI))
    for key, o in out.items():
        assert cities.point_in_polygon(o["camera_lon"], o["camera_lat"], c.safe_region), key
        assert cities.point_in_polygon(o["lon"], o["lat"], c.safe_region), key
        for z in c.zones("camera"):
            assert not z.contains(o["camera_lon"], o["camera_lat"]) and not z.contains(o["lon"], o["lat"])


def test_the_city_layer_stays_inside_its_box_when_built():
    from naigos.demo import urban

    st = urban.load(AOI)
    if not st.available:
        pytest.skip(f"local city layer not built here; run `python -m naigos.demo.urban --aoi {AOI}`")
    p = st.payload
    w, s, e, n = cities.get_city(AOI).urban_bounds
    for ch in p["chunks"]:
        for rec in ch["b"] + ch["r"]:
            for lon, lat in urban.decode_coords(rec[1:], tuple(p["origin"])):
                assert w - 1e-4 <= lon <= e + 1e-4 and s - 1e-4 <= lat <= n + 1e-4
    assert "OpenStreetMap" in p["source"]["attribution"]


def test_presentation_geometry_is_never_the_physics_geometry():
    """The OSM layer has no route into the obstacle artifact, and the artifact no route into OSM."""
    src = (REPO / "naigos" / "research" / "sources" / "urban_physics.py").read_text()
    assert "osm_urban_visual" not in src and "overpass" not in src.lower() and "visual/urban" not in src
    assert "urban_physics" not in (REPO / "naigos" / "demo" / "urban.py").read_text()
    g = artifact()
    assert not any("openstreetmap" in (s.name + s.url + s.citation).lower() for s in g.meta.sources)


def test_the_scene_says_notional_zero_shot_and_uses_the_local_profile():
    ck = scenario.checkpoint_theatre(REPO / "checkpoints" / "theatre_1000")
    p = presentation.resolve(AOI, "urban-presentation", checkpoint=ck)
    f = p.scene_fields()
    assert f["scenario"]["label"] == "notional contested-airspace simulation"
    assert f["scenario"]["checkpoint"]["relation"] == "zero_shot" and "ZERO-SHOT" in f["scenario"]["checkpoint"]["text"].upper()
    assert f["atmosphere"]["key"] == "warm_coastal_basin"
    with pytest.raises(presentation.PresentationError):
        presentation.resolve(AOI, "physics", ambience="conflict_ambience")


def test_the_page_is_handed_the_zones_to_cut_out_and_guard():
    from naigos.demo import camera

    s = viewer_sampler(AOI)
    p = camera.presets(s.m, s.m, city=cities.get_city(AOI), sample=s)
    zones = {z["name"] for z in p["protected"]}
    assert zones >= {z.name for z in cities.get_city(AOI).zones("render_cutout")}
