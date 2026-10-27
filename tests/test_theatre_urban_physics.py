"""Every file-defined theatre that declares ``urban_physics`` is held to the same contract.

A future city adds its AOI file, its city config, its component snapshot and its committed obstacle
artifact, and these parametrised checks pick it up with no edit here. Offline: nothing below reads
the network or the research cache (the artifact comes from tests/fixtures/urban_physics_root/ through
``offline_theatre``).

What is held to the standard:

* the artifact is a distinct, hashed, fail-closed thing, in its own cache namespace;
* a theatre that declares ``urban_physics`` cannot run without it, and nothing presentation-only
  (OSM extrusions, provider tiles) stands in for it, in the data layer or in the viewer;
* every cited input has a licence, a hash and three honest dates;
* camera presets keep out of the physical obstacles they fly among;
* no credential, path or screenshot reaches a scene, an export or a committed file.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from _theatre_fixtures import offline_theatre, viewer_sampler

from naigos.data import urban_obstacles as uo
from naigos.demo import cities, physics_urban
from naigos.research import allowlist, cache
from naigos.research.aoi import AOIS, aoi_from_file

REPO = Path(__file__).resolve().parents[1]
THEATRES = sorted(a.name for a in AOIS.values() if a.urban_physics is not None)
SHA = re.compile(r"^[0-9a-f]{64}$")


def snapshot(aoi: str) -> Path:
    return REPO / "components" / "aoi" / aoi


def comp(aoi: str, cid: str) -> dict:
    return json.loads((snapshot(aoi) / f"{cid}.json").read_text())


def fixture_dir(aoi: str) -> Path:
    return REPO / "tests" / "fixtures" / "urban_physics_root" / comp(aoi, "data.urban_obstacles")["parameters"]["artifact_dir"]


@pytest.fixture(params=THEATRES)
def aoi(request):
    return request.param


def test_the_two_us_theatres_are_registered():
    assert {"nyc_harbor", "los_angeles_basin"} <= set(THEATRES)


# --- the AOI file ------------------------------------------------------------------------


def test_the_aoi_file_is_strict(aoi, tmp_path):
    a = AOIS[aoi]
    doc = json.loads(Path(a.definition_file).read_text())
    assert aoi_from_file(Path(a.definition_file)) == a
    bad = tmp_path / f"{aoi}.json"
    bad.write_text(json.dumps({**doc, "typo_key": 1}))
    with pytest.raises(ValueError, match="unknown"):
        aoi_from_file(bad)
    bad.write_text(json.dumps({**doc, "urban_physics": {"bounds": doc["urban_physics"]["bounds"]}}))
    with pytest.raises(ValueError, match="exactly bounds and cell_m"):
        aoi_from_file(bad)
    w, s, e, n = doc["urban_physics"]["bounds"]
    bad.write_text(json.dumps({**doc, "urban_physics": {"bounds": [w, s, doc["east"] + 1, n], "cell_m": 10}}))
    with pytest.raises(ValueError, match="inside the AOI"):
        aoi_from_file(bad)


def test_the_box_is_a_broad_theatre_and_the_urban_box_is_a_small_part_of_it(aoi):
    a = AOIS[aoi]
    ew, ns = a.span_km()
    assert 40.0 < ew < 100.0 and 40.0 < ns < 100.0, "broad enough for procedural threats and multi-leg missions"
    w, s, e, n = a.urban_physics.bounds
    area = (e - w) * (n - s) / ((a.east - a.west) * (a.north - a.south))
    assert 0.005 < area < 0.05, "the high-resolution urban box is a sub-box, not the AOI"
    c = cities.get_city(aoi)
    assert c is not None and a.west <= c.urban_bounds[0] and c.urban_bounds[2] <= a.east
    assert a.country == "US" and a.dem_source == "usgs_3dep"
    assert a.scenario == "notional contested-airspace simulation"
    assert "digital twin" not in (a.rationale + a.bounds_policy).lower().replace("not a digital twin", "")


def test_protected_zones_name_a_box_and_a_reason_only(aoi):
    for z in AOIS[aoi].protected_zones:
        assert z.reason and set(z.policies) <= set(__import__("naigos.research.aoi", fromlist=["x"]).ZONE_POLICIES)
        assert not re.search(r"(?i)air.?defen|radar site|missile|military|naval|army|air force|fort ", z.name), z.name


# --- components: cited, hashed, fingerprinted ---------------------------------------------------


def test_every_component_is_cited_and_every_artifact_hashed(aoi):
    a = AOIS[aoi]
    comps = sorted(p.stem for p in snapshot(aoi).glob("*.json"))
    assert "data.urban_obstacles" in comps and "data.terrain_dem" in comps
    for cid in comps:
        c = comp(aoi, cid)
        assert c["sources"] and all(s["key"] in allowlist.ALLOWLIST for s in c["sources"]), cid
        assert c["license"], cid
        for art in c["cached_artifacts"]:
            assert SHA.match(art["sha256"]), (cid, art["key"])
    assert comp(aoi, "env.aoi")["parameters"]["fingerprint"] == a.fingerprint
    assert comp(aoi, "env.aoi")["parameters"]["urban_physics"]["cell_m"] == a.urban_physics.cell_m
    for cid in ("data.terrain_dem", "data.atmosphere"):
        assert all(a.fingerprint in x["key"] for x in comp(aoi, cid)["cached_artifacts"]), cid
    for x in comp(aoi, "data.urban_obstacles")["cached_artifacts"]:
        assert a.fingerprint in x["key"] or x["key"].startswith(f"urban_physics/{aoi}/"), x["key"]


def test_the_obstacle_component_cites_and_hashes_the_artifact(aoi):
    c = comp(aoi, "data.urban_obstacles")
    p = c["parameters"]
    d = fixture_dir(aoi)
    grid = uo.load_urban_obstacles(d)
    assert p["content_sha256"] == grid.hashes.content_sha256 and p["aoi_fingerprint"] == AOIS[aoi].fingerprint
    assert p["artifact_dir"] == f"urban_physics/{aoi}/{grid.hashes.content_id}"
    assert "visual" not in Path(p["artifact_dir"]).parts, "physics data never lives under visual/"
    by_name = {Path(x["path"]).name: x for x in c["cached_artifacts"] if x["path"].startswith("urban_physics/")}
    for fname, want in ((uo.PAYLOAD_FILE, grid.hashes.payload_sha256),):
        assert by_name[fname]["sha256"] == want
        assert cache.sha256_bytes((d / fname).read_bytes()) == want
    assert by_name[uo.METADATA_FILE]["sha256"] == cache.sha256_bytes((d / uo.METADATA_FILE).read_bytes())
    assert {s["key"] for s in c["sources"]} >= {"usgs_3dep", "usgs_3dep_lidar"}
    assert "osm_urban_visual" not in {x["key"] for x in c["sources"]}, "OSM is presentation-only"
    assert not any("visual/" in x["path"] for x in c["cached_artifacts"])


def test_sources_carry_licence_hash_and_three_dates(aoi):
    g = uo.load_urban_obstacles(fixture_dir(aoi))
    keys = {s.key for s in g.meta.sources}
    assert {"usgs_3dep_dem", "usgs_3dep_lidar"} <= keys
    for s in g.meta.sources:
        assert s.licence and s.citation and SHA.match(s.sha256) and s.redistribution == "repository"
        assert s.captured and s.published and s.retrieved
        assert "unknown" not in s.retrieved.lower()
    lidar = next(s for s in g.meta.sources if s.key == "usgs_3dep_lidar")
    assert re.search(r"20\d\d", lidar.captured), "the lidar capture date is recorded, not assumed"
    assert lidar.captured != lidar.published, "capture and publication are different dates"
    assert "public domain" in lidar.licence.lower()


# --- the artifact: valid, bounded, honest ----------------------------------------------------------


def test_the_artifact_matches_the_aoi_and_the_documented_box(aoi):
    a = AOIS[aoi]
    g = uo.load_urban_obstacles(fixture_dir(aoi), uo.ExpectedFrame(
        aoi_name=aoi, aoi_fingerprint=a.fingerprint, utm_epsg=comp(aoi, "env.aoi")["parameters"]["utm_epsg"],
        aoi_bounds_wgs84=a.bbox))
    m = g.meta
    assert m.physics_bounds_wgs84 == tuple(a.urban_physics.bounds) and m.cell_m == a.urban_physics.cell_m
    w, s, e, n = a.urban_physics.bounds
    ew = (e - w) * 111_320.0 * np.cos(np.radians((s + n) / 2)) / 1000.0
    assert abs(m.nx * m.cell_m / 1000.0 - ew) < 0.6, "the grid fills the documented box"
    assert 100_000 < m.nx * m.ny < 1_000_000, "bounded: useful for LOS and collision, not an unbounded JAX state"
    assert g.coverage_fraction > 0.9
    assert 0.1 < g.building_cells / g.coverage.sum() < 0.9


def test_unknown_cells_are_never_zero_height_buildings(aoi):
    g = uo.load_urban_obstacles(fixture_dir(aoi))
    unk = g.coverage == 0
    assert np.isnan(g.top_amsl_m[unk]).all() and not g.building[unk].any()
    assert g.top_with_nodata(-np.inf)[unk].max() == -np.inf


def test_a_wrong_aoi_or_zone_is_refused(aoi):
    a = AOIS[aoi]
    good = dict(aoi_name=aoi, aoi_fingerprint=a.fingerprint, aoi_bounds_wgs84=a.bbox,
                utm_epsg=comp(aoi, "env.aoi")["parameters"]["utm_epsg"])
    for over in (dict(aoi_fingerprint="0" * 10), dict(utm_epsg=good["utm_epsg"] + 1),
                 dict(aoi_name="mecca_urban")):
        with pytest.raises(uo.UrbanObstacleError):
            uo.load_urban_obstacles(fixture_dir(aoi), uo.ExpectedFrame(**{**good, **over}))


def test_corruption_and_version_changes_are_refused(aoi, tmp_path):
    d = tmp_path / "urban_physics" / aoi / fixture_dir(aoi).name
    shutil.copytree(fixture_dir(aoi), d)
    p = d / uo.PAYLOAD_FILE
    raw = bytearray(p.read_bytes())
    raw[len(raw) // 2] ^= 0x01
    p.write_bytes(bytes(raw))
    with pytest.raises(uo.UrbanObstacleError, match="payload hash"):
        uo.load_urban_obstacles(d)
    shutil.copy2(fixture_dir(aoi) / uo.PAYLOAD_FILE, p)
    m = d / uo.METADATA_FILE
    m.write_bytes(m.read_bytes().replace(b'"schema_version":1', b'"schema_version":2'))
    with pytest.raises(uo.UrbanObstacleError):
        uo.load_urban_obstacles(d)
    (d / uo.COMPLETE_FILE).unlink()
    with pytest.raises(uo.UrbanObstacleUnavailable):
        uo.load_urban_obstacles(d)


# --- fail closed ------------------------------------------------------------------------------------


def test_the_theatre_fails_closed_without_its_artifact(aoi, tmp_path):
    from naigos.data.theatre import load_theatre, load_urban_obstacles_for
    from naigos.research.roots import research_roots

    with offline_theatre(aoi) as root:
        assert load_theatre(aoi).urban_obstacles is not None
        art = root / comp(aoi, "data.urban_obstacles")["parameters"]["artifact_dir"]
        (art / uo.PAYLOAD_FILE).write_bytes(b"corrupt")
        with pytest.raises(uo.UrbanObstacleError):
            load_theatre(aoi)
        shutil.rmtree(art)
        with pytest.raises(uo.UrbanObstacleUnavailable):
            load_theatre(aoi)
        with pytest.raises(uo.UrbanObstacleUnavailable):
            load_urban_obstacles_for(aoi)


def test_a_theatre_without_the_block_has_no_obstacles():
    from naigos.data.theatre import load_urban_obstacles_for

    assert load_urban_obstacles_for("dubai_urban") is None and load_urban_obstacles_for("mecca_urban") is None


def test_the_viewer_never_substitutes_presentation_buildings(aoi, tmp_path):
    from naigos.research.roots import research_roots

    with offline_theatre(aoi):
        ok = physics_urban.load(aoi)
        assert ok.available and ok.scene_block()["geometry_source"] == "physics_obstacle_artifact"
        assert ok.scene_block()["evidence_grade"] is True
    with research_roots(cache_dir=tmp_path):                 # an empty cache: no artifact anywhere
        gone = physics_urban.load(aoi)
    assert gone.state == "unavailable" and not gone.available
    blk = gone.scene_block()
    assert blk["evidence_grade"] is False and blk["geometry_source"] is None and "evidence" not in blk
    assert "unavailable" in blk["note"] and "naigos-research" in blk["reason"]
    assert gone.page_payload() is None
    assert str(tmp_path) not in json.dumps(blk), "no filesystem path in a scene"


def test_the_physics_payload_is_the_artifact_and_leaks_nothing(aoi, monkeypatch):
    monkeypatch.setenv("NAIGOS_CESIUM_ION_TOKEN", "SECRET-ion-token-123")
    monkeypatch.setenv("NAIGOS_GOOGLE_MAPS_API_KEY", "SECRET-google-key-456")
    with offline_theatre(aoi) as root:
        pu = physics_urban.load(aoi)
        blob = json.dumps({"scene": pu.scene_block(), "page": pu.page_payload()})
    assert "SECRET" not in blob and str(root) not in blob and str(REPO) not in blob
    page = pu.page_payload()
    assert page["schema"] == physics_urban.PAGE_SCHEMA and page["evidence"]["content_sha256"] == pu.grid.hashes.content_sha256
    import base64
    top = np.frombuffer(base64.b64decode(page["top_dm"]), "<i2").reshape(page["ny"], page["nx"])
    flags = np.frombuffer(base64.b64decode(page["flags"]), "|u1").reshape(page["ny"], page["nx"])
    g = pu.grid
    cov = g.coverage == 1
    assert np.array_equal(flags & 1, g.coverage) and np.array_equal((flags >> 1) & 1, g.building)
    assert np.abs(top[cov] / 10.0 - g.top_amsl_m[cov]).max() <= 0.051
    assert (top[~cov] == physics_urban.NODATA_DM).all()
    for c in page["corners_wgs84"]:
        assert AOIS[aoi].west < c[0] < AOIS[aoi].east and AOIS[aoi].south < c[1] < AOIS[aoi].north


def test_committed_files_hold_no_secret_and_no_screenshot(aoi):
    tracked = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True, text=True).stdout.split()
    bad_img = [f for f in tracked if re.search(r"(?i)screenshot|\.(png|jpe?g|heic|webp)$", f)
               and ("fixtures" in f or "theatres" in f or "components" in f or "contracts" in f)]
    assert not bad_img, f"reference images must not be committed: {bad_img}"
    for f in (f"naigos/research/aois/{aoi}.json", f"naigos/demo/cities/{aoi}.json",
              f"docs/theatres/{aoi}.md", f"docs/theatres/{aoi}/DATA.md",
              *[str(p.relative_to(REPO)) for p in snapshot(aoi).glob("*.json")]):
        text = (REPO / f).read_text()
        assert not re.search(r"(?i)(token|api[_-]?key|secret)\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{16,}", text), f
        assert "/Users/" not in text and "Screenshot 20" not in text, f


# --- cameras ------------------------------------------------------------------------------------------


def test_camera_presets_pass_the_standard_checks(aoi):
    c = cities.get_city(aoi)
    out = cities.city_presets(c, viewer_sampler(aoi))        # raises on any broken rule
    assert {"urban_overview", "street_canyon"} <= set(out) and c.default_preset == "urban_overview"
    for key, o in out.items():
        assert o["camera_agl_m"] >= cities.MIN_AGL_M[cities.PRESET_KIND[key]], key


def test_camera_presets_keep_out_of_the_physical_obstacles(aoi):
    """A presentation camera must not sit inside the structure surface it is looking at."""
    g = uo.load_urban_obstacles(fixture_dir(aoi))
    c = cities.get_city(aoi)
    s = viewer_sampler(aoi)
    from pyproj import Transformer

    tf = Transformer.from_crs("EPSG:4326", f"EPSG:{g.meta.utm_epsg}", always_xy=True)
    x0, y0 = g.meta.env_origin_utm_m[0] + g.meta.grid_origin_enu_m[0], g.meta.env_origin_utm_m[1] + g.meta.grid_origin_enu_m[1]
    checked = 0
    for key, o in cities.city_presets(c, s).items():
        e, n = tf.transform(o["camera_lon"], o["camera_lat"])
        i, j = int((e - x0) // g.meta.cell_m), int((n - y0) // g.meta.cell_m)
        if not (0 <= i < g.meta.nx and 0 <= j < g.meta.ny) or not g.coverage[j, i]:
            continue
        checked += 1
        cam_h = o["camera_height_m"]
        assert cam_h > g.top_amsl_m[j, i] + 5.0, f"{key} camera is inside a structure"
    assert checked >= 1, "at least the street canyon is inside the physics box"


def test_zones_are_enforced_by_the_policies_they_name(aoi):
    a = AOIS[aoi]
    kinds = {p for z in a.protected_zones for p in z.policies}
    assert {"airfield", "ambience"} <= kinds
    # extraction zones never overlap the visual urban box (the standard), so no zone is half-cut
    c = cities.get_city(aoi)
    w, s, e, n = c.urban_bounds
    for z in c.zones("extraction"):
        zw, zs, ze, zn = z.buffered()
        assert ze < w or zw > e or zn < s or zs > n, z.name
    # a zone's cells, if any, are recorded as no-data in the artifact, never as open air
    g = uo.load_urban_obstacles(fixture_dir(aoi))
    for ex in g.meta.exclusions:
        assert ex.cells > 0 and any(z.name == ex.name for z in a.protected_zones)
