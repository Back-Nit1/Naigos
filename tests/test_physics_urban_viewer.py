"""The viewer draws the simulation's own urban obstacle surface in physics mode, and says what it is.

Offline: no browser, no GPU, no network. Under test:

* the three states -- ``not_declared``, ``available``, ``unavailable`` -- and that a missing artifact is
  reported, never replaced by presentation geometry;
* ``/physics_urban`` is served to the physics page only (404 elsewhere) and carries the artifact, with
  no path, token or presentation field;
* a physics export embeds the surface, a presentation export embeds only its status;
* the page's pure block (decode, quads, HUD lines) runs under node when node is installed;
* the HUD names the four kinds of surface on separate lines, evidence and presentation distinct.
"""

from __future__ import annotations

import base64
import json
import shutil
import subprocess
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest
import _urban_fixtures as fx

from naigos.demo import imagery as imagery_mod
from naigos.demo import live, physics_urban as pu

REPO = Path(__file__).resolve().parents[1]
PAGE = (REPO / "naigos" / "demo" / "assets" / "cesium.html").read_text()
PHYS_JS = PAGE[PAGE.index("/*__PHYS_URBAN_BEGIN__*/"):PAGE.index("/*__PHYS_URBAN_END__*/")]
NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="node not installed")


@pytest.fixture(scope="module")
def grid():
    return fx.make_grid()


@pytest.fixture
def available(grid):
    return pu.PhysicsUrban("available", None, grid)


def test_states_and_scene_block(grid, available):
    nd = pu.PhysicsUrban("not_declared", "x").scene_block()
    assert nd["state"] == "not_declared" and nd["evidence_grade"] is False and nd["geometry_source"] is None
    ok = available.scene_block()
    assert ok["state"] == "available" and ok["evidence_grade"] is True
    assert ok["geometry_source"] == "physics_obstacle_artifact"
    assert ok["evidence"]["content_sha256"] == grid.hashes.content_sha256
    assert ok["evidence"]["cell_m"] == fx.CELL and ok["evidence"]["nx"] == fx.NX
    assert 0.0 < ok["evidence"]["coverage_fraction"] < 1.0
    un = pu.PhysicsUrban("unavailable", "no validated artifact").scene_block()
    assert un["evidence_grade"] is False and "evidence" not in un and "unavailable" in un["note"]


def test_loading_a_theatre_without_the_block_is_not_declared():
    assert pu.load("dubai_urban").state == "not_declared"
    assert pu.load(None).state == "not_declared"
    assert pu.load("no_such_theatre").state == "not_declared"


def test_an_artifact_attached_to_the_env_wins(grid):
    got = pu.load("dubai_urban", {"urban_obstacles": grid})
    assert got.available and got.grid is grid


def test_the_page_payload_is_the_artifact_and_only_the_artifact(grid, available):
    p = available.page_payload()
    blob = json.dumps(p)
    assert p["schema"] == pu.PAGE_SCHEMA and p["artifact_schema"] == "naigos.urban_obstacles"
    assert "SECRET" not in blob and "/Users/" not in blob and "overpass" not in blob.lower()
    top = np.frombuffer(base64.b64decode(p["top_dm"]), "<i2").reshape(fx.NY, fx.NX)
    flags = np.frombuffer(base64.b64decode(p["flags"]), "|u1").reshape(fx.NY, fx.NX)
    cov = grid.coverage == 1
    assert (top[~cov] == pu.NODATA_DM).all() and (flags[~cov] == 0).all()
    assert np.allclose(top[cov] / 10.0, grid.top_amsl_m[cov], atol=0.051)
    assert ((flags & pu.FLAG_BUILDING) > 0).sum() == grid.building_cells


def _serve(handler):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def _get(url):
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, b""


def test_physics_urban_is_served_to_the_physics_page_only(available):
    physics = imagery_mod.resolve_visual_config("physics", ion_token=None, google_api_key=None, imagery="osm")
    srv, base = _serve(live.make_handler(None, {}, None, visual=physics, physics_urban=available))
    try:
        code, body = _get(base + "/physics_urban")
        assert code == 200 and json.loads(body)["schema"] == pu.PAGE_SCHEMA
        assert _get(base + "/")[0] == 200
    finally:
        srv.shutdown()
    urban = imagery_mod.resolve_visual_config("urban-presentation", ion_token=None, google_api_key=None,
                                              imagery="osm", local_urban=False)
    srv, base = _serve(live.make_handler(None, {}, None, visual=urban, physics_urban=available))
    try:
        assert _get(base + "/physics_urban")[0] == 404, "presentation pages never hold the bytes"
    finally:
        srv.shutdown()
    srv, base = _serve(live.make_handler(None, {}, None, visual=physics,
                                         physics_urban=pu.PhysicsUrban("unavailable", "gone")))
    try:
        assert _get(base + "/physics_urban")[0] == 404
    finally:
        srv.shutdown()


def test_the_static_export_keeps_the_two_kinds_of_urban_geometry_apart(tmp_path, monkeypatch, available):
    pytest.importorskip("jax")
    import test_viewer_export as tve

    demo = tve.demo_json.__wrapped__(tmp_path_factory=_Factory(tmp_path)) if hasattr(tve.demo_json, "__wrapped__") else None
    if demo is None:
        pytest.skip("export fixture not reusable here")
    monkeypatch.setattr(pu, "load", lambda *a, **k: available)
    from naigos.demo import viewer

    out = viewer.build(demo, tmp_path / "phys.html", visual_mode="physics")
    emb = tve._embedded(out)
    assert emb["/scene"]["physics_urban"]["state"] == "available"
    assert emb["/scene"]["physics_urban"]["drawn_in_this_mode"] is True
    assert emb["/physics_urban"]["evidence"]["content_sha256"] == available.grid.hashes.content_sha256
    text = out.read_text()
    assert "SECRET" not in text and "ion_token\":\"" not in text


class _Factory:
    def __init__(self, p):
        self.p = p

    def mktemp(self, name):
        d = self.p / name
        d.mkdir(exist_ok=True)
        return d


def test_static_payload_presentation_export_carries_status_only(tmp_path, available):
    pytest.importorskip("jax")
    import test_viewer_export as tve

    demo = tve.demo_json.__wrapped__(tmp_path_factory=_Factory(tmp_path))
    payload = live.static_payload(demo, physics_urban=available, embed_physics_urban=False)
    assert payload["/scene"]["physics_urban"]["state"] == "available"
    assert payload["/scene"]["physics_urban"]["drawn_in_this_mode"] is False
    assert "/physics_urban" not in payload
    payload = live.static_payload(demo, physics_urban=pu.PhysicsUrban("unavailable", "gone"),
                                  embed_physics_urban=True)
    assert "/physics_urban" not in payload and payload["/scene"]["physics_urban"]["state"] == "unavailable"


def test_the_hud_has_four_separate_surface_lines():
    assert 'id="layerprov"' in PAGE and "Surfaces" in PAGE
    for key in ("physics_terrain", "physics_urban_obstacles", "presentation_local_buildings",
                "presentation_provider_buildings"):
        assert f'key: "{key}"' in PHYS_JS
    assert "not physical" in PAGE and "UNAVAILABLE" in PHYS_JS
    # the physics drawing never reads the presentation layer or its palette
    block = PAGE[PAGE.index("physics urban obstacles\n"):PAGE.index("// --------------------------------------------------------------- city layer")]
    assert "URBAN_THEMES" not in block and "THEME" not in block and 'getRoute("/urban")' not in block
    assert 'VISUAL.mode === "physics"' in block and 'getRoute("/physics_urban")' in block


def test_physics_page_draws_nothing_when_unavailable():
    block = PAGE[PAGE.index("physics urban obstacles\n"):PAGE.index("// --------------------------------------------------------------- city layer")]
    assert 'PHYSU.state === "available"' in block
    assert "unavailable" in block and 'getRoute("/urban")' not in block


def _node(js: str):
    r = subprocess.run([NODE, "-e", js], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


@needs_node
def test_pure_block_decodes_and_builds_the_surface(available):
    p = available.page_payload()
    js = PHYS_JS + f"""
const d = {json.dumps(p)};
const g = physDecode(d);
const all = physQuads(g, 0, g.nx, 0, g.ny);
const st = physStats(g);
// every quad: 4 corners x [kind, i, j, h]; count tops
let tops = 0, walls = 0;
for (let k = 0; k < all.length; k += 16) {{ if (all[k] === 0) tops++; else walls++; }}
// a corner round-trips to the SW corner of the artifact
const c0 = physCorner(g, 0, 0), c1 = physCorner(g, g.nx, g.ny);
console.log(JSON.stringify({{nx: g.nx, ny: g.ny, tops, walls, built: st.built, covered: st.covered,
  maxH: st.max_height_m, sw: c0, ne: c1}}));
"""
    out = _node(js)
    g = available.grid
    assert out["nx"] == fx.NX and out["ny"] == fx.NY
    assert out["tops"] == g.building_cells == out["built"], "one top per obstacle cell, nothing invented"
    assert out["walls"] > 0 and out["covered"] == int(g.coverage.sum())
    assert abs(out["maxH"] - float(np.nanmax(g.height_agl_m))) < 0.2
    sw = g.meta.grid_corners_wgs84[0]
    assert abs(out["sw"][0] - sw[0]) < 1e-9 and abs(out["sw"][1] - sw[1]) < 1e-9


@needs_node
def test_pure_block_draws_walls_to_the_neighbour_and_ignores_unknown_cells():
    js = PHYS_JS + """
// 3x1 grid: [building 50 m] [open] [unknown]
const g = {nx: 3, ny: 1, ground: Int16Array.of(100, 100, -32768), top: Int16Array.of(600, 100, -32768),
           flags: Uint8Array.of(3, 1, 0), corners: [[0,0],[3,0],[3,1],[0,1]], cell: 10};
const q = physQuads(g, 0, 3, 0, 1);
const kinds = []; for (let k = 0; k < q.length; k += 16) kinds.push([q[k], q[k+3]]);
console.log(JSON.stringify(kinds));
"""
    kinds = _node(js)
    assert [k for k, _ in kinds].count(0) == 1, "only the building cell has a top"
    walls = sorted(k for k, _ in kinds if k)
    assert walls == [1, 2, 3, 4], "west/south/north walls stand on its own ground; east wall to the open neighbour"
    lows = {k: h for k, h in kinds if k}
    assert all(h == 10.0 or h == 60.0 for h in lows.values())


@needs_node
def test_pure_block_hud_lines():
    js = PHYS_JS + """
const ev = {nx: 5, ny: 4, cell_m: 10, coverage_fraction: 0.97, content_sha256: "abcdef0123456789"};
const out = {
  ok: layerLines({visualMode: "physics", terrainDrawn: true, physUrban: {state: "available", evidence: ev}, physUrbanDrawn: true}),
  pres: layerLines({visualMode: "urban-presentation", terrainDrawn: true, physUrban: {state: "available", evidence: ev},
                    physUrbanDrawn: false, localText: "3 of 9 buildings drawn (OSM, ODbL) — not physical"}),
  gone: layerLines({visualMode: "physics", terrainDrawn: true, physUrban: {state: "unavailable"}, physUrbanDrawn: false}),
  none: layerLines({visualMode: "physics", terrainDrawn: true, physUrban: {state: "not_declared"}}),
};
console.log(JSON.stringify(out));
"""
    out = _node(js)
    for lines in out.values():
        assert [l["key"] for l in lines] == ["physics_terrain", "physics_urban_obstacles",
                                              "presentation_local_buildings", "presentation_provider_buildings"]
        assert lines[0]["cls"].startswith("ev") and lines[1]["cls"].startswith("ev")
        assert lines[2]["cls"] == "pr" and lines[3]["cls"] == "pr"
    assert out["ok"][1]["text"].startswith("drawn") and "97.0%" in out["ok"][1]["text"]
    assert out["pres"][1]["text"].startswith("not drawn in this mode") and "not physical" in out["pres"][2]["text"]
    assert "UNAVAILABLE" in out["gone"][1]["text"] and "un" in out["gone"][1]["cls"].split()
    assert out["none"][1]["text"].startswith("none declared")
