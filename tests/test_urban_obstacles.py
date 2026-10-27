"""The physics-grade urban obstacle contract (docs/contracts/urban-obstacles-v1.md)."""

from __future__ import annotations

import ast
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from naigos.data import urban_obstacles as uo
import _urban_fixtures as fx


@pytest.fixture
def grid():
    return fx.make_grid()


@pytest.fixture
def published(tmp_path, grid):
    return uo.write_urban_obstacles(grid, tmp_path)


def _expected(**over):
    kw = dict(aoi_name=fx.AOI_NAME, aoi_fingerprint=fx.aoi_fingerprint(), utm_epsg=fx.UTM_EPSG,
              aoi_bounds_wgs84=fx.AOI_BOUNDS)
    kw.update(over)
    return uo.ExpectedFrame(**kw)


def test_module_imports_no_framework():
    src = Path(uo.__file__).read_text()
    roots = {n.names[0].name.split(".")[0] if isinstance(n, ast.Import) else (n.module or "").split(".")[0]
             for n in ast.walk(ast.parse(src)) if isinstance(n, (ast.Import, ast.ImportFrom))}
    assert not roots & {"jax", "flax", "optax", "requests", "rasterio", "pyproj", "naigos"}, roots


def test_round_trip_is_identical_and_deterministic(tmp_path, grid, published):
    again = uo.load_urban_obstacles(published, _expected())
    assert again.hashes == grid.hashes and again.meta == grid.meta
    for n in ("ground_amsl_m", "top_amsl_m", "height_agl_m", "coverage", "building", "uncertainty_m"):
        assert np.array_equal(getattr(again, n), getattr(grid, n), equal_nan=True)
        assert not getattr(again, n).flags.writeable
    # Rebuilding from the same inputs gives the same bytes and the same directory.
    other = uo.write_urban_obstacles(fx.make_grid(), tmp_path / "second")
    assert other.name == published.name
    assert (other / uo.PAYLOAD_FILE).read_bytes() == (published / uo.PAYLOAD_FILE).read_bytes()
    assert (other / uo.METADATA_FILE).read_bytes() == (published / uo.METADATA_FILE).read_bytes()


def test_committed_fixture_matches_generator():
    d = fx.committed_dir()
    assert d.is_dir(), "run: python tests/_urban_fixtures.py"
    assert uo.load_urban_obstacles(d, _expected()).hashes == fx.make_grid().hashes


def test_cache_namespace_is_not_visual(published, tmp_path):
    rel = published.relative_to(tmp_path).parts
    assert rel[0] == "urban_physics" and rel[1] == fx.AOI_NAME and "visual" not in rel


def test_nodata_is_nan_never_a_zero_height_building(grid):
    cov = grid.coverage == 0
    assert cov.any() and np.isnan(grid.top_amsl_m[cov]).all() and np.isnan(grid.height_agl_m[cov]).all()
    assert not grid.building[cov].any()
    assert grid.building_cells == 9 + 16 and grid.coverage_fraction < 1.0
    open_top = grid.top_with_nodata(-np.inf)
    assert np.isneginf(open_top[cov]).all() and np.isfinite(open_top[~cov]).all()
    with pytest.raises(uo.UrbanObstacleError):
        grid.top_with_nodata(float("nan"))
    # surveyed open ground is covered with zero height: distinct from unknown
    assert (grid.height_agl_m[(grid.coverage == 1) & (grid.building == 0)] == 0).all()


def test_origin_rebase_is_exact(grid):
    m = grid.meta
    assert grid.origin_in_env(m.env_origin_utm_m) == m.grid_origin_enu_m
    x, y = grid.origin_in_env((m.env_origin_utm_m[0] + 1000, m.env_origin_utm_m[1] - 250))
    assert (x, y) == (m.grid_origin_enu_m[0] - 1000, m.grid_origin_enu_m[1] + 250)


def test_evidence_has_no_paths_or_secrets(grid):
    ev = json.dumps(grid.evidence())
    assert grid.hashes.content_sha256 in ev and "tmp" not in ev
    assert "token" not in ev.lower() and "key=" not in ev.lower()


# --- refusals ---------------------------------------------------------------------------


def _meta_edit(published, fn):
    """Rewrite metadata in canonical form and re-sign the descriptor, so only semantics differ."""
    doc = json.loads((published / uo.METADATA_FILE).read_bytes())
    fn(doc)
    meta_b = uo.canonical_json(doc)
    desc = json.loads((published / uo.DESCRIPTOR_FILE).read_bytes())
    desc["metadata_sha256"] = uo._sha(meta_b)
    desc["content_sha256"] = uo._content_sha(desc["metadata_sha256"], desc["payload_sha256"])
    new = published.parent / desc["content_sha256"][:16]
    shutil.copytree(published, new)
    (new / uo.METADATA_FILE).write_bytes(meta_b)
    (new / uo.DESCRIPTOR_FILE).write_bytes(uo.canonical_json(desc))
    (new / uo.COMPLETE_FILE).write_text(desc["content_sha256"] + "\n")
    return new


def test_unknown_schema_version_refused(published):
    def bump(doc):
        doc["schema_version"] = 2
    with pytest.raises(uo.UrbanObstacleError, match="schema_version"):
        uo.load_urban_obstacles(_meta_edit(published, bump))


def test_wrong_schema_refused(published):
    def rename(doc):
        doc["schema"] = "other.thing"
    with pytest.raises(uo.UrbanObstacleError, match="schema"):
        uo.load_urban_obstacles(_meta_edit(published, rename))


@pytest.mark.parametrize("edit,match", [
    (lambda d: d.update(nx=0), "dimensions|layout"),
    (lambda d: d.update(vertical_datum="EGM96"), "vertical_datum"),
    (lambda d: d.update(cell_m=0.0), "cell_m"),
    (lambda d: d.update(sources=[]), "sources"),
    (lambda d: d.update(aoi_fingerprint="zz"), "fingerprint"),
    (lambda d: d["build"].update(code_commit="HEAD"), "code_commit"),
    (lambda d: d.update(physics_bounds_wgs84=[-75.0, 40.0, -73.0, 41.0]), "not inside"),
    (lambda d: d["grid_corners_wgs84"].__setitem__(0, [-70.0, 40.7]), "corner"),
    (lambda d: d["sources"][0].update(sha256="short"), "sha256"),
])
def test_malformed_metadata_refused(published, edit, match):
    with pytest.raises(uo.UrbanObstacleError, match=match):
        uo.load_urban_obstacles(_meta_edit(published, edit))


def test_payload_corruption_refused(published):
    p = published / uo.PAYLOAD_FILE
    b = bytearray(p.read_bytes())
    b[100] ^= 0xFF
    p.write_bytes(bytes(b))
    with pytest.raises(uo.UrbanObstacleError, match="payload hash"):
        uo.load_urban_obstacles(published)


def test_metadata_tamper_refused(published):
    p = published / uo.METADATA_FILE
    p.write_bytes(p.read_bytes().replace(b"synthetic fixture", b"edited fixture!!"))
    with pytest.raises(uo.UrbanObstacleError, match="metadata hash"):
        uo.load_urban_obstacles(published)


def test_truncated_payload_refused(published):
    p = published / uo.PAYLOAD_FILE
    p.write_bytes(p.read_bytes()[:-4])
    with pytest.raises(uo.UrbanObstacleError):
        uo.load_urban_obstacles(published)


@pytest.mark.parametrize("victim", [uo.COMPLETE_FILE, uo.PAYLOAD_FILE, uo.DESCRIPTOR_FILE, uo.METADATA_FILE])
def test_partial_artifact_is_unavailable(published, victim):
    (published / victim).unlink()
    with pytest.raises(uo.UrbanObstacleUnavailable):
        uo.load_urban_obstacles(published)


def test_missing_directory_is_unavailable(tmp_path):
    with pytest.raises(uo.UrbanObstacleUnavailable):
        uo.load_urban_obstacles(tmp_path / "nope")


def test_extra_file_refused(published):
    (published / "extra.txt").write_text("x")
    with pytest.raises(uo.UrbanObstacleError, match="unexpected files"):
        uo.load_urban_obstacles(published)


def test_renamed_directory_refused(published):
    moved = published.parent / "0123456789abcdef"
    published.rename(moved)
    with pytest.raises(uo.UrbanObstacleError, match="content id"):
        uo.load_urban_obstacles(moved)


def test_wrong_aoi_directory_refused(published, tmp_path):
    dest = tmp_path / "urban_physics" / "other_aoi"
    dest.mkdir(parents=True)
    moved = dest / published.name
    shutil.copytree(published, moved)
    with pytest.raises(uo.UrbanObstacleError, match="stored under"):
        uo.load_urban_obstacles(moved)


@pytest.mark.parametrize("over,match", [
    (dict(aoi_name="other_aoi"), "aoi"),
    (dict(aoi_fingerprint="0" * 10), "fingerprint"),
    (dict(utm_epsg=32619), "utm_epsg"),
    (dict(env_origin_utm_m=(1.0, 2.0)), "env origin"),
    (dict(aoi_bounds_wgs84=(-74.1, 40.6, -73.9, 40.81)), "bounds"),
    (dict(content_sha256="0" * 64), "content_sha256"),
])
def test_georeference_mismatch_refused(published, over, match):
    with pytest.raises(uo.UrbanObstacleError, match=match):
        uo.load_urban_obstacles(published, _expected(**over))


# --- array-level refusals through the builder -------------------------------------------


def _build(**edit):
    a = fx.make_arrays()
    for k, fn in edit.items():
        fn(a[k])
    return uo.build_urban_obstacle_grid(fx.make_meta(), **a)


@pytest.mark.parametrize("name,edit,match", [
    ("top_amsl_m", lambda a: a.__setitem__((2, 2), np.float32(1.0)), "below ground"),
    ("top_amsl_m", lambda a: a.__setitem__((2, 2), np.float32("nan")), "non-finite"),
    ("ground_amsl_m", lambda a: a.__setitem__((2, 2), np.float32("inf")), "infinite"),
    ("top_amsl_m", lambda a: a.__setitem__((0, 0), np.float32(5.0)), "no-data cells must be NaN"),
    ("building", lambda a: a.__setitem__((0, 0), 1), "without coverage"),
    ("building", lambda a: a.__setitem__((2, 2), 1), "non-positive height"),
    ("coverage", lambda a: a.__setitem__((2, 2), 2), "0/1"),
    ("uncertainty_m", lambda a: a.__setitem__((2, 2), np.float32(-1.0)), "negative"),
    ("top_amsl_m", lambda a: a.__setitem__((4, 5), np.float32(5000.0)), "outside"),
])
def test_bad_arrays_refused(name, edit, match):
    with pytest.raises(uo.UrbanObstacleError, match=match):
        _build(**{name: edit})


def test_wrong_shape_or_dtype_refused():
    a = fx.make_arrays()
    a["coverage"] = a["coverage"][:, :-1]
    with pytest.raises(uo.UrbanObstacleError, match="shape"):
        uo.build_urban_obstacle_grid(fx.make_meta(), **a)
    a = fx.make_arrays()
    with pytest.raises(uo.UrbanObstacleError, match="absent"):
        uo.build_urban_obstacle_grid(fx.make_meta(uncertainty_present=False, uncertainty_kind="none"), **a)


def test_fully_uncovered_refused():
    with pytest.raises(uo.UrbanObstacleError, match="no covered cell"):
        _build(coverage=lambda a: a.fill(0), building=lambda a: a.fill(0),
               ground_amsl_m=lambda a: a.fill(np.nan), top_amsl_m=lambda a: a.fill(np.nan),
               uncertainty_m=lambda a: a.fill(np.nan))


def test_no_uncertainty_variant_round_trips(tmp_path):
    a = fx.make_arrays()
    a.pop("uncertainty_m")
    g = uo.build_urban_obstacle_grid(fx.make_meta(uncertainty_present=False, uncertainty_kind="none"), **a)
    d = uo.write_urban_obstacles(g, tmp_path)
    again = uo.load_urban_obstacles(d)
    assert again.uncertainty_m is None and again.hashes == g.hashes


def test_republish_is_idempotent_and_incomplete_dir_is_replaced(tmp_path, grid):
    d = uo.write_urban_obstacles(grid, tmp_path)
    assert uo.write_urban_obstacles(grid, tmp_path) == d
    (d / uo.COMPLETE_FILE).unlink()
    (d / uo.PAYLOAD_FILE).write_bytes(b"junk")
    assert uo.load_urban_obstacles(uo.write_urban_obstacles(grid, tmp_path)).hashes == grid.hashes
