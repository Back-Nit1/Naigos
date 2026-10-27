# Contract: physics-grade urban obstacles, v1

Schema `naigos.urban_obstacles`, version `1`. Code: `naigos/data/urban_obstacles.py`.
Tests: `tests/test_urban_obstacles.py`. Synthetic fixture: `tests/fixtures/urban_physics_root/`
(built by `tests/_urban_fixtures.py`).

This is the **only** urban geometry the simulation may collide with, line-of-sight against, or
train on. Presentation geometry (OSM extrusions, Google/Cesium tiles) is a different thing in a
different cache namespace and never substitutes for it, in either direction.

## What it is

One immutable raster of structure tops over bare earth, in the theatre's local frame, with a
coverage mask, a building mask, optional uncertainty, full provenance and three hashes. It is
loaded without network access, from a cache directory, by numpy and the standard library only
(no JAX, Flax, RL, viewer or provider import; `tests/test_urban_obstacles.py` asserts that).

## Public API (`naigos.data.urban_obstacles`)

| name | use |
| --- | --- |
| `UrbanObstacleGrid` | the frozen artifact: `.meta`, six read-only arrays, `.hashes` |
| `UrbanObstacleMeta`, `UrbanSource`, `BuildInfo`, `Exclusion`, `ArtifactHashes` | typed, frozen metadata |
| `load_urban_obstacles(path, expected=None)` | strict read + full verification of one artifact dir |
| `ExpectedFrame(aoi_name, aoi_fingerprint, utm_epsg, aoi_bounds_wgs84, env_origin_utm_m=None, content_sha256=None)` | what a theatre requires the artifact to be georeferenced to; **always pass it when a theatre loads** |
| `build_urban_obstacle_grid(meta, *, ground_amsl_m, top_amsl_m, coverage, building, uncertainty_m=None)` | validate + hash in memory (builders) |
| `write_urban_obstacles(grid, cache_root)` | publish payload, metadata, descriptor, marker; reopen and re-verify |
| `artifact_dir(cache_root, aoi, content_sha256)` | `<cache_root>/urban_physics/<aoi>/<content-id>/` |
| `grid.top_with_nodata(fill)` | `top_amsl_m` with unknown cells set to a caller-chosen value (no default) |
| `grid.origin_in_env(env_origin_utm_m)` | exact re-expression of the grid origin in another env's ENU frame |
| `grid.evidence()` | path-free, secret-free summary for `/scene` and exports |
| `UrbanObstacleError`, `UrbanObstacleUnavailable` | refusal / absent-or-incomplete |

### Converting to JAX (Agent 2)

```python
g = load_urban_obstacles(path, ExpectedFrame(...))
top = jnp.asarray(g.top_with_nodata(-jnp.inf))     # explicit choice for unknown cells
cov = jnp.asarray(g.coverage, dtype=bool)          # keep the mask; do not infer it from values
x0, y0 = g.origin_in_env(env_origin_utm_m)         # ENU of cell (0,0) SW corner in YOUR env
cell, nx, ny = g.meta.cell_m, g.meta.nx, g.meta.ny
```

Static shape `(ny, nx)` is fixed per artifact; pad nothing, invent nothing.

## Grid conventions

Same as `naigos.data.enu.ENUGrid`: arrays `(ny, nx)`, **row 0 = south, column 0 = west**, +x east,
+y north, metres. Cell `(r, c)` covers `[x0 + c*cell, x0 + (c+1)*cell) x [y0 + r*cell, y0 + (r+1)*cell)`
where `(x0, y0)` is `grid_origin_enu_m`. Sampling is by containing cell; the surface is a
piecewise-constant height field. Cells outside the grid are **unknown**, not open.

## Arrays (all `(ny, nx)`, little-endian, C order)

| array | dtype | meaning |
| --- | --- | --- |
| `ground_amsl_m` | float32 | bare-earth elevation, metres AMSL (the DEM datum) |
| `top_amsl_m` | float32 | top of the highest structure in the cell, metres AMSL; equals ground where open |
| `height_agl_m` | float32 | `top - ground`, stored; must agree within 1 mm |
| `coverage` | uint8 | 1 = surveyed, 0 = unknown |
| `building` | uint8 | 1 = obstacle; subset of `coverage` |
| `uncertainty_m` | float32 (optional) | height uncertainty, metres, >= 0; meaning in `uncertainty.kind` |

### No-data semantics (normative)

`coverage == 0` cells hold **NaN** in every float array and `building == 0`. They are *unknown*:
not open air, not a zero-height building. `coverage == 1` cells are finite; surveyed open ground
has `building == 0`, `top == ground`, `height == 0`; an obstacle has `building == 1`, `height > 0`.
Any NaN under coverage, any non-finite value anywhere, any number under no-data, `top < ground`,
`height` outside `[0, 1000]` m, a building without coverage, or a covered non-building with height
is refused. A consumer picks its own policy for unknown cells via `top_with_nodata(fill)`.
Protected-zone exclusions are applied by setting those cells to no-data and recording the count.

## On-disk layout

```
<cache_root>/urban_physics/<aoi>/<content-id>/
    payload.bin       arrays concatenated in the order above, no padding
    metadata.json     canonical JSON (sorted keys, no whitespace, ASCII, no NaN)
    descriptor.json   schema + the three hashes + file sizes
    COMPLETE          content_sha256 and a newline; written last
```

`content-id` is the first 16 hex of `content_sha256`. Never under `data_cache/visual/`.
Write order is payload, metadata, descriptor, `COMPLETE`, then reopen and re-verify. A directory
without `COMPLETE`, with a missing file, or with an extra file is refused
(`UrbanObstacleUnavailable` / `UrbanObstacleError`); an incomplete directory is replaced on
republish, a complete one is never overwritten with different content.

## Hashes

* `metadata_sha256` = SHA-256 of the exact bytes of `metadata.json` (which must be canonical).
* `payload_sha256` = SHA-256 of the exact bytes of `payload.bin`.
* `content_sha256` = SHA-256 of `"naigos.urban_obstacles/1\n<metadata_sha256>\n<payload_sha256>\n"`.
  This is the artifact's identity; the `data.urban_obstacles` theatre component cites and hashes it.

## Metadata fields (`metadata.json`)

`schema`, `schema_version` (1 only; others refused), `aoi_name`, `aoi_fingerprint`,
`aoi_bounds_wgs84`, `physics_bounds_wgs84` (the documented urban physics box; must be inside the
AOI), `grid_corners_wgs84` (SW, SE, NE, NW as `[lon, lat]`; must lie inside the physics box),
`utm_epsg`, `env_origin_utm_m` (absolute UTM E, N of the env ENU origin the artifact was built
against), `grid_origin_enu_m` (ENU of the SW corner of cell (0,0), relative to that origin),
`cell_m` (0.5-100), `nx`, `ny` (>= 2, <= 16 M cells), `vertical_datum` (`AMSL`), `axes`, `nodata`
(the normative text above, compared verbatim), `sources[]`, `build{code_commit, dirty,
built_at}`, `uncertainty{present, kind}`, `exclusions[]{name, bbox_wgs84, cells}`, `notes`,
`payload[]` (name, dtype, shape, offset, nbytes: must equal what nx, ny and the dtype table imply).

Each `sources[]` entry: `key`, `name`, `product_id`, `role` (`ground`/`structures`/`footprints`),
`licence`, `licence_url`, `citation`, `url`, `captured`, `published`, `retrieved`, `sha256` (of the
exact input bytes), `redistribution` (`repository` | `cache_only`). **Capture, publication and
retrieval are three dates; a model older than today is never described as current conditions.**
`build.code_commit` is the 40-hex builder commit (or `unknown`) and `build.dirty` is recorded
honestly; `built_at` is an input so a rebuild can be byte-identical.

## Env origin

The env ENU origin depends on the env's grid shape and cell size, so it is not a property of the
AOI alone. The artifact records the origin it was built against (the theatre's reference env,
`env_from_theatre` at 1500 m, centred). A consumer with a different origin calls
`origin_in_env(its_origin)`; this is an exact translation within one UTM zone. Pass
`ExpectedFrame.env_origin_utm_m` only to insist on the reference frame.

## Refusals

Unknown schema or version, malformed or oversize dimensions, non-finite values, `top < ground`,
AOI name/fingerprint/UTM/bounds mismatch against `ExpectedFrame`, physics box outside the AOI,
grid outside the physics box, any hash mismatch, a non-canonical or edited `metadata.json`, a
directory name that is not the content id, an artifact filed under another AOI, a partial or
extra-file directory, an artifact with no cited source. None of these has a lenient mode.

## Theatres

An AOI file with an `urban_physics` block (`{"bounds": [w, s, e, n], "cell_m": 10}`) requires the
artifact: `bounds` is the physics box (inside the AOI), `cell_m` the raster resolution. Such a
theatre **fails closed** if its `data.urban_obstacles` component or the artifact it cites is
missing or corrupt. There is no fallback to OSM, provider tiles or the DEM. AOIs without the block
are unchanged and have no urban obstacles.

## Viewer

`naigos.demo.physics_urban` (presentation side, reads this module's loader only) reports the
artifact as `/scene` `physics_urban`: `state` (`not_declared` | `available` | `unavailable`),
`evidence_grade`, `geometry_source` (`physics_obstacle_artifact` or null), the artifact's
`evidence()` (schema, three hashes, resolution, coverage, sources, build commit) and a reason
that names the fix. No path and no credential appears. `--visual physics` also serves it at
`/physics_urban` (404 in every other mode) as `naigos.physics_urban.page/1`: decimetre int16
ground and top, a flags byte, and four WGS84 corners; a physics export embeds it, a presentation
export embeds only the status. The page draws a flat top per obstacle cell and a wall wherever
the neighbour is lower or unknown, cell for cell, and the HUD keeps it on a separate line from the
two presentation layers. A missing or corrupt artifact is `unavailable`: the HUD says so and draws
nothing in its place.

## Out of scope

Weather, aircraft dynamics, collision response, rewards and observations are consumers of this
artifact and live elsewhere. This contract defines bytes, meaning and refusal, nothing more.
