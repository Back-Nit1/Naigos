# nyc_harbor

A **notional contested-airspace simulation** over New York Harbor and its shores.
It is a generic simulation theatre on real terrain and real (dated) lidar, not a
digital twin, and it makes no claim about any real force, facility, event or
current condition. Blue is evasive only, with exactly three actions (bank,
flight-path angle, throttle) and no weapon; red ground and air entities are the
project's generic hazards, drawn at random per seed and re-rolled -- never placed
from real-world data or at named locations. Mission waypoints are anonymous
procedural coordinates. No civilians, crowds, casualties or damage are modelled
or drawn.

## AOI versus urban sub-boxes

Three boxes, kept apart on purpose:

| box | extent | what it is for |
| --- | --- | --- |
| **AOI** (`naigos/research/aois/nyc_harbor.json`) | **-74.40--73.85 E, 40.48--40.92 N**, about 46 x 49 km, UTM 18N / EPSG:32618, fingerprint `997eeb1b1c` | the whole simulation: DEM, atmosphere, airfields, procedural threats and missions |
| **urban physics box** (`urban_physics`) | -74.025--73.965 E, 40.695--40.740 N, about 5.1 x 5.0 km, 10 m cells | the only place physics-grade urban obstacles exist (Lower Manhattan) |
| **visual urban box** (`naigos/demo/cities/nyc_harbor.json`) | -74.05--73.95 E, 40.68--40.76 N, about 8.4 x 8.9 km | presentation-only OpenStreetMap buildings and roads |

The AOI is centred on the harbor, not on any site: Raritan and Upper Bay water, the
Hudson and East River channels, the Palisades and Watchung rise to the west, and
four dense boroughs plus the New Jersey waterfront. That is enough lateral extent
(46 x 49 km) for multi-leg missions, recovery gates and routing around water,
shore and ridge, and enough room for procedurally generated threats not to sit
inside each other's envelopes. Relief is low (278 m, 1.8 deg mean slope; the
radar horizon, not ridge masking, carries the geometry), so the theatre is a
test of coastal geometry and of the urban obstacle layer rather than of terrain
masking: 53% of the AOI is masked from the high point at 100 m AGL and 0% at
300 m and above.

## Protected zones

Ten zones, each a **box and a reason only** (no attribute, function or
operational detail). The east edge (-73.85) stops short of the large
international airport; the airfields inside the AOI that are civil airports are
removed from start geometry (`airfield`), and three of them are also kept out of
extraction and rendering. Military-named airfields would be dropped
(`exclude_military_airfields`); none are in the box.

| zone | policies |
| --- | --- |
| three civil airport operating areas | `airfield`, `ambience`, `extraction`, `render_cutout` |
| three civil heliports (waterfront) | `airfield`, `ambience` (a camera zone here would forbid every Lower Manhattan shot: the heliport sits inside the skyline) |
| two container-port and rail areas (west, south-west) | `camera`, `ambience`, `extraction` |
| port and industrial waterfront (Brooklyn) | `ambience` (it borders both urban boxes; cutting it would leave a hole in the harbor-facing context) |
| restricted federal property (the Narrows) | `camera`, `ambience`, `extraction`, `render_cutout` |

The airports lose `camera`: at city scale a 30 km, 110 degree view cone around any
Lower Manhattan shot contains them, so they are protected by extraction and the
render cut-out instead. Protected zones are never inferred from screenshots or
from any other image. Scene payloads carry no infrastructure semantics.

## Sources, dates, licences, hashes

Everything crosses the allowlist and the cache, with a SHA-256 and a citation:

```bash
uv run naigos-research --aoi nyc_harbor     # a second run makes no network call
```

Provenance: [`nyc_harbor/DATA.md`](nyc_harbor/DATA.md); components in
`components/aoi/nyc_harbor/`.

| input | source | licence | dates |
| --- | --- | --- | --- |
| terrain (30 m, AOI) | USGS 3DEP bare earth | public domain | 3DEP seamless mosaic |
| ground (10 m, physics box) | USGS 3DEP bare earth | public domain | 3DEP seamless mosaic |
| structure heights | USGS 3DEP lidar point cloud, EPT collection `NY_NewYorkCity`, building (6) and bridge-deck (17) returns | public domain | **captured 2013-08-05 .. 2014-04-21** (post-Sandy survey), published 2015-04-14 |
| footprints and roof heights | NYC Building Footprints, NYC Open Data `5zhs-2jue` | NYC Open Data Terms of Use | features last edited 2006-09-25 .. 2026-10-15; dataset updated 2026-10-25 |

**Staleness, said plainly.** The lidar is a 2013-14 capture; the structure surface is
not "current conditions" and says so (`sources[].captured`). The artifact build date is
a third, separate date (`build.built_at`), and the footprint dataset's own update date
a fourth. Towers completed after the capture appear only through the footprint
heights (cells with a footprint but no lidar structure return are taken from the
footprint's roof height at higher stated uncertainty). The official NYC 3D Building
Model is not used directly; the footprint dataset is derived from it.

Measured: the 10 m lidar ground agrees with the 3DEP DEM to a median of -0.03 m
(a datum or unit error would fail the build, > 5 m). Where a footprint and a lidar
structure overlap (71,987 cells) their heights differ by a median -0.63 m and a median
absolute 2.83 m.

## The physics obstacle surface

`data.urban_obstacles` cites and hashes the artifact
([contract](../contracts/urban-obstacles-v1.md)):

* **resolution**: 500 x 493 cells at 10 m, 5.0 x 4.9 km. A 10 m cell resolves city
  blocks and avenue-width gaps only marginally; it does not resolve a street canyon,
  and a cell is the maximum structure top inside it, which is conservative (a gap
  narrower than a cell is closed). 246,500 cells are about 4.4 MB of state,
  bounded, and fine enough for collision and line of sight at the speeds this
  airframe flies. One resolution only, so there is no edge-selection rule to get wrong.
* **coverage**: 99.99% of cells are surveyed; cells with no lidar return within 30 m are
  unknown (NaN), never open air and never zero-height buildings. Protected-zone
  `extraction` cells would be unknown too; none of that kind falls in this box.
* **obstacles**: 131,579 cells; heights p50 20.1 m, p90 55.6 m, p99 155.9 m, tallest
  cell 547.9 m. 17 isolated cells that stood more than 60 m and three times above all
  eight neighbours were repaired as classification noise (an aircraft or a bird
  filed as a building). Thin spires and masts are lower bounds.
* **artifact**: content `c68d8582bf27f84f...` -- the full hashes are in
  [`nyc_harbor/DATA.md`](nyc_harbor/DATA.md) and in the component. A copy is committed
  under `tests/fixtures/urban_physics_root/` (4.3 MB) so tests run on a fresh clone.
* **fail-closed**: the theatre cannot load without it
  (`naigos.data.theatre.load_urban_obstacles_for`); there is no fallback to OSM, provider
  tiles or the DEM.

## Cameras and look

All three are checked before the page sees them (above the DEM, sightline clear,
inside the safe region -- which here includes the Upper Bay, where the harbor shot
stands -- and out of protected zones), and the street canyon is also checked against
the obstacle surface (it must sit above the structure under it).

* `urban_overview` (opening): from 941 m above the Upper Bay, looking north-north-east
  across the harbor at the Lower Manhattan skyline.
* `coastal_corridor`: from 606 m over the Battery, north along the Hudson waterfront.
* `street_canyon`: 89 m up and 700 m south of the Financial District, looking up the
  street grid. A **presentation camera only**; the fixed-wing model is not certified
  for street-level flight.

Atmosphere `cool_atlantic_coast` (clear mid-morning light from the south-east, cool
blue-grey haze) and `cool_atlantic_dusk` (low sun from the west over the Hudson, rose-grey
light); both presentation-only profiles. Select one with `--atmosphere`. Palette
`nyc_harbor`: stone, brick, steel and cool glass over a darker, desaturated ground, with
no swatch as bright and saturated as the red threat or green detection overlays.

## Commands

Run, record and replay it with the [shared commands](README.md#commands-for-any-theatre-aoi), `<aoi>` = `nyc_harbor`.

## Known limitations

* Lidar is 2013-14 (12-13 years old at build); recent construction is footprint-only.
* The official NYC 3D Building Model is not used; footprints and roof heights stand in.
* 10 m cells: no street canyons, no bridge cables, thin spires are lower bounds.
* Water cells inside the lidar collection with no returns are surveyed open water; a
  data void over land is unknown. A bridge deck (class 17) is an obstacle; its piers over
  water are only as good as the provider's classification.
* The harbor camera stands over water. The shared standard asks safe regions to be land;
  here the region includes the bay on purpose, bounded by the Narrows zone and the AOI.
* Low relief: terrain masking is a weak mechanic here.
* The provider tile path was not exercised (no credential in this environment).
