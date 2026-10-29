# los_angeles_basin

A **notional contested-airspace simulation** over the Los Angeles coastal basin and
its ranges. It is a generic simulation theatre on real terrain and real (dated)
lidar, not a digital twin, and it makes no claim about any real force, facility,
event or current condition. Blue is evasive only, with exactly three actions (bank,
flight-path angle, throttle) and no weapon; red ground and air entities are the
project's generic hazards, drawn at random per seed and re-rolled -- never placed
from real-world data or at named locations. Mission waypoints are anonymous
procedural coordinates. No civilians, crowds, casualties or damage are modelled or
drawn.

## AOI versus urban sub-boxes

| box | extent | what it is for |
| --- | --- | --- |
| **AOI** (`naigos/research/aois/los_angeles_basin.json`) | **-118.62--118.00 E, 33.80--34.30 N**, about 57 x 55 km, UTM 11N / EPSG:32611, fingerprint `89ce98e489` | the whole simulation: DEM, atmosphere, airfields, procedural threats and missions |
| **urban physics box** (`urban_physics`) | -118.280--118.215 E, 34.025--34.075 N, about 6.0 x 5.6 km, 10 m cells | the only place physics-grade urban obstacles exist (Downtown Los Angeles) |
| **visual urban box** (`naigos/demo/cities/los_angeles_basin.json`) | -118.30--118.20 E, 34.00--34.09 N, about 9.2 x 10.0 km | presentation-only OpenStreetMap buildings and roads |

The box is large enough because the simulation needs room, not a city: it holds the
Pacific shore and Santa Monica Bay, the Santa Monica Mountains (to about 1,000 m),
the Palos Verdes rise, the basin floor and downtown, and the San Gabriel front (to
about 1,700 m within the box) -- 1.9 km of relief, 6.4 deg mean slope. That is what the
wind work that consumes this theatre needs: onshore coastal flow, basin routing,
terrain masking from real ridgelines (12% of the AOI is masked from the high point
at 100 m AGL, 5% at 300 m, 0% at 1 km), and the lee of the east-west ranges for
mountain-wave and rotor scenarios. The south edge stops north of the port basins
and the naval installations on the southern coast; the box is not centred on any site.

## Protected zones

Eight zones, each a **box and a reason only**: six civil airport operating areas
(`airfield`, `ambience`, `extraction`, `render_cutout`), one restricted federal property
on the south-western coast (`ambience`, `extraction`, `render_cutout`) and one
restricted research campus in the foothills (`camera`, `ambience`, `extraction`,
`render_cutout`). Airports and the coastal property lose `camera` because at city
scale a 30 km, 110 degree cone around a downtown shot contains them; they are protected
by extraction and the render cut-out instead. Airfields inside the box that are civil
fields and fall in a zone are not start points (4 remain: all outside the zones);
military-named fields would be dropped. Zones are never inferred from screenshots or
from any other image, and no scene payload carries infrastructure semantics.

## Sources, dates, licences, hashes

```bash
uv run naigos-research --aoi los_angeles_basin     # a second run makes no network call
```

Provenance: [`los_angeles_basin/DATA.md`](los_angeles_basin/DATA.md); components in
`components/aoi/los_angeles_basin/`.

| input | source | licence | dates |
| --- | --- | --- | --- |
| terrain (30 m, AOI) and ground (10 m, physics box) | USGS 3DEP bare earth | public domain | 3DEP seamless mosaic |
| structure heights | USGS 3DEP lidar point cloud, EPT collection `CA_LosAngeles_1_B23`, unclassified (1) and bridge-deck (17) returns | public domain | **captured 2023-09-25 .. 2024-01-08**; EPT copy built 2025-11-21 (publisher release date not listed by the National Map) |

**LARIAC: evaluated, not used -- a licensing gate.** LA County's building outlines and
heights (LARIAC) were considered. A redistribution licence for derived data that can be
committed to this repository could not be established offline, so by the fail-closed
rule the source is not allowlisted, not fetched and not in any artifact. The structure
surface comes from public-domain USGS 3DEP lidar alone; no footprint source is needed
because the lidar itself locates the structures. OSM (ODbL) stays presentation-only
and, even with height tags, is never physics evidence.

Measured: the lidar ground agrees with the 3DEP DEM to a median of -0.035 m (a datum or
unit error would fail the build, > 5 m).

## The physics obstacle surface

`data.urban_obstacles` cites and hashes the artifact
([contract](../contracts/urban-obstacles-v1.md)):

* **resolution**: 592 x 546 cells at 10 m, 5.9 x 5.5 km (323,232 cells, about 5.6 MB of
  state), one resolution, so there is no edge-selection rule. A cell is the maximum
  structure top inside it (conservative: a gap narrower than a cell is closed); a 10 m
  grid does not resolve a street canyon.
* **coverage**: 99.98% of cells are surveyed; unknown cells are NaN, never open air and
  never zero-height buildings.
* **what an obstacle is here -- read this**: the 2023 collection has *no building class*
  (only unclassified, ground, noise and bridge deck). The obstacle mask is therefore every
  non-ground surface return at least 3 m above local ground: buildings **and tree canopy**
  and other surface objects. It is an obstacle surface, not a building inventory, which is
  why 261k of 323k cells (81%) are obstacles at a median height of 9.7 m (p90 23.8 m, p99
  79.0 m, tallest cell 310.2 m). Compare New York: 53% obstacle cells, median 20.1 m, 1.6%
  of cells above 100 m against LA's 0.5%. Los Angeles is low, broad and canopied with a few
  spikes, not Manhattan in a warmer palette.
* **artifact**: content `6f792719e131499f...` -- full hashes in
  [`los_angeles_basin/DATA.md`](los_angeles_basin/DATA.md). A copy is committed under
  `tests/fixtures/urban_physics_root/` (5.6 MB).
* **fail-closed**: the theatre cannot load without it; no fallback to OSM, provider tiles or
  the DEM.

## Cameras and look

Checked before the page sees them (above the DEM, clear sightline, inside the safe region,
out of protected zones; the street canyon also sits above the obstacle surface):

* `urban_overview` (opening): from 1,282 m over the Hollywood side, looking south-east
  across Downtown and the lower grid.
* `valley_overview`: from 1,582 m over the eastern basin floor, looking west over the
  basin with Downtown and the Santa Monica Mountains beyond.
* `street_canyon`: 78 m up, looking south-east along a downtown grid axis. A
  **presentation camera only**; the fixed-wing model is not certified for street-level flight.
* no `coastal_corridor`: the shore inside the AOI is flanked by two civil airport zones, so
  an honest coastal shot would frame them. The coast is in the physics terrain, not a preset.

Atmosphere `warm_coastal_basin` (warm early-afternoon light from the west-south-west, a pale
warm haze); palette `los_angeles_basin`: concrete, pale stone, steel, blue-green glass and warm
highlights, clear of the red and green overlay colours.

## Commands

Run, record and replay it with the [shared commands](README.md#commands-for-any-theatre-aoi), `<aoi>` = `los_angeles_basin`.

## Known limitations

* Tree canopy and other surface objects are obstacles here (no building class exists in the
  source); the mask is not a building inventory.
* Lidar is a 2023-24 capture; later construction is absent.
* 10 m cells: no street canyons; thin spires and masts are lower bounds.
* LARIAC is unused (licence gate above); heights come from lidar only.
* Terrain masking is real but modest in the basin; the ranges carry it.
* The provider tile path was not exercised (no credential in this environment).
