# Urban weather and mission architecture

Status: **implemented unless a section is explicitly labelled proposed or
unavailable**. This is a physics-grade deterministic simulation with a fast JAX
training model and an independent 6-DoF validation path. It is not flight
certification, a digital twin, or evidence that a controller is ready for a real
aircraft.

## Compatibility and model identities

`baseline-v0` remains the default and retains its historical transition,
observation, checkpoint and action contracts. New behavior is cumulative and
opt-in:

| environment model | implemented behavior |
| --- | --- |
| `baseline-v0` | historical point-mass environment |
| `resource-world-v1/endurance` | binding fuel, reserve and diversion outcomes |
| `resource-world-v1/vehicle-weather` | endurance plus wind-relative motion, density and correlated gusts |
| `resource-world-v1/urban-vehicle-weather` | vehicle-weather plus validated obstacle collision, LOS and mechanical-wake surrogate |

Blue is purely evasive in every model. Its complete action remains bank
command, flight-path-angle command and throttle command. There is no weapon,
target, firing, sensor-mode, threat-control or fourth action.

## Weather and turbulence

**Implemented.** `WeatherConfig` is immutable and fixed shape. It carries
altitude knots; ENU mean wind; density ratio; temperature; sensing attenuation;
background turbulence intensity; three scale lengths; scenario identity;
enable masks and severity scales. Construction validates equal knot counts and
finite physical ranges. Training, evaluation, replay and verification never
fetch live weather.

The actor receives only the local body-axis wind estimate, turbulence
magnitude, density estimate and uncertainty. The full profile, true gust and
exact density are privileged evidence. Scenario components can be staged from
an external source upstream, but the simulator accepts only normalized,
hash-identified configuration.

The current turbulence model is honestly identified as
`bounded-ar1-gust-v1`: a three-component, temporally correlated AR(1) shaping
filter with airspeed- and scale-length-dependent correlation and explicit
clipping. It is an engineering surrogate, not Dryden turbulence and not white
noise. Reset and step noise use named weather stream 106 and stateless counters,
so enabling weather does not consume threat, kill, fuel, sensing or mission
draws. Stream 108 is reserved for missions. A pure NumPy implementation
independently reconstructs profile interpolation and gust recursion.

Obstacle-induced mechanical turbulence is separate from background turbulence.
It uses validated obstacle top/height, incoming wind, downwind distance and
height above the obstacle, with deterministic decay. It is a roughness/wake
surrogate, not CFD. Aircraft wake turbulence is unavailable. The interface can
later be replaced by a hash-pinned CFD field without changing the actor action
contract.

## Fast vehicle physics

**Implemented.** The training transition keeps the fixed three-action interface
and full 3D position, true airspeed, heading, flight-path angle, bank, actuator
lag and load factor. Air-relative velocity is computed from vehicle state;
inertial velocity is:

```text
ground velocity = air-relative velocity + mean wind + gust
```

Ground velocity advances position, while heading and attitude remain
air-relative. Density ratio scales executable longitudinal/climb authority,
equivalent-airspeed stall margin and propulsive demand. Wind changes time and
fuel to reach a gate through the resulting ground track; diversion logic uses
the actor's estimate rather than future weather truth. Neutral weather is
numerically equivalent to the prior resource transition, and `baseline-v0`
does not enter this path.

This remains a fast point-mass/kinematic training model, not 6-DoF. Collision
uses swept substep sampling rather than only the final position, preventing an
aircraft from jumping through a narrow obstacle between decisions.

## Physics urban surface

**Implemented, physics evidence.** The only accepted urban input is Agent 1's
`naigos.urban_obstacles/1` artifact. Loading verifies schema, AOI, UTM frame,
origin, extent, resolution, coverage semantics and descriptor/content hashes.
NumPy arrays become JAX arrays once at environment construction; no file I/O
occurs under `jit`, `vmap` or `scan`.

The artifact preserves bare ground and obstacle top as separate surfaces.
Covered open cells use terrain; covered obstacle cells use the conservative
cell top. Outside the artifact's valid coverage, collision and LOS use terrain
only—unknown coverage is never reinterpreted as zero elevation. A tile offset
inside the theatre AOI is explicit in the artifact frame.

The environment reports obstacle-top clearance and separate raw terrain and
building-collision events. A swept segment is sampled at up to 128 points per
decision interval, terminal aircraft freeze after loss, and one primary outcome
is selected while simultaneous raw events remain recorded.

LOS explicitly selects either `terrain-only` or
`terrain+naigos.urban_obstacles/1`; the checkpoint records that surface and its
digest. Curvature/refraction remains in the terrain LOS implementation. Visual
OpenStreetMap buildings and provider 3D tiles are presentation only and cannot
enter collision, LOS, observations, rewards or verification.

## Missions and terminal outcomes

**Implemented.** `MissionConfig` uses a fixed maximum number of stages and
masks. It supports anonymous procedural transit, ordered multi-leg, time-on-
station, survey-corridor and recovery/diversion missions. State contains stage
volumes, current stage, dwell, deadline, completion/failure/recovery flags,
exclusive outcome and scenario identity. The legacy single destination is
unchanged when missions are disabled.

The actor sees current-stage bearing/range/vertical relation, dwell/deadline
progress and already-permitted recovery information. Future hidden stages are
not leaked. Outcomes distinguish primary completion, safe recovery, survival,
joint mission-and-survival, timeout, shotdown, terrain, building, bounds, fuel
and envelope loss. Timeout is not recovery; diversion is not primary mission
completion; a later loss invalidates joint success. Metrics also report route
length, duration, fuel, gust/authority diagnostics and CBF intervention.

Terminal precedence for the new model is documented in the mission transition
and retained alongside all raw indicators: threat loss, building collision,
terrain collision, bounds loss, fuel loss, primary completion, recovery, then
timeout. Historical baseline precedence is not changed.

## CBF and independent evidence

**Implemented as an attempted backstop, not a guarantee.** The new-model CBF
uses current executable authority, estimated wind/density, actuator lag,
current randomized bounds, sensed threats and the evidence-grade urban surface.
It emits feasibility, correction norm and reason bits:

```text
terrain=1, urban=2, threat=4, bounds=8, authority=16, resource=32
```

A pure NumPy active-set reference is used for held-out comparisons. The old
horizontal feasibility bit is not promoted as proof of vertical or urban
feasibility.

`naigos.rl.physics_verifier` does not call the JAX transition. It reconstructs
weather interpolation, gust state, air/ground velocity, density and fuel
coupling, obstacle sampling and first crossing, mission stage/dwell/deadline,
exclusive outcome, requested/executed control and identity hashes. Altered
weather, mission, collision or provenance evidence is rejected.

## Frozen city scenarios

**Implemented assumptions, not current weather claims.** The immutable scenario
bank contains five NYC Harbor cases (calm, harbor crosswind, building wake,
low-density demand and marginal-fuel headwind) and six Los Angeles Basin cases
(calm, coastal flow, basin crosswind, mountain-wave/rotor surrogate,
marginal-fuel headwind and reduced-authority crosswind). Training and held-out
seeds are disjoint. These cases are notional engineering stress cases; they do
not describe present conditions in either city.

Every scenario records the weather component/config digest, turbulence model,
urban artifact digest, mission-bank digest, theatre component hashes, master
seed, named-stream version, aircraft model, policy identity and resource model.

## Checkpoints, trajectories and replay

**Implemented.** Checkpoint schema 6 records environment model, weather and
turbulence digests, obstacle digest and LOS schema, aircraft physics model,
mission schema/config/bank digests, seed-stream version, CBF contract,
policy/resource identities, code commit and dirty state. Resume fails closed on
a mismatch. The historical shipped checkpoint remains loadable as
`baseline-v0`.

The trajectory extension records before/after mission state; local weather
observation; true mean wind, gust and density as privileged evidence;
air-relative and ground velocity; obstacle clearance; separate building loss;
weather/mission scenario identities through immutable environment metadata;
CBF reason bits; requested/executed action; resource state; and recurrent memory
when present. Old fields retain their old meaning.

## High-fidelity validation boundary

**Interface implemented; nonlinear adapter unavailable/fail-closed.**
`naigos.validation.sixdof` defines canonical, deterministic conversion and ten
controlled maneuver cases: level, bank, climb, descent, throttle, headwind,
crosswind, gust, density variation, recovery and obstacle clearance (combined
where appropriate). A fixture adapter proves the boundary and hashes.

The optional JSBSim boundary requires version `1.2.3` and an immutable aircraft
model manifest with configuration hashes. It refuses to run if the package,
version or hash-pinned model is absent. No external aircraft model is bundled,
and no comparison against a nonlinear 6-DoF model is therefore claimed. The
fixture is explicitly labelled non-6-DoF.

## Verification status and limitations

**Measured locally:** focused weather, mission, urban, CBF, checkpoint,
trajectory, GRU/resource and validator tests pass on CPU; a one-iteration PPO
physics smoke completed locally. These are implementation checks, not learned
performance results. No remote training or Modal job was submitted.

Known limitations are the assumed airframe/fuel coefficients, AR rather than
Dryden turbulence, a mechanical-wake surrogate rather than CFD, raster swept
collision rather than mesh contact, no aircraft wake, no uncertain-map rung,
and no installed hash-pinned nonlinear aircraft model. None of the implemented
simulation or validation paths establishes operational safety or real-aircraft
readiness.
