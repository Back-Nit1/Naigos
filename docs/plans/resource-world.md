# Resource world: a model ladder for constrained autonomous routing

Status: **L1 endurance and the vehicle-weather/urban-mission increments are
implemented; sensing, communications and uncertain-map increments remain
proposed.** Historical audit text below is retained as the pre-implementation
baseline and is superseded where the implementation amendment says so.

## Implementation amendment (2026-10-27)

The following cumulative identities now exist without changing the default:

- `baseline-v0`: historical behavior and shipped checkpoint contract;
- `resource-world-v1/endurance`: binding fuel, reserve and diversion;
- `resource-world-v1/vehicle-weather`: wind-relative motion, density-scaled
  authority/burn and deterministic bounded AR(1) gusts;
- `resource-world-v1/urban-vehicle-weather`: validated raster obstacle collision
  and LOS, mechanical-wake surrogate and fixed-shape anonymous missions.

The older “does not exist” entries in the audit describe the state from which
the ladder was designed. Current details and limitations are in
[urban-weather-missions](../urban-weather-missions.md). In particular, sensing
and communications are not silently claimed by the later weather identity;
they remain independent proposed rungs.

The objective is to turn Naigos from a route-through-threats task into a
resource-management task without weakening its existing evidence. The current
controller remains the immutable first rung of the ladder. Every later rung
keeps blue purely evasive and keeps the complete blue action space exactly:

```text
(bank command, flight-path-angle command, throttle command)
```

There is no weapon, target-selection, sensor-mode, communications, abort, or
mission-mode action. A diversion is expressed by flying to a recovery gate with
the same three controls. Sensor, communications, actuator, compute, weather,
and map effects change the world or the information available to the policy;
they do not add action axes.

## Evidence labels used here

- **Existing** means the behavior is in the current repository and is covered
  by code or tests.
- **Measured** means a committed artifact or documented experiment reports a
  number. It does not mean the result generalizes beyond its stated scenario.
- **Proposed** means an unimplemented design choice. Numerical ranges in this
  document are initial experiment ranges, not calibrated facts.
- **Assumption** means the repository has no empirical basis for the value yet.

This distinction is especially important for weather, fuel, tracking, and
failure rates. The repository has useful calibration inputs, but it does not
currently have a defensible empirical distribution for resource failures.

## Audit of the current environment

| area | what exists now | what does not exist or is misleading | evidence |
| --- | --- | --- | --- |
| Fuel and endurance | **Existing:** `AircraftState.fuel`; deterministic burn from base burn, normalized speed, and excess load factor; exact fuel fraction in ego observation; per-step fuel shaping; `out_of_fuel` reward term; fuel-loss fraction in the CBF sweep. | `ASSUMED_FUEL_KG=3000` and the burn law are not calibrated. Fuel does not bind in normal sorties. Running dry does **not** remove an aircraft from `alive`; it only reduces speed toward the stall floor and raises `out_of_fuel` on every later live step. The independent verifier does not check fuel. There is no reserve, endurance estimate, alternate destination, or diversion outcome. | `env/airframe.py`, `env/flight_env.py`, `env/theatre_bridge.py`; `test_airframe.py`, `test_reward_shaping.py`, `test_cbf_sweep.py`; next-steps E-5. |
| Sensor visibility and tracking | **Existing:** terrain/refracted LOS, range/aspect/altitude-dependent detection probability, and a deterministic red track-quality accumulator with gain and decay. The actor receives exact threat-relative state, exact `pd`, and exact red lock quality for threats admitted by the visibility/track-memory mask. Red pursuit only follows a contact with existing track. | There is no sampled blue measurement process, false negative, false positive, measurement noise, blue track estimate, covariance/confidence, track age, dropout state, or stale propagation. `pd` is integrated as a continuous expected value rather than sampled into detections. The observation's memory channel uses red's track on blue as a proxy for blue knowing the emitter. | `env/detection.py`, `env/obs.py`, `env/threats.py`; `test_detection_physics.py`, `test_env_contract.py`, `test_interceptor_3d.py`. |
| Communications | **Existing:** a fixed-size nearest-friendly set. | Friendly position, velocity, and alive status are instantaneous ground truth. There are no packets, link states, loss, delay, age, queue, range, or bandwidth limits. | `obs.friend_features` and `obs.build`; `test_env_contract.py`, `test_networks_ctde.py`. |
| Weather | **Existing:** the research layer records a three-day, one-site vertical atmosphere profile. Measured refractivity directly sets effective Earth radius. Surface density influenced the research-time detection component. The theatre object loads density profile data. | Wind does not enter motion or fuel. The derived component retains wind speed summaries but drops wind direction. `airframe_from_theatre` does not apply the loaded density profile to stall, climb, thrust, or burn. Weather does not alter sensing dynamically. The presentation atmosphere is intentionally visual only. The statement in next-steps D-4 that air density is dynamically used is broader than the current code supports. | `research/sources/atmosphere.py`, `data/theatre.py`, `env/theatre_bridge.py`; `test_atmosphere.py`; next-steps D-4. |
| Airframe degradation | **Existing:** nominal first-order bank, flight-path-angle, and speed lags; static g, climb, speed, floor, and ceiling limits. | There is no health state, wear, partial failure, authority loss, changing lag, biased actuator, or recovery. A threat outcome is terminal; there is no partial combat-damage model, and none is proposed here. | `env/config.py`, `env/airframe.py`; `test_airframe.py`. |
| Compute and decision latency | **Existing:** a fixed 2 s decision interval and deterministic actuator lag. The live viewer reports achieved simulation speed as a presentation metric. | Every policy output is available and applied in the same environment step. There is no simulated inference time, deadline, held action, action queue, compute budget, or missed update. Wall-clock viewer throughput does not affect simulation state and must remain separate from simulated decision latency. | `EnvConfig.dt`, `flight_env.step`; `test_achieved_rate.py`. |
| Map uncertainty | **Existing:** physics uses one exact DEM; the actor sees exact current AGL and optionally exact distances to the play-area edges. It does not receive a terrain map or terrain lookahead. Map bounds and route geometry can be randomized. | There is no uncertain onboard map, position error, terrain-height error, confidence, missing tile, or map-belief update. The CBF terrain barrier reads the exact simulation DEM, which is acceptable for the current baseline but would be privileged in a map-uncertainty experiment. | `env/terrain.py`, `env/obs.py`, `env/flight_env.py`; `test_map_randomization.py`, `test_edge_obs.py`, `test_route_generalization.py`. |
| CBF feasibility | **Existing:** an optional observation-limited HOCBF filter for known lethal envelopes, terrain, and map boundaries. It returns and logs a feasibility bit. There is a paired, provenance-carrying sweep over margin, threat count, and red level. **Measured:** the committed trained-policy comparison reports 0.227 horizontal-QP infeasibility, zero terrain losses, and lower objective completion with the filter. | The feasibility bit covers the horizontal envelope/boundary projection only; the terrain command clamp is not included in that test. Boundary constraints use the full DEM extent, not `EnvState.bounds`, so they do not represent a randomized inner play area. The acceleration box is nominal and does not know actuator degradation, delay, wind, or fuel state. Terrain uses exact truth and threat constraints use exact reconstructed positions from current observations, not uncertain tracks. The cyclic projection is not an exact QP solver. | `rl/cbf.py`, `rl/cbf_sweep.py`; `test_cbf.py`, `test_cbf_sweep.py`, `test_theatre_bridge.py`; `docs/cbf-sweep.md`; next-steps S-1/S-3/S-4. |
| Diversion and abort | **Existing:** one primary objective, arrival, terminal losses, timeout, and continuous live-viewer respawn. | There is no alternate recovery point, mission abort state, reserve rule, safe-diversion outcome, or reward ordering between completing, recovering, and losing the aircraft. | `flight_env.reset`, `flight_env.step`, `flight_env.respawn`; `test_persistent_rollouts.py`. |

Two cross-cutting conclusions follow from the audit:

1. Fuel is the cheapest real resource to make consequential because its state,
   observation, burn, reward term, metrics, and checkpoint serialization already
   exist. Its terminal and verification semantics are the missing pieces.
2. Observation uncertainty cannot be implemented by adding noise to the current
   threat feature array. The current array mixes ground truth, red's belief, and
   blue's display logic. A separate blue track state is required.

## Non-negotiable design rules

1. `baseline-v0` is the current environment, observation, actor, seed stream,
   and checkpoint contract. Its default behavior and golden rollouts do not
   change.
2. New resource dynamics live in a versioned environment model, initially
   `resource-world-v1`. Results always name the environment model and resource
   rung. “Naigos” alone is not an adequate configuration label.
3. The three flight-control actions never change. Resource management emerges
   from route, speed, altitude, maneuver timing, action stability, and physical
   choice of destination.
4. Exogenous bad luck is not directly punished. A random sensor dropout,
   communications outage, gust, or actuator fault changes the transition or
   observation; reward evaluates the resulting routing outcome. This prevents
   the policy from being penalized for an event it could not control.
5. Physics and terminal outcomes use privileged truth. The actor receives only
   the declared observation. The critic may receive privileged state during
   training, but evaluation always runs the actor alone and records the split.
6. All capacities are static at construction. State updates use `lax.scan`,
   `vmap`, masks, ring-buffer indices, and `jnp.where`; no resource creates a
   data-dependent shape or Python branch over traced values.
7. Randomness is deterministic and resource-local. Adding a later resource may
   not shift the random sequence of an earlier one.
8. The independent verifier checks logged traces and does not import JAX env
   transition functions. Distribution checks and per-trace transition checks
   are separate; one trajectory cannot prove that a random generator has the
   intended distribution.
9. Training and held-out scenario banks are immutable, digest-checked, and seed
   disjoint, following `route-bank-v1`. A held-out seed from the same generator
   is reported as such; a held-out combination is reported separately.

## The model ladder

Each rung is cumulative, but every experiment also keeps single-resource
ablations so a compound failure does not hide which mechanism mattered.

| rung | environment model | enabled resources | purpose |
| --- | --- | --- | --- |
| L0 | `baseline-v0` | Current environment exactly as shipped. | Preserve the working controller, shipped checkpoint, published numbers, and all current regression tests. |
| L1 | `resource-world-v1/endurance` | Binding fuel/endurance, reserve margins, two recovery gates, primary-versus-divert outcomes. | Minimum viable genuine resource-routing problem. |
| L2 | `resource-world-v1/sensing` (**proposed**) | L1 plus noisy blue measurements, confidence, dropout, track aging, and stale-track propagation. | Make information quality a time-dependent resource without changing red truth. |
| L3 | `resource-world-v1/coordination` (**proposed**) | L2 plus lossy/delayed friendly packets and simulated decision deadlines. | Test robust decentralized coordination and control persistence. |
| L4 | `resource-world-v1/vehicle-weather` (**implemented independently of proposed L2/L3**) | L1 plus wind/gusts, density-scaled authority/burn and local weather observations. | Couple route, energy, controllability, and environment. |
| L4U | `resource-world-v1/urban-vehicle-weather` (**implemented**) | L4 plus evidence-grade urban collision/LOS, wake surrogate and opt-in missions. | Test mission and recovery under weather and urban constraints. |
| L5 | `resource-world-v1/uncertain-map-cbf` (**proposed**) | L4U plus onboard terrain uncertainty. | Measure safety-backstop value when its information and authority are not oracles. |

Do not train only L5 and compare it with L0. The ladder is the experiment:
every rung gets the same policy classes, route splits, scenario accounting, and
resource-free control arm.

## Fixed-shape JAX contract

**Amendment (wave 2, L1 MVP).** `baseline-v0` does *not* keep the literal
`EnvState`/`RewardTerms` treedefs: both are runtime-only NamedTuples and gained
one trailing neutral field each (`EnvState.resource`, `RewardTerms.diverted`),
so one jitted code path serves L0 and L1 without a duplicated state class.
What baseline-v0 preserves is everything persisted or consumed by a trained
actor: the `Learner` checkpoint pytree (no env state is ever serialized), the
checkpoint keys the demo reads, `Observation` and the 10/14-wide ego vector,
all RNG streams, and the transition numerics (baseline never reads the resource
latents). `tests/test_baseline_contract.py` pins the field lists, round-trips a
baseline checkpoint, runs the shipped `checkpoints/theatre_1000` on the amended
runtime, and perturbs every resource latent to show the baseline step ignores it.
Forward compatibility (new baseline checkpoints on old code) is not claimed:
`env_cfg` now carries a `resource` entry.

`baseline-v0` retains the current `Observation`, actor and checkpoint layout. The first
resource rung introduces a new fixed schema and allocates the complete v1
capacity immediately, even while later resources are disabled. This avoids a
new actor/checkpoint shape at every rung.

Proposed compile-time capacities:

```text
D = 2       diversion/recovery gates per aircraft
K_t = 6     threat tracks exposed to the actor (same current top-K capacity)
K_f = 3     friendly tracks exposed to the actor
Q_c = 9     communications slots: current plus eight delayed steps
Q_a = 5     requested-action slots: current plus four delayed steps
Z_w = 6     weather-profile altitude knots
M = 8       map-error control grid per axis
```

The ground-truth threat capacity remains `EnvConfig.n_threat`; it is static for
one compiled environment. Proposed resource state, conceptually:

```text
ResourceState
  blue_tracks
    pos_hat            (B, T, 3)
    vel_hat            (B, T, 3)
    confidence         (B, T)
    age_s              (B, T)
    valid              (B, T) bool
  communications
    packet_ring        (B_receiver, B_sender, Q_c, 7)
    packet_valid       (B_receiver, B_sender, Q_c) bool
    write_index        () int32
    link_up             (B_receiver, B_sender) bool
    outage_age_s        (B_receiver, B_sender)
  actuators
    authority_true     (B, 3)
    lag_scale_true     (B, 3)
    authority_est      (B, 3)
  decisions
    action_ring        (B, Q_a, 3)
    action_valid       (B, Q_a) bool
    executed_action    (B, 3)
    action_age_s       (B,)
    tokens             (B,)
    deadline_misses    (B,)
  weather
    altitude_knots_m   (Z_w,)
    wind_enu_ms        (Z_w, 3)
    density_ratio      (Z_w,)
    sensing_factor     (Z_w,)
    gust_enu_ms        (B, 3)
  map
    error_control_m    (M, M)
    position_bias_m    (B, 2)
  mission
    diversion_pos      (B, D, 3)
    diversion_active   (B, D) bool
    reserve_fuel_kg    (B,)
    outcome            (B,) int32
  safety
    feasible           (B,) bool
    correction_norm    (B,)
    active_constraints (B,) int32
```

Fuel remains in `AircraftState`; duplicating it would create two sources of
truth. Every later-rung field exists at L1 with a neutral value and an enable
mask. Severity and enable values should be traced `ResourceScales` in state,
like `RedScales`, so curriculum changes do not require new array shapes.

The v1 actor receives a fixed `ResourceObservation`:

```text
legacy_ego       (B, 14)       current ten ego features plus four edge features
threat_tracks    (B, K_t, 19)  estimated rel pos/vel, range, envelope,
                                confidence, age, fresh flag, 8-wide kind code
threat_mask      (B, K_t)
friendly_tracks  (B, K_f, 11)  delayed rel pos/vel, range, alive, age,
                                link quality, validity
friendly_mask    (B, K_f)
resources        (B, 20)       fuel/reserve margins, estimated authority,
                                last executed action, action age/tokens/miss,
                                local estimated wind, turbulence/map/CBF status
diversions       (B, D, 8)     relative position, range, bearing sin/cos,
                                estimated fuel margin, active bit
diversion_mask   (B, D)
```

The exact 20 resource scalars must be named in code and versioned; no anonymous
“misc features” tail. An encoder is added for resources and diversions, while
threat and friendly sets remain permutation invariant. Recurrent memory is not
required for v1 because age, confidence, executed-action age, and queues make
the relevant persistence explicit; an RNN is a later A/B experiment, not a
silent architectural confound.

### Deterministic random streams

At reset, derive resource keys with stable integer IDs rather than taking one
long positional split:

```text
fuel       = fold_in(episode_key, 101)
sensor     = fold_in(episode_key, 102)
comms      = fold_in(episode_key, 103)
actuator   = fold_in(episode_key, 104)
compute    = fold_in(episode_key, 105)
weather    = fold_in(episode_key, 106)
map        = fold_in(episode_key, 107)
```

L1 consumes only the `fuel` stream; its budget, reserve, and burn-multiplier
latents are `fold_in(fuel, 1..3)`, so IDs 102-107 stay free
(`env.resource.RNG_STREAM_IDS`, `tests/test_resource_rng.py`).

Terminal outcomes are exclusive. Simultaneous events within one step resolve as
shot > terrain > bounds > fuel > primary > diversion > timeout
(`env.resource.OUTCOME_PRECEDENCE`); the env refuses arrival/diversion on a loss
step. Raw per-event flags stay in `RewardTerms`, while metrics, the verifier and
replay count the exclusive outcome and report `raw_*` rates beside it.

Per-step draws are `fold_in(resource_key, t)`. Enabling communications must not
change the fuel, weather, threat, or kill draws for the same master scenario.
The scenario record stores master seed, resource-world version, config digest,
and realized episode-level latent parameters.

## Resource definitions

All ranges below are **proposed assumptions** for initial robustness experiments.
They must be replaced or narrowed when provenance or sensitivity results exist.

### 1. Fuel and propulsive-energy budget

| requirement | proposed definition |
| --- | --- |
| State | Existing `fuel_kg`; cumulative burn; reserve fuel; derived specific mechanical energy `g*z + V^2/2`; estimated fuel-to-primary and fuel-to-each-diversion margins. Mechanical energy is derived, not a second consumable. |
| Actor observes | Exact own fuel fraction; estimated endurance; estimated fuel margin to the primary; best diversion margin; reserve fraction; current speed and altitude already present. The actor does not see future wind or future threat-driven detour cost. |
| Privileged | True burn coefficients and episode multiplier; exact future weather field; true minimum-fuel path estimates used only for diagnostics; remaining fuel of other aircraft unless received over communications. |
| Transition | Keep the existing base/speed/load burn structure for the first rung. Apply an episode burn multiplier. Fuel exhaustion becomes one terminal recovery failure because the point-mass model does not implement a defensible glide-to-landing model. Do not repeatedly emit `out_of_fuel`. |
| Reward versus hard constraint | Fuel cannot go below zero. First dry crossing is a terminal airframe loss in the CMDP channel. Fuel used remains weak efficiency shaping. Reserve breach is an observable warning and diagnostic, not a hard constraint. Primary completion reward is greater than safe diversion reward, which is greater than timeout and far greater than loss. |
| Training randomization | Define budget by geometry, not a fixed 3000 kg: initial still-air range budget / direct-route range in `[1.25, 2.0]`; reserve / direct-route requirement in `[0.10, 0.20]`; burn multiplier `[0.85, 1.15]`; maneuver-burn multiplier `[0.75, 1.25]`. |
| Held-out failures | Budget ratio `[1.05, 1.20]`; high burn multiplier with a threat detour; later, low reserve plus headwind; throttle degradation plus marginal fuel. Keep single-axis extrapolation separate from unseen combinations. |
| Verification | Pure NumPy integrates logged burn from prior fuel, executed speed/load, `dt`, and logged episode coefficients; checks monotonicity, first-crossing uniqueness, terminal outcome, and reward/CMDP agreement. A population test checks sampled ranges. A tampered fuel sample must fail verification. |

### 2. Sensor noise, confidence, dropout, and stale threat tracks

| requirement | proposed definition |
| --- | --- |
| State | One blue track slot per padded ground-truth threat: estimated position/velocity, confidence, age, last-update flag, and valid mask. Internal truth-to-slot association is simulator bookkeeping; the actor receives no global threat ID. Red's existing `lock` remains separate. |
| Actor observes | Only the top `K_t` estimated tracks, including confidence, age, and whether the track was refreshed this step. Geometry is the estimate, not truth. |
| Privileged | True threat state, true LOS, noiseless `pd`, measurement availability, dropout regime, injected error, and red's track on blue. The critic may use these during training if the run declares CTDE privileged fields. |
| Transition | Sample measurement availability from visibility, signal strength, and a two-state dropout process. On update, add seeded position/velocity noise and raise confidence. Otherwise propagate constant velocity, age the track, decay confidence, and invalidate after a fixed stale horizon. False positives are deferred beyond v1 sensing. |
| Reward versus hard constraint | No penalty for noise or dropout. Exposure and engagement rewards continue to use ground truth so corrupt sensors cannot make danger disappear from the objective. Track invalidation is a hard observation rule, not a physical terminal. |
| Training randomization | Position sigma `50-500 m`; velocity sigma `2-20 m/s`; independent dropout `0-0.20`; burst entry probability `0-0.02/step`; burst duration `2-10` steps; confidence half-life `6-30 s`; stale expiry `10-60 s`. |
| Held-out failures | `0.35` independent dropout; `20-40 s` bursts; low-confidence reacquisition near a lethal envelope; sensor dropout combined with communications outage. |
| Verification | NumPy verifier checks propagation, age, confidence bounds, update/invalidation rules, mask zeroing, and that observations equal track state rather than truth. Statistical tests check noise/dropout moments over many fixed seeds. Tampering with age or using true threat position in a stale observation must fail. |

### 3. Communications loss and delayed friendly state

| requirement | proposed definition |
| --- | --- |
| State | Fixed per-link packet ring, validity bits, write index, link-up Markov state, outage age, and last delivered packet. Self-links are always masked. |
| Actor observes | At most `K_f` friend packets: delayed estimated position/velocity, reported alive bit, packet age, link-quality estimate, and validity. “No packet” is explicit, never encoded as a zero-position friend. |
| Privileged | Current true friendly state, queue contents not yet delivered, actual link state, scheduled delay, and dropped packets. |
| Transition | Each sender emits a fixed-width packet. Per-link delay selects a ring slot; loss suppresses delivery. Burst outages use a two-state chain. The receiver holds the last packet and ages it until expiry. |
| Reward versus hard constraint | No reward penalty for link loss. Coordination quality is measured through mission outcomes, separation if later introduced, and duplicated/divergent routing diagnostics. Packet capacity, delivery time, and expiry are hard simulator rules. |
| Training randomization | Delay `0-3` steps (`0-6 s`); independent loss `0-0.15`; outage entry `0-0.01/step`; outage duration `2-10` steps; expiry `12-40 s`. The queue capacity supports eight-step held-out delays without recompilation. |
| Held-out failures | Delay `4-8` steps; `20-40 s` outage; asymmetric links; communications loss combined with one aircraft's sensor dropout or actuator fault. |
| Verification | Trace verifier checks causal delivery: no packet may arrive before send time, after a drop, or with a future state. It recomputes age and expiry from logged send/delivery records. Tests poison queued and masked slots to prove the actor cannot read them. |

### 4. Actuator degradation and reduced control authority

| requirement | proposed definition |
| --- | --- |
| State | True and estimated authority for bank, flight-path angle, and throttle; lag multipliers; optional command bias; fault-mode state and onset time. This is generic component degradation, not damage from a weapon. |
| Actor observes | Noisy authority estimates and the last executed action. The difference between requested and executed response is learnable over time. Exact fault mode and future onset remain hidden. |
| Privileged | True authority, lag, bias, onset/recovery schedule, and nominal command before degradation. |
| Transition | Degradation scales command range and increases the existing first-order time constants before airframe integration. Episode-start degradation is the first implementation; one persistent, seeded in-flight onset comes later. No state-dependent structural branch. |
| Reward versus hard constraint | Reduced authority is a hard physical limit. Do not penalize the occurrence of a fault. Terrain, bounds, fuel, and objective outcomes retain their existing meanings. |
| Training randomization | Per-axis authority `[0.70, 1.00]`; lag multiplier `[1.0, 1.5]`; episode fault probability `0-0.15`; at most one affected axis initially. |
| Held-out failures | Bank authority `[0.40, 0.60]`; simultaneous throttle and flight-path degradation; onset during a turn; degradation combined with low fuel or decision delay. |
| Verification | Independent integration checks the applied command, authority clamp, lag, and resulting kinematics from the logged requested/executed actions. Tests assert unchanged nominal dynamics at authority 1, bounded response under faults, and no new action dimension. |

### 5. Missed decision deadlines and compute budget

| requirement | proposed definition |
| --- | --- |
| State | Requested-action ring, executed action, action age, simulated completion delay, deadline-miss count, and renewable decision tokens. No wall-clock timer enters the environment. |
| Actor observes | Last executed action, its age, remaining token fraction, and whether the previous update missed. It does not observe the next sampled service time. |
| Privileged | Requested actions still in flight, exact completion time, next miss, and token refill schedule. |
| Transition | A policy output is a request. If it completes by the deadline and tokens permit, it becomes the executed three-axis command; otherwise the previous executed command is held. A requested change larger than a declared epsilon consumes one token, so stable control can conserve budget without a fourth “compute” action. Tokens refill deterministically per window. |
| Reward versus hard constraint | Queue causality, deadline, and token capacity are hard. There is no direct penalty for a sampled miss. Optional tiny action-change shaping must be reported separately from fuel/manoeuvre cost and is not in the MVP. |
| Training randomization | Completion delay `0-1` step; miss probability `0-0.10`; available updates `60-100%` of steps; refill windows `10-50` steps. |
| Held-out failures | Delay `2-4` steps; `5-15` step miss bursts; 40% update budget; deadline burst during sensor reacquisition or actuator degradation. |
| Verification | NumPy replay checks that each executed action came from a causal request, held correctly through misses, consumed/refilled tokens exactly once, and stayed in `[-1,1]^3`. Re-running with the same seed must reproduce every execution time and held action. |

### 6. Weather effects on motion and sensing

| requirement | proposed definition |
| --- | --- |
| State | Six-knot vertical profiles for ENU wind, density ratio, and sensing factor; per-aircraft AR(1) gust; episode weather ID. Truth remains spatially simple in v1: altitude-dependent, not a 4D forecast cube. |
| Actor observes | Noisy local wind estimate in body axes, turbulence magnitude, and local density estimate. It does not see future wind along the route or exact sensing attenuation. |
| Privileged | Full vertical profiles, gust innovations, exact local density, and future wind. |
| Transition | Ground velocity becomes air-relative velocity plus wind. Density scales available climb/longitudinal response and the burn multiplier using one documented law. Gust follows a seeded bounded AR(1). A generic sensing factor modifies measurement availability/confidence, not terrain truth. |
| Reward versus hard constraint | Weather itself has no penalty. It changes fuel, arrival time, control feasibility, and observation quality. Airframe limits after density scaling are hard physical bounds. |
| Training randomization | Density profile is the theatre component scaled by `[0.90,1.10]` and clipped to `[0.35,1.05]`. Wind speed is `0-0.12` of cruise speed; direction is uniform and explicitly an assumption because the derived component currently discards direction. Gust sigma is `0-0.03` of cruise speed. Sensing factor `[0.8,1.0]`. |
| Held-out failures | Wind `0.15-0.25` of cruise speed; strong headwind with marginal fuel; crosswind with bank degradation; low density plus climb demand; poor sensing factor plus dropout. |
| Verification | NumPy verifier checks ground displacement from logged air-relative velocity and wind, profile interpolation, density-scaled limits, gust recursion, and fuel coupling. Theatre provenance hash and the synthetic direction assumption are both recorded. No result calls the three-day profile a climatology. |

### 7. Map uncertainty

| requirement | proposed definition |
| --- | --- |
| State | True DEM remains unchanged. An onboard map is represented by an `8x8` low-frequency height-error control grid plus per-aircraft horizontal position bias. Optional local confidence decays outside recently observed cells. |
| Actor observes | Current estimated AGL, uncertainty, and a fixed set of ego-frame terrain lookahead samples with height and sigma. It never receives the true DEM or the error grid. |
| Privileged | True DEM, error field, position bias, true AGL, and future terrain. |
| Transition | Episode-static low-frequency error is the first version. A later rung may reduce local uncertainty from radar-altimeter observations using a fixed-size update rule. Physics and collision always use the true DEM. |
| Reward versus hard constraint | Terrain collision is a hard terminal loss using truth. Map error itself has no penalty. Estimated-map clearance is information, not ground truth. |
| Training randomization | Vertical bias `[-20,20] m`; low-frequency error sigma `0-40 m`; correlation length `5-20 km`; horizontal bias `0-100 m`; lookahead dropout `0-0.10`. |
| Held-out failures | Ridge underestimation `60-100 m`; horizontal bias `200-400 m`; missing lookahead sector; map error combined with crosswind and reduced climb authority. |
| Verification | Independent sampler checks truth collision separately from estimated clearance, reconstructs the low-rank error from logged coefficients, and proves actor observations contain estimates only. Mirrored/shifted-map negative controls follow `test_dem_enu.py` and `test_viewer_export.py`. |

### 8. CBF feasibility as a managed safety resource

| requirement | proposed definition |
| --- | --- |
| State | Per-agent last feasibility, correction norm, number/type of active constraints, and reason bits for envelope, bounds, terrain, actuator, and delay infeasibility. Solver state is not hidden inside the policy. |
| Actor observes | Previous feasibility and correction magnitude only. The policy may learn not to repeatedly hand the filter impossible states. It does not see privileged threats or true-map constraints. |
| Privileged | Exact truth constraints for evaluation; reference-solver result; true authority/wind; full reason bits. The runtime CBF uses only sensed tracks, estimated terrain plus declared uncertainty margin, current play-area bounds, and current executable authority. |
| Transition | Rebuild the admissible control set each step from current authority, held-action latency, wind, bounds, terrain estimate, and track uncertainty. Inflate a stale threat envelope by a deterministic age/confidence reachability bound. Feasibility includes all enabled barriers, including the vertical command, rather than only the horizontal projection. |
| Reward versus hard constraint | The CBF is an attempted runtime constraint, never an unconditional guarantee. Physical action bounds remain hard. Infeasibility and intervention are primary metrics. A small intervention penalty may be an ablation; it must not replace reporting feasibility. |
| Training randomization | Reuse the existing margin sweep `0-3000 m`, threat counts, and red levels; add authority `[0.7,1]`, track age `0-20 s`, and map sigma `0-40 m` only after single-axis calibration. |
| Held-out failures | Stale track plus reduced bank authority; uncertain ridge plus low climb authority; randomized inner bounds; overlapping envelopes under decision delay; head-on geometry with low fuel. |
| Verification | Compare sampled states against an offline exact/reference QP and report feasibility confusion, constraint residual, and suboptimality. Trace verifier recomputes every barrier from the information actually available to the filter and checks that `EnvState.bounds`, not grid extent, was used. Keep paired CBF/no-CBF seeds and provenance from the existing sweep. |

### 9. Diversion and abort behavior

| requirement | proposed definition |
| --- | --- |
| State | Primary objective, two padded recovery gates, active masks, per-gate arrival radii, reserve requirement, and terminal outcome enum: `primary`, `diverted`, `shotdown`, `terrain`, `bounds`, `fuel`, `timeout`. |
| Actor observes | Ego-frame bearing/range and estimated fuel margin to each active gate. It receives no discrete “abort” command. Choosing to abort means physically routing to a recovery gate. |
| Privileged | Exact fuel-to-go under realized future weather, exact gate reachability, and outcome classification. |
| Transition | Recovery gates are route-relative abstract safe exits in the MVP: one near the departure side and one beyond/abeam of the objective side, always inside bounds and outside the spawn ramp. Reaching one terminates safely as `diverted`; reaching the primary remains `primary`. Real airfields are a later theatre feature. |
| Reward versus hard constraint | Proposed ordering: primary completion `+300`, safe diversion `+100`, timeout `0` before ordinary step costs, any airframe loss `-300`. Exact weights require a reward-hacking sweep. No reserve threshold automatically changes the objective or forces a diversion. |
| Training randomization | Gate lateral offset `0.15-0.45` of short-side extent; one or two active gates; gate fuel-margin estimate noise `0-10%`; fuel budgets from the L1 range. |
| Held-out failures | Only the farther gate active; near gate made unattractive by threats; biased fuel-to-go estimate; low fuel plus delayed control; recovery choice under communications loss. |
| Verification | Geometry checks gates are inside the play area and disjoint from the primary arrival region. The trace verifier recomputes the first terminal crossing and proves exactly one outcome. Evaluation reports primary rate and diversion rate separately; “survival” may not count a timeout as equivalent to a recovery. |

## Held-out evaluation protocol

Create `resource-bank-v1`, separate from `route-bank-v1`, with a canonical JSON
digest and reserved seed ranges. Every scenario records route scenario ID,
resource parameters, enabled rung, master seed, and combination class.

Report three splits separately:

1. `interpolation`: unseen seeds and parameter values inside training ranges.
2. `single_axis_extrapolation`: one resource outside its training range, all
   others nominal.
3. `compositional`: each marginal value is familiar, but the combination was
   excluded from training.

Minimum compositional bank:

| ID | frozen combination | question |
| --- | --- | --- |
| C1 | marginal fuel + persistent headwind | Does the actor preserve a recovery option rather than merely slow down? |
| C2 | sensor burst dropout + communications outage | Can local stale-track behavior remain safe without a shared picture? |
| C3 | bank-authority loss + crosswind + delayed actions | Does the route leave controllability margin? |
| C4 | map ridge underestimation + reduced climb authority | Does the policy respect uncertainty, and does the CBF report infeasibility honestly? |
| C5 | stale threat track + low fuel + inflated CBF envelope | Does the backstop consume so much route/energy margin that diversion becomes preferable? |
| C6 | marginal fuel + throttle degradation + missed-deadline burst | Can the actor avoid an unrecoverable energy state with unchanged action space? |

Each policy is evaluated on identical scenario records. Report at least:
primary completion, safe diversion, each loss cause, timeout, fuel reserve at
termination, cumulative exposure, track-age exposure, packet age, deadline
misses, actuator authority, CBF intervention/infeasibility, and route family.
Aggregate numbers never replace per-combination results.

The existing direct-route and avoid-plus-nap controllers remain baselines. Add
one transparent `reserve-divert` heuristic that flies the primary while the
estimated reserve margin is positive and otherwise flies the best recovery
gate. It uses only the actor observation. This is the competent baseline for L1;
beating a controller that never diverts would be a weak claim.

## Checkpoints, provenance, and resume

No existing checkpoint is migrated implicitly.

- A schema-4 checkpoint with no environment-model field is interpreted only as
  `baseline-v0`, matching how missing route/edge flags already mean legacy.
- `checkpoints/theatre_1000` continues to load through the current `Actor` and
  current `Observation`. Its outputs and replay remain unchanged.
- Resource-world policies use `ActorV2`, `ResourceObservation-v1`, and a new
  checkpoint schema carrying `environment_model_id`, `observation_schema_id`,
  `resource_config`, resource curriculum state, resource-bank digest, and all
  optimizer/RNG/history recovery state.
- Loading an ActorV2 checkpoint into baseline-v0, or a baseline actor into a
  resource observation, is refused with both schema IDs in the error.
- If transfer initialization is later desired, it is a named one-way conversion
  that copies compatible legacy encoder/trunk weights, initializes new branches
  explicitly, drops or reconstructs optimizer state explicitly, and records the
  parent checkpoint hash. It is not called “resume.” Training L1 from scratch
  avoids needing this migration in the minimum increment.
- Resume compatibility includes every resource capacity, enabled rung,
  randomization range, terminal/reward definition, and verifier schema. An
  override produces a new run identity; it does not append to the old curve.

`run.json`, checkpoints, evaluation artifacts, and replays record the full
resource config and realized scenario-bank version. Theatre-backed weather or
terrain inputs retain the existing component IDs and hashes. Assumed failure
distributions live in a versioned model component such as
`components/model.resource_world.json`; they are never presented as measured
platform reliability.

## Independent verification strategy

Extend the pure-NumPy verifier rather than importing resource-world functions.
The verifier consumes logged truth, actor-visible observations, executed
actions, and realized exogenous samples. It checks:

1. physical transitions: fuel, wind-adjusted motion, degraded authority,
   action queues, and terminal outcomes;
2. information boundaries: observed tracks/packets/map values derive only from
   available measurements and causal messages;
3. constraint consistency: reward/CMDP losses and CBF barriers agree with the
   declared resource model;
4. fixed-shape masks: invalid slots are zero and cannot influence the actor;
5. stochastic calibration separately: many-seed tests check specified ranges,
   correlations, and event rates without pretending one trace proves them.

Every subsystem gets a negative-control test that modifies one logged field and
requires verification to fail. Full-run verification samples episodes from
every evaluation split, not only training-distribution episodes.

## Recommended minimum viable first increment: L1 endurance and diversion

Implement only the smallest rung that changes the optimal routing decision:

1. Leave `NaigosEnv`/`Actor` defaults and `baseline-v0` checkpoints untouched.
2. Introduce the versioned resource-world state/observation containers with all
   v1 fixed capacities, but enable only fuel and mission fields. Neutral later
   fields must have no effect.
3. Keep the existing burn equation, label it an assumption, and randomize
   initial fuel relative to direct-route fuel-to-go so endurance binds across
   map sizes and route families.
4. Make the first fuel exhaustion event terminal and include it in `alive`,
   outcome classification, reward, CMDP cost, metrics, replay, and NumPy
   verification.
5. Add two abstract recovery gates and a safe-diversion terminal outcome. The
   policy chooses them through flight controls; no abort action or target action
   is added.
6. Add estimated primary/diversion fuel margins to the resource observation and
   implement the observation-only `reserve-divert` heuristic.
7. Freeze a small `resource-bank-v1` containing nominal, marginal-fuel,
   burn-multiplier, farther-gate-only, and threat-detour cases. Reserve its seeds
   and pin its digest before training.
8. Run L0 regression, L1 unit/contract/verifier tests, then CPU smoke training.
   Do not deploy Modal or start a paid run as part of this increment.

Acceptance criteria before L2:

- The complete suite still loads and replays `checkpoints/theatre_1000` as
  baseline-v0, and the existing legacy golden rollout is unchanged.
- L1 `reset`, `step`, and rollout are deterministic, `jit`/`vmap` compatible,
  and fixed-shape.
- A constructed marginal-fuel scenario proves that a substantial detour can
  exhaust fuel while a recovery gate remains reachable; a high-fuel control
  does not exhaust.
- `out_of_fuel` fires once, terminates once, and agrees across env, reward, CMDP,
  metrics, replay, and independent verifier.
- The `reserve-divert` heuristic reduces fuel losses on the frozen marginal
  bank without being credited with primary completion for a diversion.
- Training and held-out resource seeds/ranges are disjoint and digest-checked.
- Every artifact labels its numbers as baseline-v0 or resource-world-v1/L1.

### Tests to add with L1

```text
tests/test_resource_world_contract.py
  fixed shapes; jit/vmap; deterministic resource-local keys; neutral fields

tests/test_endurance.py
  burn integration; route-relative budgets; one-shot dry terminal; reserve margins

tests/test_diversion.py
  gate geometry; observation; first terminal outcome; reward ordering; heuristic

tests/test_resource_verifier.py
  NumPy agreement and tamper failures for fuel/outcomes/observations

tests/test_resource_bank.py
  pinned digest; seed/range disjointness; exact scenario reproduction

tests/test_resource_checkpoint.py
  legacy schema maps only to baseline-v0; ActorV2 resume equivalence; mismatch refusal
```

Continue to run the relevant existing tests, especially environment contract,
airframe, invariant, reward/CMDP verifier, networks, persistent rollouts,
checkpoint resume, map/route randomization, CBF, and CBF sweep tests.

## Explicit non-goals

- No environment-code change in this design task.
- No PyTorch rewrite; implementation remains JAX/Flax/Optax.
- No weapon, target selection, offensive sensor mode, engagement decision, or
  ability for blue to alter a threat.
- No new action dimension, including abort, communications, sensor, or compute
  actions.
- No real-platform fuel, sensor, reliability, or failure-rate claim from the
  proposed ranges.
- No high-fidelity engine, battery, thermal, maintenance, structural-damage,
  projectile, or weapon-damage model.
- No dynamic weather API calls during training or evaluation; inputs remain
  cached, versioned, and offline.
- No real-airfield diversion procedure in L1. Abstract recovery gates avoid
  pretending the current airfield data supplies operational constraints it
  does not contain.
- No learned map completion, SLAM, sensor fusion network, or RNN in the minimum
  increment.
- No automatic objective switch at a reserve threshold. The routing policy
  makes the physical choice.
- No claim that CBF safety is guaranteed while feasibility is false, or that
  the current feasibility bit covers terrain.
- No Modal deployment, paid compute, credential change, or external system
  mutation.

## Trajectory recording of L1 rollouts

L1 rollouts can be recorded as trajectory datasets (opt-in; `docs/TRAJECTORY_DATA.md`).
The actor's resource block is stored as `resource/observation`; absolute reserve
and initial fuel, the recovery gates, the env's exclusive outcome code and the
diversion state are `resource/*`; the true burn multiplier is declared
`privileged/burn_multiplier` for auditing and is never part of an actor input.
Fuel exhaustion is a terminal cause only under L1. Datasets embed the
resource-bank protocol and digest, so a dataset cannot be confused with one
from a different frozen bank. Collection only: no model has been trained on it.

## Recommended integration order after L1

1. Land L1 and produce a CPU-only learning curve against direct, avoid-plus-nap,
   and reserve-divert baselines on the frozen bank.
2. Add L2 sensing with a blue track state and independent verifier before adding
   communications; otherwise delayed truth and stale estimates are impossible
   to distinguish.
3. Add communications and decision latency separately, then their composition.
4. Add actuator degradation before updating the CBF, because authority defines
   the feasible control set.
5. Add weather and map uncertainty, then make the CBF consume the same estimates
   and authority the actor actually has.
6. Only after every single-resource rung is verified, train and evaluate the
   compositional L5 policy. Report the ladder, not only the final aggregate.
