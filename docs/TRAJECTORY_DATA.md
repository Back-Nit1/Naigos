# Trajectory data specification

Status: **implemented format version 1.0.0 / schema version 1.0.0**. This
document describes storage and validation primitives only. No PPO collector,
training loop, checkpoint, controller, or reported result reads or writes this
format yet.

## Purpose and boundary

The format is one common evidence layer for three later consumers:

- behavior cloning from the policy-requested or CBF-executed controls;
- failure analysis with terminal causes, raw reward terms, resources, and CBF
  intervention visible on the same transition;
- offline comparison without importing the simulator or JAX.

It preserves the current Naigos control contract. Blue has exactly three
purely evasive controls, in this fixed order:
`bank_cmd`, `gamma_cmd`, `throttle_cmd`. There is no target or offensive
action. The format records threat observations and adverse outcomes because
they are environment evidence, not actions available to blue.

The implementation is `naigos.trajectory`, which imports only NumPy and the
standard library. NPZ is the exchange container: JAX arrays cross the writer
boundary through NumPy's array protocol, and a future PyTorch data loader can
read the same arrays with NumPy before calling `torch.from_numpy`. No
framework-specific serialization is stored.

## Transition semantics

Array axis 0 is always the record axis. A record means:

1. `observation/*`, `resource/fuel_before`, `episode/id`, and `step/index`
   describe the pre-action state;
2. the behavior policy emits `action/requested`;
3. the CBF emits `action/executed`, `cbf/feasible`, and
   `cbf/intervention_l2`;
4. the environment advances once;
5. `reward/*`, `constraint_cost/*`, `resource/fuel_after`, and `status/*`
   describe that transition and its post-action state.

`status/reset` is true when the pre-action observation came directly from an
environment reset, and therefore requires `step/index == 0`. It is not a second
termination flag. `status/termination_cause` is a compact uint8 event code for
this transition, not a persistent state:

| code | cause |
| ---: | --- |
| 0 | none |
| 1 | reached |
| 2 | shot down |
| 3 | terrain |
| 4 | out of bounds |
| 5 | out of fuel |
| 6 | timeout |
| 7 | external reset |
| 255 | other (must be explained by the producing dataset's configuration) |

`status/alive` and `status/reached` are post-transition states. Thus an agent
that reached its objective can be both alive and reached. A timeout can also
leave it alive. A cause is nonzero only on the transition that created the
terminal event. If several raw terminal indicators occur on one transition,
the primary cause uses this fixed precedence:
`terrain > out_of_bounds > shot_down > reached`. All raw indicators remain in
their separate `reward/*` arrays, so the primary code never discards the
simultaneous evidence. Fuel exhaustion is currently a cost event rather than an
automatic Naigos termination; code 5 is used only if a future environment
configuration explicitly terminates the episode for it.

Records are ordered transitions, not an unordered replay bag. Each episode is a
single contiguous run beginning at `step/index == 0` with `status/reset=true`;
later steps increment by exactly one and carry `status/reset=false`. Once the
record stream switches to a different episode ID, the earlier episode may not
resume. Each agent may emit at most one nonzero terminal cause per episode.
These rules are checked across chunk boundaries as well as within a chunk, so
chunking cannot hide a duplicated, skipped, or reordered transition.

## Required arrays

Shapes below omit the leading record axis `N`. Dimension sizes are immutable
schema metadata. All floats in the core schema are float32; identifiers are
integer arrays; masks and flags are NumPy bool arrays. NaN and infinity are
invalid.

| group | field | shape | unit / encoding |
| --- | --- | --- | --- |
| identity | `episode/id` | `[]` | uint64 |
| identity | `step/index` | `[]` | uint32 |
| local observation | `observation/ego` | `[agent, ego_feature]` | normalized, component metadata |
| local observation | `observation/threats` | `[agent, threat_slot, threat_feature]` | normalized ego-frame values |
| local observation | `observation/threat_mask` | `[agent, threat_slot]` | bool |
| local observation | `observation/friends` | `[agent, friend_slot, friend_feature]` | normalized ego-frame values |
| local observation | `observation/friend_mask` | `[agent, friend_slot]` | bool |
| control | `action/requested` | `[agent, 3]` | normalized `[-1,1]` |
| control | `action/executed` | `[agent, 3]` | normalized `[-1,1]` |
| CBF | `cbf/feasible` | `[agent]` | bool |
| CBF | `cbf/intervention_l2` | `[agent]` | `||executed-requested||₂` |
| reward | `reward/<term>` | `[agent]` | raw unit declared per term |
| constraint | `constraint_cost/aircraft_loss` | `[agent]` | `[0,1]`, separate from reward |
| status | `status/alive`, `status/reached` | `[agent]` | bool |
| status | `status/reset` | `[]` | bool |
| status | `status/termination_cause` | `[agent]` | uint8 enum above |
| resource | `resource/fuel_before`, `resource/fuel_after` | `[agent]` | kg |

Every current raw `RewardTerms` member has its own array. This intentionally
does not store one pre-weighted scalar reward: a later curriculum or analysis
can reweight raw terms, while the independently defined CMDP cost remains
auditable. The schema records units, dtype, symbolic shape, description, and
normalization for every field. The current nonlinear and componentwise ego,
threat, and friendly transforms are written into metadata rather than inferred
from source code. Validation also recomputes the CBF intervention norm, fuel
used, and aircraft-loss cost from their source arrays, checks bounded indicators,
and requires masked observation padding to be exactly zero.

The historical `baseline-v0` environment reports fuel exhaustion as a repeated
aircraft-loss cost while the aircraft remains alive. A future resource-world
configuration may instead make the first exhaustion event terminal. The raw
`reward/out_of_fuel`, status cause, environment configuration, and cost array
make that distinction explicit; the format does not silently reinterpret one
environment's semantics as the other's.

### Optional privileged state and future resources

Privileged CTDE state is dataset-optional but not chunk-optional. A producer
adds fixed-shape fields under `privileged/` when it builds the schema; once the
schema contains them, every chunk must contain them. Masks are ordinary
explicit fields, for example:

```python
FieldSpec(
    "privileged/global_state", "float32", ("global_feature",), "1",
    "Centralized-critic training state.",
    {"method": "componentwise", "components": [...]},
)
FieldSpec(
    "privileged/global_mask", "bool", ("global_slot",), "1",
    "Validity mask for padded privileged slots.", {"method": "identity"},
)
```

Future resource state uses the `resource/` namespace in the same way, such as
`resource/thermal_margin` or paired before/after battery fields. New fields do
not get silently added to schema 1.0: their complete specifications become
part of that dataset's immutable schema metadata. A meaning change to an
existing core field requires a new schema version.

## Immutable provenance

`metadata.json` is canonical JSON and contains:

- theatre, route, scenario, and unsigned seed;
- code commit and dirty-tree flag;
- the complete environment configuration and its canonical SHA-256;
- policy id, kind, framework, optional checkpoint/artifact SHA-256, and policy
  configuration;
- all dimensions and field specifications.

`metadata.commit.json` records the metadata byte count and SHA-256 and is
written once. The public writer refuses an existing output path, refuses to
replace metadata, and refuses all appends after finalization. A partial writer
can resume only when its immutable metadata exactly matches the expected
metadata and it has no orphaned or temporary chunk artifacts.

The provenance is dataset-wide by design. A campaign spanning a different
theatre, route, scenario, seed, code commit, environment configuration, or
behavior policy is written as another dataset shard. This makes accidental
mixtures fail at composition time instead of repeating large strings in every
step.

## Directory and commit protocol

```text
<dataset>/
  metadata.json                 canonical immutable schema + provenance
  metadata.commit.json          byte count and SHA-256 of metadata.json
  chunks/
    00000000.npz                bounded arrays for one chunk
    00000000.json               commit descriptor: records, bytes, SHA-256
    00000001.npz
    00000001.json
  manifest.json                 written last; its presence means complete
```

Chunk publication is payload first, descriptor second. Both use a fully
flushed temporary followed by no-replace publication. A crash can therefore
leave an NPZ or `.tmp-*` without a descriptor, but can never commit a descriptor
for a half-written NPZ. Validation reports those leftovers as incomplete
artifacts. It never treats corrupt committed content as merely partial.

`TrajectoryWriter.append` holds only the supplied chunk and never retains an
earlier chunk. `TrajectoryReader.iter_chunks` likewise validates and yields one
chunk at a time. Chunk size is a collector decision; it should be bounded by a
target byte size or step count, not by campaign length. JSON stays proportional
to the number of chunks and fields, never the number of steps.

## Hashes and completion

Every chunk descriptor hashes the exact NPZ bytes. The final
`content_sha256` is SHA-256 over canonical JSON containing:

- format name and version;
- immutable metadata SHA-256;
- ordered `(chunk index, record count, chunk SHA-256)` entries.

This is a content-integrity and identity mechanism, not a cryptographic
signature against a malicious party able to rewrite every file. Published
datasets should cite the final `content_sha256` out of band.

`manifest.json` is the sole completion marker and is written only after all
committed chunks validate. Its absence makes the dataset partial. The validator
defaults to refusing partial data; `allow_partial=True` returns the validated
committed prefix plus explicit partial reasons, but no final `content_sha256`.
Readers never guess that a quiet writer has finished or expose a mutable prefix
hash as a finished-dataset identity.

## Version and migration policy

Readers accept only versions they implement. Version 1.0.0 currently uses
exact-version compatibility: an unknown format or schema version is rejected,
not best-effort decoded. Enum values are append-only within major version 1;
field removals, dtype/shape changes, action changes, and semantic changes
require a new major schema version.

There is no checkpoint migration because this work does not change checkpoint
bytes, actor inputs, or PPO. A future trajectory migration must be an explicit
read-old/write-new tool with tests that preserve record counts, provenance,
and source content hashes.

## What is measured and what is proposed

Implemented and tested here: schema construction, strict NumPy validation,
chunked NPZ round trips, immutable metadata, per-chunk and whole-dataset
hashing, corruption rejection, version rejection, partial recognition, and
clean partial resume.

Not measured or claimed: compression ratio, collector throughput, recommended
chunk size, behavior-cloning accuracy, or offline policy performance. PPO
collection and any PyTorch planner are future integrations; this format only
establishes their shared data boundary.
