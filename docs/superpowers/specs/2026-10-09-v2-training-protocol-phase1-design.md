# v2 training protocol, Phase 1: orchestration on existing generalization features

## Status

Phase 1 of a decomposed project. Later phases (not designed here):

- **Phase 2**: per-world threat-composition and sensor/dynamics randomization,
  via discrete presets (each world draws one of a small set of hand-defined
  `EnvConfig` presets; implemented as separate vmap batches per preset, no
  change to the JAX step function's data layout).
- **Phase 3**: a minimal recurrent (GRU) actor network and hidden-state
  threading through the PPO rollout and checkpoint format.
- **Phase 4**: the full hyperparameter sweep (learning rate, entropy
  coefficient, network width, recurrent-vs-feedforward), built on Phase 1's
  multi-seed/provenance/report infrastructure and Phase 3's recurrent net.

This document covers **Phase 1 only**: an orchestration layer over
generalization features that already exist and already work
(`route_mode="diverse"`, `route_eval`, `map_randomize`, `obs_edge_features`,
the red curriculum, `BASELINE_POLICIES`). It makes no change to env physics,
no change to the network architecture, and no change to PPO's update rule.

## Goal

Produce one real, reproducible measurement of the repo's existing
generalization machinery across theatres, with baselines, multiple seeds,
full provenance, and a report that states a finding only when the replicated
confidence interval supports it.

## Non-goals

- No per-world domain randomization of threats/sensors/dynamics (Phase 2).
- No recurrent network (Phase 3).
- No hyperparameter sweep (Phase 4 — needs Phase 3's recurrent axis to be a
  real option, since the original ask names it as one of the four sweep
  axes).
- No change to `naigos/env/*`, `naigos/rl/networks.py`, or `naigos/rl/ppo.py`.

## Components

### New files

- `naigos/rl/protocol_v2.py` — pure functions: the plateau predicate, the
  transfer-table builder, the provenance-manifest builder, the stdlib
  bootstrap CI, and `headline_claims` (the report's claim-gating function).
  Dependency-free beyond stdlib + what `naigos.rl.train`/`runmeta` already
  import, matching `runmeta.py`'s own "stdlib only" discipline.
- `scripts/run_protocol_v2.py` — CLI driver. Submits the run matrix (via
  `scripts/modal_runs.py`'s existing submit/status/fetch plumbing), then
  drives evaluation and report generation once every run in the matrix has a
  final checkpoint.

### Touched files (small diffs)

- `naigos/rl/train.py`:
  - add `PlateauConfig` (new dataclass) and `TrainConfig.plateau:
    PlateauConfig | None = None`.
  - when set, `run()`'s loop stops early once the red curriculum reports
    `curriculum_reason == "held"` for `min_evaluations` straight evaluations
    **at `red_level == 1.0`** — reusing `RedCurriculum`'s own smoothing and
    evidence-gate state, not a new statistic.
  - `TrainConfig.iterations` remains a hard cap: a run that never plateaus
    stops there, same as today. `plateau=None` (the default) reproduces
    today's behavior exactly — fixed-iteration stop, no new history columns.
  - history rows gain `plateau_reason` (`null` when `plateau` is unset) so a
    run can be told apart from one that hit the iteration cap.

### Untouched

`naigos/env/*`, `naigos/rl/networks.py`, `naigos/rl/ppo.py`,
`naigos/rl/red_team.py` internals (its public `RedCurriculum` surface is
read, not changed), `naigos/rl/reward.py`.

## Theatre matrix

Theatres: `owens_valley` (what the shipped checkpoint already trains on —
the in-distribution reference), `tehran_basin`, `dubai_urban`, `mecca_urban`.

Runs:
- 4 theatres × 3 seeds = 12 single-theatre policies.
- 1 multi-theatre policy × 3 seeds = 3 runs. "Multi-theatre" means each
  training world batch round-robins its AOI across iterations (one AOI per
  iteration, cycling), using `env_from_theatre` per AOI and reusing the
  existing `run()` entry point once per iteration-block — no new env
  machinery, just driver-level AOI rotation between calls.
- 15 training runs total.

Each run uses `route_mode="diverse"`, `route_eval=True`, `map_randomize=True`,
`obs_edge_features=True` (all four flags `scripts/train_theatre.py` already
exposes), `red_curriculum` defaulted (smoothed, `window`, as shipped), and
`plateau=PlateauConfig()` (red level 1.0, `min_evaluations` reused from the
curriculum's own gate) with `iterations` set as a generous hard cap.

## Transfer table

For every trained policy (15) × every theatre (4) × its 3 held-out-scenario
seeds: call `evaluate_route_bank` with that theatre's DEM, and call
`baseline()` (`direct_route_policy`, `avoid_nap_policy`) on the **same**
scenario seeds, so every cell in the table — the learner and both
baselines — is scored on identical scenarios.

Output: `transfer_table.json` — keyed `(train_theatre, eval_theatre)` →
`{policy_seed_values, policy_mean, policy_ci95, direct_mean, avoid_nap_mean}`
— plus a rendered Markdown table, in the same style as
`naigos/rl/cbf_sweep.py`'s `markdown_summary`/`format_table`.

CI caveat stated in the report itself: n=3 seeds is a rough bootstrap CI, not
a tight one. The report says so; it doesn't imply more precision than it has.

## Provenance

`protocol_v2_manifest.json`, one per run-group (each single-theatre policy,
and the multi-theatre policy), wrapping `runmeta.build_metadata` (already
has commit+dirty via `git_info`, seed, config) and adding:

- `uv.lock` sha256 (one digest, same pattern as `config.ROUTE_BANK_V1_SHA256`)
  plus explicit `jax`/`optax`/`flax` version strings, already available from
  `runmeta.device_report()`.
- route protocol digest: `config.ROUTE_BANK_V1_SHA256` and the bank version
  name, surfaced at the manifest's top level whenever `route_eval` is on (the
  value already exists per-run inside `route_eval.json`; this just copies it
  up).
- resulting artifacts: path + sha256 of the final checkpoint,
  `transfer_table.json`, and the rendered report — closing the loop from
  manifest back to what it produced.

## Report gating

`protocol_v2.headline_claims(table)`: a stdlib-only bootstrap CI
(`random.choices`, no new dependency) over each cell's 3 seeds. A "headline"
is any comparative sentence the report would otherwise assert — e.g.
"diverse routes generalize better than legacy", "multi-theatre transfer beats
single-theatre training", "the learner beats direct-route". Each candidate
computes its effect size and a 95% CI; the Markdown states the claim only
when the CI excludes zero. Otherwise it prints the raw numbers labeled "not
significant at n=3 seeds" and does not assert a winner.

## Compute

Modal, via the existing `scripts/modal_runs.py submit`/`resume`/`status`
commands — one submission per run in the 15-run matrix, driven by
`scripts/run_protocol_v2.py` rather than by hand.

## Testing

- `tests/test_protocol_v2.py` (new): the plateau predicate (reaches `held` at
  1.0 after `min_evaluations`, never fires early, `plateau=None` reproduces
  the exact fixed-iteration history of today — a golden-value check against
  the current `run()` output), the bootstrap CI (known-distribution sanity
  check), `headline_claims` (asserts a claim is withheld when seeds disagree,
  asserts it fires when they clearly don't), and the transfer-table builder
  (shape and key-set, using baseline()'s existing output on a synthetic
  config so the test runs fast and offline).
- No change to any existing test's expected values — `plateau=None` is the
  default and must be behaviorally identical to today.

## Implementation audit (2026-10-26)

The design was checked against the code before it was implemented. Where the
design and the code disagreed, the code won; each deviation is listed here.

**Deviations from the design**

- `device_report()` lives in `train.py`, not `runmeta.py`, and reports only `jax`.
  Dependency versions (jax, jaxlib, flax, optax, orbax-checkpoint, numpy) come
  from `importlib.metadata` in `protocol_v2.provenance_manifest`.
- Baselines are scored with `evaluate_route_bank(env, BASELINE_POLICIES[...])`, not
  `baseline()`. `baseline()` draws random keys and does not play the route bank, so it
  would not be "the same scenarios" as the learner. The table builder additionally
  refuses to build if the baseline value differs between two policies on one theatre.
- "plateau_reason null when unset" and "no new history columns when unset" contradict
  each other. Resolved as: the column exists only when `TrainConfig.plateau` is set;
  `null` means the run hit its `iterations` cap, a string means it plateaued.
- A plateau seen at a tick makes the *next* iteration the last (forced final evaluation +
  checkpoint), so a plateau-stopped run ends like a capped run: a final eval row and a
  checkpoint at the stop iteration. `runmeta.verify_run_dir` accepts a short run only if
  its last row carries `plateau_reason` and a checkpoint exists at that iteration.
- The streak counts distinct *evaluations* (`curriculum_evals_total`), not ticks, since
  ticks between evaluations repeat the same evaluation.
- Multi-theatre runs have no plateau inside a block; the driver stops rotating once the
  plateau predicate holds over the single history that spans all theatres.
- Evaluation is at a fixed `--eval-red-level` (default 1.0) for every policy and both
  baselines, rebuilt from the final checkpoint. The in-run `route_eval.json` rows are
  not used (next item).
- The Modal path (`modal_runs.py submit`, `modal_train.py`) does not carry
  `route_mode`/`route_eval`/`map_randomize`/`obs_edge_features`/`plateau`, so the plan
  emits local `scripts/train_theatre.py` commands. Remote plumbing is a separate step.

**Finding that predates this work (not fixed here): evaluation ignores the red level after G-4**

Since `10adb4b`, a promotion no longer rebuilds `NaigosEnv`, and `NaigosEnv.rollout`
resets from `env.cfg`, which `run()` builds once at the starting level. So
`evaluate()`, `baseline()` and the in-run route evaluation keep scoring at the starting
red difficulty, while training and the logged `red_level` move on. Measured (synthetic
16-threat env, 300 steps, 64 worlds): avoid+nap survival is 0.855 at red level 0 and
0.223 at level 1.0, so the gap is not small. Consequences: the survival statistic that
drives the curriculum (and therefore the plateau predicate) is measured at the starting
level; and the `route_*` columns in `history.json` are labelled with a level they were
not measured at. Fixing it changes fixed-iteration behaviour, so it needs its own
decision; the plateau predicate and `build_report` are written not to depend on it
(the report re-scores everything itself).

**Statistics at n=3**

The percentile bootstrap over 3 seeds is exactly [min, max] of the seeds (the all-min and
all-max resamples each have probability 1/27, above alpha/2). A claim is therefore
asserted only when every seed clears the comparator. That is conservative against false
positives from a lucky seed, but it is not a calibrated 95% interval, and the ~40 claims
tested are uncorrected for multiplicity; the report prints both caveats.
