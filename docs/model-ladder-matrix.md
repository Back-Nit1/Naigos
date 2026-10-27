# Model-ladder experiment matrices

`scripts/model_ladder.py` expands a reviewed JSON matrix locally. Its default
mode is deliberately inert: it does not import Modal, inspect credentials,
read a Volume, deploy an app, or submit a job. It prints every run plus exact
environment-step and agent-step totals.

```console
uv run python scripts/model_ladder.py /path/to/matrix.json
uv run python scripts/model_ladder.py /path/to/matrix.json --out /tmp/naigos-plan.json
```

The axes are policy family/version, theatre, seed, resource preset, and
hyperparameter preset. Only policy implementations registered in
`naigos.rl.policies` can be planned. A preset may be recorded before its worker
integration exists, but the submission path refuses every unsupported cell
before spawning the first job.

The scenario suite is currently fixed to the built-in `route-bank-v1` name and
digest. A different label or digest is rejected during planning because the
Modal worker does not yet accept a suite selector; recording a custom value
would claim provenance for scenarios that never executed.

## Example

These prices are example operator inputs, **not current Modal pricing**. Replace
the three hourly rates with a dated quote for the named GPU, reserved CPU core,
and reserved GiB. The GPU kind/count, CPU cores, and memory describe the actual
explicit reservation and submission refuses any mismatch. `timeout_s` must be
at least the already-deployed function's ceiling (currently 1,800 seconds for
smoke and 21,600 seconds for other profiles), because a smaller number would
not produce an upper-bound estimate. These assumptions and the per-job bound
are copied into each submitted run's immutable metadata.

```json
{
  "schema": "naigos-experiment-matrix/1",
  "profile": "short",
  "theatres": ["owens_valley", "tehran_basin"],
  "seeds": [0, 1, 2],
  "policies": [
    {"family": "deepsets", "version": "1"}
  ],
  "resources": [
    {"name": "baseline-v0", "config": {}}
  ],
  "hyperparameters": [
    {"name": "control", "profile_overrides": {}},
    {"name": "longer-screen", "profile_overrides": {"iterations": 300}}
  ],
  "timeout_s": 21600,
  "cost_rates": {
    "gpu_kind": "A10G",
    "gpu_hourly_usd": 1.0,
    "cpu_core_hourly_usd": 0.1,
    "memory_gib_hourly_usd": 0.01,
    "gpu_count": 1,
    "cpu_cores": 4,
    "memory_gib": 16
  },
  "dataset_sha256": null,
  "scenario_suite": {
    "name": "route-bank-v1",
    "sha256": "d3b44303697da01666e2173b8d34547d90694b026c3ccd0bcbbcbfdcef86e7fa"
  },
  "synthetic": false,
  "use_cbf": false
}
```

This example expands to 12 jobs. Its upper bound is the sum over all 12 jobs
assuming every one consumes its complete declared timeout; it is not a runtime
forecast. The plan stores every resolved profile, the exact per-run policy
specification and digest, the ordered theatre threat-class basis, the scenario
suite digest, optional dataset digest, and both named presets. Those values are
inside `run.json`'s immutable identity if a run is submitted.

## Submission gates

Submission is a separate, explicit operation:

```console
uv run python scripts/model_ladder.py /path/to/matrix.json \
  --submit --max-cost-usd 50
```

Before any job is spawned, the command requires all of the following:

- the worktree is clean and committed, and still matches the commit used to
  construct the plan;
- the calculated all-jobs upper bound does not exceed `--max-cost-usd`;
- the declared timeout covers the deployed function timeout;
- the priced GPU kind/count, CPU cores, and memory exactly match the explicit
  deployed reservation;
- every cell is supported by the current training worker;
- every non-smoke cell has a successful, clean smoke marker for that exact
  commit, produced by an image whose independently baked deploy-time source
  fingerprint matches that commit; and
- no run name already has a live writer.

It then calls `scripts.modal_runs.submit_prepared`, which uses the existing
deployed `smoke_remote` or `train_remote` function and the existing manifest,
lock, checkpoint, status, and resume machinery. The matrix command never
deploys or authenticates. Resume remains an explicit inspection-time decision
through `scripts/modal_runs.py resume`; the matrix does not auto-resume or put
two writers in one run directory.

The image fingerprint means a smoke submission against an old deployment fails
instead of blessing the current checkout under an old image. After changing
code, deploy the clean committed checkout, then run smoke; the worker records
both the baked source fingerprint and Modal image ID in runtime provenance.

## Interrupted submissions

Every submission writes an atomic receipt under
`runs/.matrix-submissions/<plan-sha256>.json` before the first spawn and after
each state transition. Re-running the exact matrix skips cells already recorded
as submitted. A process interruption or spawn exception leaves the current cell
as `submitting` or `failed`; it is never retried automatically because a lost
response does not prove that Modal rejected the request.

After checking Modal and the run manifests, `--continue-after-failure` skips
those unresolved cells and submits only cells that are still `pending`. It does
not assert that the failed cell is safe to retry and does not erase its error or
attempt timestamps from the receipt.

Immediately before each new spawn, the launcher also atomically inserts the run
name into the persistent `naigos-run-submission-claims` Modal Dict using
`put(..., skip_if_exists=True)`. Concurrent launchers therefore cannot both
cross the check-and-spawn boundary. Claims are intentionally not released after
a spawn exception: a lost response may still correspond to a queued, billable
job. If a claim is unresolved, inspect Modal plus the Volume manifest and use a
new run identity; automatic claim deletion/retry is deliberately unsupported.
Even after the Dict service expires an inactive claim, an existing Volume
manifest makes the run name permanently single-use for a new submission;
terminal runs can only enter the explicit resume path.

Run names and identities are deterministic. Changing any axis, resolved
profile, code state, policy contract, theatre policy basis, dataset digest, or
scenario digest creates a different identity. Duplicate inputs and duplicate
resolved identities fail during local planning.
