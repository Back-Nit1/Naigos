# Modal readiness audit

Audit of `naigos/rl/modal_train.py`, `naigos/rl/modal_pipeline.py`, `scripts/modal_runs.py`,
`scripts/pipeline.py`, `naigos/pipeline/*`, `naigos/rl/runmeta.py`, and next-steps C-1 / P-1.
Written 2026-10-26. **Nothing was deployed, authenticated, or billed.** Everything below is
labelled:

- **MEASURED** — reproduced locally in this audit (command or test named).
- **SOURCE** — read from the installed `modal` 1.5.0 client source. The client's behaviour is
  known; the server's is not.
- **UNVERIFIED** — can only be learned on Modal. Each one is a stop condition or a smoke check below.

No GPU number exists in this repo. Every figure below that is not a count of steps is a formula
waiting for a first `perf.json`.

## 1. Prioritized blockers

Fixed in this change (each has a test in `tests/test_modal_readiness.py`):

| # | Blocker | Evidence | Effect if left |
|---|---|---|---|
| B1 | Deploying either `naigos/rl/modal_*.py` by filesystem path cannot load it. Both use `from . import runmeta`; script mode imports them with no parent package. | **MEASURED**: `importlib.spec_from_file_location` reproduction raises `attempted relative import with no known parent package`; `modal.cli.import_refs` (**SOURCE**) raises the same and says to use `-m`. Module mode `import naigos.rl.modal_train` / `modal_pipeline` loads and registers 2 / 9 functions. | Step 2 of every documented flow fails. Free failure, but nobody gets past it. |
| B2 | The smoke marker `.smoke_ok` can never be written. The worker calls `verify_run_dir` before it marks the manifest `completed`, so it always reports ``manifest.json status is 'running'``. That also turns the `artifact_fetch` preflight check red. | **MEASURED**: ran a real 2-iteration synthetic run, then the worker's post-training sequence; output `manifest.json status is 'running'`, `would write smoke marker: False`. | `submit --profile short/full` is refused with no recourse except `--skip-smoke-gate`, and the pipeline's `training.require_smoke: true` skips every nightly. The gate that exists to protect the budget trains people to bypass it. |
| B3 | Neither GPU training image nor the CPU eval image installs `pyproj`. `env_from_theatre` -> `data.geodetic.georef_from_enu` imports it on every real-theatre run. `orbax-checkpoint` (imported directly by `rl/checkpoint.py`) was only a transitive dependency. | **MEASURED**: `env_from_theatre(aoi=None, ...)` with `pyproj` blocked raises `ModuleNotFoundError`; with only `pyproj` allowed (`rasterio`, `requests` blocked) it succeeds. Smoke's `image` preflight imports jax/flax/optax/ppo only, so it would not have named this. | Smoke (and every candidate, and every evaluation) dies after the container and GPU are already billing. Both images now install `pyproj>=3.6` and `orbax-checkpoint>=0.5`. |
| B4 | `fetch` ran `modal volume get naigos-runs /runs/<name>`. `/runs` is where a container mounts the Volume; CLI paths are relative to the Volume root. | **SOURCE**: `modal/cli/_download.py` iterates `volume.iterdir(remote_path)`. `status`, `logs`, `cancel` already used `<name>/...` correctly. | Fetch finds nothing. Fixed to `/<name>`. |
| B5 | The worker recorded `MODAL_TASK_ID` (`ta-...`) as the manifest `job_id`, and the status resolver preferred the manifest over the local submission record (`fc-...`). `FunctionCall.from_id` needs a call id. | **SOURCE**: `FunctionCall.from_id`, `modal.current_function_call_id` exist in 1.5.0 (checked). The pipeline binding already uses `current_function_call_id()`. | A job killed by its own timeout leaves a manifest saying `running`; with no Modal answer it reads `unknown` an hour later instead of `timed_out`. That is the exact case `classify_status` exists for. Worker now records the call id; the resolver ignores anything that is not `fc-`. |
| B6 | `refresh_writer_lock` is never called outside a test, so the run-directory lock goes stale at `LOCK_STALE_S` = 2 h even while a run is live. | **MEASURED**: `grep` finds no caller. | A `full` run longer than 2 h can have its backstop lock taken by a second submit. Both training progress callbacks now refresh it. The primary guard (`_check_not_live`) was unaffected. |
| B7 | The launcher re-uploaded its queued manifest after `.spawn()` returned with a call id. A fast worker could already have written `running`, so the stale launcher write erased the worker's transition, attempt record and start timestamp. | **SOURCE** plus a regression test that performs the worker transition inside `spawn()` before returning | The launcher now writes the shared queued manifest once, retains the `fc-...` id in its local job record, and lets the worker write that id to the shared manifest through `modal.current_function_call_id()`. |

Open, not fixable locally. Highest first:

| # | Open item | Class | What to do |
|---|---|---|---|
| O1 | **Working tree is dirty** (other work in flight: `docs/STACK.md`, `train.py`, `policies.py`, ...). `verify_run_dir` reports `produced from a dirty working tree`; `pipeline/candidate.py` requires `dirty is False`; `coordinator.py` refuses to dispatch on unknown/dirty code. | **MEASURED** (`git status`; code paths read) | Deploy and submit only from a clean commit. The smoke marker is stamped with the commit, so any later commit re-arms the gate. |
| O2 | **`modal deploy` of the pipeline starts three crons immediately** (06:00, 08:00 UTC daily; 10:00 Sunday). | Source read | Deploy it last, off-window, and run `pipeline.py pause` straight away. Until a snapshot is published (`seed`) nothing trains anyway. |
| O3 | **`block_network=True` with a Volume and a `modal.Dict` is untested remotely.** `restrict_modal_access` exists as a separate parameter (default `False`) in 1.5.0, which supports the P-1 reading that Dict/Volume access is governed by it, not by `block_network`. | **SOURCE** for the parameters; **UNVERIFIED** for behaviour | First candidate is the test. If leases or the Volume fail under it: pause, redeploy with `NAIGOS_PIPELINE_BLOCK_NETWORK=0`. The code path is still offline. |
| O4 | **`include_source` overlap.** In module mode Modal auto-mounts the `naigos` package at `/root/naigos`, and the images also `add_local_dir("naigos", "/root/naigos")`. Behaviour of the overlap is not documented in the client. | **UNVERIFIED** | If `modal deploy` errors on a mount conflict, set `include_source=False` on the `modal.App(...)`. If it deploys, smoke's `image` and `data_cache` checks confirm the container sees the files. |
| O5 | **DEM allowlist.** py3dep fetches 30 m 3DEP through GDAL from `prd-tnm.s3.amazonaws.com`, not in the `usgs_3dep` hosts. The snapshot child sets `GDAL_HTTP_PROXY=127.0.0.1:9` and the `socket.getaddrinfo` guard, so a cloud DEM fetch fails closed. | Source read | Not a first-run blocker: `default_config.json` has `snapshot.refresh_sources: ["open_meteo"]`, and the DEM arrives with the seed or the parent snapshot. **Do not add the bucket to the allowlist from here**; that is a data-layer decision (recording the real fetch URL in the manifest comes with it). Keep DEM sources out of `refresh_sources` until it is made. |
| O6 | **Image dependencies are unpinned.** `>=` floors only, resolved at build time. Local lock: jax/jaxlib 0.11.1, flax 0.12.9, optax 0.2.8, distrax 0.1.9, orbax-checkpoint 0.12.4, numpy 2.5.3, pyproj 3.8.0 (`.venv`). | **MEASURED** (local), **UNVERIFIED** (what the image resolves) | Smoke's `image` check records the resolved jax/flax/optax versions in the manifest. Compare them to the local lock before `short`; pin `==` to the lock in a follow-up once they are known good (the `jax[cuda12]` plugin must be pinned with it). `extra_index_url` points at an HTML find-links page and is probably ignored by pip; `jax[cuda12]` resolves from PyPI. |
| O7 | **`modal` is not a project dependency.** `.venv` has no `modal`, so `uv run python scripts/modal_runs.py ...` exits `the modal client is not installed`. The pyenv Python has modal 1.5.0 (next-steps says 1.5.5). | **MEASURED** | Run the Modal-facing scripts with the pyenv `python` (the commands below do), or `uv run --with "modal==1.5.0" ...`. Decision: add a `modal` extra to `pyproject.toml` (changes `uv.lock`). |
| O9 | **Smoke gate is shared.** `.smoke_ok` proves `modal_train.py`'s image (data baked in) at a commit, not the pipeline's `train_image` (no data, pipeline layout). | Source read | The first pipeline candidate is the only test of the pipeline image. Run it by hand, paused, with the stop conditions in section 4. |
| O10 | **A timed-out run keeps its writer lock** (the worker never reaches `finally`). Resume inside `LOCK_STALE_S` of the last refresh needs `--force-unlock`. | Source read | Confirm `status` shows no live call first, then `resume <run> --force-unlock`. |

## 2. How the pieces connect (traced)

smoke -> short -> full, as the code actually goes:

1. `modal_runs.py submit`: resolves the profile, reads `git_info`, **reads `.smoke_ok` off the Volume root** and applies `smoke_gate` (commit must match; `--skip-smoke-gate` overrides), `_check_not_live`, writes a `queued` manifest to `<run>/manifest.json`, and calls `fn.spawn(spec, meta)` on the **deployed** function (`smoke_remote` 30-min ceiling or `train_remote` 6-h ceiling; `retries=0`). The returned call id goes into the launcher's local `runs/.jobs/<run>.json`; the worker writes it to the shared manifest when it starts, avoiding a stale launcher write over a running worker.
2. Container (`_train_impl`): `volume.reload()`, `acquire_writer_lock`, manifest -> `running`, `volume.commit()`, tee stdout/stderr to `<run>/train.log` (redacted), then `_run_training`: four preflight checks (image, gpu_backend, data_cache, volume_write), fail-fast if any is red, theatre build (writes `theatre.json`), `train.run(...)` with `on_persist=volume.commit` after every history/perf/checkpoint write, then the `checkpoint` and `artifact_fetch` checks, `verify_run_dir`, and for smoke only: write `/runs/.smoke_ok` stamped with the commit. `finally`: manifest -> `completed|failed`, release lock, commit.
3. `status`: manifest from the Volume + `FunctionCall.get(timeout=0)` state -> `classify_status`. Exit 0 only for `completed`.
4. `resume`: rebuilds the spec from the run's own `run.json`, `resume_compatibility` (commit, seed, theatre, shape, length), `_check_not_live`, spawns with `resume_from: "auto"`; the worker picks `latest_resumable`, walking back past a half-written newest checkpoint. Resume continues the sample path of parameters, optimizers, RNG, curricula and history; rollouts in flight are not checkpointed (`tests/test_resume.py` states this).
5. `fetch`: `modal volume get naigos-runs /<run> <dest> --force`, then the same `verify_run_dir` logic offline. `verify <dir>` needs no Modal.

Persistence: nothing reaches another container until `volume.commit()`; the launcher's `read_file` sees committed state only. A run killed by timeout keeps everything committed before the kill (`progress`/`on_persist`), loses everything after, and cannot record its own death — hence the two-source status.

Writer locking: lock file `.writer.json` per run dir (O_EXCL, best effort on a Volume, now heartbeat-refreshed), plus the submit-time live-call check, which is the one that actually prevents a second writer. Pipeline candidates use a `modal.Dict` lease (`put(skip_if_exists=True)` is the atomic acquire) and take the run-dir lock with `force=True` on top.

## 3. First-run procedure

Run from a clean commit. Shell variables are used so each step can be pasted. Steps marked **[you]** need your credentials or a dashboard; nothing here is automated.

```bash
cd ~/Naigos && unset VIRTUAL_ENV
git status --short                       # must print nothing (O1)
export COMMIT=$(git rev-parse --short HEAD); echo $COMMIT
.venv/bin/python -m pytest tests/test_modal_readiness.py tests/test_run_metadata.py \
  tests/test_run_status.py tests/test_pipeline_cli.py tests/test_resume.py -q   # all green
```

**0. Price and ceiling [you].** In the Modal dashboard set a workspace spend limit you can afford to lose, and write down today's per-second A10G, CPU-core and GiB rates from modal.com/pricing. They are inputs to section 5; this repo does not hard-code a price. Assumption, not measured: Modal bills GPU, CPU and memory separately.

**1. Authenticate [you].**
```bash
modal token new && modal profile current        # prints the profile name, not the secret
```

**2. Deploy the training app, with the exposure capped.** The ceilings are read at deploy time; this deploy allows a 15-minute smoke and a 1-hour `short`, not the 6-hour default.
```bash
NAIGOS_MODAL_SMOKE_TIMEOUT_S=900 NAIGOS_MODAL_TIMEOUT_S=3600 \
  modal deploy -m naigos.rl.modal_train
modal app list | grep naigos                    # one row, state deployed
```
Export the same two variables in the shell used for `submit`, so the manifest records the ceiling that was actually deployed.

**3. Smoke (real theatre, so the `pyproj`/cache path is exercised).**
```bash
export NAIGOS_MODAL_SMOKE_TIMEOUT_S=900 NAIGOS_MODAL_TIMEOUT_S=3600
python scripts/modal_runs.py submit --profile smoke --run-name smoke-$COMMIT
```

**4. Status.** Poll until `completed` or a terminal state; `status` exits non-zero otherwise.
```bash
python scripts/modal_runs.py status smoke-$COMMIT        # prints the six preflight lines
python scripts/modal_runs.py logs   smoke-$COMMIT --tail 100
```

**5. Fetch and verify.**
```bash
python scripts/modal_runs.py fetch  smoke-$COMMIT --dest runs/modal
python scripts/modal_runs.py verify runs/modal/smoke-$COMMIT    # last line must be "OK:"
python scripts/modal_runs.py list /                             # .smoke_ok is at the Volume root
```
Read from the manifest: resolved jax/flax/optax versions (compare with O6), `actual_backend` is `gpu`, device kind is A10G.

**6. Short run** (only after the stop conditions in section 4 pass).
```bash
python scripts/modal_runs.py submit --profile short --run-name short-s0-$COMMIT
python scripts/modal_runs.py status short-s0-$COMMIT
python scripts/modal_runs.py fetch  short-s0-$COMMIT --dest runs/modal
python scripts/modal_runs.py verify runs/modal/short-s0-$COMMIT
```
Never pass `--skip-smoke-gate`. Emergency brake: `python scripts/modal_runs.py cancel <run>` (artifacts already committed stay).

**7. Pipeline, last.** Only after step 6 verified and section 5 produced a $/M number.
```bash
modal deploy -m naigos.rl.modal_pipeline                        # crons are live from this moment (O2)
python scripts/pipeline.py pause --reason "first deploy, no snapshot yet"
python scripts/pipeline.py status
python scripts/pipeline.py seed                                 # validates locally, uploads ~170 MB of data_cache to the Volume, publishes in the cloud
```
A paused tick refuses **all** costly work, manual `run-now` included (`coordinator.tick`, the `cfg["paused"]` branch), so the first candidate needs a short unpaused window. Pick a moment at least an hour clear of 06:00, 08:00 and Sunday 10:00 UTC, and keep the 1-per-day cap (`training.max_candidates_per_day: 1`) as the second guard:
```bash
python scripts/pipeline.py resume
python scripts/pipeline.py run-now snapshot                     # CPU only; open-meteo is the one refresh source
python scripts/pipeline.py status                               # wait here until the snapshot is `completed`
python scripts/pipeline.py run-now nightly                      # one `short` candidate on GPU (<= 4 h ceiling; lower it with NAIGOS_PIPELINE_TRAIN_TIMEOUT_S before the deploy)
python scripts/pipeline.py pause --reason "first candidate in flight"   # pause does not cancel it; it stops the next cron
python scripts/pipeline.py status
python scripts/pipeline.py inspect <candidate-id>
```
Resume the schedule (`pipeline.py resume`) only after that candidate reaches a recorded decision and its `perf.json` is costed. `auto_promote` stays `false`. Kill switches, softest first: `pipeline.py pause`; `modal app stop naigos-pipeline` (stops the crons permanently).

## 4. Stop conditions (do not spend more if any is true)

Before `short`:
1. Any of the six smoke preflight lines is `FAIL` or `not run`.
2. `verify` on the fetched smoke run does not end in `OK:`.
3. Manifest `actual_backend` is not `gpu`, or `perf.json` `device.platform` is not `gpu`.
4. `.smoke_ok` commit differs from `git rev-parse HEAD`, or `git status` is not clean.
5. Resolved jax/flax/optax in the manifest are not the versions you intend to report (O6).
6. Smoke `first_iteration_s` over 600 s, or smoke wall time over 10 minutes: something is compiling or loading that should not be. Investigate before scaling about 2,100x in env-steps (200 x 128 x 128 vs 3 x 16 x 32).
7. `status` is `unknown` for any job: find it in the dashboard first; do not resubmit the same name.

Before `full` (98.3 M env-steps, 6 h ceiling, needs a redeploy with `NAIGOS_MODAL_TIMEOUT_S` raised):
8. The `short` $/M (section 5) multiplied by 98.3 is above the amount you have approved. State the projection to yourself first.
9. `recompiles` > 2 or `recompile_s_total` > 10% of `wall_s`: throughput is being lost to compilation; fix before paying for 30x more steps.
10. The `short` GPU `env_steps_per_s` is not clearly above a CPU run **of the same shape**. The recorded CPU baseline (3.8 k steps/s) is 32 x 64; it is not comparable to 128 x 128. Re-measure the CPU shape or make no speedup claim.
11. `short` is `timed_out`, or `verify` fails for any reason other than the one you are about to fix.

Always:
12. No `retries`, no `--skip-smoke-gate`, no `--force-unlock` before `status` shows no live call, no `pipeline.py resume` before one watched candidate.
13. Pipeline: any lease error, Volume mount error, or `Dict` error under `block_network` -> `pause`, then redeploy with `NAIGOS_PIPELINE_BLOCK_NETWORK=0`; do not retry in a loop.

## 5. Dollars per million environment steps from `perf.json`

`perf.json` fields used: `env_steps_per_iteration`, `iterations_timed`, `steady_iteration_s_median`, `env_steps_per_s` (= `env_steps_per_iteration / steady_iteration_s_median`), `first_iteration_s` (compile + one step), `recompile_s_total`, `wall_s` (loop wall time including evaluation and checkpoint writes, not container start or preflight), `resumed`.

Let `r` be the **total billed rate in $/s of the container** for this function: GPU + CPU cores + memory, from the pricing page (section 3, step 0). It is an input, not a repo constant.

- **Marginal (steady state) cost**: `usd_per_Msteps = r / env_steps_per_s * 1e6`. Use this to compare devices and to project a larger run of the **same shape**.
- **All-in lower bound for one run**: `usd = r * wall_s`, so `usd_per_Msteps_run = r * wall_s / (iterations * env_steps_per_iteration / 1e6)`. It excludes image pull, container start and the preflight, so it is a floor.
- **The authoritative number** is the Modal billing page for that app and time range divided by the same step count. Report it next to the floor; do not report the floor alone.
- A resumed segment's `perf.json` covers only that segment (`resumed: true`). Sum `r * wall_s` and steps over every segment.

```bash
python - <<'EOF' RUN_DIR RATE_USD_PER_S
import json, sys, pathlib
d, r = pathlib.Path(sys.argv[1]), float(sys.argv[2])
p = json.loads((d / "perf.json").read_text())
iters = json.loads((d / "run.json").read_text())["config"]["iterations"]
steps = iters * p["env_steps_per_iteration"]
print(f"marginal  ${r / p['env_steps_per_s'] * 1e6:,.4f} per M env-steps")
print(f"all-in>=  ${r * p['wall_s'] / (steps / 1e6):,.4f} per M env-steps over {steps:,} steps")
print(f"recompiles {p['recompiles']} costing {p['recompile_s_total']} s of {p['wall_s']} s wall")
EOF
```
(`RUN_DIR` and `RATE_USD_PER_S` are positional: put them after `-`, e.g. `python - runs/modal/short-s0-$COMMIT 0.00xxx <<'EOF'`.)

Step counts for projections (arithmetic on `runmeta.PROFILES`, not measurements): smoke 1,536; short 3,276,800; full 98,304,000; the pipeline's weekly (600 iterations at 128 x 128) 9,830,400. Pipeline spend per month = 30 x short + about 4.3 x weekly candidates, each at the marginal $/M of its profile, plus CPU evaluation and the daily snapshot, whose costs are also unmeasured.

## 6. What this audit did not do

- No Modal authentication, deploy, spawn, volume read or write, or billing action. All client-side checks used the pyenv `modal` 1.5.0 install and importlib.
- The model-ladder evolution described in the first half of the request is not part of this change. Other work on `policies.py`, `protocol_v2.py`, `trajectory/`, `docs/plans/model-ladder.md`, `docs/STACK.md`, `train.py` and `test_invariant.py` was present in the working tree and was left untouched.
