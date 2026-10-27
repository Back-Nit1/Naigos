#!/usr/bin/env python
"""Plan, price and explicitly submit a Naigos model-ladder experiment matrix.

Dry-run is the default and performs no Modal import or remote call::

    uv run python scripts/model_ladder.py matrix.json

Submission is deliberately noisy and requires both an explicit flag and a
cost ceiling::

    uv run python scripts/model_ladder.py matrix.json --submit --max-cost-usd 25

The submit path reuses ``scripts.modal_runs.submit_prepared`` and therefore the
existing deployment, smoke gate, writer exclusion, manifests and checkpoints.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from naigos.rl import experiment_matrix as matrix  # noqa: E402
from naigos.rl import runmeta  # noqa: E402

RECEIPT_SCHEMA = "naigos-matrix-submission-receipt/1"
MATRIX_RECEIPTS_DIR = REPO / "runs" / ".matrix-submissions"


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _receipt_path(plan: dict) -> Path:
    return MATRIX_RECEIPTS_DIR / f"{plan['plan_sha256']}.json"


def _load_or_create_receipt(plan: dict, max_cost_usd: float) -> tuple[Path, dict]:
    path = _receipt_path(plan)
    if path.exists():
        try:
            receipt = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(f"cannot continue from submission receipt {path}: {exc}") from exc
        names = [entry.get("run_name") for entry in receipt.get("entries") or []]
        expected = [exp["run_name"] for exp in plan["experiments"]]
        if (
            receipt.get("schema") != RECEIPT_SCHEMA
            or receipt.get("plan_sha256") != plan["plan_sha256"]
            or names != expected
        ):
            raise SystemExit(f"submission receipt {path} does not match this exact plan")
        return path, receipt

    now = _utc()
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "plan_sha256": plan["plan_sha256"],
        "code": plan["code"],
        "created_utc": now,
        "updated_utc": now,
        "authorized_max_cost_usd": max_cost_usd,
        "cost_assumptions": {
            "timeout_s_per_job": plan["timeout_s_per_job"],
            "rates_and_reservation": plan["cost_rates"],
            "upper_bound_cost_usd": plan["upper_bound_cost_usd"],
        },
        "entries": [
            {"run_name": exp["run_name"], "state": "pending"}
            for exp in plan["experiments"]
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    runmeta.write_json(path, receipt)
    return path, receipt


def _write_receipt(path: Path, receipt: dict) -> None:
    receipt["updated_utc"] = _utc()
    runmeta.write_json(path, receipt)


def _load(path: str | Path) -> dict:
    try:
        value = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot read matrix configuration {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit("matrix configuration must contain a JSON object")
    return value


def _summary(plan: dict) -> str:
    return (
        f"{plan['job_count']} jobs | {plan['total_env_steps']:,} env-steps | "
        f"{plan['total_agent_steps']:,} agent-steps | upper-bound cost "
        f"${plan['upper_bound_cost_usd']:.2f} | plan {plan['plan_sha256'][:16]}"
    )


def _print_plan(plan: dict) -> None:
    print(_summary(plan))
    for exp in plan["experiments"]:
        p, r, h = exp["policy"], exp["resource"], exp["hyperparameters"]
        print(
            f"  {exp['run_name']} theatre={exp['theatre']} seed={exp['seed']} "
            f"policy={p['family']}@{p['version']} resource={r['name']} hp={h['name']} "
            f"env_steps={exp['env_steps']:,} agent_steps={exp['agent_steps']:,}"
        )


def _check_declared_timeout_covers_deployment(plan: dict, modal_runs) -> None:
    """Refuse a cost estimate whose timeout is below the deployed ceiling."""
    required = max(
        modal_runs.SMOKE_TIMEOUT_S
        if exp["profile"]["name"] == "smoke"
        else modal_runs.TIMEOUT_S
        for exp in plan["experiments"]
    )
    declared = plan["timeout_s_per_job"]
    if declared < required:
        raise SystemExit(
            f"refusing submission: declared timeout {declared}s is below the existing "
            f"deployed function ceiling {required}s, so ${plan['upper_bound_cost_usd']:.2f} "
            "is not an upper-bound cost"
        )


def _check_declared_resources_match_deployment(
    plan: dict, modal_runs, marker: dict | None = None
) -> None:
    """Make the priced reservation identical to the deployed reservation."""
    rates = plan["cost_rates"]
    expected = {
        "gpu_kind": modal_runs.GPU_KIND,
        "gpu_count": modal_runs.GPU_COUNT,
        "cpu_cores": modal_runs.CPU_CORES,
        "memory_gib": modal_runs.MEMORY_MIB / 1024.0,
    }
    mismatches = {
        key: {"declared": rates.get(key), "deployed": value}
        for key, value in expected.items()
        if rates.get(key) != value
    }
    if mismatches:
        raise SystemExit(
            "refusing submission: priced resource reservation does not match the "
            f"deployed function: {mismatches}"
        )
    recorded = ((marker or {}).get("runtime") or {}).get("resource_reservation")
    if recorded is not None and recorded != expected:
        raise SystemExit(
            "refusing submission: the verified smoke image used a different resource "
            f"reservation than this launcher: smoke={recorded}, launcher={expected}"
        )


def _submit_plan(
    plan: dict, *, max_cost_usd: float, continue_after_failure: bool = False
) -> list[dict]:
    matrix.verify_plan(plan)
    upper = float(plan["upper_bound_cost_usd"])
    if not math.isfinite(max_cost_usd) or max_cost_usd < 0:
        raise SystemExit("--max-cost-usd must be a finite non-negative number")
    if max_cost_usd < upper:
        raise SystemExit(
            f"refusing submission: declared upper-bound cost ${upper:.2f} exceeds "
            f"--max-cost-usd ${max_cost_usd:.2f}"
        )
    code = runmeta.git_info(REPO)
    if code != plan["code"]:
        raise SystemExit(f"code state changed after planning: plan={plan['code']}, current={code}")
    if code.get("commit") is None or code.get("dirty"):
        raise SystemExit("matrix submission requires a clean committed worktree")

    unsupported = {
        exp["run_name"]: matrix.submission_problems(exp)
        for exp in plan["experiments"]
        if matrix.submission_problems(exp)
    }
    if unsupported:
        detail = "\n".join(f"  {name}: {'; '.join(problems)}" for name, problems in unsupported.items())
        raise SystemExit(
            "refusing to mislabel unsupported experiments; integrate the trainer first:\n" + detail
        )

    receipt_path, receipt = _load_or_create_receipt(plan, max_cost_usd)
    entries = {entry["run_name"]: entry for entry in receipt["entries"]}
    unresolved = [
        entry for entry in receipt["entries"]
        if entry.get("state") in {"submitting", "failed"}
    ]
    if unresolved and not continue_after_failure:
        names = ", ".join(entry["run_name"] for entry in unresolved)
        raise SystemExit(
            f"submission receipt {receipt_path} has unresolved cells ({names}). "
            "Inspect Modal and the run manifests before using --continue-after-failure; "
            "failed cells are never retried automatically."
        )
    pending = [
        exp for exp in plan["experiments"]
        if entries[exp["run_name"]].get("state") == "pending"
    ]
    already = [
        {"run_name": entry["run_name"], "job_id": entry.get("job_id"), "skipped": True}
        for entry in receipt["entries"] if entry.get("state") == "submitted"
    ]
    if not pending:
        print(f"[receipt] no untouched cells remain; {receipt_path}")
        return already

    # Delayed import is a tested guarantee: an ordinary dry-run neither imports
    # Modal nor reads credentials/Volumes/deployments.
    from scripts import modal_runs

    # Preflight every run before spawning the first one. submit_prepared repeats
    # both checks at the actual call boundary to close the race.
    marker = modal_runs._read_volume_json(runmeta.SMOKE_MARKER)
    _check_declared_timeout_covers_deployment(plan, modal_runs)
    _check_declared_resources_match_deployment(plan, modal_runs, marker)
    for exp in pending:
        refusal = runmeta.verified_smoke_gate(exp["profile"]["name"], marker, code["commit"])
        if refusal:
            raise SystemExit(f"refusing matrix submission: {refusal}")
        modal_runs._check_not_live(exp["run_name"])

    submitted = list(already)
    for exp in pending:
        profile = runmeta.RunProfile(**{
            **exp["profile"], "overrides": tuple(exp["profile"].get("overrides") or ())
        })
        meta = matrix.experiment_metadata(exp, plan)
        spec = matrix.remote_spec(exp)
        entry = entries[exp["run_name"]]
        entry.update(state="submitting", attempt_started_utc=_utc())
        _write_receipt(receipt_path, receipt)
        try:
            job_id = modal_runs.submit_prepared(
                profile, exp["run_name"], meta, spec, require_verified_smoke=True
            )
        except BaseException as exc:
            entry.update(
                state="failed",
                failed_utc=_utc(),
                error=runmeta.redact(f"{type(exc).__name__}: {exc}"),
            )
            _write_receipt(receipt_path, receipt)
            print(f"[receipt] recorded failed submission in {receipt_path}")
            raise
        entry.update(state="submitted", job_id=job_id, submitted_utc=_utc())
        _write_receipt(receipt_path, receipt)
        record = {"run_name": exp["run_name"], "job_id": job_id}
        submitted.append(record)
        print(f"[submitted] {record['run_name']} job={record['job_id']}")
    return submitted


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help=f"JSON configuration with schema {matrix.SCHEMA}")
    parser.add_argument("--out", help="optional local plan JSON destination")
    parser.add_argument("--submit", action="store_true", help="perform remote submissions (off by default)")
    parser.add_argument(
        "--continue-after-failure",
        action="store_true",
        help="after manual reconciliation, skip failed/uncertain cells and submit only untouched cells",
    )
    parser.add_argument(
        "--max-cost-usd", type=float,
        help="required with --submit; must cover the computed all-jobs timeout upper bound",
    )
    args = parser.parse_args(argv)
    try:
        plan = matrix.build_plan(_load(args.config), code=runmeta.git_info(REPO))
        matrix.verify_plan(plan)
    except matrix.MatrixError as exc:
        raise SystemExit(f"invalid experiment matrix: {exc}") from exc

    _print_plan(plan)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        runmeta.write_json(out, plan)
        print(f"wrote local plan to {out}")
    if not args.submit:
        print("dry-run only; no Modal import or remote call (pass --submit to change this)")
        return 0
    if args.max_cost_usd is None:
        raise SystemExit("--submit requires a finite non-negative --max-cost-usd")
    if not math.isfinite(args.max_cost_usd) or args.max_cost_usd < 0:
        raise SystemExit("--max-cost-usd must be a finite non-negative number")
    _submit_plan(
        plan,
        max_cost_usd=args.max_cost_usd,
        continue_after_failure=args.continue_after_failure,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
