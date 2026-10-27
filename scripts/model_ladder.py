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
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from naigos.rl import experiment_matrix as matrix  # noqa: E402
from naigos.rl import runmeta  # noqa: E402


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


def _check_declared_resources_match_deployment(plan: dict, modal_runs) -> None:
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


def _submit_plan(plan: dict, *, max_cost_usd: float) -> list[dict]:
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

    # Delayed import is a tested guarantee: an ordinary dry-run neither imports
    # Modal nor reads credentials/Volumes/deployments.
    from scripts import modal_runs

    _check_declared_timeout_covers_deployment(plan, modal_runs)
    _check_declared_resources_match_deployment(plan, modal_runs)

    # Preflight every run before spawning the first one. submit_prepared repeats
    # both checks at the actual call boundary to close the race.
    marker = modal_runs._read_volume_json(runmeta.SMOKE_MARKER)
    for exp in plan["experiments"]:
        refusal = runmeta.verified_smoke_gate(exp["profile"]["name"], marker, code["commit"])
        if refusal:
            raise SystemExit(f"refusing matrix submission: {refusal}")
        modal_runs._check_not_live(exp["run_name"])

    submitted = []
    for exp in plan["experiments"]:
        profile = runmeta.RunProfile(**{
            **exp["profile"], "overrides": tuple(exp["profile"].get("overrides") or ())
        })
        meta = matrix.experiment_metadata(exp, plan)
        spec = matrix.remote_spec(exp)
        job_id = modal_runs.submit_prepared(
            profile, exp["run_name"], meta, spec, require_verified_smoke=True
        )
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
    _submit_plan(plan, max_cost_usd=args.max_cost_usd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
