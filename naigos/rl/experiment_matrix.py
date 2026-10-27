"""Pure experiment-matrix planning for the Naigos model ladder.

This module performs no Modal calls.  It turns one reviewed JSON configuration
into immutable run identities, exact work counts and an upper-bound cost.  The
CLI imports the remote submission plumbing only after an explicit ``--submit``.
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
import math
import re
from typing import Any, Mapping, Sequence

from ..env.config import (
    EnvConfig,
    ROUTE_BANK_V1_SHA256,
    ROUTE_BANK_VERSION,
    ThreatKindConfig,
)
from ..research.spec import load_component
from . import policies, runmeta

SCHEMA = "naigos-experiment-matrix/1"
PLAN_SCHEMA = "naigos-experiment-plan/1"
_LABEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,47}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class MatrixError(ValueError):
    """The matrix cannot describe unambiguous, reproducible experiments."""


def _plain(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _plain(dataclasses.asdict(value))
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    return value


def canonical_json(value: Any) -> bytes:
    return json.dumps(_plain(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _exact_keys(data: Mapping[str, Any], *, required: set[str], optional: set[str], where: str) -> None:
    missing = required - set(data)
    unknown = set(data) - required - optional
    if missing or unknown:
        details = []
        if missing:
            details.append(f"missing {sorted(missing)}")
        if unknown:
            details.append(f"unknown {sorted(unknown)}")
        raise MatrixError(f"{where}: {', '.join(details)}")


def _label(value: Any, where: str) -> str:
    if not isinstance(value, str) or not _LABEL_RE.fullmatch(value):
        raise MatrixError(f"{where} must be 1-48 safe identifier characters, got {value!r}")
    return value


def _finite_nonnegative(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MatrixError(f"{where} must be a non-negative number")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise MatrixError(f"{where} must be a finite non-negative number")
    return value


def _positive_int(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise MatrixError(f"{where} must be a positive integer")
    return value


def _axis(data: Any, where: str, *, required: set[str], optional: set[str]) -> list[dict]:
    if not isinstance(data, list) or not data:
        raise MatrixError(f"{where} must be a non-empty list")
    out = []
    seen = set()
    for i, item in enumerate(data):
        if not isinstance(item, Mapping):
            raise MatrixError(f"{where}[{i}] must be an object")
        _exact_keys(item, required=required, optional=optional, where=f"{where}[{i}]")
        normalized = _plain(item)
        identity = digest(normalized)
        if identity in seen:
            raise MatrixError(f"{where} contains duplicate entry at index {i}")
        seen.add(identity)
        out.append(normalized)
    return out


def _policy_config(
    profile: runmeta.RunProfile, theatre: str, *, synthetic: bool
) -> tuple[EnvConfig, dict]:
    kwargs = {
        "n_blue": profile.n_blue,
        "n_threat": profile.n_threat,
        "n_threat_active": profile.n_threat,
    }
    if synthetic:
        return EnvConfig(**kwargs), {"kind": "synthetic-defaults"}
    try:
        component = load_component("model.detection", theatre)
        classes = component["parameters"]["threat_classes"]
    except (FileNotFoundError, KeyError, TypeError) as exc:
        raise MatrixError(
            f"theatre {theatre!r} has no usable model.detection component: {exc}"
        ) from exc
    if not isinstance(classes, Mapping) or not classes:
        raise MatrixError(f"theatre {theatre!r} has no threat classes")
    # Policy shape depends on the number of one-hot threat classes, not their
    # engagement parameters.  Avoid loading the DEM merely to describe an
    # actor, but bind the basis to the complete component so regenerated
    # theatre data cannot silently reuse an old experiment identity.
    kinds = tuple(ThreatKindConfig(label=str(label)) for label in classes)
    return EnvConfig(**kwargs, threat_kinds=kinds), {
        "kind": "theatre-model.detection",
        "theatre": theatre,
        "component_sha256": digest(component),
        "threat_classes": list(classes),
    }


def _policy_record(
    item: Mapping[str, Any], profile: runmeta.RunProfile, theatre: str, *, synthetic: bool
) -> dict:
    family = _label(item["family"], "policies.family")
    version = _label(item["version"], "policies.version")
    cfg, basis = _policy_config(profile, theatre, synthetic=synthetic)
    try:
        spec = _plain(policies.get_policy_spec(cfg, family=family, version=version))
    except policies.PolicyNotRegisteredError as exc:
        raise MatrixError(str(exc)) from exc
    return {
        "family": family,
        "version": version,
        "spec": spec,
        "spec_sha256": digest(spec),
        "spec_basis": basis,
    }


def _run_name(parts: Sequence[str], identity_sha256: str) -> str:
    readable = "-".join(re.sub(r"[^A-Za-z0-9._-]+", "-", p).strip("-_") for p in parts)
    readable = readable[:49].rstrip("-_") or "matrix"
    return runmeta.validate_run_name(f"{readable}-{identity_sha256[:10]}")


def build_plan(config: Mapping[str, Any], *, code: Mapping[str, Any] | None = None) -> dict:
    """Build a deterministic plan and exact aggregate work/cost accounting."""
    if not isinstance(config, Mapping):
        raise MatrixError("matrix configuration must be an object")
    _exact_keys(
        config,
        required={
            "schema", "profile", "theatres", "seeds", "policies", "resources",
            "hyperparameters", "timeout_s", "cost_rates",
        },
        optional={"dataset_sha256", "scenario_suite", "synthetic", "use_cbf"},
        where="matrix",
    )
    if config["schema"] != SCHEMA:
        raise MatrixError(f"matrix.schema must be {SCHEMA!r}")
    profile_name = _label(config["profile"], "profile")
    theatres = [_label(v, f"theatres[{i}]") for i, v in enumerate(config["theatres"])] \
        if isinstance(config["theatres"], list) and config["theatres"] else []
    if not theatres:
        raise MatrixError("theatres must be a non-empty list")
    if len(theatres) != len(set(theatres)):
        raise MatrixError("theatres contains duplicates")
    seeds = config["seeds"]
    if not isinstance(seeds, list) or not seeds:
        raise MatrixError("seeds must be a non-empty list")
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in seeds):
        raise MatrixError("seeds must contain non-negative integers")
    if len(seeds) != len(set(seeds)):
        raise MatrixError("seeds contains duplicates")

    policy_axis = _axis(config["policies"], "policies", required={"family", "version"}, optional=set())
    resource_axis = _axis(config["resources"], "resources", required={"name", "config"}, optional=set())
    hyper_axis = _axis(
        config["hyperparameters"], "hyperparameters",
        required={"name", "profile_overrides"}, optional=set(),
    )
    for r in resource_axis:
        _label(r["name"], "resources.name")
        if not isinstance(r["config"], Mapping):
            raise MatrixError("resources.config must be an object")
    for h in hyper_axis:
        _label(h["name"], "hyperparameters.name")
        if not isinstance(h["profile_overrides"], Mapping):
            raise MatrixError("hyperparameters.profile_overrides must be an object")

    timeout_s = _positive_int(config["timeout_s"], "timeout_s")
    rates = config["cost_rates"]
    if not isinstance(rates, Mapping):
        raise MatrixError("cost_rates must be an object")
    _exact_keys(
        rates,
        required={"gpu_hourly_usd", "cpu_core_hourly_usd", "memory_gib_hourly_usd",
                  "gpu_count", "cpu_cores", "memory_gib"},
        optional=set(), where="cost_rates",
    )
    rates = {k: _finite_nonnegative(v, f"cost_rates.{k}") for k, v in rates.items()}
    dataset_sha = config.get("dataset_sha256")
    if dataset_sha is not None and (not isinstance(dataset_sha, str) or not _SHA256_RE.fullmatch(dataset_sha)):
        raise MatrixError("dataset_sha256 must be null or 64 lower-case hex characters")
    scenario = config.get("scenario_suite") or {
        "name": ROUTE_BANK_VERSION,
        "sha256": ROUTE_BANK_V1_SHA256,
    }
    if not isinstance(scenario, Mapping):
        raise MatrixError("scenario_suite must be an object")
    _exact_keys(scenario, required={"name", "sha256"}, optional=set(), where="scenario_suite")
    scenario = {"name": _label(scenario["name"], "scenario_suite.name"), "sha256": scenario["sha256"]}
    if not isinstance(scenario["sha256"], str) or not _SHA256_RE.fullmatch(scenario["sha256"]):
        raise MatrixError("scenario_suite.sha256 must be 64 lower-case hex characters")
    canonical_scenario = {
        "name": ROUTE_BANK_VERSION,
        "sha256": ROUTE_BANK_V1_SHA256,
    }
    if scenario != canonical_scenario:
        raise MatrixError(
            "scenario_suite must equal the executable built-in route bank "
            f"{canonical_scenario!r}; custom suites are not wired into the Modal trainer"
        )

    code_record = _plain(code if code is not None else {"commit": None, "dirty": None})
    experiments = []
    identities: set[str] = set()
    names: set[str] = set()
    for h in hyper_axis:
        try:
            profile = runmeta.resolve_profile(profile_name, **dict(h["profile_overrides"]))
        except (TypeError, ValueError) as exc:
            raise MatrixError(f"hyperparameters {h['name']!r}: {exc}") from exc
        for theatre in theatres:
            policy_records = [
                _policy_record(
                    p, profile, theatre, synthetic=bool(config.get("synthetic", False))
                )
                for p in policy_axis
            ]
            for seed in seeds:
                for policy in policy_records:
                    for resource in resource_axis:
                        identity = {
                            "profile": profile.as_dict(),
                            "theatre": theatre,
                            "seed": seed,
                            "policy": policy,
                            "resource": resource,
                            "hyperparameters": h,
                            "dataset_sha256": dataset_sha,
                            "scenario_suite": scenario,
                            "synthetic": bool(config.get("synthetic", False)),
                            "use_cbf": bool(config.get("use_cbf", False)),
                            "code": code_record,
                        }
                        identity_sha = digest(identity)
                        if identity_sha in identities:
                            raise MatrixError("two matrix cells resolve to the same immutable identity")
                        identities.add(identity_sha)
                        name = _run_name(
                            ["ladder", theatre, policy["family"], resource["name"], h["name"], f"s{seed}"],
                            identity_sha,
                        )
                        if name in names:
                            raise MatrixError(f"two matrix cells resolve to duplicate run name {name!r}")
                        names.add(name)
                        experiments.append({
                            "run_name": name,
                            "identity_sha256": identity_sha,
                            **identity,
                            "env_steps": profile.total_env_steps,
                            "agent_steps": profile.total_env_steps * profile.n_blue,
                        })

    per_hour = (
        rates["gpu_hourly_usd"] * rates["gpu_count"]
        + rates["cpu_core_hourly_usd"] * rates["cpu_cores"]
        + rates["memory_gib_hourly_usd"] * rates["memory_gib"]
    )
    per_job = per_hour * timeout_s / 3600.0
    plan = {
        "schema": PLAN_SCHEMA,
        "matrix_schema": SCHEMA,
        "code": code_record,
        "job_count": len(experiments),
        "total_env_steps": sum(e["env_steps"] for e in experiments),
        "total_agent_steps": sum(e["agent_steps"] for e in experiments),
        "timeout_s_per_job": timeout_s,
        "cost_rates": rates,
        "upper_bound_cost_usd_per_job": per_job,
        "upper_bound_cost_usd": per_job * len(experiments),
        "experiments": experiments,
    }
    plan["plan_sha256"] = digest(plan)
    return plan


def verify_plan(plan: Mapping[str, Any]) -> None:
    if not isinstance(plan, Mapping) or plan.get("schema") != PLAN_SCHEMA:
        raise MatrixError("not a supported experiment plan")
    saved = plan.get("plan_sha256")
    if not isinstance(saved, str) or saved != digest({k: v for k, v in plan.items() if k != "plan_sha256"}):
        raise MatrixError("experiment plan was edited after it was built")
    names = [e.get("run_name") for e in plan.get("experiments", [])]
    identities = [e.get("identity_sha256") for e in plan.get("experiments", [])]
    if len(names) != plan.get("job_count") or len(set(names)) != len(names):
        raise MatrixError("experiment plan has missing or duplicate run names")
    if len(set(identities)) != len(identities):
        raise MatrixError("experiment plan has duplicate immutable identities")


def experiment_metadata(experiment: Mapping[str, Any], plan: Mapping[str, Any]) -> dict:
    """Build ``run.json`` metadata; the experiment record lives inside config identity."""
    verify_plan(plan)
    profile = runmeta.RunProfile(**{
        **experiment["profile"],
        "overrides": tuple(experiment["profile"].get("overrides") or ()),
    })
    meta = runmeta.build_metadata(
        profile,
        run_name=experiment["run_name"],
        seed=experiment["seed"],
        synthetic=experiment["synthetic"],
        aoi=experiment["theatre"],
        use_cbf=experiment["use_cbf"],
        code=dict(plan["code"]),
        launcher="modal-matrix",
    )
    meta["config"]["experiment"] = {
        "identity_sha256": experiment["identity_sha256"],
        "plan_sha256": plan["plan_sha256"],
        "policy": experiment["policy"],
        "resource": experiment["resource"],
        "hyperparameters": experiment["hyperparameters"],
        "dataset_sha256": experiment["dataset_sha256"],
        "scenario_suite": experiment["scenario_suite"],
    }
    return meta


def submission_problems(experiment: Mapping[str, Any]) -> list[str]:
    """Why the current Modal worker cannot honestly execute this matrix cell."""
    problems = []
    if (experiment["policy"]["family"], experiment["policy"]["version"]) != (
        policies.DEFAULT_POLICY_FAMILY.value, policies.DEFAULT_POLICY_VERSION,
    ):
        problems.append("current Modal trainer only executes deepsets@1")
    resource = experiment["resource"]
    if resource["name"] != "baseline-v0" or resource["config"]:
        problems.append("current Modal trainer only executes the empty baseline-v0 resource preset")
    if experiment["dataset_sha256"] is not None:
        problems.append("current online trainer does not consume an offline trajectory dataset")
    return problems


def remote_spec(experiment: Mapping[str, Any]) -> dict:
    problems = submission_problems(experiment)
    if problems:
        raise MatrixError("; ".join(problems))
    return {
        "profile": experiment["profile"],
        "run_name": experiment["run_name"],
        "seed": experiment["seed"],
        "synthetic": experiment["synthetic"],
        "aoi": experiment["theatre"],
        "use_cbf": experiment["use_cbf"],
        "resume_from": None,
        "resume_overrides": [],
        "force_unlock": False,
        "job_id": None,
        "experiment_identity_sha256": experiment["identity_sha256"],
    }


__all__ = [
    "MatrixError", "PLAN_SCHEMA", "SCHEMA", "build_plan", "canonical_json", "digest",
    "experiment_metadata", "remote_spec", "submission_problems", "verify_plan",
]
