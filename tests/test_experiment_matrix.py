"""Offline model-ladder matrix planning and submission safety."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from naigos.env.config import ROUTE_BANK_V1_SHA256, ROUTE_BANK_VERSION
from naigos.rl import experiment_matrix as em
from naigos.rl import runmeta

REPO = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"_matrix_test_{name}", REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _config(**changes):
    cfg = {
        "schema": em.SCHEMA,
        "profile": "smoke",
        "theatres": ["owens_valley", "tehran_basin"],
        "seeds": [0, 1],
        "policies": [{"family": "deepsets", "version": "1"}],
        "resources": [{"name": "baseline-v0", "config": {}}],
        "hyperparameters": [
            {"name": "three", "profile_overrides": {}},
            {"name": "five", "profile_overrides": {"iterations": 5}},
        ],
        "timeout_s": 1800,
        "cost_rates": {
            "gpu_hourly_usd": 1.0,
            "cpu_core_hourly_usd": 0.1,
            "memory_gib_hourly_usd": 0.01,
            "gpu_count": 1,
            "cpu_cores": 2,
            "memory_gib": 4,
        },
        "dataset_sha256": None,
        "scenario_suite": {"name": ROUTE_BANK_VERSION, "sha256": ROUTE_BANK_V1_SHA256},
    }
    cfg.update(changes)
    return cfg


CODE = {"commit": "a" * 40, "dirty": False}


def test_cartesian_plan_has_exact_work_and_unique_stable_identities():
    plan = em.build_plan(_config(), code=CODE)
    assert plan == em.build_plan(_config(), code=CODE)
    assert plan["job_count"] == 2 * 2 * 1 * 1 * 2
    three = runmeta.resolve_profile("smoke")
    five = runmeta.resolve_profile("smoke", iterations=5)
    assert plan["total_env_steps"] == 4 * three.total_env_steps + 4 * five.total_env_steps
    assert plan["total_agent_steps"] == plan["total_env_steps"] * three.n_blue
    assert len({e["run_name"] for e in plan["experiments"]}) == 8
    assert len({e["identity_sha256"] for e in plan["experiments"]}) == 8
    assert all(runmeta.RUN_NAME_RE.fullmatch(e["run_name"]) for e in plan["experiments"])
    em.verify_plan(plan)


def test_cost_is_a_declared_timeout_upper_bound_not_a_runtime_guess():
    plan = em.build_plan(_config(), code=CODE)
    hourly = 1.0 + 2 * 0.1 + 4 * 0.01
    assert plan["upper_bound_cost_usd_per_job"] == pytest.approx(hourly * 0.5)
    assert plan["upper_bound_cost_usd"] == pytest.approx(plan["job_count"] * hourly * 0.5)


@pytest.mark.parametrize(
    "change,match",
    [
        ({"theatres": ["x", "x"]}, "theatres contains duplicates"),
        ({"seeds": [1, 1]}, "seeds contains duplicates"),
        ({"policies": [{"family": "deepsets", "version": "1"}] * 2}, "duplicate entry"),
        ({"dataset_sha256": "nope"}, "dataset_sha256"),
        ({"timeout_s": 0}, "positive integer"),
    ],
)
def test_ambiguous_or_unbounded_inputs_fail_closed(change, match):
    with pytest.raises(em.MatrixError, match=match):
        em.build_plan(_config(**change), code=CODE)


def test_plan_hash_rejects_editing_after_review():
    plan = em.build_plan(_config(), code=CODE)
    plan["experiments"][0]["seed"] = 99
    with pytest.raises(em.MatrixError, match="edited"):
        em.verify_plan(plan)


def test_run_metadata_makes_every_experiment_axis_immutable(tmp_path):
    plan = em.build_plan(_config(theatres=["owens_valley"], seeds=[0],
                                 hyperparameters=[{"name": "default", "profile_overrides": {}}]), code=CODE)
    exp = plan["experiments"][0]
    meta = em.experiment_metadata(exp, plan)
    x = meta["config"]["experiment"]
    assert x["identity_sha256"] == exp["identity_sha256"]
    assert x["policy"]["spec_sha256"] == exp["policy"]["spec_sha256"]
    assert x["scenario_suite"]["sha256"] == ROUTE_BANK_V1_SHA256
    assert x["resource"] == {"name": "baseline-v0", "config": {}}
    _, fresh = runmeta.write_metadata(tmp_path, meta)
    assert fresh
    changed = json.loads(json.dumps(meta))
    changed["config"]["experiment"]["resource"]["name"] = "other"
    with pytest.raises(runmeta.RunCollision):
        runmeta.write_metadata(tmp_path, changed)


def test_unsupported_cells_can_be_planned_but_never_mislabeled_as_submittable():
    plan = em.build_plan(_config(
        theatres=["owens_valley"], seeds=[0],
        resources=[{"name": "resource-world-v1", "config": {"endurance": True}}],
        hyperparameters=[{"name": "default", "profile_overrides": {}}],
    ), code=CODE)
    exp = plan["experiments"][0]
    assert em.submission_problems(exp)
    with pytest.raises(em.MatrixError, match="baseline-v0"):
        em.remote_spec(exp)


def test_cli_defaults_to_local_dry_run_and_never_reaches_submit(monkeypatch, tmp_path, capsys):
    cli = _load_script("model_ladder")
    path = tmp_path / "matrix.json"
    path.write_text(json.dumps(_config(theatres=["owens_valley"], seeds=[0],
                                       hyperparameters=[{"name": "default", "profile_overrides": {}}])))
    monkeypatch.setattr(cli, "_submit_plan", lambda *_a, **_k: pytest.fail("dry-run submitted remotely"))
    assert cli.main([str(path)]) == 0
    output = capsys.readouterr().out
    assert "1 jobs" in output and "dry-run only" in output


def test_submit_requires_a_cost_ceiling_before_any_submit_call(monkeypatch, tmp_path):
    cli = _load_script("model_ladder")
    path = tmp_path / "matrix.json"
    path.write_text(json.dumps(_config(theatres=["owens_valley"], seeds=[0],
                                       hyperparameters=[{"name": "default", "profile_overrides": {}}])))
    monkeypatch.setattr(cli, "_submit_plan", lambda *_a, **_k: pytest.fail("missing cost gate reached submit"))
    with pytest.raises(SystemExit, match="requires.*max-cost"):
        cli.main([str(path), "--submit"])


def test_prepared_submit_uses_existing_smoke_and_writer_gates(monkeypatch):
    mr = _load_script("modal_runs")
    profile = runmeta.resolve_profile("short")
    meta = runmeta.build_metadata(
        profile, run_name="matrix-test", seed=0, synthetic=True,
        code=CODE, launcher="modal-matrix",
    )
    spec = {
        "profile": profile.as_dict(), "run_name": "matrix-test", "seed": 0,
        "synthetic": True, "aoi": None, "use_cbf": False, "resume_from": None,
        "resume_overrides": [], "force_unlock": False, "job_id": None,
    }
    checks, spawns = [], []
    monkeypatch.setattr(mr, "_check_not_live", lambda name: checks.append(name))
    monkeypatch.setattr(mr, "_spawn", lambda *a, **k: spawns.append((a, k)) or "fc-1")
    monkeypatch.setattr(mr, "_read_volume_json", lambda _path: None)
    with pytest.raises(SystemExit, match="no smoke run"):
        mr.submit_prepared(profile, "matrix-test", meta, spec)
    assert checks == [] and spawns == []

    monkeypatch.setattr(mr, "_read_volume_json", lambda _path: {"commit": CODE["commit"]})
    assert mr.submit_prepared(profile, "matrix-test", meta, spec) == "fc-1"
    assert checks == ["matrix-test"] and len(spawns) == 1
