"""Offline model-ladder matrix planning and submission safety."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

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
            "gpu_kind": "A10G",
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
    assert x["policy"]["spec_basis"]["theatre"] == "owens_valley"
    assert len(x["policy"]["spec_basis"]["component_sha256"]) == 64
    assert x["scenario_suite"]["sha256"] == ROUTE_BANK_V1_SHA256
    assert x["resource"] == {"name": "baseline-v0", "config": {}}
    assert x["cost_assumptions"]["rates_and_reservation"] == plan["cost_rates"]
    assert x["cost_assumptions"]["timeout_s"] == 1800
    _, fresh = runmeta.write_metadata(tmp_path, meta)
    assert fresh
    changed = json.loads(json.dumps(meta))
    changed["config"]["experiment"]["resource"]["name"] = "other"
    with pytest.raises(runmeta.RunCollision):
        runmeta.write_metadata(tmp_path, changed)


def test_matrix_resume_preserves_and_compares_the_full_experiment_identity(monkeypatch):
    modal_runs = _load_script("modal_runs")
    plan = em.build_plan(
        _config(
            theatres=["owens_valley"], seeds=[0],
            hyperparameters=[{"name": "default", "profile_overrides": {}}],
        ),
        code=CODE,
    )
    recorded = em.experiment_metadata(plan["experiments"][0], plan)
    monkeypatch.setattr(runmeta, "git_info", lambda _repo: CODE)

    _profile, _code, requested = modal_runs._relaunch_from(
        recorded, recorded["run_name"]
    )
    assert requested["config"]["experiment"] == recorded["config"]["experiment"]
    assert runmeta.identity(requested) == runmeta.identity(recorded)
    assert runmeta.resume_compatibility(recorded, requested)["ok"]

    changed = json.loads(json.dumps(requested))
    changed["config"]["experiment"]["scenario_suite"]["sha256"] = "0" * 64
    verdict = runmeta.resume_compatibility(recorded, changed)
    assert not verdict["ok"]
    assert "config.experiment" in verdict["changes"]


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


def test_noncanonical_scenario_suite_cannot_be_recorded_as_if_it_executed():
    with pytest.raises(em.MatrixError, match="custom suites are not wired"):
        em.build_plan(
            _config(
                scenario_suite={"name": "invented-suite", "sha256": "0" * 64}
            ),
            code=CODE,
        )


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

    with pytest.raises(SystemExit, match="finite non-negative"):
        cli.main([str(path), "--submit", "--max-cost-usd", "nan"])


def test_submit_refuses_a_declared_timeout_below_deployed_ceiling():
    cli = _load_script("model_ladder")
    plan = em.build_plan(
        _config(
            theatres=["owens_valley"], seeds=[0], timeout_s=1799,
            hyperparameters=[{"name": "default", "profile_overrides": {}}],
        ),
        code=CODE,
    )
    deployed = SimpleNamespace(SMOKE_TIMEOUT_S=1800, TIMEOUT_S=21600)
    with pytest.raises(SystemExit, match="not an upper-bound cost"):
        cli._check_declared_timeout_covers_deployment(plan, deployed)

    covered = em.build_plan(
        _config(
            theatres=["owens_valley"], seeds=[0], timeout_s=1800,
            hyperparameters=[{"name": "default", "profile_overrides": {}}],
        ),
        code=CODE,
    )
    cli._check_declared_timeout_covers_deployment(covered, deployed)


def test_submit_prices_the_exact_deployed_resource_reservation():
    cli = _load_script("model_ladder")
    plan = em.build_plan(_config(), code=CODE)
    deployed = SimpleNamespace(
        GPU_KIND="A10G", GPU_COUNT=1, CPU_CORES=2, MEMORY_MIB=4096,
    )
    cli._check_declared_resources_match_deployment(plan, deployed)

    understated = json.loads(json.dumps(plan))
    understated["cost_rates"]["gpu_count"] = 0
    with pytest.raises(SystemExit, match="does not match the deployed function"):
        cli._check_declared_resources_match_deployment(understated, deployed)


def test_submission_receipt_skips_successes_and_never_retries_ambiguous_failures(
    monkeypatch, tmp_path
):
    import scripts

    cli = _load_script("model_ladder")
    monkeypatch.setattr(cli, "MATRIX_RECEIPTS_DIR", tmp_path)
    monkeypatch.setattr(cli.runmeta, "git_info", lambda _repo: CODE)
    plan = em.build_plan(
        _config(
            theatres=["owens_valley"],
            seeds=[0, 1, 2],
            hyperparameters=[{"name": "default", "profile_overrides": {}}],
        ),
        code=CODE,
    )
    calls = []

    def submit(_profile, name, _meta, _spec, **_kwargs):
        calls.append(name)
        if len(calls) == 2:
            raise RuntimeError("connection lost after spawn request")
        return f"fc-{len(calls)}"

    fake_modal_runs = SimpleNamespace(
        SMOKE_TIMEOUT_S=1800,
        TIMEOUT_S=21600,
        GPU_KIND="A10G",
        GPU_COUNT=1,
        CPU_CORES=2,
        MEMORY_MIB=4096,
        _read_volume_json=lambda _path: None,
        _check_not_live=lambda _name: None,
        submit_prepared=submit,
    )
    monkeypatch.setitem(sys.modules, "scripts.modal_runs", fake_modal_runs)
    monkeypatch.setattr(scripts, "modal_runs", fake_modal_runs, raising=False)

    with pytest.raises(RuntimeError, match="connection lost"):
        cli._submit_plan(plan, max_cost_usd=100)

    receipt_path = tmp_path / f"{plan['plan_sha256']}.json"
    receipt = json.loads(receipt_path.read_text())
    assert [entry["state"] for entry in receipt["entries"]] == [
        "submitted", "failed", "pending",
    ]
    assert receipt["entries"][0]["job_id"] == "fc-1"
    assert "connection lost" in receipt["entries"][1]["error"]

    with pytest.raises(SystemExit, match="never retried automatically"):
        cli._submit_plan(plan, max_cost_usd=100)
    assert len(calls) == 2

    continued = cli._submit_plan(
        plan, max_cost_usd=100, continue_after_failure=True
    )
    assert calls == [
        plan["experiments"][0]["run_name"],
        plan["experiments"][1]["run_name"],
        plan["experiments"][2]["run_name"],
    ]
    assert {record["run_name"] for record in continued} == {
        plan["experiments"][0]["run_name"],
        plan["experiments"][2]["run_name"],
    }
    receipt = json.loads(receipt_path.read_text())
    assert [entry["state"] for entry in receipt["entries"]] == [
        "submitted", "failed", "submitted",
    ]


def test_prepared_submit_uses_verified_smoke_and_writer_gates(monkeypatch):
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
        mr.submit_prepared(
            profile, "matrix-test", meta, spec, require_verified_smoke=True
        )
    assert checks == [] and spawns == []

    monkeypatch.setattr(
        mr, "_read_volume_json", lambda _path: {"commit": CODE["commit"], "dirty": None}
    )
    with pytest.raises(SystemExit, match="clean worktree"):
        mr.submit_prepared(
            profile, "matrix-test", meta, spec, require_verified_smoke=True
        )

    monkeypatch.setattr(
        mr, "_read_volume_json", lambda _path: {
            "commit": CODE["commit"],
            "dirty": False,
            "runtime": {"image_source": CODE},
        }
    )
    assert mr.submit_prepared(
        profile, "matrix-test", meta, spec, require_verified_smoke=True
    ) == "fc-1"
    assert checks == ["matrix-test"] and len(spawns) == 1


def test_verified_smoke_gate_requires_exact_clean_provenance():
    assert runmeta.verified_smoke_gate("smoke", None, None) is None
    assert "cannot be verified" in runmeta.verified_smoke_gate(
        "short", {"commit": None, "dirty": None}, None
    )
    assert "clean worktree" in runmeta.verified_smoke_gate(
        "short", {"commit": CODE["commit"], "dirty": True}, CODE["commit"]
    )
    assert "deployed image source" in runmeta.verified_smoke_gate(
        "short", {"commit": CODE["commit"], "dirty": False}, CODE["commit"]
    )
    assert runmeta.verified_smoke_gate(
        "short",
        {
            "commit": CODE["commit"],
            "dirty": False,
            "runtime": {"image_source": CODE},
        },
        CODE["commit"],
    ) is None
