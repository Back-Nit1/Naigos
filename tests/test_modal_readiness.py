"""Local defects found auditing the Modal path before its first remote run (docs/plans/modal-readiness.md).

Offline: nothing here imports a Modal account, authenticates, or deploys. Each test pins a
failure that would otherwise first appear on a paid container.
"""

from __future__ import annotations

import ast
import importlib.util
import re
import sys
from pathlib import Path

from naigos.rl import runmeta as rm

REPO = Path(__file__).resolve().parents[1]
MODAL_FILES = [REPO / "naigos" / "rl" / n for n in ("modal_train.py", "modal_pipeline.py")] + [
    REPO / "scripts" / "modal_runs.py", REPO / "scripts" / "pipeline.py",
    REPO / "naigos" / "rl" / "runmeta.py"]
INSTRUCTION_FILES = MODAL_FILES + [REPO / "README.md", *sorted((REPO / "docs").rglob("*.md"))]


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"_script_{name}", REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# --- deploy: both Modal files use relative imports, so script mode cannot load them ----------


def test_no_instruction_deploys_or_runs_a_modal_file_by_path():
    """`modal deploy naigos/rl/modal_train.py` fails with 'attempted relative import with no
    known parent package' (modal.cli.import_refs); module mode (`-m`) is the one that loads."""
    bad = [f"{p.relative_to(REPO)}: {m.group(0)}" for p in INSTRUCTION_FILES
           for m in re.finditer(r"modal (?:deploy|run) (?!-m)\S*naigos/rl/modal_\w+\.py", p.read_text())]
    assert not bad, bad


def test_the_deploy_hints_name_the_module():
    assert "modal deploy -m naigos.rl.modal_pipeline" in _load_script("pipeline").DEPLOY_HINT


# --- image: everything the real-theatre path imports must be installed ------------------------


def _pip_packages(path: Path, var: str) -> str:
    """The string constants passed to the image expression assigned to `var`."""
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == var for t in node.targets):
            return " ".join(c.value for c in ast.walk(node.value)
                            if isinstance(c, ast.Constant) and isinstance(c.value, str))
    raise AssertionError(f"{var} not assigned in {path.name}")


def test_training_and_evaluation_images_carry_the_theatre_and_checkpoint_dependencies():
    """`env_from_theatre` imports pyproj (naigos/data/geodetic.py); `checkpoint.py` imports orbax."""
    images = [(REPO / "naigos/rl/modal_train.py", "image"),
              (REPO / "naigos/rl/modal_pipeline.py", "train_image"),
              (REPO / "naigos/rl/modal_pipeline.py", "eval_image")]
    for path, var in images:
        pkgs = _pip_packages(path, var)
        for needed in ("pyproj", "orbax-checkpoint"):
            assert needed in pkgs, f"{path.name}:{var} does not install {needed}"


# --- fetch: `modal volume get` paths are relative to the Volume root, not the mount -----------


def test_fetch_asks_for_the_run_by_its_volume_path_not_its_mount_path(monkeypatch, tmp_path):
    mr = _load_script("modal_runs")
    calls = []
    monkeypatch.setattr(mr, "_modal_cli", lambda *a: calls.append(a) or 0)
    monkeypatch.setattr(mr, "_report", lambda d: 0)
    mr.cmd_fetch(type("A", (), {"run_name": "short-s0-20260101T000000Z", "dest": str(tmp_path)})())
    assert calls[0][:4] == ("volume", "get", "naigos-runs", "/short-s0-20260101T000000Z")


# --- job id: the manifest may carry a task id; only a call id can be asked about --------------


def test_status_uses_the_function_call_id_even_if_the_manifest_holds_a_task_id(monkeypatch, tmp_path):
    mr = _load_script("modal_runs")
    monkeypatch.setattr(mr, "JOBS_DIR", tmp_path)
    mr._write_job_record("short-s0", {"job_id": "fc-123"})
    assert mr._resolve_job_id("short-s0", {"job_id": "ta-456"}) == "fc-123"
    assert mr._resolve_job_id("short-s0", {"job_id": "fc-789"}) == "fc-789"
    assert mr._resolve_job_id("never-submitted-here", {"job_id": "ta-456"}) is None


def test_spawn_does_not_overwrite_a_worker_that_started_before_spawn_returned(monkeypatch):
    """The launcher's second stale manifest upload used to turn RUNNING back into QUEUED."""
    mr = _load_script("modal_runs")
    remote = {}
    records = []

    def put(_path, payload):
        remote.clear()
        remote.update(payload)

    class Call:
        object_id = "fc-new"

    class Function:
        def spawn(self, _spec, _meta):
            # Modal may schedule the worker before spawn() returns to the client.
            remote.update(status=rm.RUNNING, started_utc="2026-10-27T12:00:00Z",
                          attempts=[{"kind": "start"}], job_id="fc-new")
            return Call()

    monkeypatch.setattr(mr, "_put_volume_json", put)
    monkeypatch.setattr(mr, "_deployed_function", lambda _tag: Function())
    monkeypatch.setattr(mr, "_write_job_record", lambda name, record: records.append((name, record)))

    profile = rm.resolve_profile("smoke")
    job_id = mr._spawn(profile, "smoke-race", {"code": {}, "config": {}}, {}, gpu="A10G")

    assert job_id == "fc-new"
    assert remote["status"] == rm.RUNNING
    assert remote["started_utc"] == "2026-10-27T12:00:00Z"
    assert remote["attempts"] == [{"kind": "start"}]
    assert records[0][1]["job_id"] == "fc-new"


# --- smoke gate: the worker verifies its own run before it can mark itself completed ---------


def _running_run(tmp_path: Path) -> Path:
    d = tmp_path / "r1"
    d.mkdir()
    prof = rm.resolve_profile("smoke", iterations=4)
    code = {"commit": "abc", "dirty": False}
    rm.write_json(d / rm.META_FILENAME, rm.build_metadata(
        prof, run_name="r1", seed=0, synthetic=True, code=code, launcher="modal"))
    rm.write_json(d / rm.HISTORY_FILENAME, [{"iter": 0}, {"iter": 4}])
    rm.write_json(d / rm.PERF_FILENAME, {"device": {"platform": "gpu"}, "steady_iteration_s_median": 0.5})
    (d / "ckpt_000004").write_bytes(b"x")
    man = rm.build_manifest(run_name="r1", job_id="fc-1", profile="smoke", gpu="A10G", timeout_s=1800,
                            max_retries=0, checkpoint_every=4, code=code, config={"iterations": 4},
                            status=rm.RUNNING)
    man["preflight"] = {c: {"ok": True, "detail": "x"} for c in rm.PREFLIGHT_CHECKS}
    rm.write_json(d / rm.MANIFEST_FILENAME, man)
    return d


def test_a_fetched_run_still_marked_running_does_not_verify(tmp_path):
    assert any("status is 'running'" in p for p in rm.verify_run_dir(_running_run(tmp_path)))


def test_the_worker_can_verify_its_own_run_before_finalizing_the_manifest(tmp_path):
    """Otherwise the status check always fails, `artifact_fetch` is always red, and the smoke
    marker that arms `short`/`full` and the pipeline's `require_smoke` is never written."""
    d = _running_run(tmp_path)
    assert rm.verify_run_dir(d, assume_finalizing=True) == []


def test_finalizing_excuses_only_the_running_status(tmp_path):
    d = _running_run(tmp_path)
    (d / "ckpt_000004").unlink()
    problems = rm.verify_run_dir(d, assume_finalizing=True)
    assert problems and not any("status is" in p for p in problems)
