"""v2 training protocol, Phase 1: plateau stopping, the CI-gated report, provenance.

Everything runs on tiny synthetic envs on CPU. Nothing here submits a job.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

import jax
import numpy as np
import pytest

from naigos.env import config as C
from naigos.env.config import EnvConfig
from naigos.rl import checkpoint as ck
from naigos.rl import protocol_v2 as p2
from naigos.rl import runmeta
from naigos.rl.ppo import PPOConfig
from naigos.rl.red_team import RedCurriculum
from naigos.rl.train import PlateauConfig, RouteBankOverlap, TrainConfig, run

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("run_protocol_v2", REPO / "scripts" / "run_protocol_v2.py")
driver = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(driver)

ENV = EnvConfig(n_blue=2, n_threat=3, n_threat_active=3, max_steps=40)
PPO = PPOConfig(n_envs=2, n_steps=8, n_minibatches=2)
# Never moves the level (survival is always inside [0, 1]), so every tick says `held` at level 0.
ALWAYS_HELD = RedCurriculum.legacy(demote_survival=0.0, promote_survival=1.0)


def _row(it, reason, level, n_evals):
    return {"iter": it, "curriculum_reason": reason, "red_level_before": level, "red_level_after": level,
            "curriculum_evals_total": n_evals}


# --- plateau predicate ------------------------------------------------------


def test_plateau_needs_min_evaluations_held_at_the_level():
    h = [{"iter": 0, "phase": "baseline"}, _row(20, "held", 1.0, 5), _row(40, "held", 1.0, 6)]
    assert p2.plateau_reason(h, 1.0, 3) is None
    h.append(_row(60, "held", 1.0, 7))
    assert "red level 1 held across 3 evaluations" in p2.plateau_reason(h, 1.0, 3)


def test_plateau_is_never_declared_below_the_level_or_on_weaker_evidence():
    held_low = [_row(i, "held", 0.95, i) for i in range(1, 10)]
    assert p2.plateau_reason(held_low, 1.0, 3) is None
    for broken in ("insufficient_evidence", "promoted", "demoted"):
        h = [_row(20, "held", 1.0, 5), _row(40, broken, 1.0, 6), _row(60, "held", 1.0, 7), _row(80, "held", 1.0, 8)]
        assert p2.plateau_reason(h, 1.0, 3) is None, broken  # the streak is only 2 since `broken`


def test_ticks_between_evaluations_do_not_count_twice():
    # three ticks, but only two distinct evaluations behind them
    h = [_row(10, "held", 1.0, 4), _row(15, "held", 1.0, 4), _row(20, "held", 1.0, 5)]
    assert p2.plateau_reason(h, 1.0, 3) is None
    assert p2.plateau_reason(h + [_row(30, "held", 1.0, 6)], 1.0, 3) is not None


# --- run(): plateau stopping ------------------------------------------------


def _cfg(out, plateau, iterations=10):
    return TrainConfig(iterations=iterations, out_dir=str(out), seed=3, eval_every=2, eval_worlds=2,
                       checkpoint_every=100, curriculum_every=2, plateau=plateau)


def _leaves(learner):
    return jax.tree.leaves((learner.actor.params, learner.critic.params, learner.lam_raw))


@pytest.fixture(scope="module")
def plateaued(tmp_path_factory):
    out = tmp_path_factory.mktemp("plateau") / "p2-test-s3"
    meta = runmeta.build_metadata(
        runmeta.RunProfile(name="local", purpose="t", iterations=10, n_envs=2, n_steps=8, n_threat=3, n_blue=2,
                           cell_m=1500.0, eval_every=2, eval_worlds=2, checkpoint_every=100),
        run_name=out.name, seed=3, synthetic=True, code={"commit": "0" * 40, "dirty": False}, launcher="local")
    learner, hist = run(ENV, PPO, _cfg(out, PlateauConfig(red_level=0.0, min_evaluations=2)),
                        meta=meta, red_curriculum=ALWAYS_HELD)
    return out, learner, hist


def test_run_stops_on_the_plateau_one_iteration_after_it_is_seen(plateaued):
    out, _, hist = plateaued
    # ticks at 2 and 4 are the second and third distinct held evaluations (evals fire at 1, 2, 4); iteration 5 is last
    assert hist[-1]["iter"] == 5
    assert hist[-1]["plateau_reason"].startswith("plateau: red level 0 held")
    assert all(r["plateau_reason"] is None for r in hist[:-1] if "plateau_reason" in r)
    assert (out / "ckpt_000005").is_dir() and not (out / "ckpt_000010").exists()
    assert json.loads((out / runmeta.HISTORY_FILENAME).read_text())[-1]["plateau_reason"]


def test_a_plateau_stopped_run_verifies_as_complete_but_a_truncated_one_does_not(plateaued, tmp_path):
    out, _, _ = plateaued
    assert runmeta.verify_run_dir(out) == []
    cut = tmp_path / out.name
    shutil.copytree(out, cut)
    rows = json.loads((cut / runmeta.HISTORY_FILENAME).read_text())
    rows[-1]["plateau_reason"] = None  # same run, minus the statement that it plateaued
    (cut / runmeta.HISTORY_FILENAME).write_text(json.dumps(rows))
    assert any("truncated" in p for p in runmeta.verify_run_dir(cut))


def test_a_plateau_stop_still_requires_a_checkpoint_at_the_stop(plateaued, tmp_path):
    out, _, _ = plateaued
    cut = tmp_path / out.name
    shutil.copytree(out, cut)
    shutil.rmtree(cut / "ckpt_000005")
    assert any("no checkpoints" in p for p in runmeta.verify_run_dir(cut))


def test_plateau_completion_requires_a_recognized_reason_and_exact_checkpoint(plateaued, tmp_path):
    out, _, _ = plateaued
    bad_reason = tmp_path / "bad-reason" / out.name
    shutil.copytree(out, bad_reason)
    rows = json.loads((bad_reason / runmeta.HISTORY_FILENAME).read_text())
    rows[-1]["plateau_reason"] = "stopped early"
    (bad_reason / runmeta.HISTORY_FILENAME).write_text(json.dumps(rows))
    problems = runmeta.verify_run_dir(bad_reason)
    assert any("unrecognized plateau_reason" in p for p in problems)
    assert any("truncated" in p for p in problems)

    wrong_checkpoint = tmp_path / "wrong-checkpoint" / out.name
    shutil.copytree(out, wrong_checkpoint)
    shutil.copytree(wrong_checkpoint / "ckpt_000005", wrong_checkpoint / "ckpt_000006")
    shutil.rmtree(wrong_checkpoint / "ckpt_000005")
    assert any("no checkpoint at plateau stop iteration 5" in p
               for p in runmeta.verify_run_dir(wrong_checkpoint))


def test_resuming_a_plateaued_run_continues_nothing(plateaued):
    out, _, hist = plateaued
    _, again = run(ENV, PPO, _cfg(out, PlateauConfig(red_level=0.0, min_evaluations=2)),
                   resume_from=ck.latest_resumable(out), red_curriculum=ALWAYS_HELD)
    assert again[-1]["iter"] == 5 and len(again) == len(hist)


def test_the_cap_still_binds_when_the_level_is_never_reached(tmp_path):
    _, hist = run(ENV, PPO, _cfg(tmp_path / "cap", PlateauConfig(red_level=1.0, min_evaluations=2), iterations=6),
                  red_curriculum=ALWAYS_HELD)
    assert hist[-1]["iter"] == 6 and hist[-1]["plateau_reason"] is None  # null = hit the cap


def test_plateau_off_is_the_fixed_iteration_loop_exactly(tmp_path):
    plain_l, plain_h = run(ENV, PPO, _cfg(tmp_path / "off", None, iterations=6), red_curriculum=ALWAYS_HELD)
    armed_l, armed_h = run(ENV, PPO, _cfg(tmp_path / "armed", PlateauConfig(1.0, 2), iterations=6),
                           red_curriculum=ALWAYS_HELD)
    assert all("plateau_reason" not in r for r in plain_h)  # no new column when unset
    drop = {"plateau_reason", "wall_s", "iter_s", "env_steps_per_s", "peak_mem_mb"}
    strip = lambda rows: [{k: v for k, v in r.items() if k not in drop} for r in rows]  # noqa: E731
    assert strip(plain_h) == strip(armed_h)
    assert all(np.array_equal(a, b) for a, b in zip(_leaves(plain_l), _leaves(armed_l)))


# --- bootstrap and claim gating --------------------------------------------


def test_bootstrap_is_deterministic_under_an_explicit_seed():
    v = [0.61, 0.70, 0.58, 0.66, 0.52, 0.63]
    assert p2.bootstrap_ci(v, seed=7) == p2.bootstrap_ci(v, seed=7)
    assert p2.bootstrap_ci(v, seed=7) != p2.bootstrap_ci(v, seed=8)
    assert p2.derived_seed(1, "a|b") == p2.derived_seed(1, "a|b") != p2.derived_seed(1, "a|c")


def test_at_three_seeds_the_interval_is_the_observed_range():
    # all-min and all-max resamples each have probability 1/27 > alpha/2, so the percentile bounds ARE min and max:
    # a claim at n=3 is asserted only if every seed clears the comparator. Documented in the report, not hidden.
    assert p2.bootstrap_ci([0.61, 0.70, 0.58], seed=1) == pytest.approx((0.58, 0.70))


def test_bootstrap_known_distribution():
    lo, hi = p2.bootstrap_ci([10.0, 10.1, 9.9, 10.05, 9.95], seed=0)
    assert 9.9 <= lo < 10.0 < hi <= 10.1
    assert p2.bootstrap_ci([1.0, 1.0, 1.0], [0.0], seed=0) == (1.0, 1.0)  # constant effect, no spread
    assert p2.bootstrap_ci([0.5], seed=0) is None  # one seed has no spread to resample


def _records(learner_by_train, direct=0.4, avoid=0.5, evals=("A", "B")):
    return [{"train": t, "eval": e, "seed": s, "learner": v, "direct": direct + (e == "B") * 0.1,
             "avoid_nap": avoid + (e == "B") * 0.1}
            for t, vals in learner_by_train.items() for e in evals for s, v in enumerate(vals)]


def test_transfer_table_shape_and_key_set():
    t = p2.transfer_table(_records({"A": [0.6, 0.62, 0.64], "B": [0.5, 0.52, 0.54]}), seed=1)
    assert set(t) == {"A|A", "A|B", "B|A", "B|B"}
    c = t["A|B"]
    assert c["seeds"] == [0, 1, 2] and c["policy_seed_values"] == [0.6, 0.62, 0.64]
    assert c["direct_mean"] == pytest.approx(0.5) and c["avoid_nap_mean"] == pytest.approx(0.6)
    assert c["policy_ci95"][0] <= c["policy_mean"] <= c["policy_ci95"][1]


def test_transfer_table_refuses_baselines_scored_on_different_scenarios():
    recs = _records({"A": [0.6, 0.62, 0.64]}, evals=("A",))
    recs[1]["direct"] += 0.05
    with pytest.raises(ValueError, match="scenarios not identical"):
        p2.transfer_table(recs, seed=1)


def _claim(claims, cid):
    return next(c for c in claims if c["id"] == cid)


def test_claims_are_withheld_when_seeds_disagree_and_asserted_when_they_do_not():
    recs = _records({"A": [0.90, 0.10, 0.50], "B": [0.80, 0.82, 0.84], "C": [0.20, 0.21, 0.22]}, evals=("A",))
    claims = p2.headline_claims(p2.transfer_table(recs, seed=1), seed=1)
    noisy, clear, bad = (_claim(claims, f"learner_vs_direct@{t}|A") for t in "ABC")
    assert not noisy["asserted"] and "not significant at n=3 seeds" in noisy["text"] and "beats" not in noisy["text"]
    assert clear["asserted"] and "beats direct route" in clear["text"] and clear["ci95"][0] > 0
    assert bad["asserted"] and "is worse than direct route" in bad["text"]
    assert p2.headline_claims(p2.transfer_table(recs, seed=1), seed=1) == claims  # deterministic


def test_multi_vs_single_transfer_claim_pools_off_theatre_singles():
    recs = _records({p2.MULTI: [0.8, 0.82, 0.84], "A": [0.3, 0.31, 0.32], "B": [0.3, 0.32, 0.31]})
    claims = p2.headline_claims(p2.transfer_table(recs, seed=1), seed=1)
    c = _claim(claims, "multi_vs_single_transfer@A")
    assert c["asserted"] and c["effect"] == pytest.approx(0.82 - 0.31, abs=0.01)  # pooled over B's policies only


def test_the_markdown_states_the_multiplicity_and_every_claim():
    recs = _records({"A": [0.6, 0.62, 0.64]})
    table = p2.transfer_table(recs, seed=1)
    claims = p2.headline_claims(table, seed=1)
    md = p2.render_markdown(table, claims, metric="survival_rate", split="heldout", meta=p2.plan_matrix(
        ("A",), (0,)))
    assert "uncorrected for multiplicity" in md and md.count("- SUPPORTED") + md.count("- withheld") == len(claims)


# --- the plan ---------------------------------------------------------------


def test_plan_is_the_15_run_matrix_and_is_reproducible():
    a, b = p2.plan_matrix(driver.THEATRES), p2.plan_matrix(driver.THEATRES)
    assert a == b and len(a["runs"]) == 15
    assert sum(r["kind"] == "single" for r in a["runs"]) == 12 and len({r["name"] for r in a["runs"]}) == 15
    assert {t for r in a["runs"] if r["kind"] == "single" for t in r["theatres"]} == set(driver.THEATRES)
    assert all(runmeta.RUN_NAME_RE.match(r["name"]) for r in a["runs"])
    assert a["route_bank"]["sha256"] == C.ROUTE_BANK_V1_SHA256
    assert a["cell_m"] == 500.0
    assert all(r["iterations_cap"] == 3000 for r in a["runs"])
    assert p2.plan_matrix(driver.THEATRES, seeds=(0, 1))["plan_sha256"] != a["plan_sha256"]


def test_plan_submits_nothing_and_every_training_flag_is_present():
    for r in p2.plan_matrix(driver.THEATRES)["runs"]:
        cmd = " ".join(r["command"])
        assert "modal" not in cmd.lower() and "submit" not in cmd
        if r["kind"] == "single":
            assert all(f in r["command"] for f in ("--route-mode", "--route-eval", "--map-randomize", "--edge-obs", "--plateau"))
            assert r["command"][r["command"].index("--cell-m") + 1] == "500.0"


def test_plan_refuses_training_seeds_inside_the_route_bank_ranges():
    with pytest.raises(RouteBankOverlap):
        p2.plan_matrix(driver.THEATRES, seeds=(0, C.ROUTE_BANK_SEED_BASE["heldout"] + 3))


# --- provenance -------------------------------------------------------------


def test_provenance_records_commit_dirty_deps_lock_bank_and_artifact_hashes(tmp_path):
    f, d = tmp_path / "f.json", tmp_path / "ckpt"
    f.write_text("{}")
    d.mkdir()
    (d / "a").write_text("x")
    m = p2.provenance_manifest(REPO, artifacts={"table": f, "ckpt": d})
    assert m["code"]["commit"] and isinstance(m["code"]["dirty"], bool)
    assert m["dependencies"]["jax"] and m["dependencies"]["flax"] and m["dependencies"]["optax"]
    assert m["uv_lock_sha256"] == hashlib.sha256((REPO / "uv.lock").read_bytes()).hexdigest()
    assert m["route_bank"]["sha256"] == m["route_bank"]["recomputed_sha256"] == C.ROUTE_BANK_V1_SHA256
    assert m["artifacts"]["table"]["sha256"] == hashlib.sha256(b"{}").hexdigest()
    assert len(m["artifacts"]["ckpt"]["sha256"]) == 64
    with pytest.raises(Exception):
        p2.provenance_manifest(REPO, artifacts={"gone": tmp_path / "missing"})


# --- driver: multi-theatre rotation and the end-to-end report ---------------


def _tiny_env(aoi, *, red_level=None, **_flags):
    # two "theatres" that differ only in episode length, so the table has two distinct eval columns
    cfg = ENV.replace(max_steps={"syn_a": 40, "syn_b": 44}[aoi])
    return (cfg if red_level is None else RedCurriculum().apply(cfg, red_level)), None


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("matrix")
    plan = p2.plan_matrix(("syn_a",), (0, 1), iterations=4, block_iterations=2, root=str(root))
    plan["eval_theatres"] = ["syn_a", "syn_b"]
    plan["runs"] = [r for r in plan["runs"] if not (r["kind"] == "multi" and r["seed"] == 1)]
    plan["plan_sha256"] = hashlib.sha256(json.dumps(
        {k: v for k, v in plan.items() if k != "plan_sha256"}, sort_keys=True).encode()).hexdigest()
    kw = dict(ppo=PPO, eval_every=2, eval_worlds=2, checkpoint_every=2, make_env=_tiny_env, plateau=None)
    for r in plan["runs"]:
        if r["kind"] == "single":
            run(_tiny_env("syn_a")[0], PPO, TrainConfig(iterations=4, out_dir=str(root / r["name"]), seed=r["seed"],
                                                       eval_every=2, eval_worlds=2, checkpoint_every=2))
        else:
            driver.train_multi(root / r["name"], seed=r["seed"], iterations=4, block=2,
                               aois=("syn_a", "syn_b"), **kw)
    return root, plan


def test_multi_theatre_rotates_the_aoi_and_carries_one_history(trained):
    root, plan = trained
    multi = root / next(r["name"] for r in plan["runs"] if r["kind"] == "multi")
    assert [(b["aoi"], b["from_iteration"], b["to_iteration"]) for b in json.loads((multi / "blocks.json").read_text())] \
        == [("syn_a", 0, 2), ("syn_b", 2, 4)]
    iters = [r["iter"] for r in json.loads((multi / runmeta.HISTORY_FILENAME).read_text())]
    assert iters == sorted(iters) and iters[-1] == 4 and ck.latest_resumable(multi).name == "ckpt_000004"
    assert driver.train_multi(multi, seed=0, iterations=4, block=2, aois=("syn_a", "syn_b"), ppo=PPO, eval_every=2,
                              eval_worlds=2, checkpoint_every=2, make_env=_tiny_env, plateau=None) \
        == json.loads((multi / "blocks.json").read_text())  # already done: nothing re-run
    route_rows = json.loads((multi / "route_eval.json").read_text())["evaluations"]
    assert {r["context"]["aoi"] for r in route_rows} == {"syn_a", "syn_b"}
    meta = runmeta.read_metadata(multi)
    assert meta["config"]["map_randomize"] is True
    assert meta["config"]["obs_edge_features"] is True


def test_multi_theatre_block_ledger_recovers_from_the_checkpoint(trained, tmp_path):
    root, plan = trained
    source = root / next(r["name"] for r in plan["runs"] if r["kind"] == "multi")
    recovered = tmp_path / source.name
    shutil.copytree(source, recovered)
    (recovered / "blocks.json").unlink()
    blocks = driver.train_multi(recovered, seed=0, iterations=4, block=2, aois=("syn_a", "syn_b"),
                                ppo=PPO, eval_every=2, eval_worlds=2, checkpoint_every=2,
                                make_env=_tiny_env, plateau=None)
    assert [(b["aoi"], b["from_iteration"], b["to_iteration"]) for b in blocks] == [
        ("syn_a", 0, 2), ("syn_b", 2, 4)
    ]
    assert all(b["recovered_from_checkpoint"] is True for b in blocks)


def test_report_refuses_runs_that_do_not_verify(trained, tmp_path):
    root, plan = trained
    with pytest.raises(SystemExit, match="unverified"):
        driver.build_report(plan, root, tmp_path / "out", make_env=_tiny_env)


def test_report_end_to_end_is_deterministic_and_fully_provenanced(trained, tmp_path):
    root, plan = trained
    kw = dict(make_env=_tiny_env, splits=("heldout",), allow_unverified=True)
    m = driver.build_report(plan, root, tmp_path / "one", **kw)
    driver.build_report(plan, root, tmp_path / "two", **kw)
    tt = json.loads((tmp_path / "one" / "transfer_table.json").read_text())
    assert tt == json.loads((tmp_path / "two" / "transfer_table.json").read_text())
    cells = tt["metrics"]["survival_rate"]["cells"]
    assert set(cells) == {f"{t}|{e}" for t in ("syn_a", p2.MULTI) for e in ("syn_a", "syn_b")}
    assert cells["syn_a|syn_b"]["seeds"] == [0, 1] and "direct_mean" in cells["syn_a|syn_b"]
    assert {"learner_vs_direct@syn_a|syn_b", "learner_vs_avoid_nap@syn_a|syn_b"} <= {
        c["id"] for c in tt["metrics"]["survival_rate"]["claims"]}
    md = (tmp_path / "one" / "report.md").read_text()
    assert "UNVERIFIED RUNS INCLUDED" in md and "uncorrected for multiplicity" in md
    assert m["code"]["commit"] and "dirty" in m["code"] and m["dependencies"]["jax"] and m["uv_lock_sha256"]
    assert m["route_bank"]["sha256"] == C.ROUTE_BANK_V1_SHA256
    for name, a in m["artifacts"].items():
        assert a["sha256"] == runmeta_sha(Path(a["path"])), name
    g = m["run_groups"][plan["runs"][0]["name"]]
    assert g["verification_problems"] and g["checkpoint"]["sha256"] and g["plateau_reason"] is None


def test_report_rejects_an_edited_plan(trained, tmp_path):
    root, plan = trained
    with pytest.raises(SystemExit, match="plan_sha256"):
        driver.build_report({**plan, "eval_theatres": ["syn_a"]}, root, tmp_path / "x", make_env=_tiny_env)


def runmeta_sha(path: Path) -> str:
    from naigos.pipeline import layout

    return layout.sha256_path(path)
