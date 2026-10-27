#!/usr/bin/env python
"""v2 training protocol, Phase 1 driver (docs/superpowers/specs/2026-10-09-v2-training-protocol-phase1-design.md).

    uv run python scripts/run_protocol_v2.py plan --out runs/protocol_v2/plan.json
    uv run python scripts/run_protocol_v2.py train-multi --seed 0 --out runs/protocol_v2/p2-multi-s0
    uv run python scripts/run_protocol_v2.py report --plan runs/protocol_v2/plan.json \
        --runs runs/protocol_v2 --out runs/protocol_v2/report

This script never submits a remote job. `plan` only writes the matrix and prints
the LOCAL commands for it (the Modal path does not carry the route/edge/plateau
flags). `report` reads finished run directories; it trains nothing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from naigos.env.config import EnvConfig  # noqa: E402
from naigos.env.theatre_bridge import env_from_theatre  # noqa: E402
from naigos.env.flight_env import NaigosEnv  # noqa: E402
from naigos.rl import checkpoint as ckpt  # noqa: E402
from naigos.rl import protocol_v2 as p2  # noqa: E402
from naigos.rl import runmeta  # noqa: E402
from naigos.rl.ppo import PPOConfig, greedy_policy  # noqa: E402
from naigos.rl.red_team import RedCurriculum  # noqa: E402
from naigos.rl.train import BASELINE_POLICIES, PlateauConfig, TrainConfig, evaluate_route_bank, run  # noqa: E402

# the generalization flags every run in the matrix trains with (same four as train_theatre.py)
TRAIN_FLAGS = {"route_mode": "diverse", "map_randomize": True, "obs_edge_features": True}
# owens_valley is what the shipped checkpoint trains on (the in-distribution reference)
THEATRES = ("owens_valley", "tehran_basin", "dubai_urban", "mecca_urban")
N_THREAT, CELL_M = 16, 500.0  # publishable fidelity; runmeta's short/full profiles use 500 m
METRICS = ("survival_rate", "objective_rate")


def theatre_env(aoi: str, *, red_level: float | None = None, **flags):
    """`(EnvConfig, hmap)` for one theatre; `red_level` applies the red curriculum at that level."""
    cfg, hmap, _ = env_from_theatre(aoi=aoi, n_threat=N_THREAT, cell_m=CELL_M, **(flags or TRAIN_FLAGS))
    return (cfg if red_level is None else RedCurriculum().apply(cfg, red_level)), hmap


# --- multi-theatre training -------------------------------------------------


def train_multi(out, *, seed, iterations, block, aois=THEATRES, make_env=theatre_env, ppo=None,
                plateau=PlateauConfig(), eval_every=20, eval_worlds=64, checkpoint_every=100):
    """One policy over several theatres: the AOI rotates every `block` iterations.

    Each block is one `run()` call that resumes the previous block's checkpoint,
    so optimizer state, RNG stream, red level and history carry across theatres
    exactly as they do across an ordinary resume. A killed run continues from its
    latest checkpoint. The plateau test sees one history spanning all theatres.
    There is no `resume_compatibility`/`curriculum_compatibility` check between blocks -- the
    driver is the only writer and the curricula are the defaults; add them if blocks ever get a CLI.
    """
    out, ppo = Path(out), ppo or PPOConfig()
    profile = runmeta.RunProfile(
        name="local", purpose="protocol v2 multi-theatre run", iterations=iterations, n_envs=ppo.n_envs,
        n_steps=ppo.n_steps, n_threat=N_THREAT, n_blue=EnvConfig().n_blue, cell_m=CELL_M,
        eval_every=eval_every, eval_worlds=eval_worlds, checkpoint_every=checkpoint_every)
    meta = runmeta.build_metadata(profile, run_name=out.name, seed=seed, synthetic=False,
                                  aoi="multi:" + "+".join(aois), code=runmeta.git_info(REPO), launcher="local")
    meta["config"].update(
        route_mode="diverse", route_eval=True, map_randomize=True,
        obs_edge_features=True, plateau=True, block_iterations=block,
    )
    prior = ckpt.latest_resumable(out) if out.is_dir() else None
    done = runmeta.checkpoint_iteration(prior) if prior else 0
    blocks_path, blocks, hist = out / "blocks.json", [], []
    if blocks_path.exists():
        blocks = json.loads(blocks_path.read_text())

    # A checkpoint is committed before the block ledger. If the process dies in
    # that narrow window, rebuild the missing, deterministic AOI segments rather
    # than advancing with an unexplained hole in blocks.json.
    cursor = 0
    for i, rec in enumerate(blocks):
        if not isinstance(rec, dict):
            raise ValueError(f"blocks.json entry {i} is not an object")
        start, end = rec.get("from_iteration"), rec.get("to_iteration")
        expected_aoi = aois[(cursor // block) % len(aois)]
        boundary = min(((cursor // block) + 1) * block, iterations)
        if start != cursor or not isinstance(end, int) or not cursor < end <= boundary:
            raise ValueError(f"blocks.json entry {i} is not a contiguous protocol block: {rec}")
        if rec.get("aoi") != expected_aoi:
            raise ValueError(
                f"blocks.json entry {i} names AOI {rec.get('aoi')!r}, expected {expected_aoi!r}"
            )
        cursor = end
    if cursor > done:
        raise ValueError(f"blocks.json reaches iteration {cursor}, beyond checkpoint {done}")
    while cursor < done:
        boundary = min(((cursor // block) + 1) * block, iterations, done)
        blocks.append({
            "aoi": aois[(cursor // block) % len(aois)],
            "from_iteration": cursor,
            "to_iteration": boundary,
            "recovered_from_checkpoint": True,
        })
        cursor = boundary
    if blocks or done:
        runmeta.write_json(blocks_path, blocks)

    if prior is not None:
        prior_history = list(ckpt.load(prior)["recovery"]["history"])
        if plateau and p2.plateau_reason(prior_history, plateau.red_level, plateau.min_evaluations):
            return blocks
    while done < iterations:
        k = done // block
        aoi, target = aois[k % len(aois)], min((k + 1) * block, iterations)
        cfg, hmap = make_env(aoi)
        _, hist = run(cfg, ppo, TrainConfig(iterations=target, out_dir=str(out), seed=seed, eval_every=eval_every,
                                            eval_worlds=eval_worlds, checkpoint_every=checkpoint_every,
                                            route_eval=True, route_eval_context={"aoi": aoi}, plateau=plateau),
                      hmap=hmap, meta=meta, resume_from=ckpt.latest_resumable(out))
        actual_done = hist[-1]["iter"]
        if actual_done <= done:
            raise RuntimeError(f"multi-theatre block made no progress from iteration {done}")
        blocks.append({"aoi": aoi, "from_iteration": done, "to_iteration": actual_done})
        runmeta.write_json(blocks_path, blocks)
        done = actual_done
        if plateau and p2.plateau_reason(hist, plateau.red_level, plateau.min_evaluations):
            break
    return blocks


# --- report -----------------------------------------------------------------


def _final_row(run_dir: Path) -> dict:
    rows = json.loads((run_dir / runmeta.HISTORY_FILENAME).read_text())
    return rows[-1] if rows else {}


def build_report(plan: dict, runs_root, out_dir, *, make_env=theatre_env, eval_red_level: float = 1.0,
                 splits=("heldout", "train_geometry"), allow_unverified: bool = False) -> dict:
    """Evaluate every finished run on every theatre's frozen route bank, then write
    transfer_table.json, report.md and protocol_v2_manifest.json into `out_dir`.

    The in-training route_eval rows are NOT used: they are measured at whatever
    difficulty the run's env was built at (see the G-4 finding in the report
    notes). Every policy and both baselines are re-scored here, on identical
    scenarios, at `eval_red_level`.
    """
    runs_root, out_dir = Path(runs_root), Path(out_dir)
    bank = plan["route_bank"]
    if plan["plan_sha256"] != hashlib.sha256(json.dumps({k: v for k, v in plan.items() if k != "plan_sha256"},
                                                         sort_keys=True).encode()).hexdigest():
        raise SystemExit("plan.json was edited after it was written (plan_sha256 mismatch)")
    problems = {r["name"]: runmeta.verify_run_dir(runs_root / r["name"]) for r in plan["runs"]}
    bad = {k: v for k, v in problems.items() if v}
    if bad and not allow_unverified:
        raise SystemExit("refusing to report on unverified runs (pass --allow-unverified to record them):\n"
                         + "\n".join(f"  {k}: {'; '.join(v)}" for k, v in bad.items()))

    baselines: dict[tuple, dict] = {}  # (eval theatre, env flags) -> {baseline -> result}
    records: dict[str, list] = {m: [] for m in METRICS}
    groups, artifacts = {}, {"plan": out_dir / "plan.json"}
    out_dir.mkdir(parents=True, exist_ok=True)
    runmeta.write_json(artifacts["plan"], plan)
    for r in plan["runs"]:
        d = runs_root / r["name"]
        final = ckpt.latest_resumable(d)
        blob = ckpt.load(final)
        stored = blob["env_cfg"]  # a checkpoint from before a flag existed was trained without it
        flags = {"route_mode": stored.get("route_mode", "legacy"), "map_randomize": bool(stored.get("map_randomize")),
                 "obs_edge_features": bool(stored.get("obs_edge_features"))}
        train = p2.MULTI if r["kind"] == "multi" else r["theatres"][0]
        for ev in plan["eval_theatres"]:
            cfg, hmap = make_env(ev, red_level=eval_red_level, **flags)
            env = NaigosEnv(cfg, hmap=hmap)
            bkey = (ev, tuple(flags.values()))  # baselines are rescored if a run's env flags differ
            if bkey not in baselines:
                baselines[bkey] = {n: evaluate_route_bank(env, pol, splits=splits) for n, pol in BASELINE_POLICIES.items()}
            res = evaluate_route_bank(env, greedy_policy(blob["actor"], env.cfg), splits=splits,
                                      training_seeds=[r["seed"]])
            for m in METRICS:
                records[m].append({"train": train, "eval": ev, "seed": r["seed"],
                                   "learner": res["heldout"]["aggregate"][m],
                                   **{b: baselines[bkey][b]["heldout"]["aggregate"][m] for b in p2.BASELINES}})
        row = _final_row(d)
        groups[r["name"]] = {
            "kind": r["kind"], "seed": r["seed"], "train": train, "verification_problems": problems[r["name"]],
            "code": (runmeta.read_metadata(d) or {}).get("code"), "iterations_done": row.get("iter"),
            "iterations_cap": r["iterations_cap"], "plateau_reason": row.get("plateau_reason"),
            "checkpoint": p2.artifact_entry(final),
        }
    seed = plan["bootstrap"]["seed"]
    n_boot = plan["bootstrap"]["n_boot"]
    tables = {m: p2.transfer_table(records[m], seed=seed, n_boot=n_boot) for m in METRICS}
    claims = {m: p2.headline_claims(tables[m], seed=seed, n_boot=n_boot) for m in METRICS}
    runmeta.write_json(out_dir / "transfer_table.json", {
        "split": "heldout", "eval_red_level": eval_red_level, "metrics": {
            m: {"cells": tables[m], "claims": claims[m]} for m in METRICS}})
    md = "# v2 protocol, Phase 1: transfer report\n\n" + (
        "> UNVERIFIED RUNS INCLUDED -- see the manifest's verification_problems.\n\n" if bad else "") + (
        f"Policies and baselines scored at red level {eval_red_level:g} on the {bank['version']} held-out split.\n\n"
    ) + "\n".join(p2.render_markdown(tables[m], claims[m], metric=m, split="heldout", meta=plan) for m in METRICS)
    (out_dir / "report.md").write_text(md)
    artifacts.update(transfer_table=out_dir / "transfer_table.json", report=out_dir / "report.md")
    manifest = p2.provenance_manifest(REPO, artifacts=artifacts, run_groups=groups,
                                      extra={"eval_red_level": eval_red_level, "plan_sha256": plan["plan_sha256"]})
    runmeta.write_json(out_dir / "protocol_v2_manifest.json", manifest)
    return manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    pl = sub.add_parser("plan", help="write the run matrix and print its local commands (runs nothing)")
    pl.add_argument("--out", required=True)
    pl.add_argument("--iterations", type=int, default=3000, help="hard cap per run")
    pl.add_argument("--block", type=int, default=50, help="multi-theatre iterations per AOI before rotating")
    pl.add_argument("--cell-m", type=float, default=CELL_M, choices=(CELL_M,),
                    help="physics grid cell size; Phase 1 is pinned to publishable 500 m fidelity")
    pl.add_argument("--seeds", type=int, nargs="+", default=list(p2.SEEDS))
    tm = sub.add_parser("train-multi", help="LOCAL multi-theatre run (long)")
    tm.add_argument("--seed", type=int, required=True)
    tm.add_argument("--iterations", type=int, default=3000)
    tm.add_argument("--block", type=int, default=50)
    tm.add_argument("--out", required=True)
    rp = sub.add_parser("report", help="evaluate finished runs and write the gated report")
    rp.add_argument("--plan", required=True)
    rp.add_argument("--runs", required=True)
    rp.add_argument("--out", required=True)
    rp.add_argument("--eval-red-level", type=float, default=1.0)
    rp.add_argument("--allow-unverified", action="store_true")
    a = ap.parse_args(argv)

    if a.cmd == "plan":
        plan = p2.plan_matrix(THEATRES, seeds=tuple(a.seeds), iterations=a.iterations, block_iterations=a.block,
                              cell_m=a.cell_m, root=str(Path(a.out).parent))
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        runmeta.write_json(a.out, plan)
        for r in plan["runs"]:
            print(" ".join(r["command"]))
        print(f"\n{len(plan['runs'])} runs, plan_sha256 {plan['plan_sha256'][:16]} -> {a.out}", file=sys.stderr)
    elif a.cmd == "train-multi":
        train_multi(a.out, seed=a.seed, iterations=a.iterations, block=a.block)
    else:
        m = build_report(json.loads(Path(a.plan).read_text()), a.runs, a.out,
                         eval_red_level=a.eval_red_level, allow_unverified=a.allow_unverified)
        print(f"wrote report to {a.out} (commit {m['code']['commit']}, dirty={m['code']['dirty']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
