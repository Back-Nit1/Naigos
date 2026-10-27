"""v2 training protocol, Phase 1: the pure parts.

Design: docs/superpowers/specs/2026-10-09-v2-training-protocol-phase1-design.md.
Everything here is stdlib (plus `runmeta`, itself stdlib-only, and the pure
`naigos.env.config`), so it imports without JAX and `train.py` can import the
plateau predicate without a cycle. The JAX half -- training, evaluation -- is
`scripts/run_protocol_v2.py`.

Nothing here runs a job. `plan_matrix` only describes the runs.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import random
import statistics
import sys
from pathlib import Path

from ..env import config as env_config
from . import runmeta

PROTOCOL = "naigos-protocol-v2-phase1"
MULTI = "multi"
SEEDS = (0, 1, 2)
#: A comparison is only asserted when its bootstrap interval excludes zero.
ALPHA = 0.05
N_BOOT = 10_000
#: the two reference controllers the learner is scored against, on the same scenarios
BASELINES = ("direct", "avoid_nap")
DEPENDENCIES = ("jax", "jaxlib", "flax", "optax", "orbax-checkpoint", "numpy")


# --- plateau ----------------------------------------------------------------


def plateau_reason(history: list, red_level: float = 1.0, min_evaluations: int = 3) -> str | None:
    """Why training has plateaued, or None.

    Plateau = the red curriculum itself has reported `held` at `red_level` for
    `min_evaluations` DISTINCT evaluations in a row. Reuses the curriculum's
    smoothing and evidence gate rather than a second statistic: a `held` at the
    top level can only be reported once the gate has been met since the level
    last moved, and `insufficient_evidence` / `promoted` / `demoted` all break
    the streak. Ticks that fall between evaluations repeat the same evaluation
    count, so they are not counted twice.

    `held` includes "smoothed survival is inside the dead band", so this is a
    plateau of the curriculum, not a claim that survival is high.
    """
    seen: list[int] = []
    for row in reversed(history):
        if "curriculum_reason" not in row:  # baseline row, or an eval with no tick yet
            continue
        if (
            row["curriculum_reason"] != "held"
            or abs(row.get("red_level_before", -1.0) - red_level) > 1e-9
            or abs(row.get("red_level_after", -1.0) - red_level) > 1e-9
        ):
            break
        n = row["curriculum_evals_total"]
        if n not in seen:
            seen.append(n)
        if len(seen) >= min_evaluations:
            return f"plateau: red level {red_level:g} held across {min_evaluations} evaluations"
    return None


# --- bootstrap --------------------------------------------------------------


def derived_seed(seed: int, label: str) -> int:
    """A per-comparison seed that depends only on (seed, label), not on call order,
    so adding a cell to the table does not move any other cell's interval."""
    return int(hashlib.sha256(f"{seed}|{label}".encode()).hexdigest()[:8], 16)


def bootstrap_ci(a, b=None, *, seed: int, n_boot: int = N_BOOT, alpha: float = ALPHA):
    """Percentile bootstrap CI of mean(a), or of mean(a) - mean(b) (independent resamples).

    Deterministic for a given `seed`. Returns None below two observations in `a`:
    one seed has no spread to resample. `b` may be a single constant (a baseline
    scored on fixed scenarios), which then contributes no resampling noise.

    At n=3 this is a rough interval, not a tight one: the percentile bootstrap
    under-covers on very small samples, so treat it as a necessary gate for a
    claim, not as proof of one.
    """
    a = [float(x) for x in a]
    b = None if b is None else [float(x) for x in b]
    if len(a) < 2 or (b is not None and not b):
        return None
    rng = random.Random(seed)
    draws = sorted(
        statistics.fmean(rng.choices(a, k=len(a))) - (0.0 if b is None else statistics.fmean(rng.choices(b, k=len(b))))
        for _ in range(n_boot)
    )
    return draws[int(alpha / 2 * n_boot)], draws[int((1 - alpha / 2) * n_boot) - 1]


# --- the run matrix ---------------------------------------------------------


def plan_matrix(
    theatres, seeds=SEEDS, *, iterations: int = 3000, block_iterations: int = 50,
    cell_m: float = 500.0, root: str = "runs/protocol_v2", bootstrap_seed: int = 20261009,
) -> dict:
    """The reproducible run matrix: one policy per (theatre, seed), plus one multi-theatre
    policy per seed. A description only -- it submits nothing, and `commands` are
    LOCAL commands. The Modal path does not carry the route/edge/plateau flags.
    """
    from .train import assert_route_bank_disjoint  # lazy: pulls in JAX

    assert_route_bank_disjoint(seeds)
    flags = [
        "--cell-m", str(cell_m), "--route-mode", "diverse", "--route-eval",
        "--map-randomize", "--edge-obs", "--plateau",
    ]
    runs = []
    for seed in seeds:
        for t in theatres:
            name = f"p2-{t}-s{seed}"
            runs.append({
                "name": name, "kind": "single", "theatres": [t], "seed": seed, "iterations_cap": iterations,
                "command": ["python", "scripts/train_theatre.py", "--aoi", t, "--seed", str(seed),
                            "--iterations", str(iterations), "--out", f"{root}/{name}", *flags],
            })
        name = f"p2-{MULTI}-s{seed}"
        runs.append({
            "name": name, "kind": MULTI, "theatres": list(theatres), "seed": seed, "iterations_cap": iterations,
            "block_iterations": block_iterations,
            "command": ["python", "scripts/run_protocol_v2.py", "train-multi", "--seed", str(seed),
                        "--iterations", str(iterations), "--block", str(block_iterations),
                        "--out", f"{root}/{name}"],
        })
    plan = {
        "protocol": PROTOCOL,
        "cell_m": float(cell_m),
        "route_bank": {"version": env_config.ROUTE_BANK_VERSION, "sha256": env_config.ROUTE_BANK_V1_SHA256},
        "red_curriculum": "RedCurriculum() defaults", "plateau": {"red_level": 1.0, "min_evaluations": 3},
        "bootstrap": {"seed": bootstrap_seed, "n_boot": N_BOOT, "alpha": ALPHA},
        "eval_theatres": list(theatres), "runs": runs,
    }
    plan["plan_sha256"] = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    return plan


# --- transfer table ---------------------------------------------------------


def transfer_table(records: list[dict], *, seed: int, n_boot: int = N_BOOT) -> dict:
    """`records`: {train, eval, seed, learner, direct, avoid_nap} -- one row per
    (policy, eval theatre), all scored on the eval theatre's frozen route bank.

    The baselines are deterministic on a fixed scenario set, so every policy
    evaluated on one theatre must see the same baseline value; if not, the
    scenarios were not identical and the table is refused.
    """
    cells: dict[str, list[dict]] = {}
    for r in records:
        cells.setdefault(f"{r['train']}|{r['eval']}", []).append(r)
    base_seen: dict[tuple, float] = {}
    out = {}
    for key in sorted(cells):
        rows = sorted(cells[key], key=lambda r: r["seed"])
        for r in rows:
            for b in BASELINES:
                if abs(base_seen.setdefault((r["eval"], b), r[b]) - r[b]) > 1e-9:
                    raise ValueError(f"{b} baseline differs across policies on {r['eval']}: scenarios not identical")
        vals = [r["learner"] for r in rows]
        ci = bootstrap_ci(vals, seed=derived_seed(seed, key), n_boot=n_boot)
        out[key] = {
            "train": rows[0]["train"], "eval": rows[0]["eval"], "seeds": [r["seed"] for r in rows],
            "policy_seed_values": vals, "policy_mean": statistics.fmean(vals),
            "policy_ci95": None if ci is None else list(ci),
            "direct_mean": rows[0]["direct"], "avoid_nap_mean": rows[0]["avoid_nap"],
        }
    return out


def headline_claims(table: dict, *, seed: int, n_boot: int = N_BOOT, alpha: float = ALPHA) -> list[dict]:
    """Every comparative sentence the report could assert, each with its effect and CI.

    `asserted` is True only when the interval excludes zero; otherwise the entry
    carries the raw numbers and `text` says no winner is asserted. Claims tested
    here are uncorrected for multiplicity -- the report states the count.
    """
    claims = []

    def add(cid: str, subject: str, other: str, a, b):
        ci = bootstrap_ci(a, b, seed=derived_seed(seed, cid), n_boot=n_boot, alpha=alpha)
        effect = statistics.fmean(a) - statistics.fmean(b)
        n = len(a)
        nums = f"effect {effect:+.3f}" + ("" if ci is None else f", 95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}]")
        if ci is not None and ci[0] > 0:
            verdict, text = True, f"{subject} beats {other} ({nums}, n={n} seeds)"
        elif ci is not None and ci[1] < 0:
            verdict, text = True, f"{subject} is worse than {other} ({nums}, n={n} seeds)"
        else:
            verdict, text = False, f"{subject} vs {other}: not significant at n={n} seeds ({nums}); no winner asserted"
        claims.append({"id": cid, "effect": effect, "ci95": None if ci is None else list(ci),
                       "asserted": verdict, "text": text})

    for key, c in table.items():
        for b in BASELINES:
            add(f"learner_vs_{b}@{key}", f"policy trained on {c['train']}, eval {c['eval']},", f"{b} route",
                c["policy_seed_values"], [c[f"{b}_mean"]])
    for ev in sorted({c["eval"] for c in table.values()}):
        multi = table.get(f"{MULTI}|{ev}")
        other = [v for c in table.values() if c["eval"] == ev and c["train"] not in (MULTI, ev)
                 for v in c["policy_seed_values"]]
        if multi and other:
            add(f"multi_vs_single_transfer@{ev}", f"multi-theatre training on {ev}", "single-theatre training elsewhere",
                multi["policy_seed_values"], other)
    return claims


def render_markdown(table: dict, claims: list[dict], *, metric: str, split: str, meta: dict) -> str:
    """Transfer table (rows: trained on, columns: evaluated on) and the gated claims."""
    trains = sorted({c["train"] for c in table.values()}, key=lambda t: (t == MULTI, t))
    evals = sorted({c["eval"] for c in table.values()})
    lines = [
        f"## Transfer: {split} {metric}", "",
        f"Cell = learner mean [95% CI] / direct / avoid+nap, same frozen scenarios ({meta['route_bank']['version']}, "
        f"sha256 {meta['route_bank']['sha256'][:12]}).",
        "",
        "| trained on \\ eval | " + " | ".join(evals) + " |",
        "|---|" + "---|" * len(evals),
    ]
    for t in trains:
        cells = []
        for e in evals:
            c = table.get(f"{t}|{e}")
            if c is None:
                cells.append("-")
                continue
            ci = "n/a" if c["policy_ci95"] is None else f"[{c['policy_ci95'][0]:.3f}, {c['policy_ci95'][1]:.3f}]"
            cells.append(f"{c['policy_mean']:.3f} {ci} / {c['direct_mean']:.3f} / {c['avoid_nap_mean']:.3f}")
        lines.append(f"| {t} | " + " | ".join(cells) + " |")
    n_asserted = sum(c["asserted"] for c in claims)
    lines += [
        "", f"### Claims ({n_asserted} supported of {len(claims)} tested)", "",
        "Intervals are percentile bootstrap over training seeds (rough at n=3, under-covering) and uncorrected for "
        f"multiplicity: at alpha={ALPHA}, expect about {ALPHA * len(claims):.1f} false positives among {len(claims)} claims "
        "if nothing were true. A claim is stated only when its interval excludes zero.", "",
    ]
    lines += [f"- {'SUPPORTED' if c['asserted'] else 'withheld'}: {c['text']}" for c in claims]
    return "\n".join(lines) + "\n"


# --- provenance -------------------------------------------------------------


def _version(dist: str) -> str | None:
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return None


def artifact_entry(path: str | Path) -> dict:
    """path + sha256 (content hash of the tree, for an Orbax checkpoint directory)."""
    from ..pipeline import layout  # lazy: keeps this module's import cheap

    return {"path": str(path), "sha256": layout.sha256_path(path)}


def provenance_manifest(repo: str | Path, *, artifacts: dict[str, str | Path], run_groups: dict | None = None,
                        extra: dict | None = None) -> dict:
    """Code commit + dirty flag, dependency versions, uv.lock digest, route-bank digest,
    and path+sha256 for every artifact this protocol produced.

    A missing artifact raises: a manifest that names a file it could not hash is
    worse than none.
    """
    repo = Path(repo)
    lock = repo / "uv.lock"
    bank = env_config.route_bank()  # raises if the frozen bank no longer hashes to its pin
    return {
        "protocol": PROTOCOL,
        "code": runmeta.git_info(repo),
        "python": sys.version.split()[0],
        "host": platform.platform(),
        "dependencies": {d: _version(d) for d in DEPENDENCIES},
        "uv_lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest() if lock.exists() else None,
        "route_bank": {"version": env_config.ROUTE_BANK_VERSION, "sha256": env_config.ROUTE_BANK_V1_SHA256,
                       "recomputed_sha256": env_config.route_bank_digest(bank), "n_scenarios": len(bank)},
        "run_groups": run_groups or {},
        "artifacts": {k: artifact_entry(v) for k, v in artifacts.items()},
        **(extra or {}),
    }
