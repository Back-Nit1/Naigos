"""Deterministic CMDP verifier -- pure NumPy, zero JAX.

Nomos's rule, kept verbatim: **verify the trace, don't re-simulate it.** The
verifier is handed the logged states (positions, headings, threat positions, the
DEM) and independently recomputes every hard-constraint quantity: terrain
line-of-sight, detection probability, track build-up, lethal-envelope dwell,
terrain/ceiling/stall/bounds violations. It then compares its own numbers with
what the environment reported.

It shares no code with the JAX env by design. If a JAX rewrite quietly changes a
broadcast, a clamp or a unit, this catches it -- which a re-simulation using the
same code path never would.

Because it is pure NumPy it runs standalone: `pytest tests/test_verifier.py`
needs no accelerator, no jit warm-up and no env import beyond the config
dataclasses.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

from ..env.config import DetectionConfig, EnvConfig, TerrainConfig, ThreatKindConfig

_LOG10 = np.log(10.0)


# --------------------------------------------------------------------- terrain
def sample_height(hmap: np.ndarray, tcfg: TerrainConfig, x, y) -> np.ndarray:
    fx = np.clip(np.asarray(x) / tcfg.cell, 0.0, tcfg.nx - 1.0)
    fy = np.clip(np.asarray(y) / tcfg.cell, 0.0, tcfg.ny - 1.0)
    x0 = np.floor(fx).astype(int)
    y0 = np.floor(fy).astype(int)
    x1 = np.minimum(x0 + 1, tcfg.nx - 1)
    y1 = np.minimum(y0 + 1, tcfg.ny - 1)
    tx, ty = fx - x0, fy - y0
    top = hmap[y0, x0] * (1 - tx) + hmap[y0, x1] * tx
    bot = hmap[y1, x0] * (1 - tx) + hmap[y1, x1] * tx
    return top * (1 - ty) + bot * ty


def los_clearance(hmap, tcfg: TerrainConfig, dcfg: DetectionConfig, p_from, p_to) -> np.ndarray:
    p_from = np.asarray(p_from, dtype=np.float64)
    p_to = np.asarray(p_to, dtype=np.float64)
    s = (np.arange(dcfg.los_samples) + 0.5) / dcfg.los_samples
    s = s.reshape((1,) * (p_from.ndim - 1) + (dcfg.los_samples,))
    seg = p_to - p_from
    px = p_from[..., None, 0] + s * seg[..., None, 0]
    py = p_from[..., None, 1] + s * seg[..., None, 1]
    pz = p_from[..., None, 2] + s * seg[..., None, 2]
    ground = sample_height(hmap, tcfg, px, py)
    d_h = np.sqrt(np.sum(seg[..., :2] ** 2, axis=-1))[..., None]
    drop = (s * d_h) * ((1.0 - s) * d_h) / (2.0 * dcfg.earth_radius_eff)
    return np.min((pz - drop) - ground, axis=-1)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


# ------------------------------------------------------------------- detection
def detection_probability(hmap, cfg: EnvConfig, blue_pos, blue_psi, threat_pos, kp, active):
    """(T, B) detection probability, recomputed from first principles."""
    dcfg = cfg.detection
    bp = np.asarray(blue_pos, dtype=np.float64)[None, :, :]
    tp = np.asarray(threat_pos, dtype=np.float64)[:, None, :]
    d = bp - tp
    slant = np.sqrt(np.sum(d**2, axis=-1) + 1e-6)

    bearing = np.arctan2(-d[..., 1], -d[..., 0])
    aspect = bearing - np.asarray(blue_psi)[None, :]
    w = 0.5 * (1.0 - np.cos(2.0 * aspect))
    rcs = dcfg.rcs_head_on + w * (dcfg.rcs_beam - dcfg.rcs_head_on)

    snr = (
        kp["snr_ref_db"][:, None]
        + 40.0 * np.log(kp["detect_range"][:, None] / np.maximum(slant, 1.0)) / _LOG10
        + 10.0 * np.log(np.maximum(rcs / dcfg.rcs_head_on, 1e-12)) / _LOG10
    )
    pd_snr = _sigmoid(kp["snr_logistic_k"][:, None] * (snr - kp["snr_threshold_db"][:, None]))

    emitter = np.asarray(threat_pos, dtype=np.float64).copy()
    emitter[:, 2] += 10.0
    vis = _sigmoid(
        los_clearance(
            hmap, cfg.terrain, dcfg, emitter[:, None, :], np.broadcast_to(bp, d.shape)
        )
        / dcfg.los_clearance_scale
    )

    ground = sample_height(hmap, cfg.terrain, np.asarray(blue_pos)[:, 0], np.asarray(blue_pos)[:, 1])
    alt_agl = np.asarray(blue_pos)[:, 2] - ground
    lo = _sigmoid((alt_agl[None, :] - kp["alt_min"][:, None]) / np.maximum(0.25 * kp["alt_min"][:, None], 25.0))
    hi = _sigmoid((kp["alt_max"][:, None] - np.asarray(blue_pos)[None, :, 2]) / 500.0)

    pd = pd_snr * vis * lo * hi * np.asarray(active, dtype=np.float64)[:, None]
    return pd, slant, vis, alt_agl


def kind_params_numpy(cfg: EnvConfig, kind: np.ndarray) -> dict[str, np.ndarray]:
    # numeric fields only: `label` is a string and would poison the float table.
    fields = [f.name for f in dataclasses.fields(ThreatKindConfig) if f.type in ("float", "bool")]
    table = {f: np.array([float(getattr(k, f)) for k in cfg.threat_kinds], dtype=np.float64) for f in fields}
    table["detect_range"] *= cfg.red_detect_scale
    table["lethal_range"] *= cfg.red_lethal_scale
    table["reaction_latency"] *= cfg.red_latency_scale
    table["speed"] *= cfg.red_speed_scale
    return {f: table[f][np.asarray(kind)] for f in table}


# ------------------------------------------------------------------ the report
@dataclasses.dataclass
class VerificationReport:
    steps: int
    shootdowns: int
    terrain_violations: int
    bounds_violations: int
    fuel_losses: int
    diversions: int
    ceiling_violations: int
    envelope_dwell_steps: int
    total_cost: float  # terminal airframe losses only; matches reward.constraint_cost
    mean_exposure: float
    max_pd_error: float
    mismatches: list[str]
    raw_loss_events: int = 0  # L1: loss flags before exclusive-outcome precedence

    @property
    def ok(self) -> bool:
        return not self.mismatches

    def __str__(self) -> str:
        head = (
            f"steps={self.steps} shootdowns={self.shootdowns} "
            f"terrain={self.terrain_violations} bounds={self.bounds_violations} "
            f"fuel={self.fuel_losses} diversions={self.diversions} "
            f"ceiling={self.ceiling_violations} envelope_dwell={self.envelope_dwell_steps} "
            f"cost={self.total_cost:.2f} mean_pd={self.mean_exposure:.4f} "
            f"max_pd_err={self.max_pd_error:.2e}"
        )
        if self.mismatches:
            return head + "\nMISMATCHES:\n  " + "\n  ".join(self.mismatches)
        return head + "\nOK"


def verify_trace(
    cfg: EnvConfig,
    hmap: np.ndarray,
    traj: dict[str, Any],
    threat_kind: np.ndarray,
    threat_active: np.ndarray,
    tol: float = 2e-3,
) -> VerificationReport:
    """Recheck one logged episode.

    `traj` needs, on a leading time axis: `pos` (S, B, 3), `psi` (S, B),
    `speed` (S, B), `alive` (S, B), `threat_pos` (S, T, 3), `lock` (S, T, B),
    and `terms` (the env's own RewardTerms) for cross-checking.
    """
    pos = np.asarray(traj["pos"], dtype=np.float64)
    psi = np.asarray(traj["psi"], dtype=np.float64)
    speed = np.asarray(traj["speed"], dtype=np.float64)
    alive = np.asarray(traj["alive"])
    tpos = np.asarray(traj["threat_pos"], dtype=np.float64)
    env_lock = np.asarray(traj["lock"], dtype=np.float64)
    terms = traj["terms"]

    S, B, _ = pos.shape
    kp = kind_params_numpy(cfg, threat_kind)
    af = cfg.airframe

    mismatches: list[str] = []
    shoot = terr = bounds = fuel_losses = diversions = ceil = dwell_steps = 0
    pd_sum = 0.0
    pd_n = 0
    max_pd_err = 0.0
    lock = np.zeros((cfg.n_threat, B))
    dwell = np.zeros((cfg.n_threat, B))
    reached = np.zeros(B, dtype=bool)
    diverted = np.zeros(B, dtype=bool)
    prev_alive = np.ones(B, dtype=bool)
    raw_losses = cost_units = 0

    for s in range(S):
        live = prev_alive & ~reached & ~diverted
        pd, slant, _vis, alt_agl = detection_probability(
            hmap, cfg, pos[s], psi[s], tpos[s], kp, threat_active
        )
        pd = pd * live[None, :]

        lock = np.clip(
            lock + cfg.dt * (kp["lock_gain"][:, None] * pd - kp["lock_decay"][:, None] * (1 - pd) * lock),
            0.0,
            1.0,
        )
        locked = lock >= cfg.detection.lock_threshold
        dwell = np.where(locked, dwell + cfg.dt, 0.0)

        in_env = (slant <= kp["lethal_range"][:, None]) & (
            (alt_agl[None, :] >= kp["alt_min"][:, None]) & (pos[s][None, :, 2] <= kp["alt_max"][:, None])
        )
        firing = in_env & locked & (dwell >= kp["reaction_latency"][:, None])

        # --- independent recomputation of the hard constraints -----------------
        terr_v = (alt_agl < af.floor_agl) & live
        box = np.asarray(traj.get("bounds", [0, cfg.terrain.extent_x, 0, cfg.terrain.extent_y]))
        box = box[s] if box.ndim == 2 else box
        oob_v = (
            (pos[s][:, 0] < box[0])
            | (pos[s][:, 0] > box[1])
            | (pos[s][:, 1] < box[2])
            | (pos[s][:, 1] > box[3])
        ) & live
        ceil_v = (pos[s][:, 2] >= af.ceiling - 1e-3) & live
        dwell_v = np.any(in_env & (lock > 0.05), axis=0) & live

        ceil += int(ceil_v.sum())
        dwell_steps += int(dwell_v.sum())

        now_alive = np.asarray(alive[s], dtype=bool)
        died = prev_alive & ~now_alive
        dry_v = np.asarray(terms.out_of_fuel[s], dtype=bool) if cfg.resource.enabled else np.zeros(B, bool)
        divert_v = np.asarray(traj.get("diverted", np.zeros_like(alive))[s], dtype=bool) & ~diverted
        if cfg.resource.enabled:
            # Exclusive first outcome, same precedence as env.resource.OUTCOME_PRECEDENCE
            # (shot > terrain > bounds > fuel); raw simultaneous flags are kept as evidence.
            killed_by_fire = np.asarray(terms.shotdown[s], dtype=bool) & live
            terr_x = terr_v & ~killed_by_fire
            oob_x = oob_v & ~killed_by_fire & ~terr_x
            dry_x = dry_v & ~killed_by_fire & ~terr_x & ~oob_x
            raw_losses += int(killed_by_fire.sum() + terr_v.sum() + oob_v.sum() + dry_v.sum())
            if not np.array_equal(died, killed_by_fire | terr_x | oob_x | dry_x):
                mismatches.append(f"t={s}: alive transition does not match the exclusive loss outcome")
            cost_units += int(died.sum())
        else:
            terr_x, oob_x, dry_x = terr_v, oob_v, dry_v
            killed_by_fire = died & ~terr_v & ~oob_v & ~dry_v
        terr += int(terr_x.sum())
        bounds += int(oob_x.sum())
        shoot += int(killed_by_fire.sum())
        fuel_losses += int(dry_x.sum())
        diversions += int(divert_v.sum())

        # A kill can only be logged where a threat actually had a firing
        # solution. This is the check that catches a phantom-shootdown bug.
        for b in np.nonzero(killed_by_fire)[0]:
            if not firing[:, b].any():
                mismatches.append(f"t={s} agent={b}: logged shootdown with no firing solution")

        # cross-check the env's own reported quantities
        err = np.abs(np.max(pd, axis=0) - np.asarray(terms.exposure[s], dtype=np.float64))
        err = np.where(live, err, 0.0)
        max_pd_err = max(max_pd_err, float(err.max()))
        if err.max() > tol:
            mismatches.append(f"t={s}: exposure mismatch, max |dpd|={err.max():.3e} > tol")

        lock_err = np.abs(lock - env_lock[s])[:, live].max() if live.any() else 0.0
        if lock_err > 10 * tol:
            mismatches.append(f"t={s}: lock mismatch, max |dlock|={lock_err:.3e}")

        stall_v = (speed[s] < af.v_stall - 1e-3) & live
        if stall_v.any():
            mismatches.append(f"t={s}: airspeed below stall on agents {np.nonzero(stall_v)[0].tolist()}")

        pd_sum += float(pd[:, live].max(axis=0).sum()) if live.any() else 0.0
        pd_n += int(live.sum())

        reached = reached | np.asarray(traj["reached"][s], dtype=bool)
        diverted = diverted | divert_v
        prev_alive = now_alive

    # The CMDP cost channel is TERMINAL ONLY -- it must match
    # naigos.rl.reward.constraint_cost exactly, or the verifier is checking a
    # different constraint from the one being optimised. Lethal-envelope dwell
    # is reported separately as a diagnostic, not summed in.
    total_cost = float(shoot + terr + bounds)
    if cfg.resource.enabled:
        # one unit per lost aircraft, however many loss flags fired together
        total_cost = float(cost_units)
    return VerificationReport(
        steps=S,
        shootdowns=shoot,
        terrain_violations=terr,
        bounds_violations=bounds,
        fuel_losses=fuel_losses,
        diversions=diversions,
        ceiling_violations=ceil,
        envelope_dwell_steps=dwell_steps,
        total_cost=total_cost,
        mean_exposure=pd_sum / max(pd_n, 1),
        max_pd_error=max_pd_err,
        mismatches=mismatches,
        raw_loss_events=raw_losses,
    )


@dataclasses.dataclass
class ResourceVerificationReport:
    """Independent L1 fuel, gate, outcome, and actor-observation checks."""

    steps: int
    fuel_losses: int
    diversions: int
    max_fuel_error: float
    max_observation_error: float
    mismatches: list[str]
    total_cost: float = 0.0  # exclusive losses; equals reward.constraint_cost summed

    @property
    def ok(self) -> bool:
        return not self.mismatches


_LATENTS = ("initial_fuel", "reserve_fuel", "burn_multiplier", "diversion_pos", "diversion_active", "objective")


def _latent_problems(cfg: EnvConfig, traj: dict[str, Any], start, tol: float = 1e-3) -> list[str]:
    """Episode latents/gates must be constant in time and drawn inside the configured ranges."""
    problems = []
    for name in _LATENTS:
        a = np.asarray(traj[name], dtype=np.float64)
        if np.ptp(a, axis=0).max(initial=0.0) > 0.0:
            problems.append(f"{name} changes during the episode")
    if start is None:
        return problems
    af, rc = cfg.airframe, cfg.resource
    start = np.asarray(start, dtype=np.float64)
    objective = np.asarray(traj["objective"], dtype=np.float64)[0]
    throttle = np.clip((af.v_init - af.v_stall) / (af.v_max - af.v_stall), 0.0, 1.0)
    direct_fuel = (
        np.linalg.norm(objective[:, :2] - start[:, :2], axis=-1) / af.v_init
        * (af.burn_base + af.burn_throttle * throttle)
    )
    draws = {
        "fuel_budget_ratio": np.asarray(traj["initial_fuel"], dtype=np.float64)[0] / direct_fuel,
        "reserve_ratio": np.asarray(traj["reserve_fuel"], dtype=np.float64)[0] / direct_fuel,
        "burn_multiplier": np.asarray(traj["burn_multiplier"], dtype=np.float64)[0],
    }
    for name, value in draws.items():
        lo, hi = getattr(rc, name)
        if np.any(value < lo * (1 - tol)) or np.any(value > hi * (1 + tol)):
            problems.append(f"{name} {np.round(value, 4).tolist()} outside configured range [{lo}, {hi}]")
    if not np.array_equal(np.asarray(traj["diversion_active"])[0], np.broadcast_to(rc.active_gate_mask, (len(start), len(rc.active_gate_mask)))):
        problems.append("diversion_active differs from resource.active_gate_mask")
    box = np.asarray(traj["bounds"], dtype=np.float64)
    box = box[0] if box.ndim == 2 else box
    gates = np.asarray(traj["diversion_pos"], dtype=np.float64)[0]
    inside = (
        (gates[..., 0] >= box[0]) & (gates[..., 0] <= box[1])
        & (gates[..., 1] >= box[2]) & (gates[..., 1] <= box[3])
    )
    if not inside.all():
        problems.append("a recovery gate lies outside the play area")
    return problems


def _reward_cost_problems(terms, outcomes, weights: dict, reward, cost, tol: float = 1e-3) -> list[str]:
    """Recompute reward and CMDP cost in NumPy and compare with the production values.

    ``weights`` is ``dataclasses.asdict(RewardWeights)``; ``reward``/``cost`` are the
    unmasked ``reward.compute(terms, w)`` and ``reward.constraint_cost(terms)`` arrays
    (S, B). Also proves each terminal outcome is paid or charged exactly once.
    """
    f = lambda name: np.asarray(getattr(terms, name), dtype=np.float64)  # noqa: E731
    building = (f("building_collision") if hasattr(terms, "building_collision")
                else np.zeros_like(f("shotdown")))
    lost = np.clip(
        f("shotdown") + building + f("terrain_violation") + f("bounds_violation") + f("out_of_fuel"),
        0.0,
        1.0,
    )
    w = weights
    expected = (
        w["progress"] * f("progress") / 1_000.0 + w["arrived"] * f("arrived") + w["diverted"] * f("diverted")
        - w["exposure"] * f("exposure") - w["lock"] * f("lock_level") - w["envelope"] * f("envelope_dwell")
        - w["aircraft_loss"] * lost - w["fuel"] * f("fuel_used") - w["g_excess"] * np.maximum(f("g_excess"), 0.0)
        - w["step_cost"] - w["ceiling"] * f("ceiling_violation") - w["stall"] * f("stall_violation")
        - w["edge"] * f("edge_proximity")
    )
    problems = []
    if reward is not None and np.abs(expected - np.asarray(reward, dtype=np.float64)).max() > tol:
        problems.append("reward does not match the NumPy recomputation from RewardTerms")
    if cost is not None and np.abs(lost - np.asarray(cost, dtype=np.float64)).max() > tol:
        problems.append("CMDP cost does not match terminal-loss indicator")
    # exactly-once: payments/charges happen on the step the exclusive outcome is set
    first = np.concatenate([np.zeros_like(outcomes[:1]), outcomes[:-1]], axis=0) == 0
    first &= outcomes != 0
    for name, codes in (("arrived", (1,)), ("diverted", (2,)), ("loss", (3, 4, 5, 6, 8))):
        flag = lost > 0 if name == "loss" else f(name) > 0
        if not np.array_equal(flag, first & np.isin(outcomes, codes)):
            problems.append(f"{name} term is not paid exactly once on the first matching terminal outcome")
    if lost.sum(axis=0).max(initial=0.0) > 1.0:
        problems.append("an aircraft is charged CMDP cost more than once")
    return problems


def _resource_observation_numpy(cfg: EnvConfig, pos, psi, speed, fuel, objective,
                                initial, reserve, gates, active):
    """Independent NumPy implementation of ResourceObservation-v1."""
    af = cfg.airframe

    onboard_multiplier = cfg.resource.onboard_burn_multiplier

    def rate(v):
        throttle = np.clip((v - af.v_stall) / (af.v_max - af.v_stall), 0.0, 1.0)
        return (af.burn_base + af.burn_throttle * throttle) * onboard_multiplier

    scale = np.maximum(initial, 1e-3)
    direct_distance = np.linalg.norm(objective[:, :2] - pos[:, :2], axis=-1)
    primary_need = direct_distance / np.maximum(speed, af.v_stall) * rate(speed)
    gate_distance = np.linalg.norm(gates[..., :2] - pos[:, None, :2], axis=-1)
    gate_need = gate_distance / np.maximum(speed[:, None], af.v_stall) * rate(speed)[:, None]
    burn = rate(speed)
    best_margin = np.max(
        np.where(active, fuel[:, None] - reserve[:, None] - gate_need, -1e9), axis=1
    )
    scalar = np.stack([
        fuel / scale,
        reserve / scale,
        np.clip(fuel / np.maximum(burn * cfg.dt * cfg.max_steps, 1e-3), 0.0, 4.0),
        (fuel - reserve - primary_need) / scale,
        best_margin / scale,
        burn / max(af.burn_base + af.burn_throttle, 1e-3),
    ], axis=-1)

    rel = gates - pos[:, None, :]
    c, s = np.cos(psi)[:, None], np.sin(psi)[:, None]
    rel_x = rel[..., 0] * c + rel[..., 1] * s
    rel_y = -rel[..., 0] * s + rel[..., 1] * c
    bearing = np.arctan2(rel_y, rel_x)
    gate = np.stack([
        rel_x / 50_000.0,
        rel_y / 50_000.0,
        gate_distance / 90_000.0,
        np.sin(bearing),
        np.cos(bearing),
        (fuel[:, None] - reserve[:, None] - gate_need) / scale[:, None],
        active.astype(np.float64),
    ], axis=-1)
    gate *= active[..., None]
    return np.concatenate([scalar, gate.reshape(pos.shape[0], -1)], axis=-1)


def verify_resource_trace(
    cfg: EnvConfig,
    traj: dict[str, Any],
    tol: float = 3e-3,
    *,
    start=None,
    weights: dict | None = None,
    reward=None,
    cost=None,
) -> ResourceVerificationReport:
    """Verify L1 from logged state without importing JAX resource code.

    The log carries realized episode samples so this routine checks transitions,
    not the pseudorandom generator. Seed/provenance checks belong to the frozen
    resource-bank contract.
    """
    if not cfg.resource.enabled:
        raise ValueError("resource trace verification requires resource-world-v1/endurance")

    pos = np.asarray(traj["pos"], dtype=np.float64)
    psi = np.asarray(traj["psi"], dtype=np.float64)
    speed_log = np.asarray(traj["speed"], dtype=np.float64)
    phi_log = np.asarray(traj["phi"], dtype=np.float64)
    fuel_log = np.asarray(traj["fuel"], dtype=np.float64)
    actions = np.asarray(traj["action"], dtype=np.float64)
    initial = np.asarray(traj["initial_fuel"], dtype=np.float64)
    reserve = np.asarray(traj["reserve_fuel"], dtype=np.float64)
    multiplier = np.asarray(traj["burn_multiplier"], dtype=np.float64)
    gates = np.asarray(traj["diversion_pos"], dtype=np.float64)
    active = np.asarray(traj["diversion_active"], dtype=bool)
    objective = np.asarray(traj["objective"], dtype=np.float64)
    observed = np.asarray(traj["resource_observation"], dtype=np.float64)
    outcomes = np.asarray(traj["outcome"], dtype=np.uint8)
    diverted_log = np.asarray(traj["diverted"], dtype=bool)
    terms = traj["terms"]

    steps, agents = fuel_log.shape
    fuel = initial[0].copy()
    speed = np.full(agents, cfg.airframe.v_init, dtype=np.float64)
    phi = np.zeros(agents, dtype=np.float64)
    terminal = np.zeros(agents, dtype=bool)
    expected_outcome = np.zeros(agents, dtype=np.uint8)
    max_fuel_error = 0.0
    max_obs_error = 0.0
    fuel_losses = diversions = 0
    mismatches: list[str] = []
    af = cfg.airframe

    for t in range(steps):
        was_fuel = fuel.copy()
        was_terminal = terminal.copy()
        for _ in range(cfg.substeps):
            act = np.clip(actions[t], -1.0, 1.0)
            phi_cmd = act[:, 0] * np.arccos(1.0 / af.n_max)
            phi = phi + (phi_cmd - phi) * min((cfg.dt / cfg.substeps) / af.tau_bank, 1.0)
            speed_cmd = af.v_stall + 0.5 * (act[:, 2] + 1.0) * (af.v_max - af.v_stall)
            speed = speed + (speed_cmd - speed) * min((cfg.dt / cfg.substeps) / af.tau_speed, 1.0)
            speed = np.clip(speed, af.v_stall, af.v_max)
            throttle = (speed - af.v_stall) / (af.v_max - af.v_stall)
            load = 1.0 / np.maximum(np.cos(phi), 1e-3)
            burn = af.burn_base + af.burn_throttle * throttle + af.burn_maneuver * np.abs(load - 1.0)
            fuel = np.maximum(fuel - (cfg.dt / cfg.substeps) * burn * multiplier[t], 0.0)
            dry_substep = fuel <= 0.0
            speed = np.where(dry_substep, np.maximum(speed - 5.0 * (cfg.dt / cfg.substeps), af.v_stall), speed)
        fuel = np.where(was_terminal, was_fuel, fuel)
        speed = np.where(was_terminal, speed_log[t], speed)
        phi = np.where(was_terminal, phi_log[t], phi)

        ferr = np.abs(fuel - fuel_log[t])
        max_fuel_error = max(max_fuel_error, float(ferr.max()))
        if ferr.max() > tol:
            mismatches.append(f"t={t}: fuel mismatch, max error {ferr.max():.3e}")

        dry = (was_fuel > 0.0) & (fuel <= 0.0) & ~was_terminal
        env_dry = np.asarray(terms.out_of_fuel[t], dtype=bool)
        if not np.array_equal(dry, env_dry):
            mismatches.append(f"t={t}: out_of_fuel does not match first zero crossing")

        arrived = np.asarray(terms.arrived[t], dtype=bool)
        shot = np.asarray(terms.shotdown[t], dtype=bool)
        terrain = np.asarray(terms.terrain_violation[t], dtype=bool)
        building = np.asarray(getattr(terms, "building_collision", np.zeros_like(terms.shotdown))[t], dtype=bool)
        bounds = np.asarray(terms.bounds_violation[t], dtype=bool)
        gate_distance = np.linalg.norm(gates[t, ..., :2] - pos[t, :, None, :2], axis=-1)
        at_gate = np.any((gate_distance <= cfg.resource.gate_arrival_radius) & active[t], axis=-1)
        expected_divert = at_gate & ~arrived & ~shot & ~building & ~terrain & ~bounds & ~dry & ~was_terminal
        env_divert = np.asarray(terms.diverted[t], dtype=bool)
        if not np.array_equal(expected_divert, env_divert):
            mismatches.append(f"t={t}: diversion does not match first active-gate crossing")

        # Independent copy of the documented precedence (env.resource.
        # OUTCOME_PRECEDENCE): shot > building > terrain > bounds > fuel > primary > divert.
        new = np.zeros(agents, dtype=np.uint8)
        for code, flag in (
            (2, expected_divert), (1, arrived), (6, dry), (5, bounds), (4, terrain), (8, building), (3, shot)
        ):
            new[flag] = code
        timeout = (t + 1 >= cfg.max_steps) & ~was_terminal & (new == 0)
        new[timeout] = 7
        expected_outcome = np.where(expected_outcome == 0, new, expected_outcome)
        if not np.array_equal(expected_outcome, outcomes[t]):
            mismatches.append(f"t={t}: terminal outcome label mismatch")

        expected_obs = _resource_observation_numpy(
            cfg, pos[t], psi[t], speed_log[t], fuel_log[t], objective[t], initial[t],
            reserve[t], gates[t], active[t],
        )
        oerr = np.abs(expected_obs - observed[t])
        max_obs_error = max(max_obs_error, float(oerr.max()))
        if oerr.max() > 10 * tol:
            mismatches.append(f"t={t}: resource observation mismatch, max error {oerr.max():.3e}")

        fuel_losses += int(dry.sum())
        diversions += int(expected_divert.sum())
        terminal |= new != 0
        fuel = fuel_log[t].copy()
        speed = speed_log[t].copy()
        phi = phi_log[t].copy()

        if not np.array_equal(diverted_log[t], expected_outcome == 2):
            mismatches.append(f"t={t}: cumulative diverted flag disagrees with outcome")

    mismatches += _latent_problems(cfg, traj, start)
    if weights is not None:
        mismatches += _reward_cost_problems(terms, outcomes, weights, reward, cost)
    lost = np.clip(
        sum(np.asarray(getattr(terms, n), dtype=np.float64)
            for n in ("shotdown", "terrain_violation", "building_collision", "bounds_violation", "out_of_fuel")),
        0.0,
        1.0,
    )
    return ResourceVerificationReport(
        steps=steps,
        fuel_losses=fuel_losses,
        diversions=diversions,
        max_fuel_error=max_fuel_error,
        max_observation_error=max_obs_error,
        mismatches=mismatches,
        total_cost=float(lost.sum()),
    )
