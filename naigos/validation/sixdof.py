"""Optional, offline nonlinear 6-DoF validation boundary.

The fast JAX point-mass model remains the training model.  This module defines
deterministic scenario bytes and result contracts for a separately installed
nonlinear validator.  No network access, model download, or simulator import
occurs at module import.  The JSBSim gate is deliberately unavailable until an
immutable, hash-pinned aircraft model pack and the pinned Python package are
provided locally; it never substitutes the point-mass fixture and calls it
6-DoF.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib
import json
from pathlib import Path
from typing import Protocol

import numpy as np


SIXDOF_BOUNDARY_VERSION = "naigos-sixdof-validation-v1"
JSBSIM_REQUIRED_VERSION = "1.2.3"
AIRCRAFT_MODEL_CLASS = "generic-research-aircraft; not a deployed-aircraft representation"


class SixDofUnavailable(RuntimeError):
    """The optional validator cannot run with the supplied offline evidence."""


def _canonical(value) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclasses.dataclass(frozen=True)
class ValidationScenario:
    scenario_id: str
    dt_s: float
    initial_position_enu_m: tuple[float, float, float]
    initial_tas_mps: float
    initial_euler_rad: tuple[float, float, float]
    actions: tuple[tuple[float, float, float], ...]
    mean_wind_enu_mps: tuple[float, float, float] = (0.0, 0.0, 0.0)
    density_ratio: float = 1.0
    gust_enu_mps: tuple[tuple[float, float, float], ...] = ()
    obstacle_top_amsl_m: float | None = None
    note: str = "notional controlled validation case"

    def __post_init__(self) -> None:
        if not self.scenario_id or self.dt_s <= 0 or not self.actions:
            raise ValueError("validation scenario requires an id, positive dt, and actions")
        if any(len(action) != 3 or any(abs(v) > 1 for v in action) for action in self.actions):
            raise ValueError("validation actions must be exactly bank/gamma/throttle in [-1,1]")
        if self.density_ratio <= 0:
            raise ValueError("density_ratio must be positive")
        if self.gust_enu_mps and len(self.gust_enu_mps) != len(self.actions):
            raise ValueError("gust history must be empty or have one vector per action")

    def record(self) -> dict:
        return {"protocol": SIXDOF_BOUNDARY_VERSION, **dataclasses.asdict(self)}

    @property
    def sha256(self) -> str:
        return hashlib.sha256(_canonical(self.record())).hexdigest()


@dataclasses.dataclass(frozen=True)
class ValidationTrace:
    scenario_sha256: str
    adapter_id: str
    fidelity: str
    aircraft_model_sha256: str
    position_enu_m: np.ndarray
    tas_mps: np.ndarray
    euler_rad: np.ndarray

    def validate(self, steps: int) -> None:
        expected = ((steps + 1, 3), (steps + 1,), (steps + 1, 3))
        got = (self.position_enu_m.shape, self.tas_mps.shape, self.euler_rad.shape)
        if got != expected:
            raise ValueError(f"validation trace shapes {got} do not match {expected}")
        if not all(np.isfinite(x).all() for x in (self.position_enu_m, self.tas_mps, self.euler_rad)):
            raise ValueError("validation trace contains non-finite values")


class SixDofAdapter(Protocol):
    adapter_id: str
    fidelity: str
    aircraft_model_sha256: str

    def run(self, scenario: ValidationScenario) -> ValidationTrace: ...


class JSBSimAdapter:
    """Fail-closed boundary for a future offline JSBSim execution adapter.

    Construction verifies package version and every declared model byte.  Run
    remains intentionally unavailable in this repository because no licensed,
    cited, immutable generic aircraft model pack is shipped.  Supplying a
    presentation asset or an unpinned model directory cannot bypass the gate.
    """

    adapter_id = f"jsbsim-python-{JSBSIM_REQUIRED_VERSION}"
    fidelity = "nonlinear-6dof"

    def __init__(self, model_root: str | Path, expected_files: dict[str, str]):
        try:
            module = importlib.import_module("jsbsim")
        except ImportError as exc:
            raise SixDofUnavailable(
                f"JSBSim {JSBSIM_REQUIRED_VERSION} is not installed; core PPO remains available"
            ) from exc
        version = str(getattr(module, "__version__", "unknown"))
        if version != JSBSIM_REQUIRED_VERSION:
            raise SixDofUnavailable(
                f"JSBSim version {version!r} does not match pinned {JSBSIM_REQUIRED_VERSION!r}"
            )
        root = Path(model_root).resolve()
        if not root.is_dir() or not expected_files:
            raise SixDofUnavailable("a non-empty, immutable offline aircraft model manifest is required")
        records = []
        for relative, expected in sorted(expected_files.items()):
            path = (root / relative).resolve()
            if root not in path.parents or not path.is_file():
                raise SixDofUnavailable(f"aircraft model file is absent or escapes its root: {relative}")
            actual = _sha256_file(path)
            if actual != expected:
                raise SixDofUnavailable(f"aircraft model hash mismatch for {relative}")
            records.append((relative, actual))
        self.aircraft_model_sha256 = hashlib.sha256(_canonical(records)).hexdigest()

    def run(self, scenario: ValidationScenario) -> ValidationTrace:
        raise SixDofUnavailable(
            "JSBSim execution is fail-closed until a cited generic model pack and maneuver mapping are reviewed"
        )


class DeterministicFixtureAdapter:
    """Independent NumPy fixture for boundary tests; explicitly not 6-DoF."""

    adapter_id = "numpy-validation-fixture-v1"
    fidelity = "test-fixture-not-6dof"
    aircraft_model_sha256 = hashlib.sha256(b"abstract fixture point mass v1").hexdigest()

    def run(self, scenario: ValidationScenario) -> ValidationTrace:
        n = len(scenario.actions)
        pos = np.empty((n + 1, 3), dtype=np.float64)
        speed = np.empty(n + 1, dtype=np.float64)
        euler = np.empty((n + 1, 3), dtype=np.float64)
        pos[0] = scenario.initial_position_enu_m
        speed[0] = scenario.initial_tas_mps
        euler[0] = scenario.initial_euler_rad
        rho_authority = np.clip(np.sqrt(scenario.density_ratio), 0.55, 1.10)
        wind = np.asarray(scenario.mean_wind_enu_mps, dtype=np.float64)
        gust = (np.zeros((n, 3), dtype=np.float64) if not scenario.gust_enu_mps
                else np.asarray(scenario.gust_enu_mps, dtype=np.float64))
        for i, action in enumerate(np.asarray(scenario.actions, dtype=np.float64)):
            bank_cmd, gamma_cmd, throttle = action
            euler[i + 1, 0] = euler[i, 0] + 0.4 * rho_authority * (bank_cmd * 1.2 - euler[i, 0])
            euler[i + 1, 1] = euler[i, 1] + 0.5 * rho_authority * (gamma_cmd * 0.35 - euler[i, 1])
            target_speed = 65.0 + 0.5 * (throttle + 1.0) * 235.0
            speed[i + 1] = np.clip(speed[i] + 0.25 * rho_authority * (target_speed - speed[i]), 65.0, 300.0)
            yaw_rate = 9.80665 * np.tan(euler[i + 1, 0]) / max(speed[i + 1], 65.0)
            euler[i + 1, 2] = euler[i, 2] + scenario.dt_s * yaw_rate
            gamma, yaw = euler[i + 1, 1], euler[i + 1, 2]
            air = speed[i + 1] * np.array([
                np.cos(gamma) * np.cos(yaw), np.cos(gamma) * np.sin(yaw), np.sin(gamma)
            ])
            pos[i + 1] = pos[i] + scenario.dt_s * (air + wind + gust[i])
        trace = ValidationTrace(
            scenario.sha256, self.adapter_id, self.fidelity, self.aircraft_model_sha256,
            pos, speed, euler,
        )
        trace.validate(n)
        return trace


def compare_traces(fast: ValidationTrace, reference: ValidationTrace) -> dict:
    """Deterministic comparison record; never upgrades either fidelity label."""
    if fast.scenario_sha256 != reference.scenario_sha256:
        raise ValueError("cannot compare traces from different scenario bytes")
    if fast.position_enu_m.shape != reference.position_enu_m.shape:
        raise ValueError("trace lengths differ")
    dp = np.asarray(fast.position_enu_m) - np.asarray(reference.position_enu_m)
    dv = np.asarray(fast.tas_mps) - np.asarray(reference.tas_mps)
    return {
        "protocol": SIXDOF_BOUNDARY_VERSION,
        "scenario_sha256": fast.scenario_sha256,
        "candidate": fast.adapter_id,
        "reference": reference.adapter_id,
        "reference_fidelity": reference.fidelity,
        "position_rms_m": float(np.sqrt(np.mean(dp * dp))),
        "position_max_m": float(np.max(np.linalg.norm(dp, axis=-1))),
        "tas_rms_mps": float(np.sqrt(np.mean(dv * dv))),
    }


def _actions(action, steps=24):
    return tuple(tuple(float(v) for v in action) for _ in range(steps))


def standard_validation_suite() -> tuple[ValidationScenario, ...]:
    """Frozen controlled maneuvers required at the validation boundary."""
    base = dict(
        dt_s=1.0,
        initial_position_enu_m=(0.0, 0.0, 2_000.0),
        initial_tas_mps=180.0,
        initial_euler_rad=(0.0, 0.0, 0.0),
    )
    return (
        ValidationScenario("steady-level", actions=_actions((0.0, 0.0, 0.0)), **base),
        ValidationScenario("commanded-bank", actions=_actions((0.55, 0.0, 0.0)), **base),
        ValidationScenario("climb-descent", actions=_actions((0.0, 0.6, 0.2), 12) + _actions((0.0, -0.5, 0.2), 12), **base),
        ValidationScenario("throttle-speed", actions=_actions((0.0, 0.0, 0.9)), **base),
        ValidationScenario("headwind", actions=_actions((0.0, 0.0, 0.2)), mean_wind_enu_mps=(-30.0, 0.0, 0.0), **base),
        ValidationScenario("crosswind", actions=_actions((0.0, 0.0, 0.2)), mean_wind_enu_mps=(0.0, 30.0, 0.0), **base),
        ValidationScenario("gust-response", actions=_actions((0.0, 0.0, 0.2)), gust_enu_mps=tuple(((-1.0) ** i * 5.0, 2.0, 0.0) for i in range(24)), **base),
        ValidationScenario("low-density", actions=_actions((0.4, 0.4, 0.7)), density_ratio=0.72, **base),
        ValidationScenario("bounded-recovery", actions=_actions((-0.6, 0.5, 0.4), 12) + _actions((0.0, 0.0, 0.2), 12), **base),
        ValidationScenario("obstacle-clearance", actions=_actions((0.0, 0.6, 0.4)), obstacle_top_amsl_m=1_500.0, **base),
    )
