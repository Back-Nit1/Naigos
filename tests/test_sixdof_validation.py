"""The optional high-fidelity boundary is deterministic and fail-closed."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from naigos.validation.sixdof import (
    DeterministicFixtureAdapter,
    JSBSIM_REQUIRED_VERSION,
    JSBSimAdapter,
    SIXDOF_BOUNDARY_VERSION,
    SixDofUnavailable,
    compare_traces,
    standard_validation_suite,
)


def test_standard_suite_covers_every_required_controlled_maneuver():
    suite = standard_validation_suite()
    assert {case.scenario_id for case in suite} == {
        "steady-level", "commanded-bank", "climb-descent", "throttle-speed",
        "headwind", "crosswind", "gust-response", "low-density",
        "bounded-recovery", "obstacle-clearance",
    }
    assert all(case.record()["protocol"] == SIXDOF_BOUNDARY_VERSION for case in suite)
    assert len({case.sha256 for case in suite}) == len(suite)


def test_fixture_adapter_is_deterministic_and_never_claims_sixdof():
    case = standard_validation_suite()[6]
    adapter = DeterministicFixtureAdapter()
    first, second = adapter.run(case), adapter.run(case)
    np.testing.assert_array_equal(first.position_enu_m, second.position_enu_m)
    np.testing.assert_array_equal(first.euler_rad, second.euler_rad)
    assert first.fidelity == "test-fixture-not-6dof"
    assert first.scenario_sha256 == case.sha256


def test_comparison_rejects_different_scenario_bytes():
    adapter = DeterministicFixtureAdapter()
    first = adapter.run(standard_validation_suite()[0])
    second = adapter.run(standard_validation_suite()[1])
    with pytest.raises(ValueError, match="different scenario"):
        compare_traces(first, second)
    same = compare_traces(first, adapter.run(standard_validation_suite()[0]))
    assert same["position_max_m"] == 0.0 and same["reference_fidelity"] == "test-fixture-not-6dof"


def test_scenario_tamper_changes_digest_and_action_contract_fails_closed():
    case = standard_validation_suite()[0]
    assert dataclasses.replace(case, density_ratio=0.9).sha256 != case.sha256
    with pytest.raises(ValueError, match="exactly bank/gamma/throttle"):
        dataclasses.replace(case, actions=((0.0, 0.0, 0.0, 0.0),))


def test_jsbsim_boundary_is_optional_and_fail_closed(tmp_path):
    with pytest.raises(SixDofUnavailable, match=f"JSBSim {JSBSIM_REQUIRED_VERSION}|model manifest"):
        JSBSimAdapter(tmp_path, {})
