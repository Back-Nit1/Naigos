"""Stable contract tests for the opt-in urban throughput benchmark."""

import pytest

from naigos.bench.urban_physics import SCHEMA, summarize


def test_urban_benchmark_summary_has_unambiguous_world_step_units():
    got = summarize([2.0, 1.0, 1.5], n_worlds=4, n_steps=8)
    assert got == {
        "samples": 3, "best_s": 1.0, "median_s": 1.5, "world_steps_per_s": 32.0,
    }
    assert SCHEMA == "naigos.bench.urban_physics/1"


@pytest.mark.parametrize("samples,worlds,steps", [([], 1, 1), ([0.0], 1, 1), ([1.0], 0, 1)])
def test_urban_benchmark_rejects_ambiguous_or_invalid_measurements(samples, worlds, steps):
    with pytest.raises(ValueError):
        summarize(samples, n_worlds=worlds, n_steps=steps)
