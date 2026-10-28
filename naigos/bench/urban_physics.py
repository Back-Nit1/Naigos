"""Reproducible local throughput measurement for the urban physics hot loop.

This module never downloads data and never changes a simulator default.  Callers
must supply an already validated obstacle artifact to ``NaigosEnv``.  Timing is
kept here, outside the environment transition, so wall clock cannot affect a
rollout.
"""

from __future__ import annotations

import platform
import time
from dataclasses import dataclass

import numpy as np

SCHEMA = "naigos.bench.urban_physics/1"


@dataclass(frozen=True)
class Budget:
    """Advisory CPU smoke budget, deliberately loose enough for CI variance."""

    min_world_steps_per_s: float = 5.0
    max_compile_s: float = 120.0


def summarize(samples_s, *, n_worlds: int, n_steps: int) -> dict:
    """Summarize steady-state samples without importing JAX."""
    samples = np.asarray(samples_s, dtype=np.float64)
    if samples.size == 0 or n_worlds < 1 or n_steps < 1:
        raise ValueError("positive worlds/steps and at least one sample are required")
    if not np.isfinite(samples).all() or np.any(samples <= 0):
        raise ValueError("timings must be finite and positive")
    work = float(n_worlds * n_steps)
    best = float(samples.min())
    return {
        "samples": int(samples.size),
        "best_s": best,
        "median_s": float(np.median(samples)),
        "world_steps_per_s": work / best,
    }


def measure(env, *, n_worlds: int = 8, n_steps: int = 32, repeats: int = 3,
            seed: int = 0, budget: Budget = Budget()) -> dict:
    """Compile and time a vmapped fixed-length rollout on an already-built env."""
    if repeats < 1:
        raise ValueError("repeats must be positive")
    import jax
    import jax.numpy as jnp

    keys = jax.random.split(jax.random.PRNGKey(seed), n_worlds)
    action = jnp.zeros((env.cfg.n_blue, 3), dtype=jnp.float32)

    def one(key):
        state, _ = env.reset(key)

        def body(current, _):
            after, _, _, _, _ = env.step(current, action)
            return after, None

        state, _ = jax.lax.scan(body, state, xs=None, length=n_steps)
        return state.air.pos

    lowered = jax.jit(jax.vmap(one)).lower(keys)
    start = time.perf_counter()
    compiled = lowered.compile()
    compile_s = time.perf_counter() - start
    compiled(keys).block_until_ready()  # warm execution is never a timing sample
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        out = compiled(keys)
        out.block_until_ready()
        samples.append(time.perf_counter() - start)
    timing = summarize(samples, n_worlds=n_worlds, n_steps=n_steps)
    return {
        "schema": SCHEMA,
        "platform": platform.platform(),
        "jax_backend": jax.default_backend(),
        "n_worlds": n_worlds,
        "n_steps": n_steps,
        "n_blue": env.cfg.n_blue,
        "compile_s": compile_s,
        "timing": timing,
        "budget": {
            "min_world_steps_per_s": budget.min_world_steps_per_s,
            "max_compile_s": budget.max_compile_s,
            "passed": timing["world_steps_per_s"] >= budget.min_world_steps_per_s
            and compile_s <= budget.max_compile_s,
        },
    }
