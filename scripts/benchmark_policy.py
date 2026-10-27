#!/usr/bin/env python3
"""Measure warm-jitted actor latency without running an environment.

Example:
    python scripts/benchmark_policy.py --family recurrent --calls 10000

The benchmark is intentionally one world per call: it measures the deployment
control-loop boundary, not batched training throughput. Compilation is reported
separately from warm p50/p95/p99 latency.
"""

from __future__ import annotations

import argparse
import json
import time

import jax
import jax.numpy as jnp
import numpy as np

from naigos.env.config import EnvConfig
from naigos.rl import policies


def _inputs(cfg: EnvConfig):
    return (
        jnp.zeros((cfg.n_blue, cfg.ego_dim), dtype=jnp.float32),
        jnp.zeros(
            (cfg.n_blue, cfg.k_threat_obs, cfg.threat_feat_dim), dtype=jnp.float32
        ),
        jnp.ones((cfg.n_blue, cfg.k_threat_obs), dtype=bool),
        jnp.zeros(
            (cfg.n_blue, cfg.k_friend_obs, cfg.friend_feat_dim), dtype=jnp.float32
        ),
        jnp.ones((cfg.n_blue, cfg.k_friend_obs), dtype=bool),
    )


def benchmark(
    family: str,
    *,
    version: str = "1",
    calls: int = 10_000,
    warmup: int = 100,
) -> dict:
    if calls < 1 or warmup < 0:
        raise ValueError("calls must be positive and warmup must be non-negative")
    cfg = EnvConfig()
    actor = policies.create_actor(cfg, family=family, version=version)
    inputs = _inputs(cfg)
    memory = policies.initial_memory(cfg, family=family, version=version)
    key = jax.random.PRNGKey(0)
    variables = actor.init(key, *inputs) if memory is None else actor.init(
        key, *inputs, memory
    )

    if memory is None:
        fn = jax.jit(lambda: actor.apply(variables, *inputs)[0])
    else:
        def recurrent_call(state):
            mean, _, next_state = actor.apply(variables, *inputs, state)
            return mean, next_state

        fn = jax.jit(recurrent_call)

    t0 = time.perf_counter_ns()
    result = fn() if memory is None else fn(memory)
    jax.block_until_ready(result)
    compile_ms = (time.perf_counter_ns() - t0) / 1e6
    state = memory
    for _ in range(warmup):
        result = fn() if state is None else fn(state)
        jax.block_until_ready(result)
        if state is not None:
            state = result[1]

    samples = np.empty(calls, dtype=np.float64)
    for i in range(calls):
        start = time.perf_counter_ns()
        result = fn() if state is None else fn(state)
        jax.block_until_ready(result)
        samples[i] = (time.perf_counter_ns() - start) / 1e6
        if state is not None:
            state = result[1]

    spec = policies.get_policy_spec(cfg, family=family, version=version)
    return {
        "family": spec.family.value,
        "version": spec.version,
        "device": jax.default_backend(),
        "n_blue": cfg.n_blue,
        "calls": calls,
        "warmup_calls": warmup,
        "parameter_count": spec.parameter_count,
        "fp32_parameter_bytes": spec.parameter_count * 4,
        "persistent_bytes_per_agent": spec.memory_requirements.persistent_bytes_per_agent,
        "compile_ms": round(compile_ms, 4),
        "latency_ms": {
            "p50": round(float(np.percentile(samples, 50)), 4),
            "p95": round(float(np.percentile(samples, 95)), 4),
            "p99": round(float(np.percentile(samples, 99)), 4),
        },
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--family",
        choices=[family.value for family, _ in policies.registered_policies()],
        default="deepsets",
    )
    parser.add_argument("--version", default="1")
    parser.add_argument("--calls", type=int, default=10_000)
    parser.add_argument("--warmup", type=int, default=100)
    args = parser.parse_args(argv)
    print(json.dumps(benchmark(
        args.family, version=args.version, calls=args.calls, warmup=args.warmup
    ), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
