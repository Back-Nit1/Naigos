"""Deployable policy specifications and the small model-ladder factory.

This module is deliberately actor-only.  ``networks.Critic`` is a centralized
training baseline: it consumes privileged, whole-scene inputs and must never be
needed to fly a checkpoint.  The objects built here consume only an
``Observation``'s local fields and produce the three normalized flight-control
actions.

The current checkpoint surface is unchanged.  DeepSets v1 is exactly
``networks.Actor(cfg)`` and its parameters remain under the top-level
``"actor"`` checkpoint key.  PPO and the demos may therefore continue to use
``networks.Actor`` directly while model-ladder integration is added
incrementally.

There is no mutable global registry and no registration work at import time.
``_definitions`` constructs the built-in table only when a factory or metadata
query is made.  New families should be added there only when their actor and
checkpoint behavior exist and are tested; enum membership alone is not an
implementation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Literal, Mapping, TypeAlias

import jax
import jax.numpy as jnp
from flax import linen as nn

from ..env.config import BLUE_ACTION_NAMES, EnvConfig
from .networks import Actor

Axis: TypeAlias = int | str
DTypeName: TypeAlias = Literal["float32", "bool"]


class PolicyFamily(str, Enum):
    """Stable family names used when comparing rungs of the model ladder."""

    DEEPSETS = "deepsets"
    RECURRENT = "recurrent"
    NUMERIC_TRANSFORMER = "numeric_transformer"
    SCRIPTED = "scripted"


DEFAULT_POLICY_FAMILY = PolicyFamily.DEEPSETS
DEFAULT_POLICY_VERSION = "1"


@dataclass(frozen=True)
class TensorContract:
    """One named tensor at the deployable actor boundary."""

    name: str
    shape: tuple[Axis, ...]
    dtype: DTypeName
    description: str


@dataclass(frozen=True)
class ObservationContract:
    """The local, per-aircraft information a deployable policy may consume."""

    local_only: bool
    tensors: tuple[TensorContract, ...]


@dataclass(frozen=True)
class ActionContract:
    """The distribution outputs and normalized flight-control action."""

    control_names: tuple[str, ...]
    actor_outputs: tuple[TensorContract, ...]
    action: TensorContract
    normalized_bounds: tuple[float, float]
    bounds_inclusive: bool
    distribution: str


@dataclass(frozen=True)
class MemoryRequirements:
    """Persistent state that must be carried between policy calls."""

    recurrent: bool
    state_tensors: tuple[TensorContract, ...]
    persistent_bytes_per_agent: int
    reset_on_episode: bool


@dataclass(frozen=True)
class PolicySpec:
    """Typed identity and deployment contract for one policy implementation."""

    family: PolicyFamily
    version: str
    parameter_count: int
    memory_requirements: MemoryRequirements
    observation_contract: ObservationContract
    action_contract: ActionContract
    checkpoint_parameter_key: str
    requires_critic_at_deployment: bool


class PolicyNotRegisteredError(ValueError):
    """The requested family/version has no implementation in this build."""


@dataclass(frozen=True)
class _PolicyDefinition:
    actor_factory: Callable[[EnvConfig], nn.Module]
    spec_factory: Callable[[EnvConfig], PolicySpec]


def count_parameters(params) -> int:
    """Count scalar leaves in a Flax parameter pytree."""

    return sum(math.prod(leaf.shape) for leaf in jax.tree.leaves(params))


def _actor_input_shapes(cfg: EnvConfig) -> tuple[jax.ShapeDtypeStruct, ...]:
    return (
        jax.ShapeDtypeStruct((cfg.n_blue, cfg.ego_dim), jnp.float32),
        jax.ShapeDtypeStruct(
            (cfg.n_blue, cfg.k_threat_obs, cfg.threat_feat_dim), jnp.float32
        ),
        jax.ShapeDtypeStruct((cfg.n_blue, cfg.k_threat_obs), jnp.bool_),
        jax.ShapeDtypeStruct(
            (cfg.n_blue, cfg.k_friend_obs, cfg.friend_feat_dim), jnp.float32
        ),
        jax.ShapeDtypeStruct((cfg.n_blue, cfg.k_friend_obs), jnp.bool_),
    )


def _deepsets_parameter_count(cfg: EnvConfig) -> int:
    """Trace shapes only; do not allocate a model merely to describe it."""

    variables = jax.eval_shape(
        Actor(cfg).init,
        jax.random.PRNGKey(0),
        *_actor_input_shapes(cfg),
    )
    return count_parameters(variables["params"])


def _deepsets_spec(cfg: EnvConfig) -> PolicySpec:
    agent = cfg.n_blue
    observation = ObservationContract(
        local_only=True,
        tensors=(
            TensorContract(
                "ego", (agent, cfg.ego_dim), "float32", "local aircraft and route features"
            ),
            TensorContract(
                "threats",
                (agent, cfg.k_threat_obs, cfg.threat_feat_dim),
                "float32",
                "padded sensed-threat set",
            ),
            TensorContract(
                "threat_mask",
                (agent, cfg.k_threat_obs),
                "bool",
                "valid sensed-threat slots",
            ),
            TensorContract(
                "friends",
                (agent, cfg.k_friend_obs, cfg.friend_feat_dim),
                "float32",
                "padded local-friendly set",
            ),
            TensorContract(
                "friend_mask",
                (agent, cfg.k_friend_obs),
                "bool",
                "valid local-friendly slots",
            ),
        ),
    )
    output_shape = (agent, cfg.action_dim)
    action = ActionContract(
        control_names=BLUE_ACTION_NAMES,
        actor_outputs=(
            TensorContract(
                "mean", output_shape, "float32", "pre-tanh diagonal-Gaussian location"
            ),
            TensorContract(
                "log_std", output_shape, "float32", "pre-tanh diagonal-Gaussian log scale"
            ),
        ),
        action=TensorContract(
            "action", output_shape, "float32", "bank, flight-path angle, and throttle commands"
        ),
        normalized_bounds=(-1.0, 1.0),
        bounds_inclusive=False,
        distribution="tanh-squashed diagonal Gaussian",
    )
    return PolicySpec(
        family=PolicyFamily.DEEPSETS,
        version=DEFAULT_POLICY_VERSION,
        parameter_count=_deepsets_parameter_count(cfg),
        memory_requirements=MemoryRequirements(
            recurrent=False,
            state_tensors=(),
            persistent_bytes_per_agent=0,
            reset_on_episode=False,
        ),
        observation_contract=observation,
        action_contract=action,
        checkpoint_parameter_key="actor",
        requires_critic_at_deployment=False,
    )


def _definitions() -> Mapping[tuple[PolicyFamily, str], _PolicyDefinition]:
    """Return built-ins without mutating process-global registration state."""

    return {
        (PolicyFamily.DEEPSETS, DEFAULT_POLICY_VERSION): _PolicyDefinition(
            actor_factory=Actor,
            spec_factory=_deepsets_spec,
        )
    }


def _key(family: PolicyFamily | str, version: str) -> tuple[PolicyFamily, str]:
    try:
        normalized = PolicyFamily(family)
    except ValueError as exc:
        raise PolicyNotRegisteredError(f"unknown policy family {family!r}") from exc
    key = (normalized, str(version))
    if key not in _definitions():
        available = ", ".join(f"{f.value}@{v}" for f, v in registered_policies())
        raise PolicyNotRegisteredError(
            f"policy {normalized.value}@{version} is not registered; available: {available}"
        )
    return key


def registered_policies() -> tuple[tuple[PolicyFamily, str], ...]:
    """Implemented family/version pairs, in stable order."""

    return tuple(_definitions())


def create_actor(
    cfg: EnvConfig,
    *,
    family: PolicyFamily | str = DEFAULT_POLICY_FAMILY,
    version: str = DEFAULT_POLICY_VERSION,
) -> nn.Module:
    """Build a deployable actor.  DeepSets v1 is the backward-compatible default."""

    return _definitions()[_key(family, version)].actor_factory(cfg)


def get_policy_spec(
    cfg: EnvConfig,
    *,
    family: PolicyFamily | str = DEFAULT_POLICY_FAMILY,
    version: str = DEFAULT_POLICY_VERSION,
) -> PolicySpec:
    """Describe an implemented actor without creating concrete parameters."""

    return _definitions()[_key(family, version)].spec_factory(cfg)


__all__ = [
    "ActionContract",
    "DEFAULT_POLICY_FAMILY",
    "DEFAULT_POLICY_VERSION",
    "MemoryRequirements",
    "ObservationContract",
    "PolicyFamily",
    "PolicyNotRegisteredError",
    "PolicySpec",
    "TensorContract",
    "count_parameters",
    "create_actor",
    "get_policy_spec",
    "registered_policies",
]
