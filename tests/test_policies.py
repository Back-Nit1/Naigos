"""The model-ladder seam preserves DeepSets v1 and stays deployment-only."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from naigos.env.config import BLUE_ACTION_NAMES, EnvConfig
from naigos.rl import networks
from naigos.rl import policies


CFG = EnvConfig(n_blue=2, n_threat=7, n_threat_active=5, k_threat_obs=4, k_friend_obs=1)


def _inputs(cfg: EnvConfig = CFG):
    keys = jax.random.split(jax.random.PRNGKey(23), 3)
    ego = jax.random.normal(keys[0], (cfg.n_blue, cfg.ego_dim))
    threats = jax.random.normal(
        keys[1], (cfg.n_blue, cfg.k_threat_obs, cfg.threat_feat_dim)
    )
    friends = jax.random.normal(
        keys[2], (cfg.n_blue, cfg.k_friend_obs, cfg.friend_feat_dim)
    )
    threat_mask = jnp.array([[True, True, False, False], [True, False, True, False]])
    friend_mask = jnp.ones((cfg.n_blue, cfg.k_friend_obs), dtype=bool)
    return ego, threats, threat_mask, friends, friend_mask


def test_default_factory_is_the_existing_actor_with_the_same_parameter_tree_and_outputs():
    """Fixed parameters and inputs take the exact pre-abstraction code path."""

    inputs = _inputs()
    direct = networks.Actor(CFG)
    factory = policies.create_actor(CFG)
    assert type(factory) is type(direct) is networks.Actor
    assert factory.cfg == direct.cfg == CFG

    key = jax.random.PRNGKey(41)
    direct_params = direct.init(key, *inputs)
    factory_params = factory.init(key, *inputs)
    assert jax.tree.structure(factory_params) == jax.tree.structure(direct_params)
    for actual, expected in zip(jax.tree.leaves(factory_params), jax.tree.leaves(direct_params)):
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))

    expected = direct.apply(direct_params, *inputs)
    actual = factory.apply(direct_params, *inputs)
    for got, want in zip(actual, expected):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))


@pytest.mark.parametrize("edge_features", [False, True])
def test_default_spec_matches_the_concrete_actor_parameter_count(edge_features):
    cfg = CFG.replace(obs_edge_features=edge_features)
    params = networks.Actor(cfg).init(jax.random.PRNGKey(5), *_inputs(cfg))
    spec = policies.get_policy_spec(cfg)
    assert spec.parameter_count == policies.count_parameters(params["params"])


def test_default_spec_pins_the_local_observation_and_three_control_contract():
    spec = policies.get_policy_spec(CFG)
    assert spec.family is policies.PolicyFamily.DEEPSETS
    assert spec.version == "1"
    assert spec.observation_contract.local_only
    assert [tensor.name for tensor in spec.observation_contract.tensors] == [
        "ego", "threats", "threat_mask", "friends", "friend_mask"
    ]
    assert spec.action_contract.control_names == BLUE_ACTION_NAMES
    assert spec.action_contract.action.shape == (CFG.n_blue, 3)
    assert spec.action_contract.normalized_bounds == (-1.0, 1.0)


def test_deployment_spec_requires_no_memory_or_critic_and_keeps_the_checkpoint_key():
    spec = policies.get_policy_spec(CFG)
    assert spec.memory_requirements == policies.MemoryRequirements(
        recurrent=False,
        state_tensors=(),
        persistent_bytes_per_agent=0,
        reset_on_episode=False,
    )
    assert spec.checkpoint_parameter_key == "actor"
    assert spec.requires_critic_at_deployment is False
    assert "critic" not in {field.name for field in dataclasses.fields(spec.observation_contract)}


def test_future_family_names_do_not_silently_fall_back_to_deepsets():
    assert set(policies.PolicyFamily) == {
        policies.PolicyFamily.DEEPSETS,
        policies.PolicyFamily.RECURRENT,
        policies.PolicyFamily.NUMERIC_TRANSFORMER,
        policies.PolicyFamily.SCRIPTED,
    }
    assert policies.registered_policies() == (
        (policies.PolicyFamily.DEEPSETS, "1"),
        (policies.PolicyFamily.RECURRENT, "1"),
    )
    for family in (
        policies.PolicyFamily.NUMERIC_TRANSFORMER,
        policies.PolicyFamily.SCRIPTED,
    ):
        with pytest.raises(policies.PolicyNotRegisteredError, match="not registered"):
            policies.create_actor(CFG, family=family)


def test_recurrent_actor_spec_memory_and_concrete_parameter_count():
    actor = policies.create_actor(CFG, family=policies.PolicyFamily.RECURRENT)
    memory = policies.initial_memory(CFG, family=policies.PolicyFamily.RECURRENT)
    variables = actor.init(jax.random.PRNGKey(9), *_inputs(), memory)
    spec = policies.get_policy_spec(CFG, family=policies.PolicyFamily.RECURRENT)
    assert type(actor) is networks.RecurrentActor
    assert memory.shape == (CFG.n_blue, policies.RECURRENT_HIDDEN)
    assert spec.parameter_count == policies.count_parameters(variables["params"])
    assert spec.memory_requirements.recurrent
    assert spec.memory_requirements.persistent_bytes_per_agent == 128 * 4
    assert spec.checkpoint_parameter_key == "actor"
    assert len(spec.sha256) == 64
    assert spec.to_dict()["family"] == "recurrent"


def test_recurrent_actor_updates_explicit_memory_and_keeps_three_controls():
    actor = policies.create_actor(CFG, family="recurrent")
    inputs = _inputs()
    memory = policies.initial_memory(CFG, family="recurrent")
    variables = actor.init(jax.random.PRNGKey(10), *inputs, memory)
    mean, log_std, next_memory = actor.apply(variables, *inputs, memory)
    assert mean.shape == log_std.shape == (CFG.n_blue, 3)
    assert next_memory.shape == memory.shape
    assert not np.array_equal(np.asarray(next_memory), np.asarray(memory))
    assert policies.initial_memory(CFG) is None


def test_recurrent_actor_is_permutation_invariant_over_threat_slots():
    actor = policies.create_actor(CFG, family="recurrent")
    ego, threats, mask, friends, friend_mask = _inputs()
    memory = policies.initial_memory(CFG, family="recurrent")
    variables = actor.init(
        jax.random.PRNGKey(11), ego, threats, mask, friends, friend_mask, memory
    )
    order = jnp.array([2, 0, 3, 1])
    a = actor.apply(variables, ego, threats, mask, friends, friend_mask, memory)
    b = actor.apply(
        variables, ego, threats[:, order], mask[:, order], friends, friend_mask, memory
    )
    for actual, expected in zip(a, b):
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=1e-6)
