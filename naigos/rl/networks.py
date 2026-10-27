"""Shared-parameter actor + centralized critic (CTDE), permutation invariant.

Copied straight from the Nomos design because it is the part that transfers 1:1.
The only structural change is that there are now two neighbour sets (threats and
friendlies) instead of one, each with its own encoder.

Why permutation invariance matters here: the number of threats an aircraft can
currently sense changes every step as terrain masks and unmasks them. A policy
built on a fixed-slot MLP would have to learn one behaviour per slot ordering.
Deep Sets (masked mean+max pooling) plus an ego-query attention head give one
behaviour that is correct for any count and any order.

CTDE: `Actor` sees only the local observation. `Critic` sees the whole scene and
exists only at training time. That is what lets anticipation emerge without
bolting on a separate threat-prediction model.
"""

from __future__ import annotations

from typing import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp

from ..env.config import EnvConfig


def mlp(sizes: Sequence[int], out: int, name: str | None = None):
    layers = []
    for h in sizes:
        layers += [nn.Dense(h), nn.tanh]
    layers += [nn.Dense(out)]
    return nn.Sequential(layers, name=name)


class SetEncoder(nn.Module):
    """Masked Deep Sets encoder with an optional ego-query attention head.

    mean+max pooling rather than mean alone: mean answers "what is the threat
    picture on average", max answers "what is the single worst thing near me",
    and evasion is driven by the second question at least as much as the first.
    """

    hidden: int = 128
    out: int = 128
    attention: bool = True
    n_heads: int = 4

    @nn.compact
    def __call__(self, x: jnp.ndarray, mask: jnp.ndarray, ego: jnp.ndarray):
        # x: (..., K, F), mask: (..., K) bool, ego: (..., E)
        m = mask[..., None].astype(x.dtype)
        h = nn.Dense(self.hidden)(x)
        h = nn.tanh(h)
        h = nn.Dense(self.hidden)(h)
        h = nn.tanh(h) * m

        count = jnp.maximum(jnp.sum(m, axis=-2), 1.0)
        mean = jnp.sum(h, axis=-2) / count
        # -inf on masked slots so an empty set pools to zeros, not to garbage
        neg = jnp.where(m > 0, h, -jnp.inf)
        mx = jnp.max(neg, axis=-2)
        mx = jnp.where(jnp.isfinite(mx), mx, 0.0)

        pooled = jnp.concatenate([mean, mx], axis=-1)

        if self.attention:
            q = nn.Dense(self.hidden)(ego)[..., None, :]  # (..., 1, H)
            attn = nn.MultiHeadDotProductAttention(num_heads=self.n_heads, qkv_features=self.hidden)
            # mask shape for flax attention: (..., heads, q_len, kv_len)
            bias_mask = mask[..., None, None, :]
            a = attn(q, h, mask=bias_mask)
            a = jnp.squeeze(a, axis=-2)
            # a set with nothing in it must contribute nothing
            any_valid = (jnp.sum(m, axis=-2) > 0).astype(x.dtype)
            pooled = jnp.concatenate([pooled, a * any_valid], axis=-1)

        return nn.tanh(nn.Dense(self.out)(pooled))


class Actor(nn.Module):
    """Local-observation policy. Tanh-squashed Gaussian over the 3 flight controls.

    The output dimension is `cfg.action_dim`, which is 3 and stays 3:
    (bank, flight-path angle, throttle). There is no fourth head. Adding one
    would break `tests/test_invariant.py`.

    `mean` is the *pre-squash* location of the base Gaussian, not the action
    itself -- it is unbounded. The action is `tanh(sample)`, built by
    `sample_action`/`log_prob` in this module. A `tanh` baked in here (as a
    clipped-Gaussian policy once had) would double-squash and bias every
    sample off the true mode; see `sample_action`.
    """

    cfg: EnvConfig
    hidden: int = 256

    @nn.compact
    def __call__(self, ego, threats, threat_mask, friends, friend_mask):
        t = SetEncoder(name="threat_encoder")(threats, threat_mask, ego)
        f = SetEncoder(name="friend_encoder", attention=False)(friends, friend_mask, ego)
        z = jnp.concatenate([ego, t, f], axis=-1)
        z = nn.tanh(nn.Dense(self.hidden)(z))
        z = nn.tanh(nn.Dense(self.hidden)(z))
        mean = nn.Dense(self.cfg.action_dim, kernel_init=nn.initializers.orthogonal(0.01))(z)
        log_std = self.param("log_std", nn.initializers.constant(-0.5), (self.cfg.action_dim,))
        return mean, jnp.broadcast_to(log_std, mean.shape)


class RecurrentActor(nn.Module):
    """DeepSets observation encoder followed by one GRU state per aircraft.

    The recurrent state is an explicit argument/result.  Nothing is cached in
    the Linen module, which keeps evaluation, checkpointing and reset semantics
    visible to callers and safe under ``jit``/``vmap``.
    """

    cfg: EnvConfig
    hidden: int = 128

    @nn.compact
    def __call__(self, ego, threats, threat_mask, friends, friend_mask, memory):
        t = SetEncoder(name="threat_encoder")(threats, threat_mask, ego)
        f = SetEncoder(name="friend_encoder", attention=False)(friends, friend_mask, ego)
        z = jnp.concatenate([ego, t, f], axis=-1)
        z = nn.tanh(nn.Dense(self.hidden, name="input_projection")(z))
        memory, z = nn.GRUCell(features=self.hidden, name="gru")(memory, z)
        z = nn.tanh(nn.Dense(self.hidden, name="head")(z))
        mean = nn.Dense(
            self.cfg.action_dim,
            kernel_init=nn.initializers.orthogonal(0.01),
            name="mean",
        )(z)
        log_std = self.param(
            "log_std", nn.initializers.constant(-0.5), (self.cfg.action_dim,)
        )
        return mean, jnp.broadcast_to(log_std, mean.shape), memory


class Critic(nn.Module):
    """Centralized value function. Training time only.

    Two heads: the task return and the CMDP cost return. PPO-Lagrangian needs
    both, and sharing the trunk between them is strictly better than two towers
    because the features that predict "am I about to be shot" are the same ones
    that predict "am I about to lose reward".
    """

    hidden: int = 256

    @nn.compact
    def __call__(self, global_state):
        z = nn.tanh(nn.Dense(self.hidden)(global_state))
        z = nn.tanh(nn.Dense(self.hidden)(z))
        v = nn.Dense(1)(z)[..., 0]
        vc = nn.Dense(1)(z)[..., 0]
        return v, vc


def sample_action(mean, log_std, key):
    """Tanh-squashed Gaussian sample.

    `raw` is the unbounded base-Gaussian draw (what `log_prob` needs); the
    action is `tanh(raw)`, which is bounded in (-1, 1) by construction --
    no clipping, so there is no boundary mass piling up at +-1 the way a
    clipped Gaussian has.
    """
    std = jnp.exp(log_std)
    eps = jax.random.normal(key, mean.shape)
    raw = mean + std * eps
    return jnp.tanh(raw), raw


def log_prob(mean, log_std, raw):
    """log-density of `tanh(raw)` under a tanh-squashed Gaussian.

    change-of-variables: log p(action) = log N(raw; mean, std) - log|d(tanh)/d(raw)|,
    and |d(tanh)/d(raw)| = 1 - tanh(raw)**2. The Jacobian term is evaluated as
    `2*(log(2) - raw - softplus(-2*raw))`, the standard stable rewrite of
    `log(1 - tanh(raw)**2)` (it is exact, not an approximation: see the
    derivation in `tests/test_ppo_lagrangian_fixes.py`) -- the naive form
    underflows to log(0) = -inf for |raw| well past the point where tanh
    saturates in float32, which a sampled `raw` reaches routinely.
    """
    var = jnp.exp(2 * log_std)
    gaussian_lp = -0.5 * (((raw - mean) ** 2) / var + 2 * log_std + jnp.log(2 * jnp.pi))
    log_jacobian = 2.0 * (jnp.log(2.0) - raw - jax.nn.softplus(-2.0 * raw))
    return jnp.sum(gaussian_lp - log_jacobian, axis=-1)


def entropy(log_std):
    """Entropy of the base Gaussian, used as the PPO entropy bonus.

    The tanh-squashed distribution has no closed-form entropy; using the base
    Gaussian's (a biased estimate, independent of the squashing) is the
    standard choice for a bounded-action PPO entropy bonus -- it still
    rewards wide exploration in the pre-squash space, which is all the bonus
    needs to do.
    """
    return jnp.sum(log_std + 0.5 * jnp.log(2 * jnp.pi * jnp.e), axis=-1)
