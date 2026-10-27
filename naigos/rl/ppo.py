"""MAPPO under CTDE, with a PPO-Lagrangian constraint channel.

Shared-parameter actor over local observations; one centralized critic over the
whole scene with two heads (task value, cost value), evaluated once per aircraft
(see `critic_inputs`). Samples of aircraft already killed or arrived are masked
out of every loss. The Lagrange multiplier
prices the CMDP cost channel so "do not get shot down" is a *learned* constraint
rather than a reward weight somebody guessed.

    L_actor = -(A_reward - lambda_eff * A_cost) * ratio  (clipped)
    lambda_eff  = softplus(lambda_raw) + lambda_kp * max(J_cost - budget, 0)
    lambda_raw <- lambda_raw + lr_lambda * (J_cost - budget)   [integral-only, Adam]

`lambda_eff` is what the actor is actually priced against every iteration: a
PID controller (Stooke et al.) on the constraint violation, proportional term
plus integral term, not the integral term alone. `J_cost` is a lifetime rate --
total losses divided by total completed agent-episodes, accumulated across
every persistent rollout segment this run has collected (`Learner.cost_accum`,
`episode_accum`) -- so it is invariant to PPO n_steps instead of being a fresh
per-segment mean that depends on how many episodes happened to finish inside
whichever window was just collected.

The multiplier is stored as a softplus pre-activation so it stays non-negative
without a hard projection step fighting the optimizer.

Everything here is jit-compiled end to end: rollout collection uses
`env.rollout` under `vmap`, so a training iteration is one XLA program.
"""

from __future__ import annotations

import dataclasses
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from ..env.config import EnvConfig
from ..env.flight_env import NaigosEnv
from ..env.obs import Observation, flatten_global
from . import networks as nets
from . import policies
from .reward import RewardWeights, compute as compute_reward, constraint_cost


@dataclasses.dataclass(frozen=True)
class PPOConfig:
    n_envs: int = 64
    n_steps: int = 128  # rollout length per iteration
    n_epochs: int = 4
    n_minibatches: int = 8
    lr: float = 3e-4
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 3e-3
    max_grad_norm: float = 0.5

    # Static actor identity. DeepSets remains the default and keeps its exact
    # flat-timestep PPO path; recurrent PPO shuffles fixed-length sequences.
    policy_family: str = policies.DEFAULT_POLICY_FAMILY.value
    policy_version: str = policies.DEFAULT_POLICY_VERSION
    recurrent_seq_len: int = 32

    # --- CMDP / Lagrangian ---
    # Expected airframe losses per agent-episode. The cost channel is terminal
    # only (see reward.constraint_cost), so this is a rate in [0, 1] and 0.10
    # means "lose at most one sortie in ten". A budget the policy cannot meet
    # turns dual ascent into an ever-growing penalty weight.
    cost_budget: float = 0.10
    lr_lambda: float = 5e-3
    lambda_init: float = 1.0
    cost_gamma: float = 1.0  # terminal cost: no discounting, count the event
    # Proportional term on the constraint violation (Stooke et al., PID
    # Lagrangian). Pure integral control on a constraint that starts far outside
    # the feasible set overshoots and then unwinds slowly; the proportional term
    # responds to the current violation instead of its history.
    lambda_kp: float = 0.5
    lambda_max: float = 25.0  # hard cap, so a mis-set budget cannot run away


class Learner(NamedTuple):
    actor: TrainState
    critic: TrainState
    lam_raw: jnp.ndarray  # softplus pre-activation of the Lagrange multiplier
    lam_opt: Any
    # Lifetime totals, carried across persistent rollout segments so the
    # measured cost rate is "total losses / total completed agent-episodes"
    # rather than a fresh per-segment mean -- see `train_step` and
    # PPOConfig.cost_budget.
    cost_accum: jnp.ndarray
    episode_accum: jnp.ndarray


def lam_value(lam_raw):
    return jax.nn.softplus(lam_raw)


def critic_inputs(obs: Observation) -> jnp.ndarray:
    """Per-aircraft critic input, `(n_blue, G + ego_dim + n_blue)`.

    Row i is the whole scene (`flatten_global`, width G), aircraft i's own ego
    vector and a one-hot of i. The scene alone is identical for every aircraft
    in a world, so a critic fed only that can give one value per world at best;
    the ego vector and index give each aircraft its own baseline.
    """
    n_blue = obs.ego.shape[0]
    g = flatten_global(obs)
    return jnp.concatenate(
        [jnp.broadcast_to(g, (n_blue, g.shape[0])), obs.ego, jnp.eye(n_blue, dtype=g.dtype)], axis=-1
    )


def masked_mean(x, mask):
    """Mean of `x` over `mask`; zero when nothing is masked in."""
    return jnp.sum(jnp.where(mask, x, 0.0)) / jnp.maximum(jnp.sum(mask), 1)


def active_agent_mask(state) -> jax.Array:
    """Agents eligible to contribute one transition to an on-policy update.

    Arrival, safe diversion, and physical loss are all per-agent terminal
    states even when another aircraft keeps the shared world running. Keeping
    this definition outside the feedforward collector gives recurrent
    collectors one terminal convention to reuse without coupling their hidden
    state implementation to this code path.
    """
    return state.alive & ~state.reached & ~state.resource.diverted


def completed_agent_mask(live: jax.Array, agent_done: jax.Array) -> jax.Array:
    """First terminal transition only, excluding repeated frozen samples."""
    return live & agent_done


def init_learner(key, cfg: EnvConfig, ppo: PPOConfig, sample_obs: Observation) -> Learner:
    k_a, k_c = jax.random.split(key)
    actor = policies.create_actor(
        cfg, family=ppo.policy_family, version=ppo.policy_version
    )
    critic = nets.Critic()

    memory = policies.initial_memory(
        cfg, family=ppo.policy_family, version=ppo.policy_version
    )
    actor_inputs = (
        sample_obs.ego, sample_obs.threats, sample_obs.threat_mask,
        sample_obs.friends, sample_obs.friend_mask,
    )
    ap = actor.init(k_a, *actor_inputs) if memory is None else actor.init(
        k_a, *actor_inputs, memory
    )
    cp = critic.init(k_c, critic_inputs(sample_obs))

    tx = optax.chain(optax.clip_by_global_norm(ppo.max_grad_norm), optax.adam(ppo.lr))
    txc = optax.chain(optax.clip_by_global_norm(ppo.max_grad_norm), optax.adam(ppo.lr))
    lam_raw = jnp.asarray(jnp.log(jnp.expm1(jnp.maximum(ppo.lambda_init, 1e-4))), dtype=jnp.float32)
    lam_opt = optax.adam(ppo.lr_lambda)
    return Learner(
        actor=TrainState.create(apply_fn=actor.apply, params=ap, tx=tx),
        critic=TrainState.create(apply_fn=critic.apply, params=cp, tx=txc),
        lam_raw=lam_raw,
        lam_opt=lam_opt.init(lam_raw),
        cost_accum=jnp.asarray(0.0, dtype=jnp.float32),
        episode_accum=jnp.asarray(0.0, dtype=jnp.float32),
    )


def gae(rewards, values, dones, last_value, gamma, lam):
    """Generalised advantage estimation over a (S, ...) trajectory."""

    def body(carry, x):
        adv, next_value = carry
        r, v, d = x
        nonterm = 1.0 - d
        delta = r + gamma * next_value * nonterm - v
        adv = delta + gamma * lam * nonterm * adv
        return (adv, v), adv

    (_, _), advs = jax.lax.scan(
        body,
        (jnp.zeros_like(last_value), last_value),
        (rewards, values, dones),
        reverse=True,
    )
    return advs, advs + values


def init_rollout(env: NaigosEnv, ppo: PPOConfig, key, red_scales=None):
    """Fresh batched (n_envs) `(EnvState, Observation)` for a persistent rollout.

    `train_step` carries this across iterations, so an episode longer than
    `n_steps` continues into the next segment instead of being cut off.

    `red_scales`, if given, overrides `env.cfg`'s own curriculum fields for
    every world (next-steps.md G-4) -- how a curriculum promotion starts a new
    rollout at the new level without rebuilding `env` or retracing `train_step`.
    """
    keys = jax.random.split(key, ppo.n_envs)
    if red_scales is None:
        state, obs = jax.vmap(env.reset)(keys)
    else:
        state, obs = jax.vmap(env.reset, in_axes=(0, None))(keys, red_scales)
    memory = policies.initial_memory(
        env.cfg,
        family=ppo.policy_family,
        version=ppo.policy_version,
        batch_shape=(ppo.n_envs,),
    )
    return (state, obs) if memory is None else (state, obs, memory)


def _make_feedforward_train(env: NaigosEnv, ppo: PPOConfig, weights: RewardWeights):
    """Build a jit-able `train_step(learner, rollout, key) -> (learner, rollout, metrics)`."""
    cfg = env.cfg
    actor_apply = nets.Actor(cfg).apply
    critic_apply = nets.Critic().apply
    # one value per aircraft: the critic runs on each row of `critic_inputs`
    critic_per_agent = jax.vmap(critic_apply, in_axes=(None, 0))

    def policy_step(params, obs: Observation, key):
        mean, log_std = actor_apply(params, obs.ego, obs.threats, obs.threat_mask, obs.friends, obs.friend_mask)
        act, raw = nets.sample_action(mean, log_std, key)
        return act, raw, nets.log_prob(mean, log_std, raw)

    def collect(learner: Learner, rollout, key):
        """One vectorised rollout segment across `n_envs` worlds, continuing
        from `rollout`. A world whose episode ends is reset in place."""

        def one_world(state, obs, k):
            def body(carry, _):
                st, ob, kk = carry
                kk, ka, kr = jax.random.split(kk, 3)
                act, raw, lp = policy_step(learner.actor.params, ob, ka)
                gs = critic_inputs(ob)
                v, vc = critic_per_agent(learner.critic.params, gs)
                # Recorded BEFORE the step, so the step on which an aircraft is
                # killed or arrives is live; every later sample of it is not.
                live = active_agent_mask(st)
                st2, ob2, terms, done, info = env.step(st, act)
                # The first terminal step remains live. Frozen samples after a
                # per-agent terminal state carry no reward, cost, or gradient.
                r = compute_reward(terms, weights, alive_mask=live)
                c = constraint_cost(terms) * live.astype(jnp.float32)
                # The max_steps timeout is treated as terminal (not bootstrapped).
                agent_done_bool = info["agent_done"] | done
                agent_done = agent_done_bool.astype(jnp.float32)
                completed = completed_agent_mask(live, agent_done_bool).astype(jnp.float32)
                # auto-reset: a finished world starts a fresh episode in place
                # Carry the ending episode's own red_scales forward (next-steps.md
                # G-4): a mid-rollout auto-reset must respawn at the level the
                # rollout is currently annealed to, not silently fall back to cfg's.
                st_r, ob_r = env.reset(kr, st.red_scales)
                st2, ob2 = jax.tree.map(lambda a, b: jnp.where(done, a, b), (st_r, ob_r), (st2, ob2))
                out = dict(
                    ego=ob.ego, threats=ob.threats, threat_mask=ob.threat_mask,
                    friends=ob.friends, friend_mask=ob.friend_mask,
                    gs=gs, raw=raw, logp=lp, v=v, vc=vc, r=r, c=c, d=agent_done,
                    live=live, completed=completed,
                    shot=terms.shotdown, arrived=terms.arrived, exposure=terms.exposure,
                    progress=terms.progress, reset=done.astype(jnp.float32),
                )
                return (st2, ob2, kk), out

            (st_f, ob_f, _), traj = jax.lax.scan(body, (state, obs, k), None, length=ppo.n_steps)
            v_last, vc_last = critic_per_agent(learner.critic.params, critic_inputs(ob_f))
            return traj, v_last, vc_last, (st_f, ob_f)

        state, obs = rollout
        keys = jax.random.split(key, ppo.n_envs)
        traj, v_last, vc_last, rollout = jax.vmap(one_world)(state, obs, keys)
        # vmap puts env on axis 0; scan put time on axis 1 -> move time first
        traj = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), traj)
        return traj, v_last, vc_last, rollout

    def loss_fn(actor_params, critic_params, lam, batch):
        mean, log_std = actor_apply(
            actor_params, batch["ego"], batch["threats"], batch["threat_mask"],
            batch["friends"], batch["friend_mask"],
        )
        logp = nets.log_prob(mean, log_std, batch["raw"])
        ratio = jnp.exp(logp - batch["logp"])

        # samples of aircraft already killed or arrived carry no signal
        live = batch["live"]
        adv = batch["adv"] - lam * batch["adv_c"]
        adv_mean = masked_mean(adv, live)
        adv_std = jnp.sqrt(masked_mean((adv - adv_mean) ** 2, live))
        adv = (adv - adv_mean) / (adv_std + 1e-8)

        pg1 = ratio * adv
        pg2 = jnp.clip(ratio, 1 - ppo.clip_eps, 1 + ppo.clip_eps) * adv
        pg_loss = -masked_mean(jnp.minimum(pg1, pg2), live)
        ent = masked_mean(nets.entropy(log_std), live)

        v, vc = critic_apply(critic_params, batch["gs"])
        v_loss = masked_mean((v - batch["ret"]) ** 2, live) + masked_mean((vc - batch["ret_c"]) ** 2, live)

        total = pg_loss + ppo.vf_coef * v_loss - ppo.ent_coef * ent
        return total, {"pg_loss": pg_loss, "v_loss": v_loss, "entropy": ent, "ratio": jnp.mean(ratio)}

    def train_step(learner: Learner, rollout, key):
        k_roll, k_shuf = jax.random.split(key)
        traj, v_last, vc_last, rollout = collect(learner, rollout, k_roll)

        # advantages, computed per (env, agent) stream against per-agent values
        adv, ret = gae(traj["r"], traj["v"], traj["d"], v_last, ppo.gamma, ppo.gae_lambda)
        adv_c, ret_c = gae(traj["c"], traj["vc"], traj["d"], vc_last, ppo.cost_gamma, ppo.gae_lambda)

        # --- CMDP budget: lifetime totals, not a per-segment mean ---
        # A per-segment mean of a rare terminal event scales with how many
        # agent-episodes happened to finish inside this segment, i.e. with
        # PPO n_steps. Accumulating "total losses" and "total completed
        # agent-episodes" across every segment this run has ever collected
        # and reporting their ratio is invariant to that choice. Only samples
        # that were live going into their terminal step count as a completed
        # agent-episode (`live`) -- a step replayed after the agent is
        # already dead/arrived is not a new episode ending.
        live = traj["live"]
        cost_accum = learner.cost_accum + jnp.sum(traj["c"] * live)
        episode_accum = learner.episode_accum + jnp.sum(traj["completed"])
        j_cost = jnp.where(episode_accum > 0, cost_accum / jnp.maximum(episode_accum, 1.0), 0.0)
        violation = j_cost - ppo.cost_budget

        # --- PID effective lambda: the value that actually prices the actor loss ---
        # Pure integral control (dual ascent alone) overshoots a constraint
        # that starts far outside the feasible set and then unwinds slowly;
        # the proportional term responds to THIS iteration's violation. It is
        # computed once, from the lambda this iteration started with, and
        # used for every epoch's actor update below -- not recomputed after
        # lam_raw's integral update, so the weight a minibatch is trained
        # against does not change mid-epoch.
        lam_eff = jnp.clip(
            lam_value(learner.lam_raw) + ppo.lambda_kp * jnp.maximum(violation, 0.0), 0.0, ppo.lambda_max
        )

        flat = jax.tree.map(lambda x: x.reshape((-1,) + x.shape[3:]) if x.ndim >= 3 else x.reshape(-1), {
            **{k: traj[k] for k in ("ego", "threats", "threat_mask", "friends", "friend_mask", "raw", "logp",
                                    "gs", "live")},
            "adv": adv, "ret": ret, "adv_c": adv_c, "ret_c": ret_c,
        })

        n = flat["adv"].shape[0]
        mb = n // ppo.n_minibatches

        def epoch(carry, ek):
            actor, critic = carry
            perm = jax.random.permutation(ek, n)

            def minibatch(carry2, i):
                actor, critic = carry2
                idx = jax.lax.dynamic_slice(perm, (i * mb,), (mb,))
                batch = jax.tree.map(lambda x: x[idx], flat)
                grads, aux = jax.grad(
                    lambda ap, cp: loss_fn(ap, cp, lam_eff, batch),
                    argnums=(0, 1), has_aux=True,
                )(actor.params, critic.params)
                actor = actor.apply_gradients(grads=grads[0])
                critic = critic.apply_gradients(grads=grads[1])
                return (actor, critic), aux

            (actor, critic), aux = jax.lax.scan(minibatch, (actor, critic), jnp.arange(ppo.n_minibatches))
            return (actor, critic), aux

        (actor, critic), aux = jax.lax.scan(
            epoch, (learner.actor, learner.critic), jax.random.split(k_shuf, ppo.n_epochs)
        )

        # --- Lagrange multiplier: integral-only dual ascent on the violation ---
        # The proportional term above is applied fresh every iteration and
        # never fed into the optimizer state, so it can vanish the instant
        # the constraint is satisfied; only the integral term accumulates.
        lam_grad = -violation * jax.nn.sigmoid(learner.lam_raw)  # d(-lam*viol)/d(raw)
        opt = optax.adam(ppo.lr_lambda)
        updates, lam_opt = opt.update(lam_grad, learner.lam_opt, learner.lam_raw)
        lam_raw = optax.apply_updates(learner.lam_raw, updates)
        lam_raw = jnp.clip(lam_raw, -10.0, jnp.log(jnp.expm1(ppo.lambda_max)))

        # reward-head fit on the live samples, with the values the rollout used
        ret_var = masked_mean((ret - masked_mean(ret, live)) ** 2, live)
        err = ret - traj["v"]
        err_var = masked_mean((err - masked_mean(err, live)) ** 2, live)

        metrics = {
            "reward": jnp.mean(jnp.sum(traj["r"], axis=0)),
            "cost": j_cost,  # lifetime losses per completed agent-episode
            "lambda": lam_eff,  # the PID value actually used to train the actor this iteration
            "lambda_integral": lam_value(lam_raw),
            "cost_violation": violation,
            "shootdown_rate": jnp.mean(jnp.sum(traj["shot"], axis=0)),
            "arrival_rate": jnp.mean(jnp.sum(traj["arrived"], axis=0)),
            "exposure": jnp.mean(traj["exposure"]),
            "progress_km": jnp.mean(jnp.sum(traj["progress"], axis=0)) / 1000.0,
            "episodes_completed": jnp.sum(traj["reset"]),  # world resets in this segment
            "live_frac": jnp.mean(live.astype(jnp.float32)),
            # 0.0 when the live returns have no variance to explain
            "explained_var": jnp.where(ret_var > 0, 1.0 - err_var / jnp.where(ret_var > 0, ret_var, 1.0), 0.0),
            **jax.tree.map(jnp.mean, aux),
        }
        return Learner(
            actor=actor, critic=critic, lam_raw=lam_raw, lam_opt=lam_opt,
            cost_accum=cost_accum, episode_accum=episode_accum,
        ), rollout, metrics

    train_step.collect = collect  # exposed so tests can inspect the raw segment
    train_step.loss_fn = loss_fn  # exposed so tests can check the live mask
    return train_step


def _make_recurrent_train(env: NaigosEnv, ppo: PPOConfig, weights: RewardWeights):
    """Sequence-minibatched PPO for the explicit-state GRU actor."""

    cfg = env.cfg
    seq_len = int(ppo.recurrent_seq_len)
    if seq_len < 1 or ppo.n_steps % seq_len:
        raise ValueError(
            f"recurrent_seq_len ({seq_len}) must divide n_steps ({ppo.n_steps})"
        )
    n_sequences = (ppo.n_steps // seq_len) * ppo.n_envs * cfg.n_blue
    if n_sequences % ppo.n_minibatches:
        raise ValueError(
            f"recurrent sequence count ({n_sequences}) must be divisible by "
            f"n_minibatches ({ppo.n_minibatches})"
        )

    actor_apply = policies.create_actor(
        cfg, family=ppo.policy_family, version=ppo.policy_version
    ).apply
    critic_apply = nets.Critic().apply
    critic_per_agent = jax.vmap(critic_apply, in_axes=(None, 0))

    def policy_step(params, obs: Observation, memory, key):
        mean, log_std, next_memory = actor_apply(
            params, obs.ego, obs.threats, obs.threat_mask,
            obs.friends, obs.friend_mask, memory,
        )
        act, raw = nets.sample_action(mean, log_std, key)
        return act, raw, nets.log_prob(mean, log_std, raw), next_memory

    def collect(learner: Learner, rollout, key):
        """Collect one segment while carrying one GRU state per aircraft."""

        def one_world(state, obs, memory, k):
            def body(carry, _):
                st, ob, mem, kk = carry
                kk, ka, kr = jax.random.split(kk, 3)
                act, raw, lp, mem2 = policy_step(
                    learner.actor.params, ob, mem, ka
                )
                gs = critic_inputs(ob)
                v, vc = critic_per_agent(learner.critic.params, gs)
                live = st.alive & ~st.reached
                st2, ob2, terms, done, info = env.step(st, act)
                r = compute_reward(terms, weights)
                c = constraint_cost(terms)
                agent_done_bool = info["agent_done"] | done
                agent_done = agent_done_bool.astype(jnp.float32)
                # A completed agent remains zero while its world finishes; a
                # completed world is zero before the reset observation is used.
                mem2 = policies.reset_memory(mem2, agent_done_bool | ~live)
                st_r, ob_r = env.reset(kr, st.red_scales)
                st2, ob2 = jax.tree.map(
                    lambda a, b: jnp.where(done, a, b),
                    (st_r, ob_r), (st2, ob2),
                )
                out = dict(
                    ego=ob.ego, threats=ob.threats,
                    threat_mask=ob.threat_mask, friends=ob.friends,
                    friend_mask=ob.friend_mask, memory=mem,
                    gs=gs, raw=raw, logp=lp, v=v, vc=vc, r=r, c=c,
                    d=agent_done, live=live, shot=terms.shotdown,
                    arrived=terms.arrived, exposure=terms.exposure,
                    progress=terms.progress, reset=done.astype(jnp.float32),
                )
                return (st2, ob2, mem2, kk), out

            (st_f, ob_f, mem_f, _), traj = jax.lax.scan(
                body, (state, obs, memory, k), None, length=ppo.n_steps
            )
            v_last, vc_last = critic_per_agent(
                learner.critic.params, critic_inputs(ob_f)
            )
            return traj, v_last, vc_last, (st_f, ob_f, mem_f)

        state, obs, memory = rollout
        keys = jax.random.split(key, ppo.n_envs)
        traj, v_last, vc_last, rollout = jax.vmap(one_world)(
            state, obs, memory, keys
        )
        traj = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), traj)
        return traj, v_last, vc_last, rollout

    def loss_fn(actor_params, critic_params, lam, batch):
        # Each item is a chronological sequence. Recompute recurrent outputs
        # from the exact hidden state that collection stored at its first step.
        xs = tuple(
            jnp.swapaxes(batch[name], 0, 1)
            for name in (
                "ego", "threats", "threat_mask", "friends", "friend_mask", "d"
            )
        )

        def actor_step(memory, x):
            ego, threats, threat_mask, friends, friend_mask, done = x
            mean, log_std, memory = actor_apply(
                actor_params, ego, threats, threat_mask, friends, friend_mask, memory
            )
            memory = policies.reset_memory(memory, done)
            return memory, (mean, log_std)

        _, (mean, log_std) = jax.lax.scan(actor_step, batch["memory"], xs)
        mean, log_std = jnp.swapaxes(mean, 0, 1), jnp.swapaxes(log_std, 0, 1)
        logp = nets.log_prob(mean, log_std, batch["raw"])
        ratio = jnp.exp(logp - batch["logp"])

        live = batch["live"]
        adv = batch["adv"] - lam * batch["adv_c"]
        adv_mean = masked_mean(adv, live)
        adv_std = jnp.sqrt(masked_mean((adv - adv_mean) ** 2, live))
        adv = (adv - adv_mean) / (adv_std + 1e-8)
        pg1 = ratio * adv
        pg2 = jnp.clip(ratio, 1 - ppo.clip_eps, 1 + ppo.clip_eps) * adv
        pg_loss = -masked_mean(jnp.minimum(pg1, pg2), live)
        ent = masked_mean(nets.entropy(log_std), live)

        v, vc = critic_apply(critic_params, batch["gs"])
        v_loss = masked_mean((v - batch["ret"]) ** 2, live) + masked_mean(
            (vc - batch["ret_c"]) ** 2, live
        )
        total = pg_loss + ppo.vf_coef * v_loss - ppo.ent_coef * ent
        return total, {
            "pg_loss": pg_loss, "v_loss": v_loss, "entropy": ent,
            "ratio": jnp.mean(ratio),
        }

    def sequence_batch(tree):
        """(time, env, agent, ...) -> (sequence, seq_len, ...)."""
        chunks = ppo.n_steps // seq_len

        def one(x):
            tail = x.shape[3:]
            x = x.reshape((chunks, seq_len) + x.shape[1:])
            x = jnp.transpose(x, (0, 2, 3, 1, *range(4, x.ndim)))
            return x.reshape((-1, seq_len) + tail)

        return jax.tree.map(one, tree)

    def train_step(learner: Learner, rollout, key):
        k_roll, k_shuf = jax.random.split(key)
        traj, v_last, vc_last, rollout = collect(learner, rollout, k_roll)
        adv, ret = gae(
            traj["r"], traj["v"], traj["d"], v_last,
            ppo.gamma, ppo.gae_lambda,
        )
        adv_c, ret_c = gae(
            traj["c"], traj["vc"], traj["d"], vc_last,
            ppo.cost_gamma, ppo.gae_lambda,
        )

        live = traj["live"]
        cost_accum = learner.cost_accum + jnp.sum(traj["c"] * live)
        episode_accum = learner.episode_accum + jnp.sum(traj["d"] * live)
        j_cost = jnp.where(
            episode_accum > 0,
            cost_accum / jnp.maximum(episode_accum, 1.0), 0.0,
        )
        violation = j_cost - ppo.cost_budget
        lam_eff = jnp.clip(
            lam_value(learner.lam_raw)
            + ppo.lambda_kp * jnp.maximum(violation, 0.0),
            0.0, ppo.lambda_max,
        )

        sequences = sequence_batch({
            **{
                k: traj[k]
                for k in (
                    "ego", "threats", "threat_mask", "friends", "friend_mask",
                    "raw", "logp", "gs", "live", "d",
                )
            },
            "adv": adv, "ret": ret, "adv_c": adv_c, "ret_c": ret_c,
        })
        # Only the chunk-entry state is an input to truncated BPTT.
        sequences["memory"] = sequence_batch(traj["memory"])[:, 0]
        n = sequences["adv"].shape[0]
        mb = n // ppo.n_minibatches

        def epoch(carry, ek):
            actor, critic = carry
            perm = jax.random.permutation(ek, n)

            def minibatch(carry2, i):
                actor, critic = carry2
                idx = jax.lax.dynamic_slice(perm, (i * mb,), (mb,))
                batch = jax.tree.map(lambda x: x[idx], sequences)
                grads, aux = jax.grad(
                    lambda ap, cp: loss_fn(ap, cp, lam_eff, batch),
                    argnums=(0, 1), has_aux=True,
                )(actor.params, critic.params)
                actor = actor.apply_gradients(grads=grads[0])
                critic = critic.apply_gradients(grads=grads[1])
                return (actor, critic), aux

            (actor, critic), aux = jax.lax.scan(
                minibatch, (actor, critic), jnp.arange(ppo.n_minibatches)
            )
            return (actor, critic), aux

        (actor, critic), aux = jax.lax.scan(
            epoch, (learner.actor, learner.critic),
            jax.random.split(k_shuf, ppo.n_epochs),
        )
        lam_grad = -violation * jax.nn.sigmoid(learner.lam_raw)
        opt = optax.adam(ppo.lr_lambda)
        updates, lam_opt = opt.update(lam_grad, learner.lam_opt, learner.lam_raw)
        lam_raw = optax.apply_updates(learner.lam_raw, updates)
        lam_raw = jnp.clip(
            lam_raw, -10.0, jnp.log(jnp.expm1(ppo.lambda_max))
        )

        ret_var = masked_mean((ret - masked_mean(ret, live)) ** 2, live)
        err = ret - traj["v"]
        err_var = masked_mean((err - masked_mean(err, live)) ** 2, live)
        metrics = {
            "reward": jnp.mean(jnp.sum(traj["r"], axis=0)),
            "cost": j_cost, "lambda": lam_eff,
            "lambda_integral": lam_value(lam_raw),
            "cost_violation": violation,
            "shootdown_rate": jnp.mean(jnp.sum(traj["shot"], axis=0)),
            "arrival_rate": jnp.mean(jnp.sum(traj["arrived"], axis=0)),
            "exposure": jnp.mean(traj["exposure"]),
            "progress_km": jnp.mean(jnp.sum(traj["progress"], axis=0)) / 1000.0,
            "episodes_completed": jnp.sum(traj["reset"]),
            "live_frac": jnp.mean(live.astype(jnp.float32)),
            "explained_var": jnp.where(
                ret_var > 0,
                1.0 - err_var / jnp.where(ret_var > 0, ret_var, 1.0), 0.0,
            ),
            **jax.tree.map(jnp.mean, aux),
        }
        return Learner(
            actor=actor, critic=critic, lam_raw=lam_raw, lam_opt=lam_opt,
            cost_accum=cost_accum, episode_accum=episode_accum,
        ), rollout, metrics

    train_step.collect = collect
    train_step.loss_fn = loss_fn
    return train_step


def make_train(env: NaigosEnv, ppo: PPOConfig, weights: RewardWeights):
    """Build the policy-family-specific jittable PPO update."""

    spec = policies.get_policy_spec(
        env.cfg, family=ppo.policy_family, version=ppo.policy_version
    )
    if spec.memory_requirements.recurrent:
        return _make_recurrent_train(env, ppo, weights)
    return _make_feedforward_train(env, ppo, weights)


def greedy_policy(
    actor_params,
    cfg: EnvConfig,
    *,
    family: policies.PolicyFamily | str = policies.DEFAULT_POLICY_FAMILY,
    version: str = policies.DEFAULT_POLICY_VERSION,
):
    """Deterministic policy for evaluation and for the demo replay.

    `mean` is the actor's pre-squash location (see `nets.Actor`); the
    deterministic action is its mode after squashing, `tanh(mean)`.
    """
    actor = policies.create_actor(cfg, family=family, version=version)
    apply = actor.apply
    memory = policies.initial_memory(cfg, family=family, version=version)

    if memory is not None:
        def recurrent_pol(obs: Observation, key, memory):
            mean, _, memory = apply(
                actor_params, obs.ego, obs.threats, obs.threat_mask,
                obs.friends, obs.friend_mask, memory,
            )
            return jnp.tanh(mean), memory

        # `NaigosEnv.rollout` detects this explicit deployment lifecycle.
        recurrent_pol.initial_memory = memory
        return recurrent_pol

    def pol(obs: Observation, key):
        mean, _ = apply(actor_params, obs.ego, obs.threats, obs.threat_mask, obs.friends, obs.friend_mask)
        return jnp.tanh(mean)

    return pol
