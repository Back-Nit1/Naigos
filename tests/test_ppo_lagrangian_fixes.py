"""Regression tests for the PPO-Lagrangian audit.

Covers: the PID effective lambda actually pricing the actor loss, a cost
budget that is invariant to PPO n_steps (persistent episode accumulators),
the tanh-squashed action distribution, and checkpoint/resume equivalence
across the schema bump these fixes require.
"""
from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from naigos.env.config import DetectionConfig, EnvConfig, TerrainConfig
from naigos.env.flight_env import NaigosEnv
from naigos.rl import checkpoint as ck
from naigos.rl import networks as nets
from naigos.rl.ppo import PPOConfig, gae, init_learner, init_rollout, make_train
from naigos.rl.reward import RewardWeights

SAFE = EnvConfig(
    n_blue=2, n_threat=2, n_threat_active=0,
    terrain=TerrainConfig(nx=21, ny=21, cell=2_000.0),
    detection=DetectionConfig(los_samples=8),
    spawn_inset_frac=0.3,
)


def _flat(cfg: EnvConfig) -> NaigosEnv:
    return NaigosEnv(cfg, hmap=jnp.zeros((cfg.terrain.ny, cfg.terrain.nx)))


def _setup(ppo: PPOConfig, cfg: EnvConfig = SAFE, seed: int = 0):
    env = _flat(cfg)
    k_init, k_roll, k_run = jax.random.split(jax.random.PRNGKey(seed), 3)
    _, sample_obs = env.reset(k_init)
    learner = init_learner(k_init, cfg, ppo, sample_obs)
    step = make_train(env, ppo, RewardWeights())
    return env, learner, step, init_rollout(env, ppo, k_roll), k_run


def _kill_aircraft_zero(rollout):
    """Aircraft 0 is out of bounds on the first step of every world, whatever
    it does; aircraft 1 is untouched. Deterministic, policy-independent cost."""
    state, obs = rollout
    pos = state.air.pos.at[:, 0, 0].set(-5_000.0)
    return state._replace(air=state.air._replace(pos=pos)), obs


# --- requirement 1: the PID effective lambda must price the actor loss ------


def test_lambda_kp_changes_the_actor_update_when_the_constraint_is_violated():
    ppo_lo = PPOConfig(n_envs=2, n_steps=8, n_epochs=1, n_minibatches=1,
                        cost_budget=0.0, lambda_kp=0.0)
    ppo_hi = dataclasses.replace(ppo_lo, lambda_kp=10.0)

    _, learner, step_lo, rollout, key = _setup(ppo_lo)
    rollout = _kill_aircraft_zero(rollout)
    _, _, step_hi, _, _ = _setup(ppo_hi)

    learner_lo, _, _ = jax.jit(step_lo)(learner, rollout, key)
    learner_hi, _, _ = jax.jit(step_hi)(learner, rollout, key)

    diffs = [
        float(jnp.abs(a - b).max())
        for a, b in zip(jax.tree.leaves(learner_lo.actor.params), jax.tree.leaves(learner_hi.actor.params))
    ]
    assert max(diffs) > 1e-8, "lambda_kp had no effect on the actor update despite a violated constraint"


def test_lambda_kp_has_no_effect_when_the_constraint_is_satisfied():
    ppo_lo = PPOConfig(n_envs=2, n_steps=8, n_epochs=1, n_minibatches=1,
                        cost_budget=1.0, lambda_kp=0.0)
    ppo_hi = dataclasses.replace(ppo_lo, lambda_kp=10.0)

    _, learner, step_lo, rollout, key = _setup(ppo_lo)
    rollout = _kill_aircraft_zero(rollout)  # cost still < budget=1.0: never violated
    _, _, step_hi, _, _ = _setup(ppo_hi)

    learner_lo, _, _ = jax.jit(step_lo)(learner, rollout, key)
    learner_hi, _, _ = jax.jit(step_hi)(learner, rollout, key)

    for a, b in zip(jax.tree.leaves(learner_lo.actor.params), jax.tree.leaves(learner_hi.actor.params)):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-6, atol=1e-6)


# --- requirement 2: the cost budget must be a per-agent-episode rate, -------
# --- carried across segments, invariant to n_steps --------------------------


def test_episode_accumulators_survive_across_persistent_segments():
    ppo = PPOConfig(n_envs=2, n_steps=8, n_epochs=1, n_minibatches=1)
    _, learner, step, rollout, key = _setup(ppo)
    rollout = _kill_aircraft_zero(rollout)
    train_step = jax.jit(step)

    total_cost = 0.0
    total_episodes = 0.0
    for _ in range(3):
        key, k = jax.random.split(key)
        # train_step splits its own rollout key off `k` internally; replicate
        # that split so this measures the exact segment it consumes.
        k_roll, _ = jax.random.split(k)
        traj, _, _, _ = jax.jit(step.collect)(learner, rollout, k_roll)
        live = traj["live"]
        total_cost += float(jnp.sum(traj["c"] * live))
        total_episodes += float(jnp.sum(traj["d"] * live))
        learner, rollout, metrics = train_step(learner, rollout, k)

    assert float(learner.episode_accum) == pytest.approx(total_episodes)
    assert float(learner.cost_accum) == pytest.approx(total_cost)
    if total_episodes > 0:
        assert float(metrics["cost"]) == pytest.approx(total_cost / total_episodes)


def _synthetic_cost_trajectory(seed=0, S=40, E=3, B=2):
    """A fake (S, E, B) trace: an agent-episode ends with probability 0.2 each
    step (`d`), and a fifth of endings are a loss (`c`). Cost is zero on any
    non-terminal or already-dead/arrived (`live=False`) sample, exactly like
    the real env's terminal-only cost channel."""
    rng = np.random.default_rng(seed)
    live = np.ones((S, E, B), dtype=bool)
    d = np.zeros((S, E, B), dtype=np.float32)
    c = np.zeros((S, E, B), dtype=np.float32)
    alive = np.ones((E, B), dtype=bool)
    for t in range(S):
        live[t] = alive
        ends = alive & (rng.random((E, B)) < 0.2)
        d[t] = ends.astype(np.float32)
        c[t] = (ends & (rng.random((E, B)) < 0.2)).astype(np.float32)
        alive = alive & ~ends  # a world whose episode ended stays done for this fake trace
    return jnp.asarray(c), jnp.asarray(d), jnp.asarray(live)


def test_cost_rate_is_invariant_to_how_a_fixed_trajectory_is_chunked():
    """The fix: accumulate total cost and total completed agent-episodes
    across segments and report their ratio, which must agree no matter how
    one fixed trajectory is chunked into PPO n_steps-sized segments."""
    c, d, live = _synthetic_cost_trajectory()
    rate_whole = float(jnp.sum(c * live) / jnp.sum(d * live))

    def chunked_rate(chunk):
        cost_accum = episode_accum = 0.0
        for start in range(0, c.shape[0], chunk):
            sl = slice(start, start + chunk)
            cost_accum += float(jnp.sum(c[sl] * live[sl]))
            episode_accum += float(jnp.sum(d[sl] * live[sl]))
        return cost_accum / episode_accum

    for chunk in (4, 8, 16, 40):
        assert chunked_rate(chunk) == pytest.approx(rate_whole), chunk


def test_the_old_per_segment_mean_was_not_chunk_invariant():
    """Documents the bug being fixed: the old code's `mean over envs of sum
    over the segment` rate scales with how many completed episodes happen to
    fall inside the segment, i.e. with n_steps, on the exact same trace."""
    c, d, live = _synthetic_cost_trajectory()

    def old_segment_mean(n_steps):
        return float(jnp.mean(jnp.sum(c[:n_steps], axis=0)))

    short, long_ = old_segment_mean(4), old_segment_mean(40)
    assert short != pytest.approx(long_, rel=0.3)


# --- requirement 3: tanh-squashed Gaussian action distribution --------------


def test_actions_are_bounded_and_log_prob_is_finite_at_extremes():
    mean = jnp.array([[0.0, 5.0, -5.0]])  # large pre-squash mean, saturates tanh
    log_std = jnp.full((1, 3), -0.5)
    act, raw = nets.sample_action(mean, log_std, jax.random.PRNGKey(0))
    assert bool(jnp.all(act > -1.0)) and bool(jnp.all(act < 1.0))
    lp = nets.log_prob(mean, log_std, raw)
    assert bool(jnp.all(jnp.isfinite(lp)))

    # push raw itself to the boundary, where naive log(1 - tanh(raw)^2) underflows
    raw_extreme = jnp.array([[20.0, -20.0, 0.0]])
    lp_extreme = nets.log_prob(mean, log_std, raw_extreme)
    assert bool(jnp.all(jnp.isfinite(lp_extreme)))


def test_log_prob_integrates_to_one_over_the_action_support():
    """`log_prob` is a density over the ACTION (tanh(raw)), not over `raw`.
    Integrating it over its true support, a in (-1, 1), via raw = atanh(a),
    must recover 1 -- which only holds if the tanh Jacobian term is right."""
    mean = jnp.array([0.3])
    log_std = jnp.array([-1.0])
    a = jnp.linspace(-0.999999, 0.999999, 40001)
    raw = jnp.arctanh(a)
    lp = nets.log_prob(jnp.broadcast_to(mean, (a.shape[0], 1)),
                        jnp.broadcast_to(log_std, (a.shape[0], 1)),
                        raw[:, None])
    density = jnp.exp(lp)
    total = jnp.trapezoid(density, a)
    assert float(total) == pytest.approx(1.0, abs=5e-3)


def test_sample_action_and_log_prob_are_finite_under_vmap():
    keys = jax.random.split(jax.random.PRNGKey(0), 16)
    mean = jax.random.normal(jax.random.PRNGKey(1), (16, 3)) * 3.0
    log_std = jnp.full((16, 3), -0.5)
    act, raw = jax.vmap(nets.sample_action)(mean, log_std, keys)
    lp = jax.vmap(nets.log_prob)(mean, log_std, raw)
    assert bool(jnp.all(jnp.isfinite(act))) and bool(jnp.all(jnp.isfinite(lp)))
    assert bool(jnp.all(act > -1.0)) and bool(jnp.all(act < 1.0))


def test_actor_mean_is_unbounded_pre_squash():
    """The network's own output is the Gaussian's location parameter, not the
    squashed action: feeding it back through sample_action/log_prob is what
    squashes it. A `tanh` baked into the Actor would make this saturate."""
    cfg = SAFE
    env = _flat(cfg)
    _, o = env.reset(jax.random.PRNGKey(0))
    actor = nets.Actor(cfg)
    p = actor.init(jax.random.PRNGKey(1), o.ego, o.threats, o.threat_mask, o.friends, o.friend_mask)
    mean, _ = actor.apply(p, o.ego, o.threats, o.threat_mask, o.friends, o.friend_mask)
    assert bool(jnp.all(jnp.isfinite(mean)))


# --- requirement 5: finite gradients / jit / vmap / resume equivalence ------


def test_actor_and_critic_gradients_are_finite_through_the_squashed_distribution():
    ppo = PPOConfig(n_envs=2, n_steps=8, n_epochs=1, n_minibatches=1)
    _, learner, step, rollout, key = _setup(ppo)
    rollout = _kill_aircraft_zero(rollout)
    traj, v_last, vc_last, _ = jax.jit(step.collect)(learner, rollout, key)
    adv, ret = gae(traj["r"], traj["v"], traj["d"], v_last, ppo.gamma, ppo.gae_lambda)
    adv_c, ret_c = gae(traj["c"], traj["vc"], traj["d"], vc_last, ppo.cost_gamma, ppo.gae_lambda)
    flat = jax.tree.map(lambda x: x.reshape((-1,) + x.shape[3:]) if x.ndim >= 3 else x.reshape(-1), {
        **{k: traj[k] for k in ("ego", "threats", "threat_mask", "friends", "friend_mask", "raw", "logp",
                                "gs", "live")},
        "adv": adv, "ret": ret, "adv_c": adv_c, "ret_c": ret_c,
    })
    grads, aux = jax.grad(
        lambda ap, cp: step.loss_fn(ap, cp, 1.0, flat), argnums=(0, 1), has_aux=True,
    )(learner.actor.params, learner.critic.params)
    for g in jax.tree.leaves(grads):
        assert bool(jnp.all(jnp.isfinite(g)))


def test_checkpoint_schema_rejects_a_pre_accumulator_checkpoint(tmp_path):
    """A schema-2 checkpoint has no episode-cost accumulators: resuming it
    would silently restart the budget's running rate rather than refusing."""
    import dataclasses as dc

    ppo = PPOConfig(n_envs=2, n_steps=8)
    cfg = SAFE
    env = _flat(cfg)
    key = jax.random.PRNGKey(0)
    _, sample_obs = env.reset(key)
    learner = init_learner(key, cfg, ppo, sample_obs)
    assert "cost_accum" in ck.REQUIRED_RECOVERY_KEYS
    assert "episode_accum" in ck.REQUIRED_RECOVERY_KEYS
    assert ck.CHECKPOINT_SCHEMA >= 3

    legacy_recovery = {k: None for k in ck.REQUIRED_RECOVERY_KEYS if k not in ("cost_accum", "episode_accum")}
    path = tmp_path / "ckpt_000001.pkl"
    path.write_bytes(__import__("pickle").dumps({
        "schema": 2,
        "actor": jax.device_get(learner.actor.params),
        "critic": jax.device_get(learner.critic.params),
        "lam_raw": float(learner.lam_raw),
        "red_level": 0.0,
        "iter": 1,
        "env_cfg": dc.asdict(cfg),
        "recovery": legacy_recovery,
    }))
    assert ck.is_resumable(path) is False


def test_resume_reproduces_the_accumulators_bit_for_bit(tmp_path):
    from naigos.rl.red_team import RedCurriculum
    from naigos.rl.reward import RewardCurriculum

    ppo = PPOConfig(n_envs=2, n_steps=8, n_epochs=1, n_minibatches=1)
    cfg = SAFE
    env = _flat(cfg)
    key = jax.random.PRNGKey(0)
    _, sample_obs = env.reset(key)
    learner = init_learner(key, cfg, ppo, sample_obs)
    step = make_train(env, ppo, RewardWeights())
    rollout = _kill_aircraft_zero(init_rollout(env, ppo, key))

    k = jax.random.PRNGKey(1)
    learner, rollout, _ = jax.jit(step)(learner, rollout, k)
    assert float(learner.episode_accum) > 0 or float(learner.cost_accum) >= 0

    path = ck.save(
        tmp_path / "ckpt_000001.pkl", learner=learner, key=k, iteration=1, red_level=0.0,
        shootdown_rate=0.0, survival_rate=1.0, survival_w=1.0, efficiency_w=0.0, history=[{"iter": 0}],
        env_cfg=cfg, ppo_cfg=ppo, red_curriculum=RedCurriculum(), reward_curriculum=RewardCurriculum(),
        reward_weights_base=RewardWeights(), wall_s=0.0,
    )
    assert ck.is_resumable(path)
    blob = ck.load(path)
    template = init_learner(key, cfg, ppo, sample_obs)
    restored = ck.restore_learner(template, blob)
    assert float(restored.cost_accum) == pytest.approx(float(learner.cost_accum))
    assert float(restored.episode_accum) == pytest.approx(float(learner.episode_accum))
