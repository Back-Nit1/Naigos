"""Complete recovery state: what a training run needs to be picked up mid-flight.

The checkpoints this repository wrote before were *policy exports*. They carried
the actor and critic parameters, the Lagrange multiplier's pre-activation, the
red curriculum level and the iteration number -- everything the demo needs to
fly a route, and not enough to continue training. Restarting from one would have
silently reset:

  * **both optimizer states.** Adam's first and second moment estimates are the
    optimizer. Dropping them restarts from a zero-momentum, zero-variance state
    with a bias correction that thinks it is at step 1, so the first few updates
    after a "resume" are effectively a different learning rate.
  * **the multiplier's optimizer state.** Same problem on the constraint
    channel, where it matters more: the Lagrange multiplier is an integrator, so
    losing its Adam state loses the accumulated price of the cost constraint.
  * **the CMDP cost/episode accumulators.** The measured cost rate is "total
    losses / total completed agent-episodes" across every segment this run has
    ever collected, not a fresh per-segment mean (see `naigos/rl/ppo.py`).
    Dropping these at resume would silently reset that lifetime rate to zero
    and reintroduce the very n_steps-dependence they exist to remove.
  * **the RNG stream.** Every rollout, every minibatch permutation and every
    evaluation draws from one key chain seeded once. Re-seeding at resume makes
    the second half of the run a different sample path, so the run is no longer
    reproducible from its seed and an interrupted run cannot be shown to equal
    an uninterrupted one.
  * **the reward curriculum.** Its dial is derived from the measured shootdown
    rate, which is only measured at an eval boundary. Without it a resumed run
    reverts to the survival-dominant weights it had at iteration 1.
  * **`history.json` continuity.** `run()` rewrites the file from its in-memory
    list, so a resume that started with an empty list would truncate the curve
    to the resumed segment.

So the checkpoint carries all of it. The top-level `actor`, `critic`,
`lam_raw`, `red_level`, `iter` and `env_cfg` keys are kept exactly where they
were, because `naigos/demo/replay.py` and `naigos/demo/live.py` read them
through `load()`, and a checkpoint format change must not break the demo.
Everything new lives under `recovery`.

Restoring is grafting, not unpickling live objects: a `TrainState` holds an
`apply_fn` and an optax `GradientTransformation`, neither of which survives a
checkpoint round trip in a form worth trusting. The learner is rebuilt by
`init_learner` and the saved arrays are grafted onto that structure, with the
pytree structure, shapes and dtypes checked first. A checkpoint whose structure
does not match the code trying to load it is refused rather than silently
half-loaded.

**Schema 4 replaced pickle with Orbax** (next-steps.md G-3). `pickle.load` runs
arbitrary code to reconstruct whatever the bytes claim to be, which makes a
checkpoint from an untrusted source a code-execution vector, not just a data
format risk. Orbax's `PyTreeCheckpointer` restores a pytree of plain arrays and
JSON-safe scalars -- no `__reduce__`, no arbitrary class reconstruction -- and
it is the JAX ecosystem's own answer to exactly this problem, so there is no
hand-rolled array format to get subtly wrong. A checkpoint is now a directory
(`ckpt_NNNNNN/`, written to a temporary name and `os.replace`d into place, same
atomicity reasoning as before) rather than a single pickle file.
`scripts/convert_checkpoint.py` migrates a pre-schema-4 pickle one-way; nothing
in this module reads pickle, by design.

Schema 5 adds explicit environment, observation, and actor identifiers plus the
L1 resource configuration. Schema-4 recovery checkpoints remain resumable only
as baseline-v0; missing fields never imply a resource-world upgrade.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp

from . import runmeta
from . import policies
from .ppo import Learner
from .red_team import RedCurriculum
from .reward import RewardCurriculum, RewardWeights
from ..env.config import (
    RESOURCE_WORLD_BASELINE,
    RESOURCE_WORLD_L1,
    ResourceWorldConfig,
)

#: 1 = the policy-export format written before resume existed. 2 added
#: `recovery`. 3 added the lifetime CMDP accumulators (`cost_accum`,
#: `episode_accum`) that make the measured cost rate invariant to PPO
#: n_steps -- see `naigos/rl/ppo.py`. 4 replaced the pickle file with an Orbax
#: checkpoint directory (next-steps.md G-3). 5 adds explicit resource-world
#: identity and configuration. A pre-4 checkpoint must go through the one-way
#: converter; schema 4 remains a supported baseline recovery format.
CHECKPOINT_SCHEMA = 5
SUPPORTED_CHECKPOINT_SCHEMAS = (4, CHECKPOINT_SCHEMA)

BASELINE_OBSERVATION_SCHEMA = "Observation-v0"
RESOURCE_OBSERVATION_SCHEMA = "L1EnduranceObservation-v1"
BASELINE_ACTOR_SCHEMA = "ActorV1"
RESOURCE_ACTOR_SCHEMA = "ActorV1+L1EnduranceInputs"

REQUIRED_RECOVERY_KEYS = (
    "iteration",
    "rng_key",
    "actor_params",
    "actor_opt_state",
    "actor_step",
    "critic_params",
    "critic_opt_state",
    "critic_step",
    "lam_raw",
    "lam_opt_state",
    "cost_accum",
    "episode_accum",
    "red_level",
    "shootdown_rate",
    "survival_rate",
    "history",
    "red_curriculum",
    "reward_curriculum",
    "reward_weights_base",
)


class CheckpointError(RuntimeError):
    """A checkpoint cannot be read, or does not match the code reading it."""


# --- RNG --------------------------------------------------------------------


def _key_to_record(key) -> dict:
    """Serialise a PRNG key without assuming which JAX key representation is in use.

    JAX has two: a raw `uint32[2]` array and a typed key array. Storing the raw
    data plus the implementation name round-trips both, and a checkpoint written
    under one and read under the other is caught by the impl mismatch rather
    than by producing a different random stream.
    """
    try:
        if jnp.issubdtype(key.dtype, jax.dtypes.prng_key):
            return {
                "impl": str(jax.random.key_impl(key)),
                "data": np.asarray(jax.device_get(jax.random.key_data(key))),
            }
    except (AttributeError, TypeError):  # pragma: no cover - version dependent
        pass
    return {"impl": None, "data": np.asarray(jax.device_get(key))}


def _key_from_record(record: dict):
    if not isinstance(record, dict) or "data" not in record:
        raise CheckpointError("checkpoint has no usable RNG key record")
    data = jnp.asarray(record["data"], dtype=jnp.uint32)
    impl = record.get("impl")
    if impl:
        try:
            return jax.random.wrap_key_data(data, impl=impl)
        except (TypeError, ValueError) as e:  # pragma: no cover - version dependent
            raise CheckpointError(
                f"checkpoint used PRNG implementation {impl!r}, which this JAX cannot "
                f"reconstruct ({e}). Resuming would change the random stream."
            ) from e
    return data


# --- pytree grafting --------------------------------------------------------


def _retype(template, generic):
    """Recover a container type Orbax's templateless restore collapsed to plain
    dict/list (an optax state is a namedtuple; Orbax hands back a list).

    Orbax preserves pytree *flattening order*, just not the container class, so
    the leaves that came off disk are already in the order `template`'s own
    fields declare. Repacking under `template`'s treedef recovers the original
    type exactly -- it is the same trick `_graft` already relies on, run before
    the structural comparison rather than during it.
    """
    return jax.tree_util.tree_unflatten(
        jax.tree_util.tree_structure(template), jax.tree_util.tree_leaves(generic)
    )


def _graft(fresh, saved, what: str):
    """Put `saved`'s leaves into `fresh`'s structure, or refuse.

    The structure comes from the *code* and the values come from the
    *checkpoint*, so a network or optimizer that changed shape between the two
    is an error here instead of a confusing failure a thousand iterations later.
    """
    fresh_leaves, fresh_def = jax.tree.flatten(fresh)
    saved_leaves, saved_def = jax.tree.flatten(saved)
    if fresh_def != saved_def:
        raise CheckpointError(
            f"{what} in this checkpoint has a different structure to the one this code "
            f"builds. The checkpoint was written by different code; resume is refused. "
            f"(checkpoint: {saved_def}; this code: {fresh_def})"
        )
    out = []
    for i, (fl, sl) in enumerate(zip(fresh_leaves, saved_leaves)):
        arr = jnp.asarray(sl)
        want_shape, want_dtype = jnp.shape(fl), jnp.asarray(fl).dtype
        if arr.shape != want_shape:
            raise CheckpointError(
                f"{what} leaf {i}: checkpoint has shape {arr.shape}, this code expects "
                f"{want_shape}"
            )
        out.append(arr.astype(want_dtype))
    return jax.tree.unflatten(fresh_def, out)


def _to_host(tree):
    """Pull a pytree back to numpy so the checkpoint holds no device handles."""
    return jax.tree.map(lambda x: np.asarray(jax.device_get(x)), tree)


# --- write ------------------------------------------------------------------


def save(
    path: str | os.PathLike,
    *,
    learner: Learner,
    key,
    iteration: int,
    red_level: float,
    shootdown_rate: float,
    survival_rate: float,
    survival_w: float,
    efficiency_w: float,
    history: list,
    env_cfg,
    ppo_cfg,
    red_curriculum: RedCurriculum,
    reward_curriculum: RewardCurriculum,
    reward_weights_base: RewardWeights,
    wall_s: float,
    run_name: str | None = None,
    code_commit: str | None = None,
    resumed_from: list | None = None,
) -> Path:
    """Write one checkpoint atomically.

    Atomic because the newest checkpoint is exactly the one a killed container
    was most likely mid-write on. Writing to a temporary directory name and
    renaming means a run that dies here loses the *new* checkpoint, not the
    last good one -- the case `runmeta.select_checkpoint` walks backwards for,
    and it means a partially-written checkpoint never appears under its real
    name at all (`CHECKPOINT_RE` only matches the bare `ckpt_NNNNNN` name).
    """
    # Orbax requires an absolute checkpoint path; callers (train.py, CLI tools)
    # commonly pass a relative one.
    path = Path(path).resolve()
    policy_spec = policies.get_policy_spec(
        env_cfg,
        family=ppo_cfg.policy_family,
        version=ppo_cfg.policy_version,
    )
    blob = {
        # --- unchanged policy-export surface: the demo reads these ---
        "schema": CHECKPOINT_SCHEMA,
        "actor": _to_host(learner.actor.params),
        "critic": _to_host(learner.critic.params),
        "lam_raw": float(learner.lam_raw),
        "red_level": float(red_level),
        "iter": int(iteration),
        "env_cfg": dataclasses.asdict(env_cfg),
        "policy": {
            "family": policy_spec.family.value,
            "version": policy_spec.version,
            "spec": policy_spec.to_dict(),
            "spec_sha256": policy_spec.sha256,
        },
        # Environment/actor semantics. Their absence in an existing schema-4
        # checkpoint means baseline-v0; resource checkpoints never rely on that
        # fallback and carry the full fixed-capacity configuration explicitly.
        "environment_model_id": env_cfg.resource.model,
        "observation_schema_id": (
            RESOURCE_OBSERVATION_SCHEMA if env_cfg.resource.enabled else BASELINE_OBSERVATION_SCHEMA
        ),
        "actor_schema_id": RESOURCE_ACTOR_SCHEMA if env_cfg.resource.enabled else BASELINE_ACTOR_SCHEMA,
        "resource_config": dataclasses.asdict(env_cfg.resource),
        "resource_bank": (
            _resource_bank_record(env_cfg) if env_cfg.resource.enabled else None
        ),
        # --- everything needed to continue ---
        "recovery": {
            "iteration": int(iteration),
            "rng_key": _key_to_record(key),
            "actor_params": _to_host(learner.actor.params),
            "actor_opt_state": _to_host(learner.actor.opt_state),
            "actor_step": int(learner.actor.step),
            "critic_params": _to_host(learner.critic.params),
            "critic_opt_state": _to_host(learner.critic.opt_state),
            "critic_step": int(learner.critic.step),
            "lam_raw": np.asarray(jax.device_get(learner.lam_raw)),
            "lam_opt_state": _to_host(learner.lam_opt),
            "cost_accum": np.asarray(jax.device_get(learner.cost_accum)),
            "episode_accum": np.asarray(jax.device_get(learner.episode_accum)),
            "red_level": float(red_level),
            # Both curricula are driven by these two measured rates and they are
            # only refreshed at an eval boundary, so they are state, not derived.
            "shootdown_rate": float(shootdown_rate),
            "survival_rate": float(survival_rate),
            "survival_w": float(survival_w),
            "efficiency_w": float(efficiency_w),
            "history": list(history),
            # The curricula themselves are frozen dataclasses living in code.
            # Storing their fields makes a code change that moves a promotion
            # threshold a detectable incompatibility rather than an invisible
            # one, which is the whole point of refusing an incompatible resume.
            "red_curriculum": dataclasses.asdict(red_curriculum),
            "reward_curriculum": dataclasses.asdict(reward_curriculum),
            "reward_weights_base": dataclasses.asdict(reward_weights_base),
            "ppo": dataclasses.asdict(ppo_cfg),
            "wall_s": float(wall_s),
            "run_name": run_name,
            "code_commit": code_commit,
            "resumed_from": list(resumed_from or []),
            "resource_curriculum": {"kind": "fixed-v1", "state": None},
        },
    }
    tmp = path.with_name(path.name + ".tmp")
    if tmp.exists():
        import shutil

        shutil.rmtree(tmp)
    ocp.PyTreeCheckpointer().save(tmp, blob)
    os.replace(tmp, path)
    return path


# --- read -------------------------------------------------------------------


def load(path: str | os.PathLike) -> dict:
    try:
        blob = ocp.PyTreeCheckpointer().restore(Path(path).resolve())
    except Exception as e:
        raise CheckpointError(f"{path} is not a readable Naigos checkpoint: {e}") from e
    if not isinstance(blob, dict) or "actor" not in blob:
        raise CheckpointError(f"{path} is not a Naigos checkpoint")
    return blob


def policy_identity(blob: dict) -> dict:
    """Return validated actor identity, migrating metadata-free checkpoints.

    Schema-4 checkpoints written before the model ladder are unambiguously
    ``deepsets@1``. They remain deployable/resumable under that family, while a
    caller asking for a recurrent actor is refused before parameter grafting.
    """

    record = blob.get("policy")
    if record is None:
        return {
            "family": policies.DEFAULT_POLICY_FAMILY.value,
            "version": policies.DEFAULT_POLICY_VERSION,
            "spec": None,
            "spec_sha256": None,
            "migrated_legacy_default": True,
        }
    if not isinstance(record, dict):
        raise CheckpointError("checkpoint policy metadata is not an object")
    family, version = record.get("family"), str(record.get("version", ""))
    try:
        policies.PolicyFamily(family)
    except (TypeError, ValueError) as exc:
        raise CheckpointError(f"checkpoint has unknown policy family {family!r}") from exc
    spec, digest = record.get("spec"), record.get("spec_sha256")
    if spec is not None:
        actual = hashlib.sha256(
            json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if digest != actual:
            raise CheckpointError(
                "checkpoint policy metadata digest does not match its specification"
            )
    return {
        "family": family,
        "version": version,
        "spec": spec,
        "spec_sha256": digest,
        "migrated_legacy_default": False,
    }


def assert_policy_compatible(blob: dict, expected: policies.PolicySpec) -> None:
    """Refuse loading actor weights under a different family/version/spec."""

    saved = policy_identity(blob)
    want = (expected.family.value, expected.version)
    got = (saved["family"], saved["version"])
    if got != want:
        raise CheckpointError(
            f"checkpoint policy is {got[0]}@{got[1]}, but this run requested "
            f"{want[0]}@{want[1]}"
        )
    # Legacy DeepSets checkpoints have no full contract to compare. Concrete
    # PyTree grafting remains the final shape/dtype compatibility check.
    digest = saved.get("spec_sha256")
    if digest is not None and digest != expected.sha256:
        raise CheckpointError(
            "checkpoint policy specification differs from the requested actor "
            f"(checkpoint {digest}, requested {expected.sha256})"
        )


def policy_kwargs(blob: dict) -> dict[str, str]:
    """Keyword arguments for ``ppo.greedy_policy`` from checkpoint metadata."""

    identity = policy_identity(blob)
    return {"family": identity["family"], "version": identity["version"]}


def is_resumable(path: str | os.PathLike) -> bool:
    """True if this file is a complete, schema-4 checkpoint.

    Used as `runmeta.select_checkpoint(..., is_valid=is_resumable)`, so a
    checkpoint orphaned mid-write (still named `*.tmp`, so it never reaches
    here) or a pre-schema-4 pickle is skipped over and the newest genuinely
    resumable checkpoint is chosen instead of being masked by it. Deliberately
    swallows every read error: the caller's next candidate is the right
    response to any of them.
    """
    try:
        blob = load(path)
    except Exception:
        return False
    rec = blob.get("recovery")
    if not isinstance(rec, dict):
        return False
    if blob.get("schema") not in SUPPORTED_CHECKPOINT_SCHEMAS:
        return False
    if any(k not in rec for k in REQUIRED_RECOVERY_KEYS):
        return False
    # the directory name is what `select_checkpoint` sorts on, so it has to
    # agree with the contents or the "most recent" checkpoint is not the most
    # recent one
    named = runmeta.checkpoint_iteration(path)
    return named is None or named == rec.get("iteration")


def latest_resumable(directory: str | os.PathLike) -> Path | None:
    return runmeta.select_checkpoint(directory, is_valid=is_resumable)


def curriculum_compatibility(
    blob: dict,
    *,
    red_curriculum: RedCurriculum,
    reward_curriculum: RewardCurriculum,
    reward_weights_base: RewardWeights,
) -> list[str]:
    """Differences between the curricula in the checkpoint and the ones in code.

    A resume across one of these produces a `history.json` whose two halves were
    trained against different promotion thresholds or different reward weights.
    Reported under the `curriculum` override key.
    """
    rec = blob.get("recovery") or {}
    problems = []
    for label, saved, current in (
        ("red curriculum", rec.get("red_curriculum"), dataclasses.asdict(red_curriculum)),
        ("reward curriculum", rec.get("reward_curriculum"), dataclasses.asdict(reward_curriculum)),
        ("reward weights", rec.get("reward_weights_base"), dataclasses.asdict(reward_weights_base)),
    ):
        if saved is None:
            problems.append(f"{label}: not recorded in this checkpoint")
            continue
        # `diverted` is unreachable in baseline-v0. A schema-4 baseline
        # checkpoint written before that reward field existed is therefore
        # semantically unchanged; normalize only this neutral addition.
        if (
            label == "reward weights"
            and environment_model_from_blob(blob) == RESOURCE_WORLD_BASELINE
            and "diverted" not in saved
        ):
            saved = {**saved, "diverted": current["diverted"]}
        changed = sorted(
            k for k in set(saved) | set(current) if saved.get(k) != current.get(k)
        )
        if changed:
            problems.append(
                f"{label} changed since the checkpoint: {', '.join(changed)} "
                f"(checkpoint {[saved.get(k) for k in changed]}, "
                f"code {[current.get(k) for k in changed]})"
            )
    return problems


def restore_learner(
    template: Learner,
    blob: dict,
    *,
    expected_policy: policies.PolicySpec | None = None,
) -> Learner:
    """Graft a checkpoint's arrays onto a freshly initialised learner.

    `template` must come from `init_learner` with the same configuration, so the
    optimizer transformations, the network structure and the multiplier's Adam
    state are the ones this code builds. Only the values come from disk.
    """
    if expected_policy is not None:
        assert_policy_compatible(blob, expected_policy)
    rec = blob.get("recovery")
    if not isinstance(rec, dict):
        raise CheckpointError(
            "this checkpoint predates resume support (no `recovery` block): it holds the "
            "policy but not the optimizer state, the RNG stream or the curriculum state. "
            "Training cannot be continued from it; it can only be replayed."
        )
    missing = [k for k in REQUIRED_RECOVERY_KEYS if k not in rec]
    if missing:
        raise CheckpointError(f"checkpoint recovery block is missing {missing}")

    # `load` restores without a template, so an optax state (a namedtuple) came
    # back as a plain list -- see `_retype`.
    actor_opt_state = _retype(template.actor.opt_state, rec["actor_opt_state"])
    critic_opt_state = _retype(template.critic.opt_state, rec["critic_opt_state"])
    lam_opt_state = _retype(template.lam_opt, rec["lam_opt_state"])

    actor = template.actor.replace(
        params=_graft(template.actor.params, rec["actor_params"], "actor parameters"),
        opt_state=_graft(template.actor.opt_state, actor_opt_state, "actor optimizer state"),
        step=jnp.asarray(rec["actor_step"], dtype=jnp.asarray(template.actor.step).dtype),
    )
    critic = template.critic.replace(
        params=_graft(template.critic.params, rec["critic_params"], "critic parameters"),
        opt_state=_graft(template.critic.opt_state, critic_opt_state, "critic optimizer state"),
        step=jnp.asarray(rec["critic_step"], dtype=jnp.asarray(template.critic.step).dtype),
    )
    return Learner(
        actor=actor,
        critic=critic,
        lam_raw=_graft(template.lam_raw, rec["lam_raw"], "lagrange multiplier"),
        lam_opt=_graft(template.lam_opt, lam_opt_state, "multiplier optimizer state"),
        cost_accum=_graft(template.cost_accum, rec["cost_accum"], "CMDP cost accumulator"),
        episode_accum=_graft(template.episode_accum, rec["episode_accum"], "CMDP episode accumulator"),
    )


def restore_key(blob: dict):
    return _key_from_record((blob.get("recovery") or {}).get("rng_key"))


# --- observation layout -----------------------------------------------------


def obs_config_from_blob(blob: dict) -> dict:
    """The observation flags the checkpoint's actor was trained with, as
    `EnvConfig.replace` overrides.

    A checkpoint written before a flag existed (checkpoints/theatre_1000) does
    not record it, and was trained without it.
    """
    env_cfg = blob.get("env_cfg") or {}
    out = {"obs_edge_features": bool(env_cfg.get("obs_edge_features", False))}
    if environment_model_from_blob(blob) == RESOURCE_WORLD_L1:
        out["resource"] = resource_config_from_blob(blob)
    return out


def _resource_bank_record(env_cfg) -> dict:
    from ..env.resource_bank import base_config_digest, resource_protocol

    protocol = resource_protocol()
    return {
        "version": protocol["bank"],
        "sha256": protocol["bank_sha256"],
        "base_config_sha256": base_config_digest(env_cfg),
    }


def environment_model_from_blob(blob: dict) -> str:
    """Missing schema-4 field means baseline-v0, never an implicit upgrade."""
    return str(blob.get("environment_model_id") or RESOURCE_WORLD_BASELINE)


def resource_config_from_blob(blob: dict) -> ResourceWorldConfig:
    model = environment_model_from_blob(blob)
    if model == RESOURCE_WORLD_BASELINE:
        return ResourceWorldConfig()
    if model != RESOURCE_WORLD_L1:
        raise CheckpointError(f"checkpoint names unsupported environment model {model!r}")
    raw = blob.get("resource_config")
    if not isinstance(raw, dict):
        raise CheckpointError(
            f"{RESOURCE_WORLD_L1} checkpoint is missing its resource_config; implicit migration is refused"
        )
    fields = {field.name for field in dataclasses.fields(ResourceWorldConfig)}
    values = {name: value for name, value in raw.items() if name in fields}
    for name in ("fuel_budget_ratio", "reserve_ratio", "burn_multiplier", "active_gate_mask"):
        if name in values:
            values[name] = tuple(values[name])
    config = ResourceWorldConfig(**values)
    if config.model != model:
        raise CheckpointError(
            f"checkpoint environment_model_id {model!r} disagrees with resource_config.model {config.model!r}"
        )
    return config


def require_environment_compatible(blob: dict, cfg, source) -> None:
    """Refuse cross-loading L0/ActorV1 and L1/ActorV2 checkpoints."""
    saved_model = environment_model_from_blob(blob)
    saved_obs = str(blob.get("observation_schema_id") or BASELINE_OBSERVATION_SCHEMA)
    wanted_obs = RESOURCE_OBSERVATION_SCHEMA if cfg.resource.enabled else BASELINE_OBSERVATION_SCHEMA
    saved_actor = str(blob.get("actor_schema_id") or BASELINE_ACTOR_SCHEMA)
    wanted_actor = RESOURCE_ACTOR_SCHEMA if cfg.resource.enabled else BASELINE_ACTOR_SCHEMA
    if (saved_model, saved_obs, saved_actor) != (cfg.resource.model, wanted_obs, wanted_actor):
        raise CheckpointError(
            f"{source}: checkpoint is {saved_model}/{saved_obs}/{saved_actor}, but the environment is "
            f"{cfg.resource.model}/{wanted_obs}/{wanted_actor}; cross-schema loading is refused"
        )
    if cfg.resource.enabled and dataclasses.asdict(resource_config_from_blob(blob)) != dataclasses.asdict(cfg.resource):
        raise CheckpointError(f"{source}: checkpoint resource_config differs from the requested L1 environment")
    if cfg.resource.enabled:
        saved_bank = (blob.get("resource_bank") or {}).get("sha256")
        want_bank = _resource_bank_record(cfg)["sha256"]
        if saved_bank != want_bank:
            raise CheckpointError(
                f"{source}: checkpoint resource bank digest {saved_bank!r} differs from this "
                f"code's {want_bank!r}; the frozen L1 bank changed, resume is refused"
            )


def require_resume_compatible(blob: dict, cfg, ppo_cfg, policy_spec, source) -> None:
    """Every identity a resumed run must share with its checkpoint, checked up front.

    Resource identity (model, observation schema, config, bank digest) and policy
    identity (family, version, spec) are independent axes; each is validated on
    its own so the error names the axis that moved. A recurrent checkpoint also
    pins its truncated-BPTT sequence length, which changes the update.
    """
    require_environment_compatible(blob, cfg, source)
    assert_policy_compatible(blob, policy_spec)
    if policy_spec.memory_requirements.recurrent:
        saved = (blob.get("recovery") or {}).get("ppo", {}).get("recurrent_seq_len")
        if saved is None or int(saved) != int(ppo_cfg.recurrent_seq_len):
            raise CheckpointError(
                f"{source}: checkpoint recurrent_seq_len is {saved!r}, but this run requested "
                f"{ppo_cfg.recurrent_seq_len}"
            )


def require_actor_ego_dim(actor_params, cfg, source) -> None:
    """Refuse an actor whose ego input width is not the one `cfg` builds.

    The width is read off the threat encoder's ego-query projection, whose
    input is the ego vector alone.
    """
    got = int(np.shape(actor_params["params"]["threat_encoder"]["Dense_2"]["kernel"])[0])
    if got != cfg.ego_dim:
        raise SystemExit(
            f"{source}: the actor takes a {got}-wide ego observation but this env builds "
            f"{cfg.ego_dim} (obs_edge_features={cfg.obs_edge_features}, "
            f"environment_model_id={cfg.resource.model}). The checkpoint's "
            f"env_cfg does not describe the observation its actor was trained on."
        )
