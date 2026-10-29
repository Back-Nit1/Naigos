# Qwen route-intent planner

Status: architecture, immutable model staging, trajectory preparation, LoRA
training, and fixed-label evaluation are implemented. No planner dataset has
been branch-labeled, no adapter has been trained, and no online result is
claimed.

## Boundary

Qwen is not a flight-control policy. It selects exactly one of nine evasive
route intents every future planning interval:

| labels | cross-track request | altitude request |
| --- | --- | --- |
| A / B / C | left 15 km | 200 m AGL / 1,000 m AGL / 4,000 m AMSL |
| D / E / F | center | 200 m AGL / 1,000 m AGL / 4,000 m AMSL |
| G / H / I | right 15 km | 200 m AGL / 1,000 m AGL / 4,000 m AMSL |

The deterministic adapter is `naigos.planner.intents.route_waypoint`. The native
DeepSets/GRU actor remains responsible for bank, flight-path angle, and throttle,
and the CBF remains the runtime safety backstop. Free text is never executable.

Torch, Transformers, and PEFT are optional and isolated from JAX PPO. Ordinary
Naigos imports and training do not import them.

## Pinned base model

The selected base is `Qwen/Qwen3-0.6B-Base` at immutable Hugging Face revision
`da87bfb608c14b7cf20ba1ce41287e8de496c0cd`. The committed manifest
`naigos/planner/qwen3_0_6b_base.json` pins all nine source files by byte count
and SHA-256. Model bytes live under ignored `models/` and never enter git.

```bash
# Stage once on each training machine.
python scripts/qwen_planner.py fetch-model
python scripts/qwen_planner.py verify-model

# Install only when training/evaluating Qwen.
uv pip install -e '.[planner]'
```

After staging, training and evaluation set `local_files_only=True` and make no
network request.

## 1. Record RL trajectories

The theatre launcher now exposes the recorder directly:

```bash
python scripts/train_theatre.py \
  --aoi owens_valley --route-mode diverse \
  --record-trajectories runs/qwen-source/trajectories \
  --record-chunk-steps 512 --out runs/qwen-source
```

Do not use every online PPO transition as a positive demonstration. Prefer
fixed promoted-policy evaluation rollouts, retain failed episodes as explicit
counterexamples, and keep entire scenario seeds in one data split.

## 2. Produce counterfactual labels

A recorded trajectory says what one policy did. It cannot establish which of
nine alternative intents would have been best. For that reason `prepare`
requires an external branch-label JSONL rather than inventing labels from
actions. The future branch collector should run all nine intents for 16 steps
under common random numbers and rank them lexicographically by survival,
constraint cost, progress, exposure, and fuel.

Each label row is:

```json
{"content_sha256":"...","episode_id":12,"step_index":63,"agent":2,"label":"E","split":"train"}
```

For an immediate behavior-distillation smoke test, Naigos can derive explicitly
weak labels from what the recorded policy actually did: mean future executed
bank selects left/center/right and the horizon endpoint selects the closest
altitude band. Successful episodes are the default. These labels imitate one
behavior and do not establish the best counterfactual intent:

```bash
python scripts/qwen_planner.py label-behavior \
  --trajectory runs/qwen-source/trajectories/<dataset> \
  --split train --out runs/qwen-source/behavior-labels.jsonl
```

Use separate scenario-seed shards for validation and pass `--split validation`.
Do not randomly split rows from one trajectory dataset.

One immutable trajectory dataset may belong to only one split. Theatre, seed,
and scenario identities are checked across inputs so a scenario cannot leak
between train and validation.

```bash
python scripts/qwen_planner.py prepare \
  --trajectory runs/qwen-source/trajectories/<dataset> \
  --labels runs/qwen-source/branch-labels.jsonl \
  --out runs/qwen-source/planner.jsonl
```

Preparation streams bounded trajectory chunks, uses only actor-visible fields,
and requires a complete causal 16-observation history. Every prompt and final
JSONL file is content-hashed.

## 3. LoRA train and evaluate

```bash
python scripts/qwen_planner.py train \
  --data runs/qwen-source/planner.jsonl \
  --out runs/qwen-adapter --device auto

python scripts/qwen_planner.py verify-adapter --adapter runs/qwen-adapter

python scripts/qwen_planner.py evaluate \
  --data runs/qwen-source/planner.jsonl \
  --adapter runs/qwen-adapter --split validation --device auto
```

The fixed recipe applies rank-8, alpha-16, zero-dropout LoRA to `q_proj` and
`v_proj` only, trains next-token labels A--I, caps prompts at 512 tokenizer
tokens, and refuses an unexpected trainable-parameter count. Evaluation scores
only the nine label logits and returns accuracy plus the complete confusion
matrix; it never generates prose.

This offline accuracy is only a screen. A planner is useful only if a later
cached, deterministic, closed-loop evaluation improves held-out Naigos outcomes
against both the native numeric planner and the frozen low-level actor.
