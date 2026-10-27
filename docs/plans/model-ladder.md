# Naigos model ladder

Status: research recommendation, 2026-10-26. No architecture, dependency,
checkpoint, or remote resource was changed while producing this plan. No model
weights were downloaded. Every experiment below is proposed unless it is
explicitly labelled **repository fact**, **measured**, or **calculated**.

## Decision

Keep the shipped DeepSets actor as `deepsets@1` and make the main ladder:

1. **DeepSets v1 (control):** the current actor, unchanged.
2. **DeepSets + GRU-128:** the smallest credible test of whether temporal memory
   helps the partially observed routing task.
3. **Numeric causal transformer:** four 128-wide causal blocks over 32 numeric
   observation steps, trained from scratch with the existing online
   MAPPO-Lagrangian objective.
4. **Optional high-level language planner:** only after a native temporal model
   passes. Screen `HuggingFaceTB/SmolLM2-135M-Instruct`; test
   `Qwen/Qwen3-0.6B` only if the screen shows planner value. The language model
   may select one of nine purely evasive route intents every 16 environment
   steps. It never emits flight controls and never changes the environment's
   action space.

Do not put Falcon-7B or OpenLLaMA-7B in the main ladder. They are obsolete,
expensive controls for this responsibility. If their weights are already
available locally in a future study, at most use one for an offline planner
ceiling check; do not train it or make it a promotion candidate.

This ordering tests one hypothesis at a time: first memory, then explicit
finite-horizon attention, then whether language pretraining contributes
anything to an intentionally narrow planning interface. Deep Sets are a sound
base for the unordered sensed-neighbour inputs because permutation-invariant
set functions are exactly the responsibility that architecture was designed
for [Deep Sets paper](https://papers.nips.cc/paper/2017/file/f22e4747da1aa27e363d86d40ff442fe-Paper.pdf).

## Non-negotiable interface

Every deployable policy remains local-observation-only. The only environment
action remains a float tensor shaped `(n_blue, 3)` containing bank,
flight-path-angle, and throttle commands. All native actors retain the existing
tanh-squashed diagonal Gaussian head. Blue receives no offensive capability,
no target selector, and no extra action axis.

The optional language planner is advisory. Its complete output vocabulary is
the Cartesian product of:

- cross-track waypoint offset: `left_15km`, `center`, `right_15km`; and
- requested altitude band: `agl_0.2km`, `agl_1.0km`, `amsl_4.0km`.

That is nine route-intent IDs. A deterministic adapter clips the waypoint to the
play area, replaces only the actor's internal routing objective, and holds the
intent for 16 steps (32 simulated seconds). The low-level actor still produces
exactly the three controls. The planner sees the same local sensed information
as the aircraft, summarized over time; it receives no ground-truth threat list,
future state, verifier result, or visual data. There is no free-text execution
path: score the nine fixed labels and take `argmax`.

## What exists now

The following are repository facts unless noted otherwise.

- Naigos is JAX, Flax Linen, Optax, and Orbax. The actor sees local observations;
  the centralized critic is training-only and is excluded from every actor
  count below. Under the proposed five-kind configuration the current
  agent-conditioned critic has 172,546 parameters (the historical checkpoint
  carries its earlier 168,962-parameter critic).
- `networks.Actor` encodes sensed threats with masked DeepSets mean/max pooling
  plus ego-query attention, encodes friendly aircraft with masked DeepSets
  pooling, and emits the three-control Gaussian.
- The shipped `checkpoints/theatre_1000` checkpoint is schema 4, has five threat
  classes, a 10-wide ego vector, and **320,518 actor parameters**. This count was
  read from its actual actor PyTree. The default three-kind synthetic config has
  320,262 parameters.
- The shipped actor occupies **1.223 MiB** as raw FP32 parameters. This is not
  peak process memory and excludes the critic, optimizer, XLA executable, and
  activations.
- The current observation has no explicit temporal window. Threat track quality
  carries some environment memory, but the actor cannot directly estimate
  motion or distinguish otherwise similar observation histories.
- Environment steps are two simulated seconds. A 32-step temporal window is 64
  seconds, comfortably longer than the modelled 2.5--6 second reaction delays
  while still short relative to a 500-step sortie.
- Training already records deterministic seeds and provenance, uses complete
  optimizer/RNG/curriculum checkpoint recovery, evaluates fixed held-out seeds,
  has a frozen 96-scenario route bank, and independently verifies logged traces
  in NumPy.
- The repository's current measured result is one 1,000-iteration CPU seed on
  one training theatre. It is evidence of learning, not convergence or an
  architecture comparison. Nothing in this document upgrades that claim.
- The working tree already contains a policy-factory seam in
  `naigos/rl/policies.py`: `deepsets@1` is the only registered implementation;
  recurrent and numeric-transformer names fail closed. This plan relies on that
  seam but does not modify it.

## Common comparison table

Small-actor counts below are exact calculations for the shipped five-kind
Owens Valley observation shape. Proposed counts must be asserted against the
concrete Flax PyTree when implemented. Weight memory is raw parameter storage,
not runtime peak. Language-model sizes are the publishers' advertised counts;
INT8/INT4 columns are bit-count lower bounds before scales, metadata, untied
buffers, KV cache, and runtime workspace.

| Candidate | Responsibility | Parameters | Raw inference weights | Temporal/context requirement | Continuous-action compatibility |
|---|---:|---:|---:|---|---|
| DeepSets v1 | low-level actor | **320,518 measured** | 1.223 MiB FP32; 0.611 MiB BF16 | current observation only | Native: unchanged Gaussian over 3 controls |
| DeepSets + GRU-128 | low-level actor | **335,494 calculated** | 1.280 MiB FP32; 0.640 MiB BF16; 512 B hidden state/agent | recurrent state; train with 32-step truncated BPTT | Native: same head |
| Numeric causal transformer | low-level actor | **754,438 calculated** | 2.878 MiB FP32; 1.439 MiB BF16; about 0.125 MiB FP32 KV/agent at 32 steps | exactly 32 observation steps / 64 simulated seconds | Native: same head |
| SmolLM2-135M-Instruct | optional macro planner | 135M advertised | about 257 MiB BF16, 129 MiB INT8, 64 MiB INT4 | publisher limit 8,192 tokens; cap Naigos prompt at 512 | Indirect only: choose 1 of 9 intents |
| Qwen3-0.6B | optional macro planner follow-up | 0.6B advertised, 0.44B non-embedding | about 1.12 GiB BF16, 0.56 GiB INT8, 0.28 GiB INT4 | model card says 32,768; config declares 40,960 positions; cap at 512 | Indirect only: choose 1 of 9 intents |
| Falcon-7B-Instruct | rejected control | 7B advertised | about 13.0 GiB BF16, 6.5 GiB INT8, 3.3 GiB INT4 | 2,048 tokens | Indirect only |
| OpenLLaMA-7B-v2 | rejected control | 6.738B calculated from config | 12.55 GiB FP16, 6.28 GiB INT8, 3.14 GiB INT4 | 2,048 tokens | Indirect only |

The GRU count is for the current two set encoders, a `266 -> 128` projection, a
Flax `GRUCell(128)`, a `128 -> 128` head layer, the existing 3-wide mean, and
3 learned log standard deviations. GRUs were introduced as a gated recurrent
sequence model in [Cho et al.](https://aclanthology.org/D14-1179.pdf).

The transformer count is for the same set encoders, a `266 -> 128` projection,
32 learned positional embeddings, four pre-norm blocks with four heads and a
256-wide MLP, final layer norm, and the existing Gaussian head. It is a causal
numeric policy, not a text model. Transformers provide parallel sequence
processing [in the original paper](https://papers.neurips.cc/paper/7181-attention-isall-you-need.pdf),
and causal transformers have precedent for continuous control in
[Decision Transformer](https://papers.neurips.cc/paper/2021/hash/7f489f642a0ddb10272b5c31057f0663-Abstract.html).
Naigos should not copy Decision Transformer's offline return-conditioned
objective: the controlled comparison keeps online MAPPO-Lagrangian and changes
only the actor's temporal encoder.

### Runtime, framework, license, and export

Latency entries are **planning estimates, not measurements**. They give an
order of magnitude and a promotion budget; the implementation must record
warm-compiled p50/p95/p99 on the same CPU and accelerator, batch size one world
(four agents), over 10,000 calls. Compilation time is reported separately.

| Candidate | Expected inference latency | JAX/Flax status | Non-JAX integration cost | License | Quantization/export |
|---|---|---|---|---|---|
| DeepSets v1 | reference; sub-ms accelerator or low-single-digit-ms CPU after JIT is plausible; require p99 <= 5 ms | Native Linen today | None | Naigos-owned weights; the repository currently has no top-level license file | Orbax today; BF16/FP16 and `jax.export`/StableHLO are available paths; quantify any lower precision before adoption |
| DeepSets + GRU-128 | roughly 1--2x DeepSets; require p99 <= 7.5 ms | Native `flax.linen.GRUCell` | None | Same as Naigos | Same as DeepSets; recurrent state is an explicit deployable tensor |
| Numeric causal transformer | roughly 2--6x DeepSets for a full 32-step window, 1.5--3x with a correct KV cache; require p99 <= 20 ms | Native Linen attention | None | Same as Naigos | Orbax/StableHLO; test BF16 first. Qwix offers JAX PTQ/QAT, but quantization is a separate promoted artifact, never assumed equivalent |
| SmolLM2-135M-Instruct | tens of ms on a suitable GPU and sub-second-to-seconds on CPU for a <=512-token prompt plus one-label scoring; require batched p99 <= 500 ms | No publisher-supported Naigos/JAX checkpoint path found | High: Torch, Transformers, tokenizer, adapter tooling, separate device/RNG/checkpoint lifecycle; cannot run inside the jitted rollout | Apache-2.0 | Publisher supplies BF16 Safetensors and lists ONNX/Transformers.js; produce and hash any INT8/INT4/GGUF conversion locally, then re-evaluate it |
| Qwen3-0.6B | tens-to-hundreds of ms GPU and seconds CPU under the same short scoring workload; require batched p99 <= 500 ms | Supported by Google's JAX/Flax-NNX MaxText, but MaxText is not a drop-in Linen dependency | Medium-high via MaxText conversion; high via Torch/Transformers. Either path is a second model/checkpoint stack | Apache-2.0 | BF16 Safetensors; Qwen documents GGUF and weight quantization. First scientific comparison stays BF16 |
| Falcon-7B-Instruct | hundreds of ms GPU or multi-second CPU is plausible; outside the desired budget/cost envelope | Official card requires PyTorch 2.0; no supported Naigos JAX path identified | Very high, plus the card recommends at least 16 GB for swift inference | Apache-2.0 | Safetensors/Transformers and ecosystem low-bit paths; irrelevant unless used as an offline control |
| OpenLLaMA-7B-v2 | similar class to Falcon; 32-head KV cache is much larger | Publisher released legacy EasyLM JAX weights as well as PyTorch weights | High even on JAX: EasyLM is a separate older stack and checkpoint format | Apache-2.0 | EasyLM or Transformers export, with separately validated low-bit conversion |

`jax.export` serializes StableHLO plus call metadata, but JAX documents a
limited compatibility window; it is an export option, not an archival
replacement for the Orbax checkpoint and its provenance
([JAX export documentation](https://docs.jax.dev/en/latest/export/export.html)).
[Qwix](https://github.com/google/qwix) is the current Google JAX quantization
library replacing AQT. No quantized native policy should be compared to an
FP32 policy without a separate artifact identity and the complete held-out
evaluation.

## Does language pretraining help here?

Probably not for low-level control. The observation is 10 ego scalars plus two
small padded numeric sets, and the output is a three-dimensional continuous
distribution. Language pretraining supplies neither aerodynamic dynamics nor
the meaning of those normalized features. Tokenizing floats also discards the
compact, smooth geometry that the native actors consume directly. A language
model in the two-second control loop is therefore rejected.

There is one narrow, falsifiable reason to test language pretraining: an
instruction-tuned model may be good at mapping a short structured history and
explicit prose constraints to a stable discrete route intent. SmolLM2's model
card describes a 135M decoder trained on two trillion tokens and then SFT/DPO,
under Apache-2.0
([model card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct),
[configuration](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/blob/main/config.json)).
Qwen3-0.6B is pretrained and post-trained, supports non-thinking deterministic
use, advertises 0.6B parameters and 32,768-token context, and is Apache-2.0
([model card](https://huggingface.co/Qwen/Qwen3-0.6B),
[configuration](https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/config.json)).
Google MaxText lists Qwen3 dense 0.6B among its supported JAX models, but MaxText
has moved to Flax NNX and brings an independent training/serving stack
([MaxText](https://github.com/AI-Hypercomputer/maxtext)).

Language pretraining counts as helpful only if the pretrained planner beats a
numeric planner trained on the identical nine-way labels and input information,
then also improves paired online held-out outcomes. Better formatting accuracy
or a stronger language benchmark is not evidence for Naigos. If the numeric
planner ties or wins, stop the language branch.

## Training recipe by rung

### Shared experimental protocol

For an architecture result, retrain the DeepSets control under the same new
protocol; do not compare a new model trained on diverse routes to the historical
west-to-east checkpoint and call the difference architectural.

- Environment: Owens Valley components pinned by content hash, 500 m cells,
  `los_samples=96`, four blue aircraft, 16 threats, no CBF during the primary
  architecture comparison, `route_mode="diverse"`, and
  `obs_edge_features=False` so the input layout stays checkpoint-compatible.
- Action and critic: unchanged three-control tanh-Gaussian actor head and
  unchanged agent-conditioned centralized two-head critic.
- Optimizer/objective: current MAPPO-Lagrangian defaults. Do not tune learning
  rate, entropy, reward, curriculum, or CBF separately by architecture in the
  primary comparison.
- Seeds: training seeds `{0, 1, 2}`. These are disjoint from pipeline held-out
  seeds and route-bank reserved ranges.
- Smoke: one seed, 3 iterations x 16 worlds x 32 steps = **1,536 env-steps**.
  This proves shapes, compilation, checkpoint read-back, deterministic replay,
  and finite gradients only.
- Screen: one seed, 200 iterations x 128 worlds x 128 steps = **3,276,800
  env-steps**. It is a throughput and learning-direction screen, not a reported
  result.
- Confirmation: all three seeds, each 3,000 iterations x 256 worlds x 128 steps
  = **98,304,000 env-steps** and **393,216,000 agent-steps** per seed. One rung
  therefore costs 294,912,000 env-steps over three seeds. Do not label it
  converged unless the red curriculum reaches 1.0 and the final 10 evaluations
  (250 iterations) change survival and objective rate by no more than 0.02
  end-to-end.
- Checkpoint every 200 iterations and evaluate every 25, matching the current
  full profile. Record compile time, steady throughput, peak device memory,
  parameter count, actor state bytes, and inference percentiles.
- Early stop only for non-finite values, verifier mismatch, invariant failure,
  or failure to improve over the iteration-0 policy by iteration 200. A stopped
  arm is a failed screen, not a zero-filled result.

The proposal does not authorize a remote run. Existing smoke gates still apply,
and no Modal deployment or paid resource should occur without a separate user
request.

### Rung 0: DeepSets v1

- Preserve `networks.Actor` exactly and keep `deepsets@1` as the default when a
  checkpoint has no policy metadata.
- Keep `checkpoints/theatre_1000` immutable as the historical baseline.
- Train a fresh three-seed control under the shared protocol. This separates
  gains from route diversification, longer training, and seed variance.
- Use the current greedy `tanh(mean)` policy for evaluation and the stochastic
  policy for collection.

### Rung 1: DeepSets + GRU-128

- Apply the existing threat/friendly encoders independently at each step.
- Concatenate ego and both 128-wide set embeddings (266 wide for the shipped
  observation), project to 128, update one GRU-128 state per aircraft, apply one
  128-wide tanh head layer, then reuse the exact Gaussian head.
- Carry hidden state across persistent rollout segments; reset an individual
  aircraft's state when its episode ends. Do not share recurrent state between
  aircraft.
- Train in 32-step sequence chunks with truncated backpropagation through time.
  PPO minibatches are sequences, never shuffled individual timesteps. Store the
  initial hidden state for every chunk so PPO recomputation uses the state that
  collection used.
- Keep the critic feed-forward. This isolates actor memory and avoids making
  centralized privileged memory a deployment dependency.
- On resume, continue the current documented behavior: parameters, optimizers,
  RNG, curricula, and history restore exactly, while in-flight episodes restart.
  The restarted episodes also start with zero GRU state. Two resumes of the same
  checkpoint must still be bit-identical to one another.

### Rung 2: numeric causal transformer

- Reuse the two set encoders at every timestep. Project the 266-wide embedding
  to `d_model=128`.
- Use context 32, four causal pre-norm blocks, four 32-wide heads, GELU MLP width
  256, learned positions 0--31, no dropout, and a final layer norm. Predict the
  same Gaussian for every valid timestep; deployment consumes the last one.
- Inputs are observations and reset/valid masks only. Do not add text tokens,
  return-to-go, future rewards, future actions, absolute map coordinates, or
  ground-truth threats.
- Reset attention at episode boundaries. A test must prove that changing future
  observations cannot change an earlier action and that tokens before a reset
  cannot affect tokens after it.
- First implementation may recompute the 32-step window. Add a KV cache only
  after cached and uncached logits agree within `1e-6` on fixed traces; cache
  correctness is more important than the small latency win at a 0.5 Hz control
  rate.
- Use 32-step PPO chunks and sequence-level shuffling as for the GRU.

### Rung 3: compact pretrained planner (conditional)

This is a separate hierarchical experiment, not another low-level actor.

1. Freeze the promoted native low-level actor.
2. From training-seed rollouts only, collect **50,000 training** and **5,000
   validation** planner states. Each record contains the last 16 local
   observations, quantized deterministically into a <=512-token JSON prompt.
3. Label each state by branching the nine route intents for 32 simulated steps
   under common random numbers and choosing the lexicographic best outcome:
   alive first, then constraint cost, objective progress, exposure, and fuel.
   Labels may use the simulator; deployment prompts may not use hidden state.
4. Train a numeric control on the same tensors and labels: the rung-2 temporal
   embedding plus a nine-way linear head.
5. Screen SmolLM2-135M-Instruct in deterministic non-sampling mode, then apply
   LoRA rank 8, alpha 16, dropout 0 to `q_proj` and `v_proj` only: **460,800
   trainable parameters**, three epochs, batch 128, AdamW `2e-4`, fixed example
   order derived from seed 0. Score single-token labels `A`--`I`; generate no
   prose.
6. Test Qwen3-0.6B only if SmolLM2 beats the numeric offline control or is within
   2 percentage points while showing a clear online benefit. Use non-thinking
   mode and the same LoRA recipe: **1,146,880 trainable parameters** from the
   published layer dimensions.
7. Record the exact upstream revision, tokenizer revision, license, source
   URLs, SHA-256 of every staged file, dtype/quantization, prompt schema, LoRA
   config, and deterministic decoding settings. Training and evaluation remain
   network-blocked after the immutable assets are staged.

The 55,000 planner examples are split by scenario seed, never by row, so states
from one sortie cannot occur in both train and validation. Language models do
not enter the JAX PPO process. A separate offline process produces a
prompt-hash-to-intent cache during a generation pass. A second, network-blocked
evaluation replays from the same seeds, consumes the now-immutable cache, and
must reproduce the first pass exactly; it fails closed on a missing or malformed
decision. This avoids mixed Torch/JAX allocators in the evidence-producing pass
and makes exact re-evaluation possible.

## Held-out evaluation shared by every rung

Use the final checkpoint selected without consulting these scenarios. Evaluate
every training seed's checkpoint separately, then report the distribution over
training seeds and paired scenario outcomes.

1. **Frozen stochastic scenarios:** seeds `900001`--`900008`, 48 worlds per
   seed, at red levels `{0.0, 0.5, 1.0}`. That is 384 worlds / 1,536 sorties per
   level and 1,152 worlds / 4,608 sorties per checkpoint. Every policy and both
   scripted baselines receive identical worlds.
2. **Frozen route bank v1:** all 48 `train_geometry` and 48 `heldout` scenarios
   at red levels `{0.0, 1.0}`: 192 worlds / 768 sorties per checkpoint. Report
   every family and the held-out-minus-train-geometry generalization gap.
3. **Theatre transfer:** the same 384 stochastic worlds on `tehran_basin` at red
   level 1.0, zero-shot. This is a transfer result, never a trained-on-Tehran
   result.
4. **Independent verification:** verify at least 32 deterministically selected
   episodes per checkpoint (the first four worlds from each frozen stochastic
   seed). Any mismatch fails the checkpoint.
5. **CBF secondary A/B:** only after selecting an architecture, replay the exact
   frozen scenarios with and without the CBF. CBF results do not decide the
   primary architecture winner because the current filter materially changes
   completion and infeasibility.

For every split report survival, objective, shootdown, terrain, bounds, timeout,
early and full exposure, fuel exhaustion, mean route length/time, and inference
latency. Report raw counts and rates. The unit for bootstrap resampling is the
world, not an aircraft, because four sorties in one world share terrain and
threats.

## Promotion criteria

Promotion is sequential: Rung 1 challenges Rung 0; Rung 2 challenges the winner;
the language planner challenges the promoted native actor plus the numeric
planner. Use 10,000 paired hierarchical bootstrap replicates over training seed
and world, with a fixed bootstrap seed recorded in the evaluation artifact.

A candidate is promoted only when all of the following hold:

1. **Integrity:** zero action-contract violations, zero non-finite actions, zero
   verifier mismatches, disjoint training/evaluation seeds, complete provenance,
   clean code identity, and deterministic re-evaluation within the current
   `1e-6` tolerance.
2. **Training completeness:** all three runs reach red level 1.0, have a valid
   final checkpoint, and meet the plateau rule above. Results from fewer than
   three completed seeds are inconclusive.
3. **Existing gates:** aggregate survival >= 0.60 and objective >= 0.50; no worse
   than avoid-plus-nap on survival and no more than 0.05 worse on objective.
   Preserve the current champion tolerances for exposure, shootdown, terrain,
   bounds, survival, and objective.
4. **Paired non-regression on both primary Owens Valley splits:** the 95%
   confidence lower bound for candidate-minus-champion survival and objective is
   at least `-0.02`; the upper bound for exposure, shootdown, terrain, and bounds
   regression is at most `+0.02`.
5. **Practical superiority:** on either frozen stochastic red level 1.0 or the
   held-out route split, at least one of survival/objective improves by at least
   0.02 with a 95% lower bound above zero, or exposure/shootdown falls by at
   least 0.02 with a 95% upper bound below zero. A parameter increase with only
   a statistical tie does not climb the ladder.
6. **Transfer floor:** Tehran survival and objective may each regress by at most
   0.05 in point estimate, and no loss channel may worsen by more than 0.05.
   Transfer is a guard against brittleness, not a claim of theatre training.
7. **Deployment budget:** native actors meet their p99 latency budgets and add
   no more than 4 MiB of raw FP32 actor parameters. The planner meets 500 ms p99
   for four batched aircraft at its 32-second cadence, returns a valid label on
   every prompt, and its full runtime peak is recorded.
8. **Language-specific evidence:** the pretrained planner must beat the numeric
   nine-way planner by at least 2 percentage points of scenario-grouped
   validation accuracy and satisfy the online superiority rule. Otherwise the
   conclusion is that language pretraining did not help this responsibility.

Do not collapse the metrics into an unreviewable scalar score. When survival
improves while exposure worsens, report the trade rather than hiding it in a
weighted average.

## Checkpoint and provenance evolution

Do not rewrite the shipped schema-4 checkpoint. Before registering a second
actor family:

- Add immutable `policy = {family, version, config}` metadata to new run records
  and checkpoints. Absence means `deepsets@1`; this is the backward-compatible
  migration rule.
- Keep the top-level `actor` key for deployable parameters and `critic` for the
  training-only critic. Dispatch the actor from policy metadata, never from
  PyTree shape guessing.
- Bump the checkpoint schema only when new recovery fields are required. Test an
  actual read of `checkpoints/theatre_1000`, exact greedy outputs before/after
  dispatch, and refusal of an unknown family/version.
- Record recurrent/attention runtime state as episode state, not model weights.
  It resets when the current resume path restarts in-flight episodes. Do not
  claim bit-identical equivalence to an uninterrupted post-interruption sample
  path; preserve the current, narrower resume guarantee.
- Store architecture config, exact parameter count, dtype, software versions,
  compiler/backend, training/evaluation seeds, route-bank digest, theatre
  component hashes, and checkpoint hash. For external planners also store the
  upstream model and tokenizer revisions, file hashes, license, adapter hash,
  and prompt/intent schema.
- Keep evaluator and verifier architecture-agnostic. They consume actions and
  traces, not actor internals.

Required implementation tests include the existing permutation, masking,
three-control, CTDE, PPO, route-disjointness, checkpoint, resume, evaluator, and
promotion tests, plus:

- exact `PolicySpec.parameter_count` and state-memory contracts per family;
- per-aircraft GRU reset and sequence-minibatch reconstruction;
- transformer causal/reset masking and cached-versus-uncached equivalence;
- finite gradients under `jit`/`vmap` for every actor;
- old schema-4 defaulting to `deepsets@1`, new-family round trip, and mismatch
  refusal;
- two resumes from one recurrent/transformer checkpoint producing identical
  continuations;
- planner prompt determinism, local-information-only construction, exhaustive
  nine-label validation, missing-cache refusal, and a source scan that prevents
  offensive planner intents.

## Rejected models and designs

### Falcon-7B-Instruct

Reject as a ladder rung. Its official card describes a 7B, 32-layer,
2,048-token causal decoder, requires PyTorch 2.0, recommends at least 16 GB for
swift inference, and now points to a newer Falcon generation
([Falcon-7B-Instruct card](https://huggingface.co/tiiuae/falcon-7b-instruct),
[configuration](https://huggingface.co/tiiuae/falcon-7b-instruct/blob/main/config.json)).
Its Apache-2.0 license is acceptable, but it is about 9,000 times the proposed
numeric transformer's parameters. A positive result would not isolate language
pretraining from sheer scale; a negative result would be expensive and
uninformative.

### OpenLLaMA-7B-v2

Reject as a ladder rung and as the default 7B control. OpenLLaMA is permissively
Apache-2.0 and unusually provides both PyTorch and EasyLM JAX weights
([project README](https://github.com/openlm-research/open_llama/blob/main/README.md)),
but v2 is a base model, not an instruction planner. Its published config is a
32-layer, 4,096-wide, 2,048-context LLaMA architecture
([configuration](https://huggingface.co/openlm-research/open_llama_7b_v2/blob/main/config.json)).
EasyLM compatibility does not make it compatible with Naigos's Linen PPO or
Orbax recovery state. It would still require a separate tokenizer, checkpoint
conversion, planner adapter, and validation path.

### End-to-end language control

Reject. Text decoding is a poor continuous-control distribution, adds malformed
output and nondeterminism failure modes, is far slower than necessary, and has
no task-aligned pretraining advantage. It would also tempt the environment
interface away from the fixed three-control action.

### PyTorch rewrite of the learner

Reject. It would invalidate the working JAX `jit`/`vmap` pipeline, duplicate
checkpoint and RNG semantics, and confound architecture with framework. A
Torch language planner, if ever approved, stays offline and outside PPO.

### Large-context or return-conditioned transformer first

Reject for the initial ladder. Context 32 already spans 64 simulated seconds;
quadratic attention beyond that needs evidence. Return-to-go conditioning would
change the learning objective and require a curated offline trajectory dataset,
so it would not answer whether temporal attention improves the current online
policy.

## Recommended next integration step

Finish and merge only the backward-compatible policy metadata seam first:
thread `family/version/config` through run metadata and new checkpoints, default
missing metadata to `deepsets@1`, and prove the shipped checkpoint's outputs are
unchanged. Then implement **Rung 1 only** with sequence-aware PPO and its reset,
causality, resume, and inference-latency tests. Do not implement the transformer
or add any language-model dependency until the three-seed GRU comparison says
whether memory is worth carrying at all.

## Primary-source model notes

- SmolLM2: 135M/360M/1.7B family, 135M trained on two trillion tokens,
  BF16/Safetensors, Apache-2.0; its 135M config declares 30 layers, width 576,
  and 8,192 positions. Sources: [model card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct),
  [config](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/blob/main/config.json).
- Qwen3-0.6B: 0.6B advertised parameters, 0.44B non-embedding, 28 layers,
  grouped-query attention, Apache-2.0. Sources: [model card](https://huggingface.co/Qwen/Qwen3-0.6B),
  [config](https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/config.json).
- Qwen documents GGUF conversion and Q8/Q5/Q4 weight quantization as deployment
  options; Naigos must still validate its own converted artifact:
  [Qwen3 llama.cpp quantization guide](https://github.com/QwenLM/Qwen3/blob/main/docs/source/quantization/llama.cpp.md).
- Falcon-7B-Instruct: 7B parameters, 32 layers, 2,048 sequence length,
  PyTorch 2.0, Apache-2.0: [official model card](https://huggingface.co/tiiuae/falcon-7b-instruct).
- OpenLLaMA: Apache-2.0 PyTorch and EasyLM JAX releases:
  [official project README](https://github.com/openlm-research/open_llama/blob/main/README.md).

