"""LoRA training and fixed-label evaluation for the offline Qwen planner.

Imports of Torch, Transformers, and PEFT stay inside call boundaries. Merely
importing Naigos, running PPO, or reading a trajectory never imports a second
ML framework.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Iterable

from .assets import DEFAULT_MODEL_DIR, load_spec, sha256_file, verify_model_dir
from .dataset import DATASET_SCHEMA
from .intents import INTENT_LABELS


class QwenPlannerError(RuntimeError):
    pass


def _stack():
    try:
        import torch
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise QwenPlannerError(
            "Qwen planner dependencies are not installed; run `uv pip install -e '.[planner]'`"
        ) from exc
    return torch, LoraConfig, PeftModel, get_peft_model, AutoModelForCausalLM, AutoTokenizer


def read_examples(path: str | os.PathLike, split: str | None = None) -> list[dict]:
    source = Path(path)
    manifest_path = source.with_suffix(source.suffix + ".manifest.json")
    if not manifest_path.is_file():
        raise QwenPlannerError(f"{path}: missing planner dataset manifest {manifest_path.name}")
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as exc:
        raise QwenPlannerError(f"{manifest_path}: invalid JSON: {exc}") from exc
    actual_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    if manifest.get("schema") != DATASET_SCHEMA or manifest.get("sha256") != actual_sha:
        raise QwenPlannerError(f"{path}: planner dataset bytes do not match their manifest")
    rows: list[dict] = []
    with source.open() as fh:
        for lineno, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise QwenPlannerError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
            expected_prompt_sha = hashlib.sha256(str(row.get("prompt", "")).encode()).hexdigest()
            if (row.get("schema") != DATASET_SCHEMA or row.get("label") not in INTENT_LABELS
                    or row.get("prompt_sha256") != expected_prompt_sha):
                raise QwenPlannerError(f"{path}:{lineno}: not a valid {DATASET_SCHEMA} example")
            if split is None or row.get("split") == split:
                rows.append(row)
    if not rows:
        raise QwenPlannerError(f"{path}: no examples for split {split!r}")
    return rows


def label_token_ids(tokenizer) -> dict[str, int]:
    ids: dict[str, int] = {}
    for label in INTENT_LABELS:
        encoded = tokenizer.encode(label, add_special_tokens=False)
        if len(encoded) != 1:
            raise QwenPlannerError(f"intent label {label!r} is {len(encoded)} tokenizer tokens, expected one")
        ids[label] = int(encoded[0])
    if len(set(ids.values())) != len(ids):
        raise QwenPlannerError("intent labels do not have distinct tokenizer ids")
    return ids


def _device(torch, requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_base(model_dir, device, *, train: bool):
    torch, LoraConfig, PeftModel, get_peft_model, AutoModel, AutoTokenizer = _stack()
    verify_model_dir(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    model = AutoModel.from_pretrained(model_dir, local_files_only=True, dtype=dtype)
    model.to(device)
    if train:
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
        lora = LoraConfig(
            r=8, lora_alpha=16, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
            target_modules=["q_proj", "v_proj"],
        )
        model = get_peft_model(model, lora)
    return torch, PeftModel, tokenizer, model


def _encoded(rows: Iterable[dict], tokenizer, label_ids: dict[str, int], max_tokens: int) -> list[dict]:
    result = []
    for row in rows:
        ids = tokenizer.encode(row["prompt"], add_special_tokens=False)
        if len(ids) > max_tokens:
            raise QwenPlannerError(
                f"prompt {row['prompt_sha256']} is {len(ids)} tokens, above the {max_tokens}-token contract"
            )
        result.append({"input_ids": ids + [label_ids[row["label"]]], "prompt_len": len(ids), "label": row["label"]})
    return result


def _collate(torch, batch: list[dict], pad_id: int):
    width = max(len(row["input_ids"]) for row in batch)
    input_ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    attention = torch.zeros((len(batch), width), dtype=torch.long)
    labels = torch.full((len(batch), width), -100, dtype=torch.long)
    for i, row in enumerate(batch):
        n = len(row["input_ids"])
        input_ids[i, :n] = torch.tensor(row["input_ids"], dtype=torch.long)
        attention[i, :n] = 1
        labels[i, n - 1] = row["input_ids"][-1]
    return input_ids, attention, labels


def train_lora(
    data_path: str | os.PathLike,
    output_dir: str | os.PathLike,
    *,
    model_dir: str | os.PathLike = DEFAULT_MODEL_DIR,
    epochs: int = 3,
    micro_batch_size: int = 1,
    effective_batch_size: int = 128,
    learning_rate: float = 2e-4,
    seed: int = 0,
    max_tokens: int = 512,
    device: str = "auto",
) -> dict:
    """Train only q_proj/v_proj LoRA adapters; base weights stay immutable."""

    if epochs < 1 or micro_batch_size < 1 or effective_batch_size < micro_batch_size:
        raise QwenPlannerError("epochs/batch sizes must be positive and effective batch >= micro batch")
    if effective_batch_size % micro_batch_size:
        raise QwenPlannerError("effective_batch_size must be divisible by micro_batch_size")
    out = Path(output_dir)
    if out.exists() and any(out.iterdir()):
        raise QwenPlannerError(f"refusing to train over non-empty output directory {out}")
    rows = read_examples(data_path, "train")
    torch, *_ = _stack()
    selected_device = _device(torch, device)
    torch, PeftModel, tokenizer, model = _load_base(model_dir, selected_device, train=True)
    labels = label_token_ids(tokenizer)
    encoded = _encoded(rows, tokenizer, labels, max_tokens)
    trainable = [p for p in model.parameters() if p.requires_grad]
    trainable_count = sum(p.numel() for p in trainable)
    if trainable_count != 1_146_880:
        raise QwenPlannerError(f"unexpected Qwen LoRA parameter count {trainable_count}, expected 1,146,880")
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate)
    accumulation = effective_batch_size // micro_batch_size
    losses: list[float] = []
    model.train()
    optimizer.zero_grad(set_to_none=True)
    update = 0
    for epoch in range(epochs):
        order = list(range(len(encoded)))
        random.Random(seed + epoch).shuffle(order)
        epoch_loss = 0.0
        for start in range(0, len(order), micro_batch_size):
            batch = [encoded[i] for i in order[start:start + micro_batch_size]]
            input_ids, attention, targets = _collate(torch, batch, tokenizer.pad_token_id)
            output = model(
                input_ids=input_ids.to(selected_device), attention_mask=attention.to(selected_device),
                labels=targets.to(selected_device),
            )
            if not torch.isfinite(output.loss):
                raise QwenPlannerError(f"non-finite loss at epoch {epoch + 1}, batch {start}")
            (output.loss / accumulation).backward()
            epoch_loss += float(output.loss.detach().cpu()) * len(batch)
            last = start + micro_batch_size >= len(order)
            if ((start // micro_batch_size) + 1) % accumulation == 0 or last:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                update += 1
        losses.append(epoch_loss / len(order))
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    tokenizer.save_pretrained(out)
    spec = load_spec()
    data_sha = hashlib.sha256(Path(data_path).read_bytes()).hexdigest()
    manifest = {
        "schema": "naigos.qwen-planner-adapter/1",
        "base": {"repo_id": spec["repo_id"], "revision": spec["revision"]},
        "dataset_sha256": data_sha,
        "intent_labels": list(INTENT_LABELS),
        "lora": {"r": 8, "alpha": 16, "dropout": 0.0, "targets": ["q_proj", "v_proj"]},
        "trainable_parameters": trainable_count,
        "training": {
            "epochs": epochs, "micro_batch_size": micro_batch_size,
            "effective_batch_size": effective_batch_size, "learning_rate": learning_rate,
            "seed": seed, "max_tokens": max_tokens, "device": selected_device, "loss_by_epoch": losses,
            "optimizer_updates": update,
        },
    }
    artifact_files = [
        {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(out.iterdir()) if path.is_file() and path.name != "naigos_training_manifest.json"
    ]
    manifest["files"] = artifact_files
    manifest["content_sha256"] = hashlib.sha256(
        json.dumps(artifact_files, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    (out / "naigos_training_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def verify_adapter_dir(adapter_dir: str | os.PathLike) -> dict:
    root = Path(adapter_dir)
    path = root / "naigos_training_manifest.json"
    try:
        manifest = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise QwenPlannerError(f"cannot read adapter manifest {path}: {exc}") from exc
    spec = load_spec()
    if (manifest.get("schema") != "naigos.qwen-planner-adapter/1"
            or manifest.get("base") != {"repo_id": spec["repo_id"], "revision": spec["revision"]}):
        raise QwenPlannerError(f"{path}: adapter is not for the pinned Naigos Qwen base")
    files = manifest.get("files") or []
    for row in files:
        artifact = root / row["name"]
        if (not artifact.is_file() or artifact.stat().st_size != row["bytes"]
                or sha256_file(artifact) != row["sha256"]):
            raise QwenPlannerError(f"adapter artifact failed verification: {artifact}")
    expected = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if not files or manifest.get("content_sha256") != expected:
        raise QwenPlannerError(f"{path}: invalid adapter content identity")
    return manifest


def evaluate_adapter(
    data_path: str | os.PathLike,
    adapter_dir: str | os.PathLike,
    *, model_dir: str | os.PathLike = DEFAULT_MODEL_DIR,
    split: str = "validation", max_tokens: int = 512, device: str = "auto",
) -> dict:
    rows = read_examples(data_path, split)
    verify_adapter_dir(adapter_dir)
    torch, *_ = _stack()
    selected_device = _device(torch, device)
    torch, PeftModel, tokenizer, model = _load_base(model_dir, selected_device, train=False)
    model = PeftModel.from_pretrained(model, adapter_dir, local_files_only=True).to(selected_device)
    label_ids = label_token_ids(tokenizer)
    id_order = torch.tensor([label_ids[label] for label in INTENT_LABELS], device=selected_device)
    encoded = _encoded(rows, tokenizer, label_ids, max_tokens)
    correct = 0
    confusion = {truth: {pred: 0 for pred in INTENT_LABELS} for truth in INTENT_LABELS}
    model.eval()
    with torch.inference_mode():
        for row in encoded:
            prompt_ids = row["input_ids"][:-1]
            logits = model(input_ids=torch.tensor([prompt_ids], device=selected_device)).logits[0, -1]
            pred = INTENT_LABELS[int(torch.argmax(logits[id_order]).cpu())]
            confusion[row["label"]][pred] += 1
            correct += pred == row["label"]
    return {
        "schema": "naigos.qwen-planner-evaluation/1", "split": split,
        "records": len(rows), "correct": int(correct), "accuracy": correct / len(rows),
        "confusion": confusion,
    }
