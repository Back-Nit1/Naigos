#!/usr/bin/env python
"""Stage, prepare, train, and evaluate the offline Qwen route-intent planner."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from naigos.planner.assets import DEFAULT_MODEL_DIR, fetch_model, verify_model_dir  # noqa: E402
from naigos.planner.dataset import derive_behavior_labels, prepare_dataset  # noqa: E402
from naigos.planner.qwen import evaluate_adapter, train_lora, verify_adapter_dir  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)

    for name in ("fetch-model", "verify-model"):
        p = sub.add_parser(name)
        p.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))

    p = sub.add_parser("verify-adapter")
    p.add_argument("--adapter", required=True)

    p = sub.add_parser("prepare")
    p.add_argument("--trajectory", action="append", required=True, help="completed trajectory dataset root")
    p.add_argument("--labels", required=True, help="counterfactual branch-label JSONL")
    p.add_argument("--out", required=True, help="planner JSONL output")

    p = sub.add_parser("label-behavior")
    p.add_argument("--trajectory", action="append", required=True, help="completed trajectory dataset root")
    p.add_argument("--out", required=True, help="weak-label JSONL output")
    p.add_argument("--split", required=True, choices=("train", "validation", "test"))
    p.add_argument("--stride", type=int, default=16)
    p.add_argument("--horizon", type=int, default=16)
    p.add_argument("--include-failures", action="store_true")

    p = sub.add_parser("train")
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--micro-batch-size", type=int, default=1)
    p.add_argument("--effective-batch-size", type=int, default=128)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))

    p = sub.add_parser("evaluate")
    p.add_argument("--data", required=True)
    p.add_argument("--adapter", required=True)
    p.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    p.add_argument("--split", default="validation", choices=("train", "validation", "test"))
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))

    args = ap.parse_args(argv)
    if args.command == "fetch-model":
        result = fetch_model(args.model_dir)
    elif args.command == "verify-model":
        result = verify_model_dir(args.model_dir)
    elif args.command == "verify-adapter":
        result = verify_adapter_dir(args.adapter)
    elif args.command == "prepare":
        result = prepare_dataset(args.trajectory, args.labels, args.out)
    elif args.command == "label-behavior":
        result = derive_behavior_labels(
            args.trajectory, args.out, split=args.split, stride=args.stride,
            horizon=args.horizon, successful_only=not args.include_failures,
        )
    elif args.command == "train":
        result = train_lora(
            args.data, args.out, model_dir=args.model_dir, epochs=args.epochs,
            micro_batch_size=args.micro_batch_size, effective_batch_size=args.effective_batch_size,
            learning_rate=args.learning_rate, seed=args.seed, device=args.device,
        )
    else:
        result = evaluate_adapter(
            args.data, args.adapter, model_dir=args.model_dir, split=args.split, device=args.device,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
