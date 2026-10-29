"""Deterministic actor-visible trajectory windows for the Qwen planner."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence

import numpy as np

from .intents import INTENTS

PROMPT_SCHEMA = "naigos.qwen-planner-prompt/1"
HISTORY_STEPS = 16
_BINS = "0123456789abcdefghijklmnopqrstuvwxyz_"  # 37 values: stored integer -18..18 plus 18


def _q(values, scale: float = 9.0) -> list[int]:
    arr = np.asarray(values, dtype=np.float32)
    if not np.isfinite(arr).all():
        raise ValueError("planner prompt inputs must be finite")
    return np.rint(np.clip(arr, -2.0, 2.0) * scale).astype(np.int16).tolist()


def _frame_summary(frame: Mapping[str, np.ndarray]) -> list[int]:
    ego = np.asarray(frame["ego"], dtype=np.float32)
    threats = np.asarray(frame["threats"], dtype=np.float32)
    threat_mask = np.asarray(frame["threat_mask"], dtype=bool)
    friends = np.asarray(frame["friends"], dtype=np.float32)
    friend_mask = np.asarray(frame["friend_mask"], dtype=bool)
    if ego.ndim != 1 or threats.ndim != 2 or friends.ndim != 2:
        raise ValueError("one planner frame must contain one agent's unbatched observations")
    if threat_mask.shape != threats.shape[:1] or friend_mask.shape != friends.shape[:1]:
        raise ValueError("planner observation masks do not match their padded sets")

    # The first ten values are the stable baseline actor contract. Explicit
    # resource/weather/mission observation extensions, when present, are actor
    # visible and are appended by the dataset builder before this function.
    # Keep the stable ten-field ego contract verbatim. Optional actor-visible
    # extensions can be much wider, so summarize their distribution rather
    # than allowing one environment rung to break the 512-token contract.
    ego_q = _q(ego[:10])
    if ego.size > 10:
        extra = ego[10:]
        ego_q += _q([np.min(extra), np.mean(extra), np.max(extra)])
    valid = np.flatnonzero(threat_mask)
    if valid.size:
        candidates = threats[valid]
        # Most urgent actor-visible contact: track, then Pd, then proximity.
        score = candidates[:, 8] * 4.0 + candidates[:, 7] * 2.0 - candidates[:, 5]
        row = candidates[int(np.argmax(score))]
        kind = int(np.argmax(row[9:])) if row.shape[0] > 9 else 0
        # Position, range, Pd and track are enough for the planner's slower
        # route decision. Velocity and envelope width remain available to the
        # low-level native actor and are not redundantly serialized here.
        threat_q = [int(valid.size), *_q(row[[0, 1, 2, 5, 7, 8]]), kind]
    else:
        threat_q = [0] * 8
    friend_count = int(np.count_nonzero(friend_mask))
    friend_range = float(np.min(friends[friend_mask, 5])) if friend_count else 0.0
    return [*ego_q, *threat_q, friend_count, *_q([friend_range])]


def _pair_summary(a: Mapping[str, np.ndarray], b: Mapping[str, np.ndarray]) -> list[int]:
    """Compress two consecutive observations without introducing future data."""

    sa, sb = _frame_summary(a), _frame_summary(b)
    # The later value plus its signed two-second delta preserves level and trend
    # while halving prompt length. Both members of every pair influence output.
    return [*sb, *[int(np.clip(y - x, -18, 18)) for x, y in zip(sa, sb)]]


def _encode_bins(values: Sequence[int]) -> str:
    """One printable symbol per quantized value, keeping prompts bounded."""

    return "".join(_BINS[int(np.clip(value, -18, 18)) + 18] for value in values)


def build_prompt(history: Sequence[Mapping[str, np.ndarray]]) -> str:
    """Build the exact causal prompt consumed by planner training/inference."""

    if len(history) != HISTORY_STEPS:
        raise ValueError(f"planner prompts require exactly {HISTORY_STEPS} observations")
    pairs = [_encode_bins(_pair_summary(history[i], history[i + 1])) for i in range(0, HISTORY_STEPS, 2)]
    payload = {
        "schema": PROMPT_SCHEMA,
        "dt_s": 2,
        "history_steps": HISTORY_STEPS,
        "encoding": "each character indexes 0123456789abcdefghijklmnopqrstuvwxyz_; decoded integer=index-18",
        "pair_end_and_delta": pairs,
        "intents": {intent.label: intent.description for intent in INTENTS},
    }
    compact = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return (
        "Select the safest useful evasive route intent. Output exactly one label A-I; no prose.\n"
        + compact
        + "\nintent="
    )


def prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()
