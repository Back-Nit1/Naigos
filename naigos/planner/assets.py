"""Pinned, hash-verified local assets for the offline Qwen planner."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import urllib.request
from pathlib import Path

SPEC_PATH = Path(__file__).with_name("qwen3_0_6b_base.json")
DEFAULT_MODEL_DIR = Path("models/qwen3-0.6b-base")


class ModelAssetError(RuntimeError):
    pass


def load_spec() -> dict:
    spec = json.loads(SPEC_PATH.read_text())
    if spec.get("schema") != "naigos.planner-model-assets/1":
        raise ModelAssetError(f"unsupported planner model asset spec in {SPEC_PATH}")
    return spec


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(block_size):
            h.update(block)
    return h.hexdigest()


def verify_model_dir(model_dir: str | os.PathLike = DEFAULT_MODEL_DIR) -> dict:
    root = Path(model_dir)
    spec = load_spec()
    problems: list[str] = []
    for row in spec["files"]:
        path = root / row["name"]
        if not path.is_file():
            problems.append(f"missing {row['name']}")
            continue
        size = path.stat().st_size
        if size != row["bytes"]:
            problems.append(f"{row['name']}: {size} bytes, expected {row['bytes']}")
            continue
        got = sha256_file(path)
        if got != row["sha256"]:
            problems.append(f"{row['name']}: sha256 {got}, expected {row['sha256']}")
    if problems:
        raise ModelAssetError("invalid local Qwen assets:\n  - " + "\n  - ".join(problems))
    return {
        "model_dir": str(root), "repo_id": spec["repo_id"], "revision": spec["revision"],
        "files": len(spec["files"]), "bytes": sum(row["bytes"] for row in spec["files"]),
    }


def fetch_model(model_dir: str | os.PathLike = DEFAULT_MODEL_DIR) -> dict:
    """Explicitly download the immutable revision, then verify every byte."""

    root = Path(model_dir)
    root.mkdir(parents=True, exist_ok=True)
    spec = load_spec()
    base = f"https://huggingface.co/{spec['repo_id']}/resolve/{spec['revision']}"
    for row in spec["files"]:
        target = root / row["name"]
        if target.exists() and target.stat().st_size == row["bytes"] and sha256_file(target) == row["sha256"]:
            continue
        fd, tmp = tempfile.mkstemp(dir=root, prefix=f".{row['name']}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as out, urllib.request.urlopen(f"{base}/{row['name']}") as response:
                while block := response.read(1024 * 1024):
                    out.write(block)
                out.flush()
                os.fsync(out.fileno())
            staged = Path(tmp)
            if staged.stat().st_size != row["bytes"] or sha256_file(staged) != row["sha256"]:
                raise ModelAssetError(f"downloaded {row['name']} failed its pinned size/hash")
            os.replace(staged, target)
        except BaseException:
            try:
                os.unlink(tmp)
            except FileNotFoundError:
                pass
            raise
    return verify_model_dir(root)
