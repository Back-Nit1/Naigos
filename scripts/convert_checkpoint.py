#!/usr/bin/env python
"""One-way converter: a pre-schema-4 pickle checkpoint -> a schema-4 Orbax directory.

    uv run python scripts/convert_checkpoint.py checkpoints/theatre_1000.pkl checkpoints/theatre_1000

`naigos/rl/checkpoint.py` stopped reading pickle entirely (next-steps.md G-3):
`pickle.load` runs arbitrary code to reconstruct whatever the bytes claim to
be, which makes a checkpoint from an untrusted source a code-execution vector.
A checkpoint already in this repository, written by this repository's own
training code, is trusted by construction -- that is what "one-way converter
for trusted existing checkpoints" means here. It is one-way because there is
deliberately no path back: nothing in this codebase writes pickle again.

The blob's *shape* does not change, only how it is written to disk, so this is
a direct pickle.load -> orbax.save, with the schema field bumped to 4. A
pre-resume (schema 1) checkpoint converts too; it simply has no `recovery`
block, exactly as it did not before, and `naigos.rl.checkpoint.is_resumable`
still refuses it on that ground.
"""

from __future__ import annotations

import argparse
import pickle
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import orbax.checkpoint as ocp  # noqa: E402

from naigos.rl.checkpoint import CHECKPOINT_SCHEMA  # noqa: E402


def convert(src: Path, dst: Path, *, force: bool = False) -> None:
    src, dst = src.resolve(), dst.resolve()  # orbax requires an absolute path
    if not src.is_file():
        raise SystemExit(f"{src}: not a file (already converted, or not a checkpoint?)")
    if dst.exists():
        if not force:
            raise SystemExit(f"{dst} already exists; pass --force to overwrite")
        shutil.rmtree(dst) if dst.is_dir() else dst.unlink()

    with open(src, "rb") as f:
        blob = pickle.load(f)
    if not isinstance(blob, dict) or "actor" not in blob:
        raise SystemExit(f"{src} does not look like a Naigos checkpoint (no top-level 'actor')")

    old_schema = blob.get("schema")
    blob["schema"] = CHECKPOINT_SCHEMA

    tmp = dst.with_name(dst.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    ocp.PyTreeCheckpointer().save(tmp, blob)
    tmp.rename(dst)
    print(f"{src} (schema {old_schema}) -> {dst} (schema {CHECKPOINT_SCHEMA})")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src", type=Path, help="the old .pkl checkpoint")
    ap.add_argument("dst", type=Path, help="the new checkpoint directory to write")
    ap.add_argument("--force", action="store_true", help="overwrite dst if it already exists")
    a = ap.parse_args(argv)
    convert(a.src, a.dst, force=a.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
