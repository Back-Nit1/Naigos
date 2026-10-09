#!/usr/bin/env python
"""The numbers docs quote, computed instead of hand-typed.

    uv run python scripts/repo_summary.py

README.md and next-steps.md used to carry a test count each -- ~1,100 and 257
-- that had drifted four-fold apart because neither was ever re-checked
against the other. A count that is read off the real collection cannot
contradict itself; this script is the thing to re-run (and re-paste) when the
suite's size changes enough to be worth saying.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def test_count() -> int:
    out = subprocess.run(
        ["uv", "run", "pytest", "--collect-only", "-q"],
        cwd=REPO, capture_output=True, text=True, check=True,
    ).stdout
    # pytest's own summary line: "N tests collected in Xs" (or "no tests ran").
    last = out.strip().splitlines()[-1]
    return int(last.split()[0])


def artifact_files(root: str) -> list[tuple[str, int]]:
    d = REPO / root
    if not d.is_dir():
        return []
    return sorted(
        (str(p.relative_to(REPO)), p.stat().st_size) for p in d.rglob("*") if p.is_file()
    )


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def main(argv=None) -> int:
    n = test_count()
    print(f"tests collected: {n}")
    for root in ("docs/artifacts", "checkpoints"):
        files = artifact_files(root)
        if not files:
            continue
        total = sum(sz for _, sz in files)
        print(f"\n{root}/  ({len(files)} files, {human(total)} total)")
        for name, sz in files:
            print(f"  {human(sz):>8}  {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
