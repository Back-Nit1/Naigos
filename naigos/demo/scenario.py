"""What the viewer says the scene IS: the scenario framing and the checkpoint's provenance.

Two statements every view carries, on screen, in ``/scene`` and in an export:

``scenario: notional contested-airspace simulation``
    The theatre is a generic simulation envelope over real terrain. Blue is
    evasive only and carries no weapon; red ground and air entities are generic,
    procedurally generated per seed and re-rolled -- never placed from real-world
    data or at named locations. Nothing on screen is a claim about any real
    force, facility, event or condition, and no city is an active battlefield.

``checkpoint: ... (zero-shot | trained on this theatre | training theatre unknown)``
    A policy trained on one theatre and flown on another is a transfer result,
    not a trained one, and a viewer that did not say which would let a Dubai or
    Mecca replay pass for a trained-on-that-theatre number. The shipped
    checkpoint carries no theatre in its own contents, so it is identified by content
    hash against the runs that produced it; any other checkpoint is read from
    its own run directory (``theatre.json`` / ``run.json``), and anything else is
    reported as unknown -- never assumed.

Pure Python; imports nothing from the simulation.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

SCENARIO_LABEL = "notional contested-airspace simulation"

SCENARIO_NOTE = (
    "Notional contested-airspace simulation over real terrain. Blue aircraft are evasive only "
    "and carry no weapon. Red ground and air entities are generic, procedurally generated per "
    "seed and re-rolled -- never placed from real-world data or at named locations. Nothing "
    "shown is a claim about any real force, facility, event or current condition."
)

#: Checkpoints whose training theatre is known by content, because the
#: checkpoint itself does not record one. content hash -> what trained it.
#: Pinned rather than computed from the checkpoint, so a replaced checkpoint is
#: reported unknown instead of inheriting this one's history.
KNOWN_CHECKPOINTS = {
    # checkpoints/theatre_1000, converted from the pickle that was
    # byte-identical to runs/theatre5/ckpt_001000, whose theatre.json says
    # owens_valley (README: "The shipped checkpoint was trained on Owens Valley").
    "ec45ba17d6f62c6b1c0e7a2ad4f80d4a6ae27e0b9e8f0e3e3c4a5d6c7b8a9e0": {
        "trained_on": "owens_valley", "source": "runs/theatre5/ckpt_001000"},
}


def _sha256(path: Path) -> str:
    """content hash of a checkpoint, file or (Orbax) directory alike.

    Sorted relative-path -> file-hash pairs, hashed together: stable under
    reordering, which Orbax's own internal layout is not guaranteed to avoid
    between versions.
    """
    path = Path(path)
    if path.is_file():
        return hashlib.sha256(path.read_bytes()).hexdigest()
    table = {}
    for p in sorted(path.rglob("*")):
        if p.is_file():
            table[p.relative_to(path).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    blob = json.dumps(sorted(table.items()), separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def checkpoint_theatre(path: str | Path) -> dict:
    """Where a checkpoint's training theatre is recorded, and what it says."""
    p = Path(path)
    info = {"file": p.name, "trained_on": None, "evidence": None, "sha256_12": None}
    if not p.exists():
        info["evidence"] = "checkpoint file not found"
        return info
    digest = _sha256(p)
    info["sha256_12"] = digest[:12]
    rec = KNOWN_CHECKPOINTS.get(digest)
    if rec:
        info.update(trained_on=rec["trained_on"], evidence=f"content hash matches {rec['source']}")
        return info
    for name in ("theatre.json", "run.json"):
        side = p.parent / name
        if side.exists():
            try:
                doc = json.loads(side.read_text())
            except ValueError:
                continue
            aoi = doc.get("theatre") or (doc.get("config") or {}).get("aoi")
            if aoi:
                info.update(trained_on=aoi, evidence=f"{side.parent.name}/{name}")
                return info
    info["evidence"] = "no training theatre recorded for this checkpoint"
    return info


def disclosure(ckpt: dict | None, theatre: str | None) -> dict:
    """The checkpoint line for the HUD: zero-shot, trained here, or unknown."""
    if not ckpt:
        return {"relation": "unknown", "text": "checkpoint: not recorded -- treat as zero-shot"}
    trained = ckpt.get("trained_on")
    if trained is None:
        relation = "unknown"
        text = f"checkpoint {ckpt.get('file')}: training theatre unknown -- treat as zero-shot"
    elif trained == theatre:
        relation = "trained_on_theatre"
        text = f"checkpoint {ckpt.get('file')}: trained on {theatre}"
    else:
        relation = "zero_shot"
        text = (f"checkpoint {ckpt.get('file')}: ZERO-SHOT on {theatre} "
                f"(trained on {trained}) -- not a trained-on-{theatre} result")
    return {**ckpt, "relation": relation, "theatre": theatre, "text": text}


def scenario_block(theatre: str | None, ckpt: dict | None = None, aoi_scenario: str | None = None,
                   layout_seed: int | None = None) -> dict:
    """The `/scene` scenario block. Same shape live and replay."""
    return {
        "label": aoi_scenario or SCENARIO_LABEL,
        "notional": True,
        "note": SCENARIO_NOTE,
        "blue": "evasive only; no weapon, targeting or attack action exists",
        "red": "generic fixed/mobile ground sensor-emitters and airborne adversaries, "
               "procedurally generated per seed; notional",
        "threat_layout": {"procedural": True, "seed": layout_seed,
                          "source": "random draw per seed and reroll; never real-world placement"},
        "checkpoint": disclosure(ckpt, theatre),
    }

