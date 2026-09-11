# SPDX-License-Identifier: GPL-3.0-only
"""Strict phase runner and artifact checks for structural Heximax."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .structuralcfg import (
    ARMS, GATES, HOLDOUT_SEED, SCREEN_GAMES, SCREEN_SEEDS, SCREEN_WORKERS,
    CONFIRMATION_SEED, source_fingerprint,
)

PHASES = {
    "confirmation": {"games": 1024, "seed": CONFIRMATION_SEED},
    "holdout": {"games": 1024, "seed": HOLDOUT_SEED},
}

PHASE_WORKERS = SCREEN_WORKERS
COLOR_KEYS = frozenset({"Color.RED", "Color.WHITE", "Color.BLUE", "Color.ORANGE"})
CANDIDATE_COLOR = "Color.RED"


def _expected_seed(phase: str, arm: str, gate: str) -> int:
    if phase == "screen":
        return SCREEN_SEEDS[gate] + ARMS.index(arm) * 10_000
    return PHASES[phase]["seed"] + (100_000 if gate == "shipped" else 0)


def _validate_results(doc: dict, expected_games: int) -> None:
    wins = doc.get("wins")
    points = doc.get("points")
    if not isinstance(wins, dict) or not isinstance(points, dict):
        raise ValueError("missing normalized duel results")
    if set(wins) != COLOR_KEYS or set(points) != COLOR_KEYS:
        raise ValueError("wins and points must contain exactly the four Color keys")
    if not all(isinstance(value, int) and not isinstance(value, bool) and value >= 0
               for value in wins.values()):
        raise ValueError("invalid normalized wins")
    completed_games = sum(wins.values())
    if completed_games > expected_games:
        raise ValueError("wins exceed requested games")
    for key, values in points.items():
        if not isinstance(values, list) or len(values) != completed_games:
            raise ValueError(f"point list length does not match completed games: {key}")
        if not all(isinstance(value, int) and not isinstance(value, bool) and value >= 0
                   for value in values):
            raise ValueError(f"invalid normalized points: {key}")


def _validate_document(doc: dict, *, expected_source: str, phase: str | None) -> dict:
    if doc.get("family") != "heximax-structural-future":
        raise ValueError("wrong artifact family")
    if doc.get("source_hash") != expected_source:
        raise ValueError("structural source fingerprint mismatch")
    if doc.get("arm") not in ARMS or doc.get("gate") not in GATES:
        raise ValueError("unknown arm/gate")
    arm, gate = doc["arm"], doc["gate"]
    actual_phase = phase or doc.get("phase")
    if actual_phase not in ("screen", *PHASES) or doc.get("phase") != actual_phase:
        raise ValueError("unknown or mismatched phase")
    expected_games = SCREEN_GAMES if actual_phase == "screen" else PHASES[actual_phase]["games"]
    if doc.get("games") != expected_games:
        raise ValueError("unexpected phase/game count")
    if doc.get("workers") != PHASE_WORKERS:
        raise ValueError("unexpected worker count")
    if doc.get("seed") != _expected_seed(actual_phase, arm, gate):
        raise ValueError("unexpected seed family")
    expected_role = "control" if arm == "control" else "candidate"
    if doc.get("role") != expected_role or doc.get("candidate") != arm:
        raise ValueError("candidate identity does not match arm")
    if doc.get("candidate_color") != CANDIDATE_COLOR:
        raise ValueError("candidate color must remain Color.RED")
    from .structuralcfg import lineup, manifest
    if doc.get("players") != lineup(arm, gate):
        raise ValueError("lineup does not match arm/gate")
    expected = manifest()["control"] if arm == "control" else manifest()["candidates"][arm]
    if doc.get("phenotype") != expected:
        raise ValueError("structural phenotype mismatch")
    _validate_results(doc, expected_games)
    return doc


def validate_artifact(path: Path, *, expected_source: str | None = None,
                      phase: str | None = None) -> dict:
    doc = json.loads(path.read_text())
    return _validate_document(doc, expected_source=expected_source or source_fingerprint(), phase=phase)


def run_phase(phase: str, arm: str, gate: str, workers: int, out: Path,
              *, seed: int | None = None) -> dict:
    if phase not in PHASES:
        raise ValueError(f"unknown phase: {phase}")
    if arm not in ARMS or gate not in GATES:
        raise ValueError("unknown arm or gate")
    from .structuralcfg import lineup, manifest
    from .structural_duel import run_duel
    settings = PHASES[phase]
    actual_seed = settings["seed"] + (100_000 if gate == "shipped" else 0) if seed is None else seed
    if workers != PHASE_WORKERS:
        raise ValueError(f"workers must be {PHASE_WORKERS}")
    if actual_seed != _expected_seed(phase, arm, gate):
        raise ValueError("seed must use the reserved phase/gate family")
    players = lineup(arm, gate)
    result = run_duel(players, settings["games"], workers, seed=actual_seed)
    doc = {
        "family": "heximax-structural-future", "phase": phase,
        "source_hash": source_fingerprint(), "arm": arm,
        "candidate": arm, "candidate_color": CANDIDATE_COLOR,
        "role": "control" if arm == "control" else "candidate",
        "gate": gate, "games": settings["games"], "workers": workers,
        "seed": actual_seed, "players": players,
        "phenotype": (manifest()["control"] if arm == "control" else manifest()["candidates"][arm]),
        "wins": {str(k): int(v) for k, v in result.wins.items()},
        "points": {str(k): list(v) for k, v in result.points.items()},
        "report": result.report(),
    }
    validate_document(doc, phase=phase)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2) + "\n")
    tmp.replace(out)
    return doc


def validate_document(doc: dict, *, phase: str | None = None) -> dict:
    """Validate an in-memory document using the same checks as a file."""
    return _validate_document(doc, expected_source=source_fingerprint(), phase=phase)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path, nargs="?")
    parser.add_argument("--phase", choices=tuple(PHASES))
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--gate", choices=GATES)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--source-hash")
    args = parser.parse_args(argv)
    if args.artifact is not None:
        doc = validate_artifact(args.artifact, expected_source=args.source_hash, phase=args.phase)
        print(json.dumps({"valid": True, "phase": doc.get("phase", "screen"),
                          "arm": doc["arm"], "gate": doc["gate"],
                          "source_hash": doc["source_hash"]}, indent=2))
        return 0
    required = (args.phase, args.arm, args.gate, args.workers, args.out)
    if any(value is None for value in required):
        parser.error("a run requires --phase --arm --gate --workers --out")
    doc = run_phase(args.phase, args.arm, args.gate, args.workers, args.out, seed=args.seed)
    print(json.dumps(doc, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
