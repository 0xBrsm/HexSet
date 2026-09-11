# SPDX-License-Identifier: GPL-3.0-only
"""Prereistered, opt-in screen for the structural Heximax family.

The ordinary ``heximax-notrade`` preset is used as the control and is never
replaced.  Importing this module explicitly registers three native Heximax
entrant kinds plus their names; it does not launch a duel.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import hexset.bots  # register ordinary presets
from hexset.arena import Entrant, PRESETS, register_preset
from hexset.bots.heximax.evaluate import NO_TRADE_WEIGHTS
from hexset.bots.heximax.structural_features import (  # registers kinds/presets
    AWARD_FRAGILITY_COEFFICIENT, BACKUP_PROGRESS_COEFFICIENT, STRUCTURAL_TEMPERATURE,
)

CONTROL = "heximax-notrade"
CANDIDATES = ("backup", "fragility", "combined")
ARMS = ("control",) + CANDIDATES
GATES = ("ab2", "shipped")
SCREEN_GAMES = 120
SCREEN_WORKERS = 30
# Structural streams are reserved separately from discovery families.
# Each gate gets its own 100,000 block; each arm gets a disjoint 10,000 slice.
SCREEN_SEEDS = {"ab2": 400_000_000, "shipped": 400_100_000}
DCP_FUTURE_SEED = 410_000_000
CONFIRMATION_SEED = 420_000_000
HOLDOUT_SEED = 440_000_000
RUNTIME_PHENOTYPE = {
    "placement": True,
    "outer_placement_wrapper": False,
    "stance": "win",
    "temperature": STRUCTURAL_TEMPERATURE,
    "port_aware": False,
}

# Keep this registration in the controller as an explicit dependency.  It
# makes ``structuralcfg`` the worker-driver import target and keeps the
# structural names out of the shipped preset module.
STRUCTURAL_PRESETS = {
    f"heximax-structural-{label}": PRESETS[f"heximax-structural-{label}"]
    for label in CANDIDATES
}


def source_fingerprint() -> str:
    """Fingerprint the complete structural evaluator/controller contract."""
    here = Path(__file__).resolve()
    files = (
        here,
        here.with_name("structural_validation.py"),
        here.parents[1] / "bots" / "heximax" / "structural_features.py",
        here.parents[1] / "bots" / "heximax" / "evaluate.py",
        here.parents[1] / "bots" / "heximax" / "search.py",
        here.with_name("duel.py"),
        here.with_name("structural_duel.py"),
        # The adapter/state bridge is part of the runtime phenotype: a
        # structural artifact cannot be resumed if translation semantics move.
        here.with_name("player.py"),
        here.with_name("state.py"),
        here.with_name("actions.py"),
        here.with_name("board.py"),
    )
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(here.parents[2])).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def manifest() -> dict:
    return {
        "family": "heximax-structural-future",
        "source_hash": source_fingerprint(),
        "weights": asdict(NO_TRADE_WEIGHTS),
        "coefficients": {
            "backup_progress": BACKUP_PROGRESS_COEFFICIENT,
            "award_fragility": AWARD_FRAGILITY_COEFFICIENT,
        },
        "control": {
            "name": CONTROL, "kind": "heximax", "mode": "notrade",
            "depth": 2, "width": 6, "max_nodes": 600, "k": 1,
            "max_trades": 0, "weights": asdict(NO_TRADE_WEIGHTS),
            "runtime": dict(RUNTIME_PHENOTYPE),
        },
        "candidates": {
            label: {
                "name": f"heximax-structural-{label}",
                "kind": f"heximax-structural-{label}",
                "mode": "notrade", "depth": 2, "width": 6,
                "max_nodes": 600, "k": 1, "max_trades": 0,
                "weights": asdict(NO_TRADE_WEIGHTS),
                "runtime": dict(RUNTIME_PHENOTYPE),
            }
            for label in CANDIDATES
        },
        "gates": {
            "ab2": "DC:<arm>,AB:2,AB:2,AB:2",
            "shipped": "DC:<arm>,DC:heximax-notrade,DC:heximax-notrade,DC:heximax-notrade",
        },
        "screen": {
            "games": SCREEN_GAMES, "workers": SCREEN_WORKERS,
            "seeds": SCREEN_SEEDS,
            "arms": ARMS,
            "controls_never_promote": True,
        },
        "dcp_future": {"seed_base": DCP_FUTURE_SEED},
        "confirmation": {"games": 1024, "seed_base": CONFIRMATION_SEED},
        "holdout": {"games": 1024, "seed_base": HOLDOUT_SEED},
    }


def _arm_name(arm: str) -> str:
    return CONTROL if arm == "control" else f"heximax-structural-{arm}"


def lineup(arm: str, gate: str) -> str:
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    if gate not in GATES:
        raise ValueError(f"unknown gate: {gate}")
    name = _arm_name(arm)
    return (
        f"DC:{name},AB:2,AB:2,AB:2" if gate == "ab2" else
        f"DC:{name},DC:heximax-notrade,DC:heximax-notrade,DC:heximax-notrade"
    )


def screen_jobs() -> list[dict]:
    """Exactly eight planned jobs with disjoint arm/gate seed streams."""
    jobs = []
    for arm_index, arm in enumerate(ARMS):
        for gate in GATES:
            # Separate arm streams by 10,000; gate streams remain distinct.
            seed = SCREEN_SEEDS[gate] + arm_index * 10_000
            jobs.append({
                "arm": arm, "role": "control" if arm == "control" else "candidate",
                "gate": gate, "games": SCREEN_GAMES,
                "workers": SCREEN_WORKERS, "seed": seed,
                "players": lineup(arm, gate),
            })
    return jobs


def run_screen(arm: str, gate: str, games: int, workers: int, seed: int, out: Path) -> dict:
    """Run one explicit job and atomically write its normalized artifact."""
    if arm not in ARMS or gate not in GATES:
        raise ValueError("unknown arm or gate")
    expected = next(
        job for job in screen_jobs() if job["arm"] == arm and job["gate"] == gate
    )
    if (games, workers, seed) != (expected["games"], expected["workers"], expected["seed"]):
        raise ValueError("screen settings must use the reserved job exactly")
    from .structural_duel import run_duel
    from .structuralcfg import manifest
    result = run_duel(lineup(arm, gate), games, workers, seed=seed)
    doc = {
        "family": "heximax-structural-future", "phase": "screen",
        "source_hash": source_fingerprint(),
        "arm": arm, "role": "control" if arm == "control" else "candidate",
        "candidate": arm, "candidate_color": "Color.RED",
        "gate": gate, "games": games, "workers": workers, "seed": seed,
        "players": lineup(arm, gate),
        "phenotype": (manifest()["control"] if arm == "control" else manifest()["candidates"][arm]),
        "wins": {str(k): int(v) for k, v in result.wins.items()},
        "points": {str(k): list(v) for k, v in result.points.items()},
        "report": result.report(),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2) + "\n")
    tmp.replace(out)
    return doc


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--gate", choices=GATES)
    parser.add_argument("--games", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    if args.plan:
        print(json.dumps({"manifest": manifest(), "jobs": screen_jobs()}, indent=2))
        return 0
    required = (args.arm, args.gate, args.games, args.workers, args.seed, args.out)
    if any(value is None for value in required):
        parser.error("a run requires --arm --gate --games --workers --seed --out")
    doc = run_screen(args.arm, args.gate, args.games, args.workers, args.seed, args.out)
    print(json.dumps(doc, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
