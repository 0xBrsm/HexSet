"""Bounded Heximax stance/temperature screen against the frozen gates.

This module registers only names in its own namespace; the shipped
``heximax-notrade`` preset is never replaced.  It is intended for one
candidate per invocation so campaigns can control workers and seeds.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import hexset.bots  # register the shipped presets
from hexset.arena import Entrant, register_preset
from hexset.bots.heximax.evaluate import NO_TRADE_WEIGHTS
from hexset.bots.stances import WIN_TEMPERATURE

from .duel import run_duel


CONFIGS = {
    "baseline-win-T2.4766": {"stance": "win", "temperature": WIN_TEMPERATURE},
    "win-T1": {"stance": "win", "temperature": 1.0},
    "win-T1.5": {"stance": "win", "temperature": 1.5},
    "win-T4": {"stance": "win", "temperature": 4.0},
    "win-T6": {"stance": "win", "temperature": 6.0},
    "own": {"stance": "own", "temperature": None},
    "relative": {"stance": "relative", "temperature": None},
    "paranoid": {"stance": "paranoid", "temperature": None},
}

COMMON = {
    "depth": 2,
    "width": 6,
    "max_nodes": 600,
    "k": 1,
    "max_trades": 0,
    "mode": "notrade",
}


def _register() -> dict[str, dict]:
    manifest = {}
    for label, cfg in CONFIGS.items():
        name = "heximax-stancecfg-" + label
        entrant = Entrant(
            name,
            kind="heximax",
            weights=NO_TRADE_WEIGHTS,
            **COMMON,
            stance=cfg["stance"],
            temperature=cfg["temperature"],
        )
        register_preset(name, entrant)
        manifest[label] = {
            **cfg,
            **COMMON,
            "mode": "notrade",
            "max_trades": 0,
            "weights": asdict(NO_TRADE_WEIGHTS),
            "target_thresholds": {
                "vs-ab2": "frozen AB:2 gate",
                "vs-shipped": "frozen heximax-notrade incumbent gate",
            },
        }
    return manifest


MANIFEST = _register()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=sorted(MANIFEST), required=True)
    parser.add_argument("--gate", choices=("both", "ab2", "shipped"), default="both")
    parser.add_argument("--games", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    name = "heximax-stancecfg-" + args.candidate
    gates = {
        "vs-ab2": f"DC:{name},AB:2,AB:2,AB:2",
        "vs-shipped": f"DC:{name},DC:heximax-notrade,DC:heximax-notrade,DC:heximax-notrade",
    }
    if args.gate != "both":
        gates = {"vs-ab2" if args.gate == "ab2" else "vs-shipped": gates["vs-ab2" if args.gate == "ab2" else "vs-shipped"]}
    rows = []
    for label, players in gates.items():
        result = run_duel(players, args.games, args.workers, seed=args.seed)
        rows.append({
            "gate": label,
            "games": args.games,
            "workers": args.workers,
            "seed": args.seed,
            "players": players,
            "report": result.report(),
            "wins": {str(k): v for k, v in result.wins.items()},
            "points": {str(k): v for k, v in result.points.items()},
        })
    document = {
        "candidate": args.candidate,
        "manifest": MANIFEST[args.candidate],
        "games": args.games,
        "workers": args.workers,
        "seed": args.seed,
        "gate": args.gate,
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(document, indent=2))
    print(json.dumps(document, indent=2), flush=True)


if __name__ == "__main__":
    main()
