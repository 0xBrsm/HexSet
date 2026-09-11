"""Bounded screen for the opt-in bank-trade horizon extension.

This is deliberately a small orchestration surface: the ordinary Heximax
presets and the no-trade policy are untouched.  Each invocation evaluates one
registered bank extension against one frozen gate, so the controller can keep
the gates and seed families independent.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import hexset.bots
import hexset.bots.heximax.bank_quiescence  # register bank extensions
from hexset.arena import Entrant, PRESETS, register_preset
from .duel import run_duel

# Controls use ordinary Heximax, keeping the extension comparison matched.
if "heximax-notrade-wide" not in PRESETS:
    register_preset("heximax-notrade-wide", Entrant(
        "heximax-notrade-wide", kind="heximax", mode="notrade",
        depth=2, width=12, max_nodes=2400, max_trades=0))
CANDIDATES = ("bankext", "bankext-wide", "control", "control-wide")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate", choices=CANDIDATES, required=True)
    p.add_argument("--gate", choices=("ab2", "shipped"), required=True)
    p.add_argument("--games", type=int, required=True)
    p.add_argument("--workers", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    if a.games < 1 or a.workers < 1:
        p.error("games/workers must be positive")
    name = {"control": "heximax-notrade", "control-wide": "heximax-notrade-wide"}.get(a.candidate, "heximax-" + a.candidate)
    lineup = (f"DC:{name},AB:2,AB:2,AB:2" if a.gate == "ab2" else
              f"DC:{name},DC:heximax-notrade,DC:heximax-notrade,DC:heximax-notrade")
    result = run_duel(lineup, a.games, a.workers, seed=a.seed)
    doc = {"candidate": a.candidate, "gate": "vs-" + a.gate,
           "games": a.games, "workers": a.workers, "seed": a.seed,
           "players": lineup, "max_trades": 0,
           "wins": {str(k): v for k, v in result.wins.items()},
           "points": {str(k): v for k, v in result.points.items()},
           "report": result.report()}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out.with_suffix(a.out.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2) + "\n")
    tmp.replace(a.out)
    print(json.dumps(doc, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
