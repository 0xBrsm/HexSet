# SPDX-License-Identifier: GPL-3.0-only
"""Controlled four-player Catanatron ablation campaign.

Each named candidate is a complete Heximax-notrade policy differing from the
shipped vector in one deliberately specified feature or local weight.  The
module registers those names, then delegates game execution to the same
sharded Catanatron duel harness used for the baseline.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import hexset.bots  # register the shipped presets
from hexset.arena import Entrant, register_preset
from hexset.bots.heximax.evaluate import NO_TRADE_WEIGHTS

from .duel import run_duel


def _register() -> dict[str, dict]:
    """Register the preregistered candidate set and return its manifest."""
    w = NO_TRADE_WEIGHTS
    terms = ("production", "buy_progress", "robber_risk", "spare_card",
             "road", "diversity", "port", "knight")
    specs: dict[str, object] = {"baseline": w}
    for term in terms:
        specs[f"drop-{term}"] = replace(w, **{term: 0.0})
    # Local ring around the already adopted values.  Production carries its
    # derived scarcity coefficient through Evaluator, so it is changed as a
    # complete production feature rather than leaving a stale subterm.
    for label, factor in (("prod-half", .5), ("prod-141", 1.41),
                          ("progress-half", .5), ("progress-141", 1.41),
                          ("risk-half", .5), ("risk-141", 1.41),
                          ("road-half", .5), ("road-zero", 0.0),
                          ("spare-half", .5), ("spare-141", 1.41)):
        term = {"prod": "production", "progress": "buy_progress",
                "risk": "robber_risk", "road": "road", "spare": "spare_card"}[label.split("-")[0]]
        specs[label] = replace(w, **{term: getattr(w, term) * factor})
    manifest = {}
    for label, weights in specs.items():
        name = "heximax-ablate-" + label
        if label == "baseline":
            # Keep this alias explicit so every run records the same preset
            # construction as the candidates.
            weights = w
        register_preset(name, Entrant(name, kind="heximax", depth=2, width=6,
                                      max_trades=0, mode="notrade", weights=weights))
        manifest[label] = {k: getattr(weights, k) for k in weights.__dataclass_fields__}
    return manifest


MANIFEST = _register()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate", choices=sorted(MANIFEST), required=True)
    p.add_argument("--games", type=int, required=True)
    p.add_argument("--workers", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    name = "heximax-ablate-" + args.candidate
    # Candidate vs AB2 and candidate vs three independent copies of the
    # shipped Heximax-notrade are separate, fresh calls with the same seed.
    # The second call is a strength gate against the incumbent, not literal
    # four-way candidate self-play, which is symmetric by construction.
    rows = []
    for label, players in (
        ("vs-ab2", f"DC:{name},AB:2,AB:2,AB:2"),
        ("vs-shipped", f"DC:{name},DC:heximax-notrade,DC:heximax-notrade,DC:heximax-notrade"),
    ):
        result = run_duel(players, args.games, args.workers, seed=args.seed)
        rows.append({"gate": label, "report": result.report(),
                     "games": args.games,
                     "wins": {str(k): v for k, v in result.wins.items()},
                     "points": {str(k): v for k, v in result.points.items()},
                     "players": players})
    document = {"candidate": args.candidate, "manifest": MANIFEST[args.candidate],
                "games": args.games, "workers": args.workers, "seed": args.seed,
                "rows": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(document, indent=2))
    print(json.dumps(document, indent=2), flush=True)


if __name__ == "__main__":
    main()
