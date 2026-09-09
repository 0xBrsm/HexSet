# SPDX-License-Identifier: GPL-3.0-only
"""Tune heximax's weights by play: one term at a time, paired boards, accept
only what clearly wins.

    python -m hexset.bench.weight_sweep --games 1024 --workers 30 \
        --out sweep.json

The outcome-likelihood fit (`hexset.fitting`) predicts winners better and
plays worse: a search consumes its coefficients as causal and they are not.
The one objective the search respects is the paired duel, so this tunes on
it directly -- but without a climb's random walk. For each free term in turn, a
handful of multiples of the incumbent's value (zero, half, double; then a
finer ring) each play the incumbent on the same boards, grouped
`[c, c, b, b]` with antithetic seat swaps (`run_cell` below). The best
cell replaces the incumbent only if its Wilson lower bound clears 50%; a
term whose every cell reads inside the interval is left where it is. `scarce`
is derived from `production` and moves with it. At the end the swept vector
plays the vector it started from over a larger confirm, on fresh boards.

What this can resolve is set by `--games`: 1,024 paired games put a cell's
standard error near 1.5 points, so an accept needs roughly 53%. A term that
matters less than that at the incumbent's scale is, for this bot, already
where it needs to be.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import replace

from hexset.arena import Entrant, Z_95, compete, wilson
from hexset.bench.throughput import default_workers, environment
from hexset.bots.evaluate import ROLLS, TERM_NAMES, Weights
from hexset.bots.heximax.evaluate import NO_TRADE_WEIGHTS, TRADING_WEIGHTS

SCARCE_PER_PRODUCTION = 0.91 / ROLLS

# Most load-bearing first, by the fit's own coefficients and the hand
# valuation's sweep, so the terms that move the most are settled before the
# ones that move relative to them.
DEFAULT_ORDER = (
    "production", "buy_progress", "robber_risk", "spare_card",
    "road", "diversity", "port", "knight",
)
PASSES = ((0.0, 0.5, 2.0), (0.71, 1.41))


def with_term(weights: Weights, term: str, value: float) -> Weights:
    changes = {term: value}
    if term == "production":
        changes["scarce"] = SCARCE_PER_PRODUCTION * value
    return replace(weights, **changes)


def run_cell(
    games: int, *, seed: int, workers: int, challenger: Entrant, baseline: Entrant
) -> dict:
    """One challenger-vs-baseline cell: `games` games, `[c, c, b, b]` seats.

    The grouping is nothing more than the lineup handed to `hexset.arena.
    compete`: its antithetic pairing swaps seats by `seats // 2` between the
    two halves of a pair, which exchanges the seat *pairs* `{0, 1}` and
    `{2, 3}` -- exactly the two sides of a `[c, c, b, b]` lineup. An
    interleaved `[c, b, c, b]` lineup does not get this: its seat pairs are
    the diagonals `{0, 2}`/`{1, 3}`, and shifting by two maps each diagonal
    onto itself, so the challenger holds the same two seats on both halves of
    a pair and the per-board seat term never cancels. That is why `--games`
    must be a multiple of 4.

    Roads per seat come from `Tournament.roads`, which the arena keeps for
    every game: a bare win rate would not show a term buying its wins by
    building differently.
    """
    challenger_seats = (0, 1)
    baseline_seats = (2, 3)

    started = time.perf_counter()
    tournament = compete(
        [challenger, challenger, baseline, baseline], games, seed=seed, workers=workers
    )
    elapsed = time.perf_counter() - started

    wins = decided = 0
    c_roads: list[int] = []
    b_roads: list[int] = []
    for winner, roads in zip(tournament.winners, tournament.roads):
        if winner is not None:
            decided += 1
            if winner in challenger_seats:
                wins += 1
        c_roads.extend(roads[i] for i in challenger_seats)
        b_roads.extend(roads[i] for i in baseline_seats)

    low, high = wilson(wins, decided, Z_95) if decided else (0.0, 1.0)
    return {
        "games": games,
        "decided": decided,
        "wins": wins,
        "win_rate": wins / decided if decided else 0.0,
        "interval_95": [low, high],
        "challenger_roads_per_game": statistics.mean(c_roads),
        "baseline_roads_per_game": statistics.mean(b_roads),
        "seconds": round(elapsed, 1),
    }


def entrant(name: str, weights: Weights, notrade: bool) -> Entrant:
    return Entrant(
        name, kind="heximax", depth=2, width=6, weights=weights,
        mode="notrade" if notrade else "honest", max_trades=0 if notrade else None,
    )


def as_dict(weights: Weights) -> dict[str, float]:
    return {name: getattr(weights, name) for name in TERM_NAMES}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("trading", "notrade"), default="trading")
    parser.add_argument("--games", type=int, default=1024, help="per cell; multiple of 4")
    parser.add_argument("--confirm", type=int, default=3072, help="final swept vs start")
    parser.add_argument("--seed", type=int, default=98000)
    parser.add_argument("--workers", type=int, default=default_workers())
    parser.add_argument("--terms", default=",".join(DEFAULT_ORDER))
    parser.add_argument(
        "--passes", default="0,0.5,2;0.71,1.41",
        help="factor rings, passes separated by ';', factors by ','",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.games % 4 or args.confirm % 4:
        parser.error("--games and --confirm must be multiples of 4")

    notrade = args.profile == "notrade"
    start = NO_TRADE_WEIGHTS if notrade else TRADING_WEIGHTS
    incumbent = start
    passes = [tuple(float(f) for f in ring.split(",")) for ring in args.passes.split(";")]
    terms = args.terms.split(",")
    common = dict(workers=args.workers)
    started = time.perf_counter()
    steps: list[dict] = []
    report: dict = {
        "environment": environment(), "settings": vars(args),
        "start": as_dict(start), "steps": steps,
    }

    def flush() -> None:
        report["incumbent"] = as_dict(incumbent)
        report["seconds"] = round(time.perf_counter() - started, 1)
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=2)

    cell_index = 0
    for pass_index, ring in enumerate(passes):
        for term in terms:
            current = getattr(incumbent, term)
            if current == 0.0 and pass_index > 0:
                continue  # a zeroed term has no ring to refine
            seed = args.seed + 1000 * cell_index
            cell_index += 1
            baseline = entrant("incumbent", incumbent, notrade)
            cells = []
            for factor in ring:
                value = current * factor if current != 0.0 else factor * 0.05
                if value == current:
                    continue
                candidate = with_term(incumbent, term, value)
                result = run_cell(
                    args.games, seed=seed, baseline=baseline,
                    challenger=entrant("candidate", candidate, notrade), **common,
                )
                low, high = result["interval_95"]
                cells.append({
                    "factor": factor, "value": value, "wins": result["wins"],
                    "decided": result["decided"], "win_rate": result["win_rate"],
                    "interval_95": [low, high],
                    "roads_per_game": [
                        result["challenger_roads_per_game"], result["baseline_roads_per_game"],
                    ],
                    "seconds": result["seconds"],
                })
                print(
                    f"  pass {pass_index} {term:<13} x{factor:<5g} = {value:9.4g}  "
                    f"{result['wins']:>4}/{result['decided']} {result['win_rate']:6.1%} "
                    f"[{low:.1%}, {high:.1%}]  ({result['seconds']:.0f}s)",
                    file=sys.stderr, flush=True,
                )
            best = max(cells, key=lambda c: c["wins"]) if cells else None
            accepted = bool(best and best["interval_95"][0] > 0.5)
            if accepted:
                incumbent = with_term(incumbent, term, best["value"])
            steps.append({
                "pass": pass_index, "term": term, "from": current, "seed": seed,
                "cells": cells, "accepted": accepted,
                "to": getattr(incumbent, term),
            })
            print(
                f"{'ACCEPT' if accepted else '  keep'} {term}: {current:.4g} -> "
                f"{getattr(incumbent, term):.4g}",
                file=sys.stderr, flush=True,
            )
            flush()

    if incumbent != start and args.confirm:
        result = run_cell(
            args.confirm, seed=args.seed + 500_000,
            baseline=entrant("start", start, notrade),
            challenger=entrant("swept", incumbent, notrade), **common,
        )
        report["confirm"] = {
            "games": args.confirm, "seed": args.seed + 500_000, "wins": result["wins"],
            "decided": result["decided"], "win_rate": result["win_rate"],
            "interval_95": result["interval_95"],
            "roads_per_game": [
                result["challenger_roads_per_game"], result["baseline_roads_per_game"],
            ],
            "seconds": result["seconds"],
        }
        low, high = result["interval_95"]
        print(
            f"CONFIRM swept vs start: {result['wins']}/{result['decided']} = "
            f"{result['win_rate']:.1%} [{low:.1%}, {high:.1%}]",
            file=sys.stderr, flush=True,
        )
    else:
        report["confirm"] = None
    flush()
    print(json.dumps(report["incumbent"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
