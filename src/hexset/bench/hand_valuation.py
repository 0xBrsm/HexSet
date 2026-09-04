# SPDX-License-Identifier: GPL-3.0-only
"""Fit and gate the redesigned hand terms against the hand valuation that shipped.

`hexset.bots.evaluate.hand_terms` replaced `progress`/`card`/`surplus_card`
with purchase progress, spare cards at the bank rate and smooth robber
exposure. The baseline every cell here plays is not a weight vector but the
old term set, frozen in `hexset.bench.shipped_hand` and checked against the
recorded choice census -- weights alone cannot express a term that no longer
exists.

Two subcommands, both on `road_sweep`'s grouped `[c, c, b, b]` seating and
its antithetic board pairing:

    python -m hexset.bench.hand_valuation fit --games 192 --json
    python -m hexset.bench.hand_valuation gates --games 384 --json

`fit` sweeps a grid over the three new weights (plus a control cell at the
fitted defaults, which must read about 50% against itself). `gates` runs the
two strength gates: (i) new heximax vs shipped-hand heximax, (ii) new heximax
vs shipped-hand search2, both reporting roads per game alongside.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace

from hexset.arena import Entrant, Z_95, wilson
from hexset.bench.road_sweep import run_cell
from hexset.bench.shipped_hand import SHIPPED_WEIGHTS
from hexset.bench.throughput import default_workers, environment
from hexset.bots.heximax.evaluate import TRADING_WEIGHTS

# The box is shared with other agents; 8 is the ceiling `road_sweep` records
# the owner setting, not a suggestion to raise if the load average looks low.
MAX_WORKERS = 8

# The grid, around the defaults the terms were written with. `buy_progress`
# is the scale on a hand that covers a purchase worth a victory point;
# `spare_card` the value of one bank-equivalent card the best purchase does
# not need; `robber_risk` the price of one bank-equivalent card expected to
# be lost to a 7.
# A cross through the values the terms were written with, one step either
# way on the two scales that decide whether cards are hoarded.
# `buy_progress` is the scale on a hand that fully covers a purchase worth a
# victory point; `robber_risk` the price of one bank-equivalent card expected
# to be lost to a 7. `spare_card` is not swept: it adds the same quarter-card
# to every hand at the table and the search only ever reads differences.
CENTRE = {"buy_progress": 0.30, "spare_card": 0.15, "robber_risk": -0.30}
# What the sweep above chose, and what `Weights` now ships.
FITTED = {"buy_progress": 0.45, "spare_card": 0.15, "robber_risk": -0.15}
CELLS: tuple[dict[str, float], ...] = (
    {**CENTRE},
    {**CENTRE, "buy_progress": 0.20},
    {**CENTRE, "buy_progress": 0.45},
    {**CENTRE, "robber_risk": -0.15},
    {**CENTRE, "robber_risk": -0.60},
)


def cells() -> list[dict[str, float]]:
    return [dict(cell) for cell in CELLS]


def _hexi(name: str, weights, depth: int, width: int | None, trades=None) -> Entrant:
    return Entrant(
        name, kind="heximax", depth=depth, width=width, weights=weights,
        max_trades=trades,
    )


def _shipped_hexi(depth: int, width: int | None, trades=None) -> Entrant:
    return Entrant(
        "heximax-shipped-hand", kind="heximax-shipped-hand", depth=depth, width=width,
        weights=SHIPPED_WEIGHTS, max_trades=trades,
    )


def _shipped_search2(depth: int, width: int | None) -> Entrant:
    return Entrant(
        "search2-shipped-hand", kind="search2-shipped-hand", depth=depth, width=width,
        weights=SHIPPED_WEIGHTS,
    )


def _line(label: str, result: dict) -> str:
    low, high = result["interval_95"]
    return (
        f"  {label:<34} win {result['wins']:>3}/{result['decided']} "
        f"{result['win_rate']:6.1%} [{low:.1%}, {high:.1%}]  "
        f"roads {result['challenger_roads_per_game']:.2f} vs "
        f"{result['baseline_roads_per_game']:.2f}  ({result['seconds']:.0f}s)"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("fit", "gates"))
    parser.add_argument("--games", type=int, default=192, help="per cell; multiple of 4")
    parser.add_argument("--seed", type=int, default=95000)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--width", type=int, default=6)
    parser.add_argument("--workers", type=int, default=min(MAX_WORKERS, default_workers()))
    parser.add_argument(
        "--only",
        choices=("i", "i-notrade", "ii", "ii-control"),
        default=None,
        help="gates mode: run one gate rather than all three, so the cells "
        "can be run separately and their JSON merged",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="write the JSON payload here after every cell, so a run cut "
        "short still leaves the cells it finished",
    )
    parser.add_argument(
        "--cells",
        default=None,
        help="fit mode: a JSON list of weight dicts to run instead of CELLS, "
        "for extending the grid when the sweep runs into an edge",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    if args.games % 4:
        parser.error("--games must be a multiple of 4 (grouped [c, c, b, b] seating)")
    if args.workers > MAX_WORKERS:
        parser.error(f"--workers must be at most {MAX_WORKERS} on this machine")

    common = dict(
        seed=args.seed, depth=args.depth, width=args.width, workers=args.workers
    )
    started = time.perf_counter()
    rows: list[dict] = []

    def flush() -> None:
        if args.out:
            with open(args.out, "w") as fh:
                json.dump(
                    {
                        "environment": environment(),
                        "settings": vars(args),
                        "seconds": round(time.perf_counter() - started, 1),
                        "cells": rows,
                    },
                    fh,
                    indent=2,
                )

    if args.mode == "fit":
        # Each candidate plays the centre of the grid, not the shipped hand
        # valuation: against the shipped one the whole grid saturates -- the
        # centre alone reads 128/128 -- so a cell's margin over the shipped
        # bot says nothing about the cell. The control is the centre against
        # itself, which must read about 50% or the harness is biased and no
        # cell below means anything.
        centre = _hexi("centre", replace(TRADING_WEIGHTS, **CENTRE), args.depth, args.width)
        control = run_cell(
            None, args.games, challenger=centre, baseline=centre, **common
        )
        control["cell"] = "control: centre vs centre"
        rows.append(control)
        print(_line("control centre/centre", control), file=sys.stderr, flush=True)
        flush()
        grid = json.loads(args.cells) if args.cells else cells()
        for cell in grid:
            if cell == CENTRE:
                continue
            result = run_cell(
                replace(TRADING_WEIGHTS, **cell), args.games, baseline=centre, **common
            )
            result["cell"] = cell
            rows.append(result)
            label = " ".join(f"{k[:4]}={v:g}" for k, v in cell.items())
            print(_line(label, result), file=sys.stderr, flush=True)
            flush()
    else:
        new = _hexi("heximax-new", TRADING_WEIGHTS, args.depth, args.width)
        # The third cell is the standing reading taken on this instrument:
        # 65.25% was measured elsewhere, and a gate that compares against it
        # needs its own control on the same boards and the same seating.
        gates = {
            "i": ("(i) heximax new vs heximax shipped hand", new,
                  _shipped_hexi(args.depth, args.width)),
            # The same gate with both sides' trading switched off: what is
            # left is how the two hand valuations rank positions, with the
            # exploitable gate taken out of the comparison.
            "i-notrade": ("(i, trading off) heximax new vs heximax shipped hand",
                          _hexi("heximax-new-notrade", TRADING_WEIGHTS, args.depth,
                                args.width, trades=0),
                          _shipped_hexi(args.depth, args.width, trades=0)),
            "ii": ("(ii) heximax new vs search2 shipped hand", new,
                   _shipped_search2(args.depth, args.width)),
            "ii-control": ("(ii control) heximax shipped hand vs search2 shipped hand",
                           _shipped_hexi(args.depth, args.width),
                           _shipped_search2(args.depth, args.width)),
        }
        for key in [args.only] if args.only else list(gates):
            label, challenger, baseline = gates[key]
            result = run_cell(
                None, args.games, challenger=challenger, baseline=baseline, **common
            )
            result["cell"] = label
            rows.append(result)
            print(_line(label, result), file=sys.stderr, flush=True)
            flush()

    elapsed = time.perf_counter() - started
    payload = {
        "environment": environment(),
        "settings": vars(args),
        "seconds": round(elapsed, 1),
        "cells": rows,
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"{len(rows)} cells, {args.games} games each, {elapsed:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
