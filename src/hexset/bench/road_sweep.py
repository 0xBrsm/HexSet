# SPDX-License-Identifier: GPL-3.0-only
"""Sweep heximax's road weight (and card weight) against itself.

Hypothesis: `Weights.road` (0.1209 per road) makes a road the cheapest way to
turn two cards into static score, and at depth 2 the search rarely sees the
settlement those cards would otherwise buy, so heximax converts wood+brick
into roads whenever it can -- more than a human would. This plays a
challenger heximax (a modified `Weights`) against the intact baseline
heximax, on identical boards with seats mirrored, exactly the way
`hexset.bench.ablate` plays a zeroed term against the full vector -- except
this also records what `ablate`/`hexset.tuning.duel` throw away: roads,
settlements and cities per seat, and game length, not just who won.

Depth 2, width 6 (the `heximax` preset), honest mode, trading on -- the
shipped configuration, unchanged except for the two weights under test.

One four-seat lineup per game, `[challenger, challenger, baseline, baseline]`
-- grouped, not interleaved; see `run_cell`'s docstring for why the grouping
matters -- antithetic-paired the way `hexset.arena.compete` pairs a duel: the
two seat pairs swap between the two halves of a board, cancelling most of the
seat term, and `--games` must be a multiple of 4 for that rotation to
complete. `--seed` fixes the board sequence, so every cell in a sweep (and
the control cell, which should read about 50%) sees the same boards.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from dataclasses import replace
from multiprocessing import Pool

from hexset.arena import MAX_ACTIONS, Entrant, Z_95, play, seat_of, spawn, wilson
from hexset.bench.throughput import default_workers, environment
from hexset.board.board import random_base_board
from hexset.bots.heximax.evaluate import TRADING_WEIGHTS
from hexset.state import city_count, road_count, settlement_count
from hexset.victory import victory_points

# The machine this sweep runs on is shared with other jobs; 8 is the ceiling
# the owner set, not a suggestion to raise if idle.
MAX_WORKERS = 8

# road x card cells: control first (must read ~50%, since it is the same
# vector as the baseline), then road alone stepped down to zero, then two
# cells that also raise `card` -- the hypothesis says a cheaper road and a
# more valuable hand both push away from road-building.
DEFAULT_CELLS: tuple[dict[str, float], ...] = (
    {"road": 0.1209},  # control: identical to baseline
    {"road": 0.08},
    {"road": 0.04},
    {"road": 0.0},
    {"road": 0.04, "spare_card": 0.02},
    {"road": 0.0, "spare_card": 0.02},
)


def _play_one(
    job: tuple[tuple[Entrant, ...], int, int],
) -> tuple[int | None, int, tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Play game `index`. Returns (winning entrant, turns, points, roads,
    settlements, cities), each of the last four in entrant order.

    Board and rotation derivation is `hexset.arena._play_one`'s, verbatim --
    same seed string keys, same antithetic pairing over the 4-seat lineup --
    so a cell here plays the identical boards `hexset.tuning.duel` would at
    the same `seed`. What is added is the per-seat build census `compete`
    does not keep: `hexset.arena.Tournament` only carries points and turns.
    """
    entrants, index, seed = job
    seats = len(entrants)
    pair, half = divmod(index, 2)
    board_index = pair
    rotation = pair + half * (seats // 2)
    board = random_base_board(random.Random(f"{seed}:{board_index}:board"))
    seats_taken = [seat_of(e, rotation, seats) for e in range(seats)]

    lineup: list = [None] * seats
    for e, entrant in enumerate(entrants):
        lineup[seats_taken[e]] = spawn(
            entrant, board, random.Random(f"{seed}:{board_index}:{e}")
        )

    game = play(
        lineup,
        board,
        random.Random(f"{seed}:{board_index}:game"),
        action_cap=MAX_ACTIONS,
    )

    points = []
    roads = []
    settlements = []
    cities = []
    for e in range(seats):
        seat = seats_taken[e]
        # true state: the same reasoning as `hexset.arena._play_one` -- the
        # terminal census includes hidden victory-point cards.
        state = game.state(seat, hidden=False)
        points.append(victory_points(state, seat))
        roads.append(road_count(state, seat))
        settlements.append(settlement_count(state, seat))
        cities.append(city_count(state, seat))

    winner = None if game.won_by is None else seats_taken.index(game.won_by)
    return winner, game.turns, tuple(points), tuple(roads), tuple(settlements), tuple(cities)


def run_cell(
    challenger_weights,
    games: int,
    *,
    seed: int,
    depth: int,
    width: int | None,
    workers: int,
    baseline: Entrant | None = None,
    challenger: Entrant | None = None,
) -> dict:
    """One challenger-vs-baseline cell: `games` games, `[c, c, b, b]` seats.

    `challenger`/`baseline` name the two entrants outright, for a caller
    playing something other than two heximax bots at different weights
    (`hexset.bench.hand_valuation` plays the shipped hand valuation, which is
    a different term set rather than a different vector, and plays search2 as
    the baseline); left unset, both are the shipped heximax and
    `challenger_weights` is the only difference between them.

    Grouped, not interleaved: `_play_one`'s antithetic pairing swaps seats by
    `seats // 2` between the two halves of a pair, which exchanges the seat
    *pairs* `{0, 1}` and `{2, 3}` -- exactly the two sides of a `[c, c, b, b]`
    lineup, per `hexset.arena._play_one`'s own docstring ("with an [a, a, b,
    b] lineup it exchanges the two sides' seat pairs exactly"). An
    interleaved `[c, b, c, b]` lineup does not get this: its seat pairs are
    the two diagonals `{0, 2}`/`{1, 3}`, and shifting by `seats // 2 == 2`
    maps each diagonal onto itself, so the challenger holds the same two
    seats on both halves of a pair and the per-board seat term never
    cancels. A first run with the interleaved lineup read the control cell
    (byte-identical weights on both sides) at 44.3% instead of the expected
    ~50% for exactly this reason.
    """
    if challenger is None:
        challenger = Entrant(
            "challenger", kind="heximax", depth=depth, width=width,
            weights=challenger_weights,
        )
    if baseline is None:
        baseline = Entrant(
            "baseline", kind="heximax", depth=depth, width=width, weights=TRADING_WEIGHTS
        )
    lineup = (challenger, challenger, baseline, baseline)
    challenger_seats = (0, 1)
    baseline_seats = (2, 3)

    jobs = [(lineup, i, seed) for i in range(games)]
    started = time.perf_counter()
    if workers > 1:
        with Pool(workers) as pool:
            outcomes = pool.map(_play_one, jobs, chunksize=1)
    else:
        outcomes = [_play_one(job) for job in jobs]
    elapsed = time.perf_counter() - started

    wins = decided = 0
    turns: list[int] = []
    c_vp: list[int] = []
    b_vp: list[int] = []
    c_roads: list[int] = []
    b_roads: list[int] = []
    c_settlements: list[int] = []
    b_settlements: list[int] = []
    c_cities: list[int] = []
    b_cities: list[int] = []
    per_game = []
    for winner, game_turns, points, roads, settlements, cities in outcomes:
        turns.append(game_turns)
        if winner is not None:
            decided += 1
            if winner in challenger_seats:
                wins += 1
        for i in challenger_seats:
            c_vp.append(points[i])
            c_roads.append(roads[i])
            c_settlements.append(settlements[i])
            c_cities.append(cities[i])
        for i in baseline_seats:
            b_vp.append(points[i])
            b_roads.append(roads[i])
            b_settlements.append(settlements[i])
            b_cities.append(cities[i])
        per_game.append(
            {
                "winner": winner,
                "turns": game_turns,
                "points": points,
                "roads": roads,
                "settlements": settlements,
                "cities": cities,
            }
        )

    low, high = wilson(wins, decided, Z_95) if decided else (0.0, 1.0)
    return {
        "games": games,
        "decided": decided,
        "unfinished": games - decided,
        "wins": wins,
        "win_rate": wins / decided if decided else 0.0,
        "interval_95": [low, high],
        "challenger_roads_per_game": statistics.mean(c_roads),
        "baseline_roads_per_game": statistics.mean(b_roads),
        "challenger_settlements_per_game": statistics.mean(c_settlements),
        "baseline_settlements_per_game": statistics.mean(b_settlements),
        "challenger_cities_per_game": statistics.mean(c_cities),
        "baseline_cities_per_game": statistics.mean(b_cities),
        "challenger_vp_mean": statistics.mean(c_vp),
        "baseline_vp_mean": statistics.mean(b_vp),
        "mean_turns": statistics.mean(turns),
        "seconds": round(elapsed, 1),
        "per_game": per_game,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=200, help="per cell; multiple of 4")
    parser.add_argument("--seed", type=int, default=90000)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--width", type=int, default=6)
    parser.add_argument(
        "--workers", type=int, default=min(MAX_WORKERS, default_workers())
    )
    parser.add_argument(
        "--cells",
        type=str,
        default=None,
        help="path to a JSON file of [{weight field: value, ...}, ...]; "
        "defaults to the six-cell road sweep built into this script",
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable output")
    args = parser.parse_args(argv)

    if args.games % 4:
        parser.error("--games must be a multiple of 4 (mirrored [c, b, c, b] seating)")
    if args.workers > MAX_WORKERS:
        parser.error(f"--workers must be at most {MAX_WORKERS} on this machine")

    if args.cells:
        with open(args.cells) as fh:
            cells = json.load(fh)
    else:
        cells = DEFAULT_CELLS

    started = time.perf_counter()
    rows = []
    for cell in cells:
        weights = replace(TRADING_WEIGHTS, **cell)
        result = run_cell(
            weights,
            args.games,
            seed=args.seed,
            depth=args.depth,
            width=args.width,
            workers=args.workers,
        )
        result["cell"] = dict(cell)
        rows.append(result)
        # Progress goes to stderr unconditionally -- a sweep runs long enough
        # that its own cadence matters, and `--json` still wants a clean
        # single document on stdout.
        low, high = result["interval_95"]
        print(
            "  " + " ".join(f"{k}={v:<8.4g}" for k, v in cell.items()) + " "
            f"win {result['wins']:>3}/{result['decided']} {result['win_rate']:6.1%} "
            f"[{low:.1%}, {high:.1%}]  "
            f"roads {result['challenger_roads_per_game']:.2f} vs "
            f"{result['baseline_roads_per_game']:.2f}  "
            f"({result['seconds']:.0f}s)",
            file=sys.stderr,
            flush=True,
        )
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
        print(f"{len(rows)} cells, {args.games} games each, depth {args.depth}, {elapsed:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
