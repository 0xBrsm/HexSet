# SPDX-License-Identifier: GPL-3.0-only
"""Split the cost of `encode` across its four blocks.

`benchmarks.model_forward` reports `encode` as one number, and needs torch to
run at all. This one is numpy-only so it runs anywhere, and it says which of
the four blocks the time is in — which is the thing an optimisation needs.

Positions share one board, because that is what a search evaluates. Giving
each its own board measures cache misses no real batch pays.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from dataclasses import asdict, dataclass

from hexset.board.board import random_base_board
from hexset.encoding import (
    _building_points,
    _encode_edges,
    _encode_globals,
    _encode_hexes,
    _encode_vertices,
    _template,
    encode,
)
from hexset.game import imagine, is_over, start
from hexset.play import step_randomly

from .throughput import environment


@dataclass
class Result:
    positions: int
    players: int
    repeats: int
    encode_us: float
    building_points_us: float
    hexes_us: float
    vertices_us: float
    edges_us: float
    globals_us: float
    accounted: float


def _positions(count: int, players: int, seed: int) -> list:
    rng = random.Random(seed)
    game = start(random_base_board(rng), players, rng)
    for _ in range(60):
        step_randomly(game, rng)

    out = []
    while len(out) < count:
        out.append(imagine(game, random.Random(seed + len(out))))
        for _ in range(3):
            if is_over(game):
                return out + [out[-1]] * (count - len(out))
            step_randomly(game, rng)
    return out


def _cycle(items):
    index = 0

    def take():
        nonlocal index
        item = items[index % len(items)]
        index += 1
        return item

    return take


def _timed(fn, repeats: int) -> float:
    """Median microseconds per call, after a warmup."""
    for _ in range(max(3, repeats // 10)):
        fn()
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - started) * 1e6)
    return round(statistics.median(samples), 2)


def run(*, positions: int, players: int, seed: int, repeats: int) -> Result:
    # true state: timing the encoder's own internals, which read the true
    # state by design (`hexset.encoding` -- the encoder is engine code).
    games = _positions(positions, players, seed)
    template = _template(games[0].state(0, hidden=False).board, players)

    block = _encode_vertices(games[0].state(0, hidden=False), 0, template)
    points = _building_points(block, players)

    whole = _timed(lambda take=_cycle(games): encode(take()), repeats)
    hexes = _timed(
        lambda take=_cycle(games): _encode_hexes(take().state(0, hidden=False), template),
        repeats,
    )
    vertices = _timed(
        lambda take=_cycle(games): _encode_vertices(take().state(0, hidden=False), 0, template),
        repeats,
    )
    points_us = _timed(lambda: _building_points(block, players), repeats)
    edges = _timed(
        lambda take=_cycle(games): _encode_edges(take().state(0, hidden=False), 0), repeats
    )
    globals_ = _timed(lambda take=_cycle(games): _encode_globals(take(), 0, points), repeats)

    parts = hexes + vertices + points_us + edges + globals_
    return Result(
        positions=len(games),
        players=players,
        repeats=repeats,
        encode_us=whole,
        building_points_us=points_us,
        hexes_us=hexes,
        vertices_us=vertices,
        edges_us=edges,
        globals_us=globals_,
        accounted=round(parts / whole, 3),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positions", type=int, default=64)
    parser.add_argument("--players", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=400)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    result = run(
        positions=args.positions,
        players=args.players,
        seed=args.seed,
        repeats=args.repeats,
    )
    payload = {"environment": environment(), **asdict(result)}

    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    env = payload["environment"]
    print(f"commit {env['commit']}  dirty {env['dirty']}  {env['machine']}")
    print(f"{result.positions} positions on one board, {result.players} players")
    print(f"  encode        {result.encode_us:>8} us")
    print(f"  building pts  {result.building_points_us:>8} us")
    print(f"  hexes         {result.hexes_us:>8} us")
    print(f"  vertices      {result.vertices_us:>8} us")
    print(f"  edges         {result.edges_us:>8} us")
    print(f"  globals       {result.globals_us:>8} us")
    print(f"  parts / whole {result.accounted:>8}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
