"""What a search-played move costs, split into engine and network.

`benchmarks.model_forward` measured a *self-play* move and found it dominated by
the forward, which is what shaped `catan.selfplay` and `catan.mcts`. A move under
a search is not that move. It enumerates, copies and steps a position for every
leaf it expands, and a phone measurement of the engine alone put that at ~298 µs
per leaf against ~25 µs to evaluate one — so the ratio is the other way round and
the conclusion drawn from `model_forward` does not carry over.

This settles it on the box, where it matters, by timing a real collector against
a real checkpoint. The split is exact rather than modelled: the evaluator is
wrapped in a timer, so network seconds are counted where they are spent and
engine seconds are the remainder. No stub run, no scaling assumption, no
separate game to compare against.

The output that decides a training run is `games_per_hour`, which is what fixes
how many expert-iteration games a night holds. It is extrapolated from a partial
game, since playing whole games at 256 simulations would cost more than the
measurement is worth; `actions_per_game` is the assumption it rests on and is
printed alongside so the extrapolation can be checked rather than trusted.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict, dataclass

from benchmarks.throughput import environment
from catan.board.board import random_base_board
from catan.expert import SearchPolicy
from catan.netbot import searcher
from catan.selfplay import Collector


class Timed:
    """The evaluator with a stopwatch, so the split needs no second run."""

    def __init__(self, inner) -> None:
        self.inner = inner
        self.seconds = 0.0
        self.leaves = 0
        self.waves = 0

    def evaluate(self, leaves):
        start = time.perf_counter()
        out = self.inner.evaluate(leaves)
        self.seconds += time.perf_counter() - start
        self.leaves += len(leaves)
        self.waves += 1
        return out


@dataclass
class Point:
    simulations: int
    wave: int
    lanes: int
    compile_mode: str
    moves: int
    seconds: float
    ms_per_move: float
    network_share: float
    us_per_leaf_network: float
    us_per_leaf_engine: float
    leaves_per_move: float
    waves_per_move: float
    games_per_hour: float


def measure(
    checkpoint: str,
    *,
    simulations: int,
    wave: int,
    lanes: int,
    moves: int,
    players: int,
    seed: int,
    device: str,
    compile_mode: str,
    max_offers: int | None,
    actions_per_game: int,
) -> Point:
    rng = random.Random(seed)
    board = random_base_board(rng)
    search = searcher(
        checkpoint,
        board,
        simulations=simulations,
        wave=wave,
        max_offers=max_offers,
        device=device,
        compile_mode=compile_mode,
        rng=random.Random(seed),
    )
    timed = Timed(search.evaluator)
    search.evaluator = timed
    policy = SearchPolicy(search, rng=random.Random(seed))
    collector = Collector(
        policy,
        lanes=lanes,
        seed=seed,
        players=players,
        board=board,
        max_offers=search.max_offers,
    )

    # One move first, so a compiled forward's warm-up is not billed to the run.
    collector.tick()
    timed.seconds = timed.leaves = timed.waves = 0

    start = time.perf_counter()
    collector.run(moves)
    seconds = time.perf_counter() - start

    engine = seconds - timed.seconds
    decisions = moves * lanes
    per_move = seconds / decisions
    return Point(
        simulations=simulations,
        wave=wave,
        lanes=lanes,
        compile_mode=compile_mode,
        moves=decisions,
        seconds=round(seconds, 3),
        ms_per_move=round(per_move * 1e3, 3),
        network_share=round(timed.seconds / seconds, 4),
        us_per_leaf_network=round(timed.seconds / max(timed.leaves, 1) * 1e6, 1),
        us_per_leaf_engine=round(engine / max(timed.leaves, 1) * 1e6, 1),
        leaves_per_move=round(timed.leaves / decisions, 2),
        waves_per_move=round(timed.waves / decisions, 2),
        games_per_hour=round(3600.0 / (per_move * actions_per_game), 2),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--simulations", type=int, nargs="+", default=[16, 64, 256])
    parser.add_argument("--wave", type=int, default=16)
    parser.add_argument("--lanes", type=int, default=1)
    parser.add_argument("--moves", type=int, default=200)
    parser.add_argument("--players", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument(
        "--compile",
        dest="compile_mode",
        default="none",
        choices=["none", "default", "reduce-overhead", "max-autotune"],
        help="torch.compile mode for checkpoint inference",
    )
    parser.add_argument(
        "--max-offers",
        type=int,
        default=None,
        help="default is the budget the checkpoint trained under",
    )
    parser.add_argument(
        "--actions-per-game",
        type=int,
        default=950,
        help="what games_per_hour extrapolates through; 950 at an offer budget of 3",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    points = [
        measure(
            args.checkpoint,
            simulations=n,
            wave=args.wave,
            lanes=args.lanes,
            moves=args.moves,
            players=args.players,
            seed=args.seed,
            device=args.device,
            compile_mode=args.compile_mode,
            max_offers=args.max_offers,
            actions_per_game=args.actions_per_game,
        )
        for n in args.simulations
    ]

    payload = {
        "environment": environment(),
        "checkpoint": args.checkpoint,
        "device": args.device,
        "actions_per_game": args.actions_per_game,
        "points": [asdict(point) for point in points],
    }
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    env = payload["environment"]
    print(f"commit {env['commit']}  dirty {env['dirty']}  {env['machine']}")
    print(
        f"{args.checkpoint} on {args.device}, wave {args.wave}, "
        f"{args.lanes} lanes, compile {args.compile_mode}, "
        f"{args.moves * args.lanes} moves"
    )
    print(f"games/hour extrapolated through {args.actions_per_game} actions a game")
    print(
        f"{'sims':>5} {'ms/move':>9} {'net %':>7} {'net us/leaf':>12} "
        f"{'eng us/leaf':>12} {'leaves':>8} {'games/h':>9}"
    )
    for point in points:
        print(
            f"{point.simulations:>5} {point.ms_per_move:>9.2f} "
            f"{point.network_share * 100:>6.1f}% {point.us_per_leaf_network:>12.1f} "
            f"{point.us_per_leaf_engine:>12.1f} {point.leaves_per_move:>8.1f} "
            f"{point.games_per_hour:>9.2f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
