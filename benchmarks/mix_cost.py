"""What an opponent mix costs a PPO iteration, per decision and per shard.

`--mix` used to accept two names. Once it accepts any arena entrant spec, the
question that decides whether the feature is usable is arithmetic: a
`search2-offers3` or `mcts:<ckpt>@64` lane opponent is far more expensive per
decision than `greedy`. Collection is about a quarter of a PPO iteration's
wall clock (the "92%" on record is the searched, expert-iteration collector's
number), so the question is how far a lane opponent can stretch that quarter.

The measurement is a **shard**, not a game and not a whole run: one PPO worker's
share of a production iteration, which at `--lanes 128 --games-per-iteration 128
--collect-workers 16` is 8 games on 8 lanes. That is the unit the iteration waits
on, because a cohort ends when the slowest worker's slowest game does.

Two modes, and they answer different halves.

*Default* runs one shard in this process with a stopwatch on the learner's policy
and on every mix opponent, so the per-decision cost of each side is **counted
where it is spent** rather than differenced between two runs. The learner's
figure is per request answered, so it already carries whatever batching the lane
count buys; an opponent's is per decision too, but `BotPolicy` answers a request
at a time and batching buys a scripted bot nothing.

*`--workers N`* runs the real `ParallelCollector` over N shards and reports wall
clock only. No stopwatch crosses the pipe, and it should not: this is the number
an affordability claim has to rest on, contention included.

**The affordability table interpolates, it does not model.** `mixed_caster`
draws independently per game index, and a cohort's cost is additive over its
games, so the expected seconds per game at fraction `f` is exactly

    S(f) = (1 - f) * S(0) + f * S(1)

and the only two points that need measuring are `f = 0` and `f = 1`. Measuring
`f = 0.15` directly instead would pay for the cast count's binomial noise --
seven cast games in a forty-eight-game cohort, standard deviation 2.6 -- to
learn nothing the endpoints do not already say.

**Use `--learner <checkpoint>`.** A random-init net flails to the action cap, so
its games run ~1400 actions against production's 865-1100, and its `S(0)` is
inflated by a tail no trained run plays. Every figure quoted for a real run
should be taken with real weights in the learner.

Two costs move together when a mix is added, and the interpolation carries both
without having to separate them: the opponent's decisions cost what they cost,
*and* the learner's own decisions get more expensive, because `Collector`
answers one `act` call per policy per tick and a cast game leaves the learner
with 2 seats instead of 4 -- half the batch, and the forward's fixed cost is
most of a small batch.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict, dataclass

import torch

from benchmarks.throughput import environment
from catan.actions import build_space
from catan.board.board import random_base_board
from catan.collect import (
    ParallelCollector,
    WorkerSpec,
    check_mix,
    mix_opponents,
    mixed_caster,
    parse_mix,
)
from catan.encoding import static_graph
from catan.model import CatanNet, ModelConfig, packing
from catan.policy import NetworkPolicy
from catan.selfplay import Collector


class Timed:
    """A `BatchPolicy` with a stopwatch, counting decisions and not calls."""

    def __init__(self, inner) -> None:
        self.inner = inner
        self.seconds = 0.0
        self.decisions = 0
        self.calls = 0

    def act(self, requests):
        start = time.perf_counter()
        out = self.inner.act(requests)
        self.seconds += time.perf_counter() - start
        self.decisions += len(requests)
        self.calls += 1
        return out


@dataclass
class Side:
    name: str
    decisions: int
    seconds: float
    ms_per_decision: float
    decisions_per_call: float


@dataclass
class Point:
    mix: str
    learner: str
    games: int
    lanes: int
    workers: int
    seconds: float
    seconds_per_game: float
    actions: int
    actions_per_game: float
    cast_share: float
    sides: list[dict]


def _frozen(path: str, board, players: int):
    from catan.collect import frozen

    return frozen(path, "cpu", board, players)


def _learner(path: str, space, graph, players: int, seed: int) -> NetworkPolicy:
    """A checkpoint as the *sampling* learner, shape from its own recorded args.

    Not `collect.frozen`, which argmaxes: a greedy learner plays shorter, less
    varied games than the run being priced, and game length is the denominator
    of every figure here.
    """
    state = torch.load(path, map_location="cpu", weights_only=False)
    stored = state.get("args", {})
    net = CatanNet(
        space,
        graph,
        players,
        ModelConfig(
            width=int(stored.get("width", 64)),
            rounds=int(stored.get("rounds", 2)),
            value_head=str(stored.get("value_head", "linear")),
            policy_head=str(stored.get("policy_head", "linear")),
        ),
    )
    net.load_state_dict(state["net"])
    return NetworkPolicy(
        net,
        space,
        packing(graph, players),
        device="cpu",
        generator=torch.Generator().manual_seed(seed),
    )


def _side(name: str, timed: Timed) -> Side:
    return Side(
        name=name,
        decisions=timed.decisions,
        seconds=round(timed.seconds, 3),
        ms_per_decision=round(timed.seconds / max(timed.decisions, 1) * 1e3, 3),
        decisions_per_call=round(timed.decisions / max(timed.calls, 1), 2),
    )


def shard(
    mix: str,
    *,
    games: int,
    lanes: int,
    players: int,
    seed: int,
    width: int,
    rounds: int,
    max_offers: int | None,
    action_cap: int,
    parent: str,
    learner: str = "",
) -> Point:
    """One worker's cohort, in this process, with every side on a stopwatch."""
    torch.set_num_threads(1)
    torch.manual_seed(seed)

    parsed = parse_mix(mix)
    check_mix(parsed, have_parent=bool(parent))

    board = random_base_board(random.Random(seed))
    topology = board.topology
    space = build_space(
        topology.num_vertices, topology.num_edges, topology.num_hexes, players
    )
    graph = static_graph(topology)
    if learner:
        policy = Timed(_learner(learner, space, graph, players, seed))
    else:
        net = CatanNet(space, graph, players, ModelConfig(width=width, rounds=rounds))
        policy = Timed(
            NetworkPolicy(
                net,
                space,
                packing(graph, players),
                device="cpu",
                generator=torch.Generator().manual_seed(seed),
            )
        )
    opponents = [
        Timed(opponent)
        for opponent in mix_opponents(
            parsed,
            seed=seed,
            max_offers=max_offers,
            lanes=lanes,
            parent=(lambda: _frozen(parent, board, players)) if parent else None,
        )
    ]
    collector = Collector(
        policy,
        lanes=min(lanes, games),
        fill=False,
        players=players,
        seed=seed,
        action_cap=action_cap,
        max_offers=max_offers,
        opponents=opponents,
        caster=(
            mixed_caster([f for _, f in parsed], players, seed) if parsed else None
        ),
    )

    start = time.perf_counter()
    episodes = collector.cohort(games)
    seconds = time.perf_counter() - start

    actions = sum(e.outcome.actions for e in episodes)
    cast = sum(1 for e in episodes if any(e.cast))
    sides = [_side("learner", policy)]
    sides += [_side(name, timed) for (name, _), timed in zip(parsed, opponents)]
    return Point(
        mix=mix or "(none)",
        learner=learner or "(random init)",
        games=len(episodes),
        lanes=min(lanes, games),
        workers=0,
        seconds=round(seconds, 3),
        seconds_per_game=round(seconds / max(len(episodes), 1), 3),
        actions=actions,
        actions_per_game=round(actions / max(len(episodes), 1), 1),
        cast_share=round(cast / max(len(episodes), 1), 4),
        sides=[asdict(s) for s in sides],
    )


def sharded(
    mix: str,
    *,
    games: int,
    lanes: int,
    workers: int,
    players: int,
    seed: int,
    width: int,
    rounds: int,
    max_offers: int | None,
    action_cap: int,
    parent: str,
) -> Point:
    """The real thing: `workers` shards over pipes, wall clock only.

    Weights are never synced. A worker builds its net from the spec and plays
    with it; the learner's parameters change which games get played and not what
    they cost, and a sync would need a net in this process for no other reason.
    """
    parsed = parse_mix(mix)
    check_mix(parsed, have_parent=bool(parent))
    shard_lanes = max(1, -(-lanes // workers))
    specs = [
        WorkerSpec(
            seed=seed,
            players=players,
            lanes=shard_lanes,
            action_cap=action_cap,
            max_offers=max_offers,
            first_game=worker,
            stride=workers,
            width=width,
            rounds=rounds,
            torch_seed=seed + 100_000 + worker,
            mix=tuple(parsed),
            parent=parent,
            cohort=True,
        )
        for worker in range(workers)
    ]
    collector = ParallelCollector(specs)
    try:
        start = time.perf_counter()
        episodes = collector.collect(games)
        seconds = time.perf_counter() - start
    finally:
        collector.close()

    actions = sum(e.outcome.actions for e in episodes)
    cast = sum(1 for e in episodes if any(e.cast))
    return Point(
        mix=mix or "(none)",
        learner="(random init)",
        games=len(episodes),
        lanes=shard_lanes,
        workers=workers,
        seconds=round(seconds, 3),
        seconds_per_game=round(seconds / max(len(episodes), 1), 3),
        actions=actions,
        actions_per_game=round(actions / max(len(episodes), 1), 1),
        cast_share=round(cast / max(len(episodes), 1), 4),
        sides=[],
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mix",
        action="append",
        default=None,
        help="one mix spec per flag; repeat to compare. '' is the no-mix baseline",
    )
    # The production PPO iteration, from `runs/klbreak-linear-r2/config`:
    # 128 games over 128 lanes and 16 workers is 8 games on 8 lanes a shard.
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--lanes", type=int, default=8)
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="0 times one shard in this process with per-side stopwatches; "
        "N times N real worker processes and reports wall clock only",
    )
    parser.add_argument("--players", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--max-offers", type=int, default=3)
    parser.add_argument("--action-cap", type=int, default=4000)
    parser.add_argument("--parent", default="", help="the 'parent' mix opponent")
    parser.add_argument(
        "--learner",
        default="",
        help="checkpoint whose weights the learner plays with. Quote no figure "
        "for a real run without it: a random-init net flails to the action cap "
        "and inflates the no-mix baseline it is compared against. Single-shard "
        "mode only -- a worker process builds its net from its own spec",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    mixes = args.mix if args.mix is not None else [""]
    points = []
    for mix in mixes:
        common = dict(
            games=args.games,
            lanes=args.lanes,
            players=args.players,
            seed=args.seed,
            width=args.width,
            rounds=args.rounds,
            max_offers=args.max_offers,
            action_cap=args.action_cap,
            parent=args.parent,
        )
        if args.workers:
            if args.learner:
                raise SystemExit("--learner is single-shard only; drop --workers")
            point = sharded(mix, workers=args.workers, **common)
        else:
            point = shard(mix, learner=args.learner, **common)
        points.append(point)
        if not args.json:
            print(
                f"{point.mix:<28} {point.seconds:>9.2f} s  "
                f"{point.seconds_per_game:>7.2f} s/game  "
                f"{point.actions_per_game:>7.1f} actions/game  "
                f"cast {point.cast_share:.2f}",
                flush=True,
            )
            for side in point.sides:
                print(
                    f"    {side['name']:<30} {side['decisions']:>7} decisions "
                    f"{side['seconds']:>8.2f} s "
                    f"{side['ms_per_decision']:>8.3f} ms/decision "
                    f"batch {side['decisions_per_call']:.2f}",
                    flush=True,
                )

    if args.json:
        json.dump(
            {"environment": environment(), "points": [asdict(p) for p in points]},
            sys.stdout,
            indent=1,
        )
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
