# SPDX-License-Identifier: GPL-3.0-only
"""What the self-play collector costs before any network is involved.

`hexset.selfplay` batches so the network's fixed dispatch toll is paid once per
tick rather than once per position. That only pays if the plumbing around it —
engine step, `encode`, legal actions, mask — is cheap relative to the forward,
so this measures the plumbing on its own with a trivial policy. Torch-free, so
it runs on the phone.

Two numbers matter and they are not the same one. Ticks/sec says how often a
batch is offered to the policy, and actions/sec says how much experience the
plumbing can produce. Sweeping the lane count separates them: actions/sec is
roughly flat, so lanes buy batch size rather than throughput, which is exactly
the trade the dispatch toll asks for.

`benchmarks.throughput` is the floor to read this against — the same engine
with no observations at all.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from typing import Sequence

from hexset.selfplay import Choice, Collector, RandomPolicy, Request

from .throughput import environment


@dataclass
class Result:
    lanes: int
    players: int
    ticks: int
    actions: int
    seconds: float
    ticks_per_second: float
    actions_per_second: float
    us_per_action: float
    policy_share: float
    games: int
    mean_actions_per_game: float


class _Timed:
    """Wraps a policy so its share of the tick can be separated from the rest."""

    def __init__(self, inner) -> None:
        self.inner = inner
        self.seconds = 0.0

    def act(self, requests: Sequence[Request]) -> Sequence[Choice]:
        started = time.perf_counter()
        out = self.inner.act(requests)
        self.seconds += time.perf_counter() - started
        return out


def run(*, lanes: int, players: int, ticks: int, seed: int, warmup: int) -> Result:
    policy = _Timed(RandomPolicy(random.Random(seed)))
    collector = Collector(policy, lanes=lanes, players=players, seed=seed)
    collector.run(warmup)

    policy.seconds = 0.0
    started = time.perf_counter()
    episodes = collector.run(ticks)
    elapsed = time.perf_counter() - started

    actions = ticks * lanes
    finished = sum(e.outcome.actions for e in episodes)
    return Result(
        lanes=lanes,
        players=players,
        ticks=ticks,
        actions=actions,
        seconds=round(elapsed, 3),
        ticks_per_second=round(ticks / elapsed, 1),
        actions_per_second=round(actions / elapsed, 1),
        us_per_action=round(elapsed / actions * 1e6, 1),
        policy_share=round(policy.seconds / elapsed, 3),
        games=len(episodes),
        mean_actions_per_game=round(finished / len(episodes), 1) if episodes else 0.0,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lanes",
        type=int,
        nargs="+",
        default=[1, 8, 32],
        help="lane counts to sweep",
    )
    parser.add_argument("--players", type=int, default=4)
    parser.add_argument("--ticks", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    results = [
        run(
            lanes=lanes,
            players=args.players,
            ticks=args.ticks,
            seed=args.seed,
            warmup=args.warmup,
        )
        for lanes in args.lanes
    ]
    payload = {"environment": environment(), "runs": [asdict(r) for r in results]}

    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    env = payload["environment"]
    print(f"commit {env['commit']}  dirty {env['dirty']}  {env['machine']}")
    print(f"{args.ticks} ticks, {args.players} players, random policy")
    print(f"  {'lanes':>6} {'ticks/s':>10} {'actions/s':>11} {'us/action':>10} {'policy':>8}")
    for result in results:
        print(
            f"  {result.lanes:>6} {result.ticks_per_second:>10} "
            f"{result.actions_per_second:>11} {result.us_per_action:>10} "
            f"{result.policy_share:>8}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
