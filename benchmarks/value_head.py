"""What the value head explains, and what is left over.

A training run's `explained_variance` is a ratio whose denominator moves. It is
`1 - Var(target - prediction) / Var(target)`, and over a self-play run the
target is every seat's terminal relative points — a quantity that shrinks as the
table gets better, because four strong seats finish closer together than four
weak ones. So explained variance can fall while the head's actual error falls
too, and the run log alone cannot tell that apart from a head going bad.

This splits the two. The collector already records the head's prediction on
every position it acted at, so no second forward is needed: the numbers here are
exactly the on-policy predictions the update would have seen, against exactly
the targets `catan.ppo` would have built.

Reported by stage of the game as well as pooled, because the pooled figure is a
mixture. At the opening nothing has happened yet and the outcome is board and
dice; near the end it is nearly determined. A head that explains the late game
and nothing early is not a broken head, it is a correct one, and the pooled
number will still look mediocre.

    python -m benchmarks.value_head --checkpoint runs/ppo-overnight/latest.pt \\
        --games 256
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time

import numpy as np
import torch

from benchmarks.throughput import environment
from catan.netbot import load
from catan.policy import NetworkPolicy
from catan.ppo import rotate
from catan.rewards import reward
from catan.selfplay import Collector, Episode
from catan.board.board import random_base_board


def explained(predicted: np.ndarray, actual: np.ndarray) -> float:
    variance = float(actual.var())
    if variance < 1e-8:
        return 0.0
    return float(1 - (actual - predicted).var() / variance)


def rows(episodes: list[Episode]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Every position's own-seat prediction, its target, and how far in it was."""
    predicted, actual, progress = [], [], []
    for episode in episodes:
        payoff = reward(episode.outcome)
        length = max(1, episode.outcome.actions - 1)
        for seat, trajectory in enumerate(episode.trajectories):
            target = rotate(payoff, seat)[0]
            for transition in trajectory:
                if not transition.value:
                    continue
                predicted.append(transition.value[0])
                actual.append(target)
                progress.append(transition.step / length)
    return (
        np.asarray(predicted, dtype=np.float64),
        np.asarray(actual, dtype=np.float64),
        np.asarray(progress, dtype=np.float64),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--games", type=int, default=256)
    parser.add_argument("--lanes", type=int, default=64)
    parser.add_argument("--players", type=int, default=4)
    parser.add_argument("--action-cap", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bins", type=int, default=5)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    board = random_base_board(random.Random(args.seed))
    loaded = load(args.checkpoint, board.topology)
    # Sampled, not greedy: the head was trained on the distribution the
    # behaviour policy visits, and scoring it on the argmax policy's positions
    # would be measuring it somewhere it never saw.
    generator = torch.Generator().manual_seed(args.seed)
    policy = NetworkPolicy(
        loaded.policy.net,
        loaded.space,
        loaded.policy.layout,
        greedy=False,
        generator=generator,
    )

    started = time.perf_counter()
    collector = Collector(
        policy,
        lanes=args.lanes,
        players=args.players,
        seed=args.seed + 1,
        action_cap=args.action_cap,
        max_offers=loaded.max_offers,
        deal=args.games,
    )
    episodes: list[Episode] = collector.drain()
    elapsed = time.perf_counter() - started

    predicted, actual, progress = rows(episodes)
    edges = np.linspace(0.0, 1.0, args.bins + 1)
    stages = []
    for low, high in zip(edges[:-1], edges[1:]):
        rows_in = (progress >= low) & (progress < high if high < 1.0 else progress <= 1.0)
        if not rows_in.any():
            continue
        stages.append(
            {
                "from": round(float(low), 2),
                "to": round(float(high), 2),
                "positions": int(rows_in.sum()),
                "target_variance": round(float(actual[rows_in].var()), 5),
                "residual_variance": round(
                    float((actual[rows_in] - predicted[rows_in]).var()), 5
                ),
                "explained_variance": round(explained(predicted[rows_in], actual[rows_in]), 4),
            }
        )

    payload = {
        "environment": environment(),
        "checkpoint": args.checkpoint,
        "iteration": loaded.iteration,
        "games": len(episodes),
        "positions": int(predicted.size),
        "seconds": round(elapsed, 1),
        "mean_turns": round(
            sum(e.outcome.turns for e in episodes) / len(episodes), 1
        ),
        "mean_actions": round(
            sum(e.outcome.actions for e in episodes) / len(episodes), 1
        ),
        "target_variance": round(float(actual.var()), 5),
        "residual_variance": round(float((actual - predicted).var()), 5),
        "mean_squared_error": round(float(((actual - predicted) ** 2).mean()), 5),
        "explained_variance": round(explained(predicted, actual), 4),
        "stages": stages,
    }

    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    print(f"{payload['games']} games, {payload['positions']} positions, {payload['seconds']}s")
    print(
        f"  target var {payload['target_variance']:.4f}"
        f"  residual var {payload['residual_variance']:.4f}"
        f"  EV {payload['explained_variance']:.3f}"
    )
    print("  by stage of the game:")
    for stage in stages:
        print(
            f"    {stage['from']:.1f}-{stage['to']:.1f}  {stage['positions']:>7} pos"
            f"  target var {stage['target_variance']:.4f}"
            f"  residual {stage['residual_variance']:.4f}"
            f"  EV {stage['explained_variance']:+.3f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
