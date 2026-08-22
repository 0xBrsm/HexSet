"""How far apart the value head puts two positions one action apart.

`benchmarks.value_head --behaviour` ruled out the explanation the search
failure was first filed under. The head is not off-policy-blind: explained
variance is +0.474 on its own positions, +0.453 on `search2`'s and +0.485 on
`greedy`'s, and flat across stages in all three. Yet the same head in a tree
loses 20:1 to the policy it shares a trunk with.

What is left is resolution. A search never asks "how is this seat doing" — it
asks which of two siblings, one action apart, is better. Those differ by a
fraction of a point. If the head's error is larger than the gap it is being
asked to resolve, then PUCT ranks children on noise, and it will do that
confidently, and no aggregate accuracy figure will ever show it.

This measures the two quantities against each other in one run, so the
comparison rests on no cross-run assumption:

    spread — the standard deviation of the head's own-seat value across every
    legal child of a probed position, which is the signal a search has to work
    with at that node.

    error — the RMS of (terminal target - prediction) at those very same
    positions, computed exactly as `benchmarks.value_head` does.

**Spread below error is the finding.** The ratio says by what factor, and
therefore how much more accurate a head would have to be before a tree could
use it — which is the number an expert-iteration run needs and does not have.

Positions are probed at random rather than exhaustively: stepping the engine
once per legal child costs ~300 µs each, so a probe of a 40-option position is
about 12 ms and probing every decision would cost more than the answer is
worth. Roll positions are skipped, because the spread across dice outcomes is
chance and not something the head is being asked to rank.

    python -m benchmarks.sibling --checkpoint runs/ppo-overnight/latest.pt \\
        --games 64 --probe 0.05
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass, replace

import numpy as np

from benchmarks.throughput import environment
from catan.actions import ActionType, apply, legal_actions, within_offer_budget
from catan.board.board import random_base_board
from catan.game import imagine, is_over, to_move
from catan.mcts import Leaf
from catan.rewards import reward, relative_points
from catan.selfplay import Collector, Episode
from catan.victory import victory_points

# Torch is imported inside `main` rather than here, which keeps `Probing` and
# `rows` — everything with arithmetic worth getting wrong — importable and
# testable on a machine with no torch. The evaluator is duck-typed anyway.


@dataclass(frozen=True)
class Spread:
    """One position's children, as the search would see them ranked."""

    seat: int
    options: int
    spread: float
    span: float
    best_gap: float


class Probing:
    """The policy, playing as usual, stopping now and then to score siblings.

    Wrapping the policy rather than replacing it keeps the probed positions on
    the distribution the head was trained on, which is the distribution most
    favourable to it — a spread that is too small *here* is too small
    everywhere.

    The measurement rides to the transition on `Choice.aux`, the pocket
    `catan.selfplay` already carries for `catan.expert`, so the terminal target
    each probe should be judged against is recoverable after the fact without a
    second bookkeeping path.
    """

    def __init__(self, policy, evaluator, *, max_offers, rate, rng) -> None:
        self.policy = policy
        self.evaluator = evaluator
        self.max_offers = max_offers
        self.rate = rate
        self.rng = rng
        self.probed = 0
        self.skipped = 0

    def act(self, requests):
        choices = self.policy.act(requests)
        for row, request in enumerate(requests):
            if self.rng.random() >= self.rate:
                continue
            spread = self._probe(request.game)
            if spread is None:
                self.skipped += 1
            else:
                self.probed += 1
                choices[row] = replace(choices[row], aux=spread)
        return choices

    def _options(self, game):
        if is_over(game):
            return ()
        return tuple(within_offer_budget(game, legal_actions(game), self.max_offers))

    def _probe(self, game) -> Spread | None:
        options = self._options(game)
        if len(options) < 2 or any(a.type is ActionType.ROLL for a in options):
            return None
        seat = to_move(game)

        # Every child encoded from the choosing seat's frame, so the only thing
        # varying across the row is the position. The tree encodes from each
        # child's own mover instead and rotates back, which adds frame changes
        # to the comparison; this is the version most favourable to the head.
        values, leaves, slots = [], [], []
        for action in options:
            child = imagine(game, self.rng)
            apply(child, action)
            if is_over(child):
                # A finished child has a known value on the same scale, which is
                # what the tree would back up; scoring it with the head instead
                # would put a guess where an answer is.
                players = child.state.num_players
                points = tuple(victory_points(child.state, s) for s in range(players))
                values.append(relative_points(points)[seat])
            else:
                slots.append(len(values))
                values.append(0.0)
                leaves.append(Leaf(child, seat, self._options(child)))

        for slot, (_, value) in zip(slots, self.evaluator.evaluate(leaves)):
            values[slot] = value[seat]

        row = np.asarray(values, dtype=np.float64)
        ordered = np.sort(row)[::-1]
        return Spread(
            seat=seat,
            options=len(options),
            spread=float(row.std()),
            span=float(ordered[0] - ordered[-1]),
            best_gap=float(ordered[0] - ordered[1]),
        )


def rows(episodes: list[Episode]) -> tuple[np.ndarray, list[Spread]]:
    """The head's error at each probed position, beside that position's spread."""
    from catan.ppo import rotate

    errors, spreads = [], []
    for episode in episodes:
        payoff = reward(episode.outcome)
        for seat, trajectory in enumerate(episode.trajectories):
            target = rotate(payoff, seat)[0]
            for transition in trajectory:
                if not isinstance(transition.aux, Spread) or not transition.value:
                    continue
                errors.append(target - transition.value[0])
                spreads.append(transition.aux)
    return np.asarray(errors, dtype=np.float64), spreads


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--games", type=int, default=64)
    parser.add_argument("--lanes", type=int, default=16)
    parser.add_argument("--players", type=int, default=4)
    parser.add_argument("--action-cap", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--probe",
        type=float,
        default=0.05,
        help="fraction of decisions to score every child of",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    import torch

    from catan.netbot import LeafEvaluator, load
    from catan.policy import NetworkPolicy

    board = random_base_board(random.Random(args.seed))
    loaded = load(args.checkpoint, board.topology)
    generator = torch.Generator().manual_seed(args.seed)
    policy = NetworkPolicy(
        loaded.policy.net,
        loaded.space,
        loaded.policy.layout,
        greedy=False,
        generator=generator,
    )
    probing = Probing(
        policy,
        LeafEvaluator(
            policy=policy,
            space=loaded.space,
        ),
        max_offers=loaded.max_offers,
        rate=args.probe,
        rng=random.Random(args.seed + 2),
    )

    started = time.perf_counter()
    collector = Collector(
        probing,
        lanes=args.lanes,
        players=args.players,
        seed=args.seed + 1,
        action_cap=args.action_cap,
        max_offers=loaded.max_offers,
        deal=args.games,
        board=board,
    )
    episodes: list[Episode] = collector.drain()
    elapsed = time.perf_counter() - started

    errors, spreads = rows(episodes)
    if not spreads:
        print("no position was probed; raise --probe or --games", file=sys.stderr)
        return 1

    spread = np.asarray([s.spread for s in spreads])
    span = np.asarray([s.span for s in spreads])
    gap = np.asarray([s.best_gap for s in spreads])
    options = np.asarray([s.options for s in spreads])
    error = float(np.sqrt((errors**2).mean()))

    payload = {
        "environment": environment(),
        "checkpoint": args.checkpoint,
        "iteration": loaded.iteration,
        "games": len(episodes),
        "probes": len(spreads),
        "skipped": probing.skipped,
        "seconds": round(elapsed, 1),
        "mean_options": round(float(options.mean()), 1),
        "rms_error": round(error, 4),
        "mean_spread": round(float(spread.mean()), 4),
        "median_spread": round(float(np.median(spread)), 4),
        "mean_span": round(float(span.mean()), 4),
        "mean_best_gap": round(float(gap.mean()), 4),
        "error_over_spread": round(error / max(float(spread.mean()), 1e-9), 2),
        "error_over_best_gap": round(error / max(float(gap.mean()), 1e-9), 2),
        "spread_quantiles": {
            str(q): round(float(np.quantile(spread, q)), 4)
            for q in (0.1, 0.25, 0.5, 0.75, 0.9)
        },
    }

    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    print(
        f"{payload['games']} games, {payload['probes']} probes "
        f"({payload['skipped']} skipped), {payload['mean_options']} options each, "
        f"{payload['seconds']}s"
    )
    print(f"  head RMS error at those positions   {payload['rms_error']:.4f}")
    print(f"  spread across siblings, mean        {payload['mean_spread']:.4f}")
    print(f"  spread across siblings, median      {payload['median_spread']:.4f}")
    print(f"  best minus second best, mean        {payload['mean_best_gap']:.4f}")
    print(
        f"  error is {payload['error_over_spread']}x the spread "
        f"and {payload['error_over_best_gap']}x the gap it has to call"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
