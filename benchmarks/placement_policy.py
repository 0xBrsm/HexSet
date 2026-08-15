"""Does a trained policy already know how to open?

`Collector._ask` makes no distinction by phase, so a setup settlement is offered
to the training policy exactly like any other action: the network has always
chosen its own openings.  Whether it has *learned* anything about them is a
separate question, and this answers it by walking a checkpoint through the eight
setup picks and comparing each choice against the fitted prior in
`catan.placement` and against the legal field it was choosing from.

Three numbers matter, and they separate the possibilities cleanly.  If the
policy's picks carry about as many pips as the average legal vertex, it never
learned to open at all and the prior is free win rate.  If they match the prior,
placement is solved and a learned model has nothing to add.  If they diverge from
the prior while still beating the field, the prior is missing something the
corpus could not see — which is the only outcome that argues for more ML here.

CPU only and deliberately small: this loads a checkpoint and ranks vertices, so
it is seconds of work, not a training run.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from typing import Any

from catan.actions import ActionType, apply, legal_actions
from catan.board.board import Board, pips, random_base_board
from catan.game import Phase, start, to_move
from catan.placement import rank, scarce_resources, score


def vertex_pips(board: Board, vertex: int) -> int:
    return sum(pips(board.tokens[h]) for h in board.topology.vertex_hexes[vertex])


def walk_setup(checkpoint: str, board: Board, seed: int) -> list[dict[str, Any]]:
    """Play the setup phase with the checkpoint and score every pick it makes."""
    # Imported here rather than at module scope because it pulls in torch, which
    # cannot be installed on the development phone at all.
    from catan.netbot import network_bot

    game = start(board, 4, rng=random.Random(seed))
    bot = network_bot(checkpoint, board)
    scarce = scarce_resources(board)
    picks: list[dict[str, Any]] = []

    while game.phase in (Phase.SETUP_SETTLEMENT, Phase.SETUP_ROAD):
        if game.phase is Phase.SETUP_ROAD:
            apply(game, bot.choose(game))
            continue

        seat = to_move(game)
        options = [
            action.a
            for action in legal_actions(game)
            if action.type is ActionType.SETUP_SETTLEMENT
        ]
        held = [v for v, owner in enumerate(game.state.vertex_owner) if owner == seat]
        ordered = rank(game.state, seat, options)
        by_vertex = {vertex: value for value, vertex in ordered}

        chosen = bot.choose(game)
        vertex = chosen.a

        # Where the policy's pick sits in the prior's ordering, 0.0 being the
        # prior's own first choice and 1.0 the worst legal vertex.  Reported
        # rather than the raw rank because the legal field shrinks by roughly a
        # third across the eight picks.
        position = [v for _, v in ordered].index(vertex)
        picks.append(
            {
                "step": len(picks),
                "seat": seat,
                "options": len(options),
                "policy_vertex": vertex,
                "prior_vertex": ordered[0][1],
                "agreed": vertex == ordered[0][1],
                "percentile": position / max(1, len(ordered) - 1),
                "policy_pips": vertex_pips(board, vertex),
                "prior_pips": vertex_pips(board, ordered[0][1]),
                "field_pips": statistics.fmean(vertex_pips(board, v) for v in options),
                "policy_score": by_vertex[vertex],
                "prior_score": ordered[0][0],
                "field_score": statistics.fmean(value for value, _ in ordered),
                # The whole opening the pick completes, so the second settlement
                # is credited with what it adds rather than judged alone.
                "opening_score": score(board, [*held, vertex], scarce),
            }
        )
        apply(game, chosen)

    return picks


def summarise(picks: list[dict[str, Any]]) -> dict[str, Any]:
    def mean(field: str) -> float:
        return round(statistics.fmean(p[field] for p in picks), 4)

    field_pips, policy_pips, prior_pips = mean("field_pips"), mean("policy_pips"), mean("prior_pips")
    headroom = prior_pips - field_pips
    return {
        "picks": len(picks),
        "agreement": round(statistics.fmean(1.0 if p["agreed"] else 0.0 for p in picks), 4),
        "percentile": mean("percentile"),
        "pips": {"field": field_pips, "policy": policy_pips, "prior": prior_pips},
        # What fraction of the gap between choosing at random and choosing by
        # the prior the policy actually captured.  Zero means it opens at
        # random; one means it opens like the prior.
        "recovered": round((policy_pips - field_pips) / headroom, 4) if headroom else None,
        "by_step": [
            {
                "step": step,
                "n": len(same),
                "agreement": round(statistics.fmean(1.0 if p["agreed"] else 0.0 for p in same), 4),
                "policy_pips": round(statistics.fmean(p["policy_pips"] for p in same), 3),
                "prior_pips": round(statistics.fmean(p["prior_pips"] for p in same), 3),
                "field_pips": round(statistics.fmean(p["field_pips"] for p in same), 3),
            }
            for step in range(8)
            if (same := [p for p in picks if p["step"] == step])
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    report: dict[str, Any] = {"games": args.games, "seed": args.seed, "checkpoints": {}}
    for checkpoint in args.checkpoint:
        picks: list[dict[str, Any]] = []
        for index in range(args.games):
            board = random_base_board(random.Random(f"{args.seed}:{index}:board"))
            picks.extend(walk_setup(checkpoint, board, seed=args.seed * 100003 + index))
        report["checkpoints"][checkpoint] = summarise(picks)

    text = json.dumps(report, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
