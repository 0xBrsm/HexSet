# SPDX-License-Identifier: GPL-3.0-only
"""Turning a finished game into a number per seat.

Deliberately not in `hexset.selfplay`: the collector emits an `Outcome` and
scalarises nothing, so what a run rewards is a choice made here and recorded,
not a default buried in the plumbing.

**The reward is relative terminal victory points**, settled 2026-08-11. Two
measurements on this engine decided it. Terminal points track wins closely
(Pearson 0.998) and carry far more per game than one bit, because a losing seat
still says how close it came. But they have to be read *relatively*: reading the
evaluation vector relatively rather than absolutely beat plain max^n 53.6%, and
more bluntly, a value fitted from absolute per-seat features scored 0/2000
despite good held-out log loss — it learned a level where a difference was
needed. Absolute points invite exactly that failure.

Two things this is not. It is not the metric: win rate is what gets reported,
because points are the training signal and a policy that farmed them without
winning would be invisible to its own reward. And it is not a licence to
discount — see `relative_points` on why γ < 1 is a trap here.
"""

from __future__ import annotations

from .selfplay import Outcome

# `relative_points` lives in `hexset.victory`: `hexset.mcts` needs it too, and
# hexset must never import this module (hexnet depends on hexset, not the
# other way around). Re-exported here, not redefined, so callers that already
# read `hexnet.rewards.relative_points` see no change and there is still only
# one definition of the quantity the value head is trained to predict.
from hexset.victory import relative_points

__all__ = ["relative_points", "win_loss", "reward"]


def win_loss(outcome: Outcome) -> tuple[float, ...]:
    """+1 to the winner, 0 to everyone else. The thing points stand in for.

    Kept so the two can be compared on the same run rather than argued about,
    and so the reported metric has a definition in the same place as the
    training signal. An unfinished game gives every seat zero.
    """
    seats = len(outcome.points)
    if outcome.winner is None:
        return (0.0,) * seats
    return tuple(1.0 if seat == outcome.winner else 0.0 for seat in range(seats))


def reward(outcome: Outcome) -> tuple[float, ...]:
    """The reward a run trains on. Relative terminal points.

    A truncated game is scored the same way. The action cap stopping a game is
    a fact about the position reached, and zeroing it would teach a policy that
    stalling escapes a loss — the one lesson this reward must not contain.
    """
    return relative_points(outcome.points)
