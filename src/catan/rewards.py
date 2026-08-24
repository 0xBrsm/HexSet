"""Turning a finished game into a number per seat.

Deliberately not in `catan.selfplay`: the collector emits an `Outcome` and
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
from .victory import WINNING_POINTS


def relative_points(points: tuple[int, ...]) -> tuple[float, ...]:
    """Each seat's terminal points less the mean of the others, over 10.

    Exactly zero-sum: the per-seat values sum to zero for any input, since
    subtracting the mean of the others is an affine transform whose total
    cancels. That is the property worth having — it says in the reward what the
    game already says, that Catan has one winner and a position is only worth
    what it is worth compared to the table. An action that lifts every seat
    equally earns nothing, which is the whole reason for reading points this
    way rather than absolutely.

    Scaled by the 10 points that win a game, so a seat's reward lands in about
    [-1, +1] and a value head does not have to learn the units.

    **Do not discount this.** With a zero-sum reward roughly half of terminal
    values are negative, and γ < 1 makes a negative terminal cheaper the later
    it arrives — which pays a losing policy to stall. Trading in circles is
    precisely that move, and it is why the action cap exists. Horizon control
    belongs in the offer budget (`actions.within_offer_budget`), which is
    measured, not in a discount factor that quietly changes the objective.
    """
    seats = len(points)
    if seats < 2:
        raise ValueError("a relative reward needs at least two seats")
    total = sum(points)
    return tuple(
        (own - (total - own) / (seats - 1)) / WINNING_POINTS for own in points
    )


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
