"""A rollout's other seats played by a model of how each of them plays.

Paired rollouts (`hexset.rollout`) play every seat of a playout with the
searching checkpoint's own policy. A caller may pass `opponents` to
`hexset.clients.netbot.rollouts_for`: a model of the other seats, read on
their moves as the table makes them (`seen`), whose `playouts(own)` is the
policy a rollout plays with. The searching seat still plays its own seat in
every playout, reads every value and scores every root.

- `SeatRouted`: that policy: the searching seat's rows to its own policy,
  every other seat's moves to the opponents' policy (`others`: `act_rows`,
  and a `conditions` mapping from seat to whatever it reads per seat). Its
  `context` (the seat and every seat's condition) travels with the worlds a
  `RolloutPool` worker plays.
"""

from __future__ import annotations

from typing import Sequence

from hexset.actions import Action
from hexset.game import Game

__all__ = ["SeatRouted"]


class SeatRouted:
    """A `hexset.clients.policy.Policy` for a playout: `own` for `seat`'s rows
    and for every value and root score, `others` for every other seat's
    moves."""

    def __init__(self, own, others, seat: int | None = None) -> None:
        self.own, self.others, self.seat = own, others, seat

    def act_rows(self, rows: Sequence[tuple[Game, int, tuple[Action, ...]]]) -> list[Action]:
        out: list[Action | None] = [None] * len(rows)
        mine = [i for i, (_, s, _) in enumerate(rows) if s == self.seat]
        theirs = [i for i, (_, s, _) in enumerate(rows) if s != self.seat]
        for idx, policy in ((mine, self.own), (theirs, self.others)):
            if idx:
                for i, action in zip(idx, policy.act_rows([rows[i] for i in idx])):
                    out[i] = action
        return out  # type: ignore[return-value]

    def value_rows(self, rows):
        return self.own.value_rows(rows)

    def score_rows(self, rows):
        return self.own.score_rows(rows)

    def context(self) -> dict:
        """What a worker's copy needs to play as this one: the seat and every seat's condition."""
        return {"seat": self.seat, "conditions": {s: v.copy() for s, v in self.others.conditions.items()}}

    def set_context(self, context: dict) -> None:
        self.seat = context["seat"]
        self.others.conditions = dict(context["conditions"])

    def __getattr__(self, name: str):
        return getattr(object.__getattribute__(self, "own"), name)
