"""Several policies as one seat: a product-of-experts committee.

A file declares its committee in its metadata: `committee`, the other member
files beside it joined by `+` (`b.onnx+c.onnx`). The file itself is the first
member and carries the seat: its player count, game types, trade gate, search
and closer are the seat's. Every member must be a file this package loads on
its own (`hexset.clients.onnxbot.load`), of any contract, and declare no
committee of its own; a member is offered at no table unless it is stamped
with a name itself.

The vote is a product. Every legal move is scored by each member's prior
(`score_rows`) and the move with the highest mean log prior is played, so a
move needs every member's support. The committee's own prior, which a search
or a robber tip reads, is that product renormalised over the options, the
members' arithmetic mean where every option has a zero product; its values are
the members' values averaged seat by seat.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from hexset.actions import Action
from hexset.game import Game

__all__ = ["Committee", "members_of"]


def members_of(meta: dict[str, str]) -> tuple[str, ...]:
    """The member files a file's `committee` key names, beside it; none for a
    file without the key."""
    value = meta.get("committee", "").strip()
    if not value:
        return ()
    names = tuple(part.strip() for part in value.split("+"))
    if any(not name or "/" in name or "\\" in name for name in names):
        raise ValueError(f"`committee` names member files beside the file, joined by '+': {value!r}")
    return names


class Committee:
    """`members` voting as one `hexset.clients.policy.Policy` (the module
    docstring). The first member's action space is the committee's."""

    def __init__(self, members: Sequence[object]) -> None:
        if len(members) < 2:
            raise ValueError("a committee has at least two members")
        self.members = tuple(members)
        self.space = self.members[0].space
        self.has_prior = all(getattr(m, "has_prior", True) for m in self.members)

    def _scored(self, rows):
        """Per member, its `(prior, value)` for every row: one batched call each."""
        return [m.score_rows(rows) for m in self.members]

    @staticmethod
    def _vote(priors: np.ndarray) -> np.ndarray:
        """The mean log prior of each option over the members (`-inf` where any
        member gives it none)."""
        with np.errstate(divide="ignore"):
            return np.log(priors).mean(axis=0)

    def act_rows(self, rows: Sequence[tuple[Game, int, tuple[Action, ...]]]) -> list[Action]:
        if not rows:
            return []
        scored = self._scored(rows)
        out = []
        for i, (_, _, options) in enumerate(rows):
            priors = np.array([s[i][0] for s in scored], dtype=np.float64)
            vote = self._vote(priors)
            if not np.isfinite(vote).any():
                vote = priors.mean(axis=0)
            out.append(options[int(np.argmax(vote))])
        return out

    def value_rows(self, rows: Sequence[tuple[Game, int]]) -> list[tuple[float, ...]]:
        if not rows:
            return []
        reads = [m.value_rows(rows) for m in self.members]
        return [tuple(float(v) for v in np.mean([r[i] for r in reads], axis=0)) for i in range(len(rows))]

    def score_rows(self, rows: Sequence[tuple[Game, int, tuple[Action, ...]]]):
        if not rows:
            return []
        scored = self._scored(rows)
        out = []
        for i, (_, _, options) in enumerate(rows):
            priors = np.array([s[i][0] for s in scored], dtype=np.float64)
            prior = np.exp(self._vote(priors))
            if not prior.sum() > 0:
                prior = priors.mean(axis=0)
            total = prior.sum()
            prior = prior / total if total > 0 else np.full(len(options), 1.0 / len(options))
            value = tuple(float(v) for v in np.mean([s[i][1] for s in scored], axis=0))
            out.append((prior, value))
        return out
