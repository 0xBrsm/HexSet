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
move needs every member's support. A file may weight its members with
`committee_weights`, one positive number per member, the file itself first,
joined by `+` (`2+1+1`): the mean is then weighted, so `2+1+1` is the nested
vote of the file against the even product of the other two. The committee's
own prior, which a search or a robber tip reads, is that product renormalised
over the options, the members' arithmetic mean where every option has a zero
product; its values are the members' values averaged seat by seat, unweighted.
Where every member lists a decision's whole options (`option_rows`,
`has_options`), the committee chooses whole decisions as a single such file
does: a decision of several moves (a setup turn, Road Building, a discard) is
voted over its whole options by the same weighted product of the members'
probabilities, its first move played and the rest kept as a plan
(`hexset.decisions.Plans`), and a search reads that listing to search the
decision whole. The members must list the same options in the same order. A
position inside a decision with no plan is voted a move at a time.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from hexset import decisions as D
from hexset.actions import Action
from hexset.game import Game

__all__ = ["Committee", "members_of", "weights_of"]


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


def weights_of(meta: dict[str, str], members: int) -> tuple[float, ...] | None:
    """The member weights a file's `committee_weights` key gives, the file
    itself first, for a committee of `members` files in all; `None` for a file
    without the key (every member alike)."""
    value = meta.get("committee_weights", "").strip()
    if not value:
        return None
    try:
        weights = tuple(float(part) for part in value.split("+"))
    except ValueError:
        raise ValueError(f"`committee_weights` is positive numbers joined by '+': {value!r}") from None
    if len(weights) != members:
        raise ValueError(
            f"`committee_weights` {value!r} gives {len(weights)} weights for a committee of {members} files"
        )
    if not all(np.isfinite(w) and w > 0 for w in weights):
        raise ValueError(f"`committee_weights` is positive numbers joined by '+': {value!r}")
    return weights


class Committee:
    """`members` voting as one `hexset.clients.policy.Policy` (the module
    docstring). The first member's action space is the committee's.
    `weights`, one positive number per member, weights the mean log prior;
    `None` weights every member alike."""

    def __init__(self, members: Sequence[object], weights: Sequence[float] | None = None) -> None:
        if len(members) < 2:
            raise ValueError("a committee has at least two members")
        self.members = tuple(members)
        if weights is None:
            weights = (1.0,) * len(self.members)
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != (len(self.members),) or not (np.isfinite(w).all() and (w > 0).all()):
            raise ValueError(f"a committee needs one positive weight per member, not {tuple(weights)!r}")
        #: `None` where every member weighs alike, which keeps the plain mean.
        self.weights = None if (w == w[0]).all() else w
        #: Whether every member lists whole options, so the committee does (`option_rows`).
        self.has_options = all(
            callable(getattr(m, "option_rows", None)) and bool(getattr(m, "has_options", False))
            for m in self.members
        )
        self._plans = D.Plans() if self.has_options else None
        self.space = self.members[0].space
        self.has_prior = all(getattr(m, "has_prior", True) for m in self.members)

    @property
    def plans(self) -> "D.Plans | None":
        """The rest of each decision the committee chose whole, played a move
        at a time; `None` where it lists no whole options."""
        return self._plans

    def _scored(self, rows):
        """Per member, its `(prior, value)` for every row: one batched call each."""
        return [m.score_rows(rows) for m in self.members]

    def _vote(self, priors: np.ndarray) -> np.ndarray:
        """The weighted mean log prior of each option over the members (`-inf`
        where any member gives it none)."""
        with np.errstate(divide="ignore"):
            logs = np.log(priors)
        if self.weights is None:
            return logs.mean(axis=0)
        return (self.weights[:, None] * logs).sum(axis=0) / self.weights.sum()

    def act_rows(self, rows: Sequence[tuple[Game, int, tuple[Action, ...]]]) -> list[Action]:
        """One move per row. Listing whole options: the plan's next move, or
        the first of the whole decision the vote chooses (each row that opens
        one is listed first, in one call per member); otherwise, and inside a
        decision with no plan, the move-at-a-time vote."""
        if not self.has_options:
            return self._atomic(rows)
        plans = self._plans
        waiting = [i for i, (game, seat, options) in enumerate(rows)
                   if len(options) > 1 and not plans.planned(game, seat, options)]
        opening = [i for i in waiting if not D.inside(rows[i][0])]
        finishing = [i for i in waiting if D.inside(rows[i][0])]
        listed = dict(zip(opening, self.option_rows([rows[i][:2] for i in opening]))) if opening else {}
        atomic = dict(zip(finishing, self._atomic([rows[i] for i in finishing]))) if finishing else {}
        out = []
        for i, (game, seat, options) in enumerate(rows):
            # A row whose plan was dropped part-way (`Plans`'s limit) asks on its own.
            decide = (lambda g, s, i=i: self._top(*listed.pop(i))) if i in listed \
                else (lambda g, s: self._top(*self.option_rows([(g, s)])[0]))
            finish = (lambda g, s, o, i=i: atomic.pop(i)) if i in atomic \
                else (lambda g, s, o: self._atomic([(g, s, o)])[0])
            out.append(plans.act(game, seat, options, decide, finish))
        return out

    @staticmethod
    def _top(listed, probs) -> "D.Option":
        """The most probable whole option, the earlier listed on a tie (as a
        decision graph chooses)."""
        return listed[int(np.argmax(probs))]

    def option_rows(self, rows: Sequence[tuple[Game, int]]) -> list[tuple[tuple["D.Option", ...], np.ndarray]]:
        """Every whole option of each row's decision with the committee's
        probability: the members' weighted product renormalised, their
        arithmetic mean where every option has a zero product. Raises where
        the committee lists no whole options or its members list different
        options."""
        if not self.has_options:
            raise ValueError("a member of this committee lists no whole options")
        if not rows:
            return []
        listings = [m.option_rows(rows) for m in self.members]
        out = []
        for i in range(len(rows)):
            listed = tuple(listings[0][i][0])
            keys = [D.key(option) for option in listed]
            for other in listings[1:]:
                if [D.key(option) for option in other[i][0]] != keys:
                    raise ValueError("the committee's members list different whole options for one decision")
            probs = np.array([np.asarray(listing[i][1], dtype=np.float64) for listing in listings])
            p = np.exp(self._vote(probs))
            if not p.sum() > 0:
                p = probs.mean(axis=0)
            total = p.sum()
            out.append((listed, p / total if total > 0 else np.full(len(listed), 1.0 / len(listed))))
        return out

    def _atomic(self, rows: Sequence[tuple[Game, int, tuple[Action, ...]]]) -> list[Action]:
        """The move-at-a-time vote: each row's move with the highest weighted
        mean log prior."""
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
