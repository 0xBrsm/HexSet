# SPDX-License-Identifier: GPL-3.0-only
"""Driver seats that play from batched policies, and the labels they keep.

`PolicySeat` answers atomic requests (`hexset.gym.driver.Request` with
`options`) for any number of policy ids from a fixed set of read policies
(`hexset.clients.policy.Policy`). Each tick it calls every read policy's
`score_rows` once, over every row it was asked with more than one option, so a
table of lanes costs one forward per policy a tick, not one per position. A
row with one legal option is played without a forward and reads nothing.

What a policy id plays is its `Play`: the product of some of the read
policies (a committee; one member is that policy alone), or a policy of its
own outside the read set, scored on that id's rows only. The product is
`hexset.clients.committee.Committee`'s: at temperature 0 the option with the
highest mean log prior over the members (their arithmetic mean where every
option has a zero product); above it, an option drawn from the renormalised
product raised to `1 / temperature`. A draw reads only the request's own key
(`Request.draw`), so a game is the same function of `(seed, index)` however
its lanes are batched, sharded or ordered.

With `record=True` each answer to a multi-option row carries a `Label`: every
read policy's prior over the row's options and its value for every seat,
board-seat order, as float32. `Labels` collects them per game from the
driver's `on_decision` hook until the game's `keep` takes them.

`batch=False` asks every policy one row at a time instead. It plays the same
games wherever the policies' forwards do not depend on the batch they are
run in, and is the reference the batched path is checked against.

`scorer`, when given, reads every policy at once: `scorer(rows)` returns one
`score_rows` answer per policy, in order, so policies served by another
process cost one round trip a tick for all of them.

On native lanes a request's `game` is a read-only view of the lane's position
(`hexset.native.playout.NativeCopy`), which a policy reads as it reads a
`Game`. The seat plays there where every policy it reads states its contract
as `native_contract` (`native_atomic`), and plays the same games and records
the same labels as on Python lanes.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from ..actions import Action
from .driver import Request

__all__ = ["Choice", "Label", "Labels", "Play", "PolicySeat", "draw_index", "product", "tempered"]

_UNIT = 2.0 ** -53


@dataclass(frozen=True)
class Play:
    """How one policy id plays. `members` indexes the seat's read policies and
    plays their product (one index plays that policy alone); `policy`, when
    given, plays that policy instead and is scored on this id's rows only.
    `temperature` 0 plays the top option, above 0 draws one."""

    members: tuple[int, ...] = ()
    policy: object | None = None
    temperature: float = 0.0

    def __post_init__(self) -> None:
        if self.policy is None and not self.members:
            raise ValueError("a play names its members or a policy of its own")
        if self.policy is not None and self.members:
            raise ValueError("a play is its members' product or one policy of its own, not both")
        if not self.temperature >= 0:
            raise ValueError(f"temperature must be at least 0, got {self.temperature!r}")


@dataclass(frozen=True)
class Label:
    """One multi-option decision's reads: the seat, the option played (its
    index in the request's options), every read policy's prior over the
    options `(policies, options)` and its values `(policies, seats)`."""

    seat: int
    chosen: int
    priors: np.ndarray
    values: np.ndarray


@dataclass(frozen=True)
class Choice:
    """A `PolicySeat` answer: the action, and the row's `Label` when the seat
    records. An atomic answer, so `decision` is None."""

    action: Action
    label: Label | None = None
    decision: None = None


def product(priors: np.ndarray) -> np.ndarray:
    """The members' product of experts over the options, renormalised:
    `exp(mean log prior)`, the arithmetic mean where every option has a zero
    product, uniform where that is zero too. `priors` is `(members, options)`."""
    with np.errstate(divide="ignore"):
        p = np.exp(np.log(priors).mean(0))
    if not p.sum() > 0:
        p = priors.mean(0)
    total = p.sum()
    return p / total if total > 0 else np.full(priors.shape[1], 1.0 / priors.shape[1])


def tempered(p: np.ndarray, temperature: float) -> np.ndarray:
    """`p ** (1 / temperature)` renormalised, in log space so a small
    temperature cannot underflow; a zero stays zero."""
    with np.errstate(divide="ignore"):
        logs = np.log(p) / temperature
    logs -= logs.max()
    q = np.exp(logs)
    return q / q.sum()


def draw_index(probabilities: np.ndarray, key: int) -> int:
    """The option a 64-bit draw key picks from `probabilities`: the first whose
    running total passes the key's uniform (its top 53 bits) times the total.
    Pure Python floats over the float64 values, so the same on every platform."""
    point = (int(key) >> 11) * _UNIT * float(probabilities.sum())
    running = 0.0
    last = 0
    for i, p in enumerate(probabilities.tolist()):
        if p > 0:
            last = i
        running += p
        if point < running:
            return i
    return last


def _top(priors: np.ndarray) -> int:
    """The product's top option, as `Committee.act_rows` picks it."""
    if priors.shape[0] == 1:
        return int(np.argmax(priors[0]))
    with np.errstate(divide="ignore"):
        vote = np.log(priors).mean(0)
    if not np.isfinite(vote).any():
        vote = priors.mean(0)
    return int(np.argmax(vote))


class PolicySeat:
    """A driver seat (`hexset.gym.driver.Seat`) for the policy ids in `plays`,
    reading `policies` at every multi-option row (the module docstring). The
    driver asks one seat object once a tick for all the ids it answers."""

    def __init__(self, policies: Sequence[object], plays: Mapping[int, Play], *, record: bool = False,
                 batch: bool = True, scorer=None) -> None:
        self.policies = tuple(policies)
        self.scorer = scorer
        self.plays = dict(plays)
        if not self.plays:
            raise ValueError("a policy seat plays at least one policy id")
        for pid, play in self.plays.items():
            if any(not 0 <= m < len(self.policies) for m in play.members):
                raise ValueError(f"policy {pid}'s members {play.members} are not all read policies")
        self.record, self.batch = record, batch
        self.rows = self.forwards = 0

    def policy_ids(self) -> tuple[int, ...]:
        return tuple(self.plays)

    @property
    def native_atomic(self) -> frozenset[str] | None:
        """The contracts this seat's policies read from a position, where every
        one of them reads a native lane's (`native_contract`); None where any
        reads only a `Game`. The driver plays the seat on native lanes only
        where it is not None (`hexset.gym.driver.NATIVE_READS`)."""
        if self.scorer is not None:
            return None
        policies = list(self.policies) + [play.policy for play in self.plays.values() if play.policy is not None]
        reads = [getattr(policy, "native_contract", None) for policy in policies]
        return None if any(read is None for read in reads) else frozenset(reads)

    def _score(self, policy, rows) -> list:
        self.forwards += 1 if self.batch else len(rows)
        if self.batch:
            return list(policy.score_rows(rows))
        return [policy.score_rows([row])[0] for row in rows]

    def choose_many(self, requests: Sequence[Request]) -> list[Choice]:
        out: list[Choice | None] = [None] * len(requests)
        asked: list[int] = []
        for i, request in enumerate(requests):
            if request.decision is not None or request.game is None:
                raise ValueError("a policy seat answers atomic requests, which carry their position")
            if request.policy not in self.plays:
                raise ValueError(f"policy {request.policy} is not played by this seat")
            if len(request.options) == 1:
                out[i] = Choice(request.options[0])
            else:
                asked.append(i)
        if not asked:
            return out  # type: ignore[return-value]
        self.rows += len(asked)
        rows = [(requests[i].game, requests[i].seat, requests[i].options) for i in asked]
        if self.scorer is None:
            scored = [self._score(policy, rows) for policy in self.policies]
        elif self.batch:
            self.forwards += len(self.policies)
            scored = list(self.scorer(rows))
        else:
            self.forwards += len(self.policies) * len(rows)
            alone = [self.scorer([row]) for row in rows]
            scored = [[read[k][0] for read in alone] for k in range(len(self.policies))]
        own: dict[int, object] = {}
        groups: dict[int, list[int]] = {}
        for k, i in enumerate(asked):
            play = self.plays[requests[i].policy]
            if play.policy is not None:
                groups.setdefault(id(play.policy), []).append(k)
                own[id(play.policy)] = play.policy
        extra: dict[int, np.ndarray] = {}
        for key, ks in groups.items():
            for k, (prior, _) in zip(ks, self._score(own[key], [rows[k] for k in ks])):
                extra[k] = np.asarray(prior, dtype=np.float64)
        for k, i in enumerate(asked):
            request = requests[i]
            play = self.plays[request.policy]
            priors = np.array([s[k][0] for s in scored], dtype=np.float64).reshape(len(scored), -1)
            if play.policy is not None:
                p = extra[k]
                chosen = (int(np.argmax(p)) if not play.temperature
                          else draw_index(tempered(p, play.temperature), request.draw))
            else:
                members = priors[list(play.members)]
                chosen = (_top(members) if not play.temperature
                          else draw_index(tempered(product(members), play.temperature), request.draw))
            label = None
            if self.record:
                values = np.array([s[k][1] for s in scored], dtype=np.float64)
                label = Label(request.seat, chosen, priors.astype(np.float32), values.astype(np.float32))
            out[i] = Choice(request.options[chosen], label)
        return out  # type: ignore[return-value]


class Labels:
    """Each game's `Label`s, in the order its decisions were answered, from a
    driver's `on_decision` hook (call this object as the hook) until `take`
    hands them over, normally in the game's `keep`."""

    def __init__(self) -> None:
        self._games: dict[int, list[Label]] = {}

    def __call__(self, request: Request, answer: object) -> None:
        label = getattr(answer, "label", None)
        if label is not None:
            self._games.setdefault(request.index, []).append(label)

    def take(self, index: int) -> list[Label]:
        return self._games.pop(index, [])

    def __len__(self) -> int:
        return len(self._games)
