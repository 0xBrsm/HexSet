# SPDX-License-Identifier: GPL-3.0-only
"""The engine's one chance source.

Registered `agents/reference/game-records.md`: every random draw the engine
makes -- the shuffled development deck, a dice roll, a robber/knight steal,
a forced discard -- goes through `hexset.game.Game.chance`, one method per
kind of event (`deck_order`, `roll`, `steal`, `discard`). `Live` is today's
behaviour, unchanged: it draws from a `random.Random` exactly as the engine
always has, so a game built with no `chance` argument is byte-identical, for
every seed, to a game built before this module existed (proved by replaying
the trade-lab bank's first game, `tests/test_record_engine.py::
test_default_chance_matches_the_seeded_stream`). `Scripted` replays a
recorded stream instead of drawing anything, so a `Record` can be replayed
without a seed, or ported in from a source (a journal, a colonist.io game)
that never had one. `Recording` wraps either and logs every outcome, which
is how `hexset.record.record_game` builds a `Record`'s `chance` stream.
`Forced` stands in for one steal outcome only, for the counterfactual
children `hexset.bench.aivat` and `hexset.bots.heximax.search` build to
score what each possible stolen card would have led to -- the direct
successor to the `_Forced` stand-in-rng each of those modules used to define
for itself.
"""

from __future__ import annotations

import random
from typing import Sequence

# Matches `hexset.game.DICE` (two six-sided dice). Duplicated rather than
# imported: `hexset.game` imports this module, so the reverse would be a
# cycle, and the die the engine plays with has never had a reason to vary.
DICE_SIDES = 6

Event = tuple[str, int]


class ChanceError(Exception):
    """Base for everything a `Chance` source raises."""


class ChanceExhausted(ChanceError):
    """`Scripted` was asked for an event past the end of its recording."""

    def __init__(self, index: int, kind: str) -> None:
        self.index = index
        self.kind = kind
        super().__init__(
            f"scripted chance stream exhausted at event {index}: "
            f"the engine asked for {kind!r}"
        )


class ChanceMismatch(ChanceError):
    """`Scripted`'s next recorded event is not the kind the engine asked for."""

    def __init__(self, index: int, expected: str, got: str) -> None:
        self.index = index
        self.expected = expected
        self.got = got
        super().__init__(
            f"scripted chance stream diverges at event {index}: the engine "
            f"asked for {expected!r}, the recording holds {got!r}"
        )


class Chance:
    """One method per chance event the engine resolves. Subclassed, never
    instantiated directly -- `Game.chance` is always one of the four below."""

    def deck_order(self, deck: list[int]) -> list[int]:
        """The development deck, in draw order (bottom of the deck first,
        since `devcards.buy` pops the end) -- called once, at `game.start`."""
        raise NotImplementedError

    def roll(self) -> int:
        """Two dice, summed."""
        raise NotImplementedError

    def steal(self, hand: Sequence[int]) -> int | None:
        """Which resource a steal takes from `hand`, or `None` if `hand` is
        empty -- an empty hand is decided by the hand alone, so it consumes
        no event: `Scripted` must not expect one recorded for it either."""
        raise NotImplementedError

    def discard(self, hand: Sequence[int], n: int) -> list[int]:
        """`n` resources to discard from `hand`, one pick at a time (so a
        multi-card discard is `n` events, not one) -- for a seat that does
        not choose its own discards (`hexset.robber.random_discard`)."""
        raise NotImplementedError


class Live(Chance):
    """The default source: draws from a `random.Random`, exactly as the
    engine always has before this module existed."""

    def __init__(self, rng: random.Random) -> None:
        self.rng = rng

    def deck_order(self, deck: list[int]) -> list[int]:
        self.rng.shuffle(deck)
        return deck

    def roll(self) -> int:
        return self.rng.randint(1, DICE_SIDES) + self.rng.randint(1, DICE_SIDES)

    def steal(self, hand: Sequence[int]) -> int | None:
        total = sum(hand)
        if total == 0:
            return None
        pick = self.rng.randrange(total)
        for resource, count in enumerate(hand):
            if pick < count:
                return resource
            pick -= count
        raise AssertionError("unreachable")

    def discard(self, hand: Sequence[int], n: int) -> list[int]:
        taken = [0] * len(hand)
        picks: list[int] = []
        for _ in range(n):
            pool = [r for r, count in enumerate(hand) if count > taken[r]]
            resource = self.rng.choice(pool)
            taken[resource] += 1
            picks.append(resource)
        return picks


class Scripted(Chance):
    """Replays a recorded event stream (`Record.chance`) in order, never
    drawing anything. Raises `ChanceMismatch` when the engine's next call is
    not the kind of event the stream holds next, and `ChanceExhausted` when
    the engine calls past the end of it -- both name the event's index, so a
    divergent replay points at exactly where."""

    def __init__(self, events: Sequence[Event]) -> None:
        self._events = list(events)
        self.index = 0

    def _next(self, kind: str) -> int:
        if self.index >= len(self._events):
            raise ChanceExhausted(self.index, kind)
        got_kind, value = self._events[self.index]
        if got_kind != kind:
            raise ChanceMismatch(self.index, kind, got_kind)
        self.index += 1
        return value

    def deck_order(self, deck: list[int]) -> list[int]:
        order = [self._next("deck") for _ in range(len(deck))]
        deck[:] = order
        return deck

    def roll(self) -> int:
        return self._next("roll")

    def steal(self, hand: Sequence[int]) -> int | None:
        if sum(hand) == 0:
            return None
        return self._next("steal")

    def discard(self, hand: Sequence[int], n: int) -> list[int]:
        return [self._next("discard") for _ in range(n)]


class Recording(Chance):
    """Wraps any `Chance` source and logs every outcome it returns, in
    order, as `("kind", value)` pairs in `self.events` -- `record_game`
    wraps `Live` with this to build a `Record`'s `chance` stream."""

    def __init__(self, inner: Chance) -> None:
        self.inner = inner
        self.events: list[Event] = []

    def deck_order(self, deck: list[int]) -> list[int]:
        order = self.inner.deck_order(deck)
        self.events.extend(("deck", card) for card in order)
        return order

    def roll(self) -> int:
        value = self.inner.roll()
        self.events.append(("roll", value))
        return value

    def steal(self, hand: Sequence[int]) -> int | None:
        resource = self.inner.steal(hand)
        if resource is not None:
            self.events.append(("steal", resource))
        return resource

    def discard(self, hand: Sequence[int], n: int) -> list[int]:
        picks = self.inner.discard(hand, n)
        self.events.extend(("discard", resource) for resource in picks)
        return picks


class Forced(Chance):
    """A chance source that returns one fixed resource from `steal` and
    raises on everything else. The direct successor to the `_Forced`
    stand-in-rng `hexset.bench.aivat` and `hexset.bots.heximax.search` each
    used to define locally: both build a counterfactual child for one
    possible steal outcome, apply the one action that consumes it, and put
    the child's real chance source back -- nothing else on that path ever
    drew, so a source that only knows how to answer `steal` is exactly as
    much as either caller has ever needed."""

    def __init__(self, resource: int) -> None:
        self.resource = resource

    def steal(self, hand: Sequence[int]) -> int | None:
        return self.resource
