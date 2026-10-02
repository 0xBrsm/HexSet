# SPDX-License-Identifier: GPL-3.0-only
"""The engine's one chance source: every random draw goes through
`hexset.game.Game.chance`, one method per kind of event. `Live` draws from a
`random.Random`, `Balanced` draws its rolls from a dice deck,
`Scripted` replays a recorded stream (so a `Record` replays without a seed),
`Recording` logs what either returns, `Hosted` takes what a remote table
showed."""

from __future__ import annotations

import random
from typing import Sequence

__all__ = [
    "DICE_SIDES",
    "Event",
    "UNSEEN",
    "ChanceError",
    "ChanceExhausted",
    "ChanceMismatch",
    "Chance",
    "Live",
    "DICE_DECK_RESHUFFLE_AT",
    "DICE_REPEAT_WEIGHT",
    "SEVEN_ODDS",
    "BALANCED_SEVEN_ODDS",
    "dice_deck",
    "Balanced",
    "for_rules",
    "Scripted",
    "Recording",
    "Hosted",
]


# Matches `hexset.game.DICE` (two six-sided dice). Duplicated rather than
# imported: `hexset.game` imports this module, so the reverse would be a cycle.
DICE_SIDES = 6

Event = tuple[str, int]

#: A draw this seat did not see: another seat's development card, a steal
#: between two other seats. Negative, so it can never be a card or resource
#: index. What a `Hosted` source returns for one, and what the engine hands
#: back from a steal it could not name.
UNSEEN = -1


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
    """One method per chance event. Subclassed, never instantiated."""

    def deck_order(self, deck: list[int]) -> list[int]:
        """The deck in draw order, bottom first since `devcards.buy` pops the
        end. Called once, at `game.start`."""
        raise NotImplementedError

    def roll(self) -> int:
        """Two dice, summed."""
        raise NotImplementedError

    def steal(self, hand: Sequence[int]) -> int | None:
        """Which resource a steal takes from `hand`, or `None` if it is empty.
        An empty hand consumes no event, so `Scripted` expects none."""
        raise NotImplementedError

    def draw(self) -> int:
        """The development card a purchase takes from a deck known only by its
        length (an observed state), or `UNSEEN` when the buyer is a seat whose
        cards this one cannot see. A referee's deck is concrete and ordered at
        the deal (`deck_order`), so only a `Hosted` source is ever asked."""
        raise NotImplementedError("a concrete deck is drawn from, not asked")

    def surrender(self, hand: Sequence[int], resource: int) -> int:
        """How many of `resource` a seat hands over to a Monopoly: read straight
        off a concrete `hand`, and the host's count for a hidden one. Not a
        random draw, but the same kind of question -- one the engine cannot
        answer from an observed state -- so it comes through the same door."""
        return hand[resource]


class Live(Chance):
    """Draws from a `random.Random`. The default source."""

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


# The dice deck: all 36 two-dice combinations, reshuffled whole once this
# many cards are left, each draw weighting a card that repeats the previous
# *sum* at `DICE_REPEAT_WEIGHT` against 1 for every other card.
DICE_DECK_RESHUFFLE_AT = 12
DICE_REPEAT_WEIGHT = 0.7

# How often each source rolls a seven, read through `rules.Rules.seven_odds`.
#
# `SEVEN_ODDS` is exact: 6 of 36 combinations.  `BALANCED_SEVEN_ODDS` has no
# closed form -- the deck is never played out, so the draw is neither with
# nor without replacement, and `DICE_REPEAT_WEIGHT` makes each draw depend on
# the previous sum -- so it is measured off this module's own `Balanced`:
# 0.16532 +- 0.00019 over 4,000,000 rolls at seed 12345.  It sits below the
# independent rate because a seven, the commonest sum, is the one most often
# discounted for repeating itself.
SEVEN_ODDS = 6 / 36
BALANCED_SEVEN_ODDS = 0.16532


def dice_deck() -> list[tuple[int, int]]:
    """All 36 ordered two-dice combinations."""
    return [(a, b) for a in range(1, DICE_SIDES + 1) for b in range(1, DICE_SIDES + 1)]


class Balanced(Live):
    """`Live`, except rolls come from a dice deck.

    Draw a card, discard it, reshuffle a whole fresh deck once
    `DICE_DECK_RESHUFFLE_AT` remain -- so a deck is never played out, and the
    36 combinations are not a hard quota over any window. Cards repeating the
    previous sum are drawn at `DICE_REPEAT_WEIGHT`.

    Everything else -- the development deck, steals -- is `Live`'s.

    Only the deck is modelled. A table that also damps streaks of sevens, or
    steers each seat's share of the sevens toward even, rolls fewer sevens
    than this source does.
    """

    def __init__(self, rng: random.Random) -> None:
        super().__init__(rng)
        self.deck: list[tuple[int, int]] = []
        self.last: int | None = None

    def roll(self) -> int:
        if len(self.deck) <= DICE_DECK_RESHUFFLE_AT:
            self.deck = dice_deck()
            self.rng.shuffle(self.deck)
        weights = [
            DICE_REPEAT_WEIGHT if a + b == self.last else 1.0 for a, b in self.deck
        ]
        index = self.rng.choices(range(len(self.deck)), weights=weights)[0]
        a, b = self.deck.pop(index)
        self.last = a + b
        return self.last


def for_rules(rules, rng: random.Random) -> Live:
    """The source a ruleset asks for: `Balanced` under `balanced_dice`,
    `Live` otherwise. The one place that choice is made, so a caller that
    wraps the source (`Recording`) cannot pick a different one from the game
    it wraps. `rules` is a `hexset.rules.Rules`, not imported: that module
    imports this one."""
    return Balanced(rng) if rules.balanced_dice else Live(rng)


class Scripted(Chance):
    """Replays a recorded event stream (`Record.chance`) in order, raising
    `ChanceMismatch` on an unexpected kind and `ChanceExhausted` past its
    end -- both naming the event's index."""

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


class Recording(Chance):
    """Wraps any `Chance` source, logging each outcome as a `("kind", value)`
    pair in `self.events`."""

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


class Hosted(Chance):
    """What a table hosted somewhere else shows one seat, fed in as it happens.

    An observed game (`hexset.game.observe`) cannot roll its own dice or pick
    its own steals: the host already has. Whoever mirrors the table queues
    each outcome with `expect` just before applying the action it belongs to,
    and the engine takes them in order through the ordinary methods -- the
    dice (`"roll"`), the card a steal moved or `UNSEEN` when this seat was not
    party to it (`"steal"`), the card a purchase drew or `UNSEEN` for another
    seat's (`"draw"`), the count a hidden seat surrendered to a Monopoly
    (`"surrender"`). Asked for an outcome nobody queued, or for the wrong
    kind, it raises exactly as `Scripted` does; `pending` lists what is still
    queued, so a mirror can tell an outcome it fed that the engine never
    consumed."""

    def __init__(self) -> None:
        self._events: list[Event] = []
        self.index = 0

    def expect(self, kind: str, value: int) -> None:
        """Queue the next outcome of `kind` for the engine to take."""
        if kind not in ("roll", "steal", "draw", "surrender"):
            raise ValueError(f"a host reveals no {kind!r}")
        self._events.append((kind, int(value)))

    @property
    def pending(self) -> list[Event]:
        """Outcomes queued and not yet taken."""
        return self._events[self.index:]

    def drain(self) -> list[Event]:
        """Discard every outcome not yet taken, and return them -- for a
        mirror that has found itself out of step with the table."""
        left = self.pending
        self.index = len(self._events)
        return left

    def _next(self, kind: str) -> int:
        if self.index >= len(self._events):
            raise ChanceExhausted(self.index, kind)
        got_kind, value = self._events[self.index]
        if got_kind != kind:
            raise ChanceMismatch(self.index, kind, got_kind)
        self.index += 1
        return value

    def deck_order(self, deck: list[int]) -> list[int]:
        # The host dealt the deck; an observed game holds it as a length.
        return deck

    def roll(self) -> int:
        return self._next("roll")

    def steal(self, hand: Sequence[int]) -> int | None:
        # The engine asks only once it has checked the hand is not empty.
        return self._next("steal")

    def draw(self) -> int:
        return self._next("draw")

    def surrender(self, hand: Sequence[int], resource: int) -> int:
        from .state import is_hidden  # local: `state` imports `rules`, which imports this

        if not is_hidden(hand):
            return hand[resource]
        return self._next("surrender")
