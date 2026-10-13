# SPDX-License-Identifier: GPL-3.0-only
"""The engine's one chance source: every random draw goes through
`hexset.game.Game.chance`, one method per kind of event. `Live` draws from a
`random.Random`, `Balanced` draws its rolls from a dice deck,
`Scripted` replays a recorded stream (so a `Record` replays without a seed),
`Recording` logs what either returns, `Hosted` takes what a remote table
showed.

A dealt game (`for_rules`) gives its dice and its steals a generator each, so
a seed's rolls are the same whatever is played in between: two players
compared on one seed meet the same dice turn for turn."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

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
    "DICE_RECENT_ROLLS",
    "DICE_RECENT_DISCOUNT",
    "SEVEN_STREAK_STEP",
    "SEVEN_ADJUST_MAX",
    "SEVEN_ODDS",
    "BALANCED_SEVEN_ODDS",
    "dice_deck",
    "DiceDeck",
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

    def roll_by(self, seat: int | None) -> int:
        """Two dice, summed, rolled by `seat` (`None`: unknown). The engine
        rolls through this; a source whose odds depend on the roller
        (`Balanced`) overrides it, every other rolls as `roll` does."""
        return self.roll()

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
    """Draws from a `random.Random`. The default source.

    Unsplit, the deck, the dice and the steals all come off `rng` in the
    order the game asks for them, so one different move -- a steal taken or
    not -- shifts every roll after it. `split` gives the dice and the steals
    a generator each, drawn off `rng` here, before the deck: the n-th roll
    and the n-th steal are then fixed by the seed alone, whatever was played
    in between. Every dealt game splits (`for_rules`); a search copy
    (`hexset.game.imagine`) does not, so a search draws as it always has.
    """

    def __init__(self, rng: random.Random, *, split: bool = False) -> None:
        self.rng = rng
        self.split = split
        # Drawn off `rng`, not hashed from its state, so two sources built
        # one after another on one generator still roll different dice.
        self.dice = random.Random(rng.getrandbits(64)) if split else rng
        self.steals = random.Random(rng.getrandbits(64)) if split else rng

    def deck_order(self, deck: list[int]) -> list[int]:
        self.rng.shuffle(deck)
        return deck

    def roll(self) -> int:
        return self.dice.randint(1, DICE_SIDES) + self.dice.randint(1, DICE_SIDES)

    def steal(self, hand: Sequence[int]) -> int | None:
        total = sum(hand)
        if total == 0:
            return None
        # Split, a steal takes exactly one draw whatever the hand's size
        # (`randrange` takes more as it rejects), so the k-th steal of a game
        # always reads the k-th draw.
        pick = int(self.steals.random() * total) if self.split else self.rng.randrange(total)
        for resource, count in enumerate(hand):
            if pick < count:
                return resource
            pick -= count
        raise AssertionError("unreachable")


# The dice deck: all 36 two-dice combinations, reshuffled whole once this
# many cards are left. A draw picks a sum by weight -- the cards of that sum
# left in the deck, each time the sum came up in the last `DICE_RECENT_ROLLS`
# rolls taking `DICE_RECENT_DISCOUNT` off (never below none) -- then one of
# its cards. A seven's weight is scaled once more for the roller (`Balanced`):
# up while it holds less than its share of the sevens so far, down while it
# holds more, and by `SEVEN_STREAK_STEP` for each seven of the current run --
# down when the run is the roller's, up when it is another seat's -- the
# scale kept within 0 and `SEVEN_ADJUST_MAX`.
DICE_DECK_RESHUFFLE_AT = 12
DICE_RECENT_ROLLS = 5
DICE_RECENT_DISCOUNT = 0.34
SEVEN_STREAK_STEP = 0.4
SEVEN_ADJUST_MAX = 2.0
#: The weight of a card repeating the previous sum before 1.4.0 (unused; kept for importers).
DICE_REPEAT_WEIGHT = 0.7

# How often each source rolls a seven, read through `rules.Rules.seven_odds`.
#
# `SEVEN_ODDS` is exact: 6 of 36 combinations.  `BALANCED_SEVEN_ODDS` has no
# closed form -- the deck is never played out, so the draw is neither with
# nor without replacement, and recent sums and the rollers' sevens move each
# draw's weights -- so it is measured off this module's own `Balanced`, two
# seats rolling in turn: 0.15584 +- 0.00036 over 4,000,000 rolls at seed
# 12345.  It sits below the independent rate because a seven, the commonest
# sum, is the one most often discounted for having come up lately.
SEVEN_ODDS = 6 / 36
BALANCED_SEVEN_ODDS = 0.15584


def dice_deck() -> list[tuple[int, int]]:
    """All 36 ordered two-dice combinations."""
    return [(a, b) for a in range(1, DICE_SIDES + 1) for b in range(1, DICE_SIDES + 1)]


def _seven_scale(sevens: Mapping[int, int], run: tuple[int | None, int], seat: int | None) -> float:
    """What a seven's weight is scaled by for `seat`'s roll, from the sevens
    each roller seen so far has rolled and the current run of them."""
    if seat is None:
        return 1.0
    sevens = dict(sevens)
    sevens.setdefault(seat, 0)          # as its roll would count it
    total = sum(sevens.values())
    balance = 1.0
    if total >= len(sevens):
        share, ideal = sevens[seat] / total, 1 / len(sevens)
        balance = 1 + (ideal - share) / ideal
    owner, length = run
    streak = SEVEN_STREAK_STEP * length * (-1 if owner == seat else 1)
    return min(SEVEN_ADJUST_MAX, max(0.0, balance + streak))


def _sum_weights(left: Iterable[tuple[int, int]], cards: int, recent: Sequence[int], scale: float) -> dict[int, float]:
    """Each sum's draw weight: its cards `left` in the deck of `cards`, less
    `DICE_RECENT_DISCOUNT` for each time it is among the `recent` sums, a
    seven's scaled by `scale`. The one formula `Balanced` draws by and
    `DiceDeck` reads."""
    w = {}
    for total, count in left:
        w[total] = count / cards * max(0.0, 1 - DICE_RECENT_DISCOUNT * recent.count(total))
    if 7 in w:
        w[7] *= scale
    return w


_FULL_DECK = tuple(sum(1 for a, b in dice_deck() if a + b == total) for total in range(2, 2 * DICE_SIDES + 1))


@dataclass(frozen=True)
class DiceDeck:
    """The dice deck's public state, by sum: what `Balanced`'s next roll is
    drawn from, read off the rolls before it and who rolled them, which
    every seat sees. A game under `balanced_dice` keeps one on its state
    (`GameState.dice`) and moves it on at every roll, whatever the source of
    the roll, so a search can draw from the deck (`Balanced.from_deck`) and
    read the next roll's odds (`odds`). Immutable: `rolled` returns the next.

    `left` is the cards left of each sum 2..12; the deck is reshuffled as
    soon as a roll leaves `DICE_DECK_RESHUFFLE_AT` or fewer, so `left` is
    always what the next roll draws from. `sevens` is each roller seen so
    far and its sevens, by seat; `run` the seat on a run of sevens and its
    length."""

    left: tuple[int, ...] = _FULL_DECK
    cards: int = DICE_SIDES * DICE_SIDES
    recent: tuple[int, ...] = ()
    sevens: tuple[tuple[int, int], ...] = ()
    run: tuple[int | None, int] = (None, 0)

    def weights(self, seat: int | None = None) -> dict[int, float]:
        """Each sum's draw weight for `seat`'s next roll, as `Balanced.weights`."""
        return _sum_weights(((total, n) for total, n in enumerate(self.left, 2)), self.cards, self.recent,
                            _seven_scale(dict(self.sevens), self.run, seat))

    def odds(self, seat: int | None = None) -> tuple[tuple[int, float], ...]:
        """`(sum, chance)` of `seat`'s next roll, over the sums left in the
        deck: the weights normalised, or the cards left where every weight
        is zero -- as `Balanced.roll_by` draws."""
        w = self.weights(seat)
        sums = [total for total, n in enumerate(self.left, 2) if n]
        mass = sum(w[t] for t in sums)
        if mass > 0:
            return tuple((t, w[t] / mass) for t in sums)
        cards = sum(self.left[t - 2] for t in sums)
        return tuple((t, self.left[t - 2] / cards) for t in sums)

    def rolled(self, seat: int | None, total: int) -> "DiceDeck":
        """The deck after `seat` rolls `total`."""
        left = list(self.left)
        left[total - 2] -= 1
        cards = self.cards - 1
        recent = (self.recent + (total,))[-DICE_RECENT_ROLLS:]
        sevens, run = dict(self.sevens), self.run
        if seat is not None:
            sevens.setdefault(seat, 0)
            if total == 7:
                sevens[seat] += 1
                run = (seat, run[1] + 1) if run[0] == seat else (seat, 1)
        if cards <= DICE_DECK_RESHUFFLE_AT:
            left, cards = list(_FULL_DECK), DICE_SIDES * DICE_SIDES
        return DiceDeck(tuple(left), cards, recent, tuple(sorted(sevens.items())), run)


class Balanced(Live):
    """`Live`, except rolls come from a dice deck (`DICE_DECK_RESHUFFLE_AT`
    and the constants after it): the 36 combinations drawn without
    replacement, reshuffled whole before they run low, so a deck is never
    played out; sums that came up lately drawn less often; and a seven drawn
    more often by a seat that has rolled fewer than its share of them and
    less often by one on a run of them. The roller is what `roll_by` names;
    `roll` (no roller) leaves the sevens unsteered.

    Everything else -- the development deck, steals, `split` -- is `Live`'s.
    The deck's draws depend only on the rolls before them and on who rolls,
    which the turn order fixes, so split it is as fixed by the seed as
    `Live`'s dice.
    """

    def __init__(self, rng: random.Random, *, split: bool = False) -> None:
        super().__init__(rng, split=split)
        self.left: dict[int, list[tuple[int, int]]] = {}
        self.cards = 0
        self.recent: list[int] = []
        self.sevens: dict[int, int] = {}       # sevens by roller, every roller seen so far
        self.run: tuple[int | None, int] = (None, 0)   # the seat on a run of sevens, and its length
        self._reshuffle()

    def _reshuffle(self) -> None:
        self.left = {}
        for a, b in dice_deck():
            self.left.setdefault(a + b, []).append((a, b))
        self.cards = DICE_SIDES * DICE_SIDES

    @classmethod
    def from_deck(cls, rng: random.Random, deck: DiceDeck, *, split: bool = False) -> "Balanced":
        """A source that rolls on from `deck`: what a search copy of a
        balanced-dice game draws from (`hexset.game.imagine`). Which of a
        sum's combinations are left is not tracked and does not matter --
        a draw reads only how many there are."""
        source = cls(rng, split=split)
        source.left = {total: [(a, b) for a, b in dice_deck() if a + b == total][:n]
                       for total, n in enumerate(deck.left, 2)}
        source.cards = deck.cards
        source.recent = list(deck.recent)
        source.sevens = dict(deck.sevens)
        source.run = deck.run
        return source

    def seven_scale(self, seat: int | None) -> float:
        """What a seven's weight is scaled by for `seat`'s roll."""
        return _seven_scale(self.sevens, self.run, seat)

    def weights(self, seat: int | None = None) -> dict[int, float]:
        """Each sum's draw weight for `seat`'s next roll."""
        return _sum_weights(((total, len(cards)) for total, cards in self.left.items()), self.cards, self.recent,
                            self.seven_scale(seat))

    def roll(self) -> int:
        return self.roll_by(None)

    def roll_by(self, seat: int | None) -> int:
        if seat is not None:
            self.sevens.setdefault(seat, 0)
        if self.cards <= DICE_DECK_RESHUFFLE_AT:
            self._reshuffle()
        w = self.weights(seat)
        sums = [t for t in sorted(w) if self.left[t]]
        total = self.dice.choices(sums, weights=[w[t] for t in sums])[0] if sum(w[t] for t in sums) > 0 \
            else self.dice.choices(sums, weights=[len(self.left[t]) for t in sums])[0]
        cards = self.left[total]
        cards.pop(self.dice.randrange(len(cards)))
        self.cards -= 1
        self.recent.append(total)
        if len(self.recent) > DICE_RECENT_ROLLS:
            self.recent.pop(0)
        if total == 7 and seat is not None:
            self.sevens[seat] += 1
            owner, length = self.run
            self.run = (seat, length + 1) if owner == seat else (seat, 1)
        return total


def for_rules(rules, rng: random.Random, *, split: bool = True) -> Live:
    """The source a ruleset asks for: `Balanced` under `balanced_dice`,
    `Live` otherwise. The one place that choice is made, so a caller that
    wraps the source (`Recording`) cannot pick a different one from the game
    it wraps. `rules` is a `hexset.rules.Rules`, not imported: that module
    imports this one.

    Split by default (`Live`): this is how every game is dealt. `split=False`
    is for rebuilding a game that was dealt before the split, from its seed
    (`Record.split_streams`, a server journal's `split_streams`)."""
    return Balanced(rng, split=split) if rules.balanced_dice else Live(rng, split=split)


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

    def roll_by(self, seat: int | None) -> int:
        value = self.inner.roll_by(seat)
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
