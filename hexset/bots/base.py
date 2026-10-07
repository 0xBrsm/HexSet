# SPDX-License-Identifier: GPL-3.0-only
"""Bot interfaces and the uniform random policy.

A bot only needs ``choose(game)``. Trading is optional: a gate implements
``gains_many(view, received, counterparties)`` and must declare a
nonnegative ``trade_floor`` when it returns positive gains. ``hexset.trading``
also accepts boolean ``accepts``/``accepts_many`` gates. ``TradesBy`` seats
one bot's moves over another's trading; ``Handoff`` seats one bot's moves up
to a round and another's after it. ``PlaysAgainst`` is the optional hook
a bot implements to play as one side against named seats
(``hexset.bots.coalition``).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Collection, Protocol, Sequence

from ..actions import Action, options_for
from ..game import Game
from ..trading import Bundle
from ..view import View

__all__ = [
    "Bot",
    "TradeGate",
    "TradesBy",
    "Handoff",
    "handed_off",
    "PlaysAgainst",
    "seat_at",
    "play_against",
    "RandomBot",
]


class Bot(Protocol):
    """Choose a legal action for the seat currently entitled to act."""

    def choose(self, game: Game) -> Action: ...


class TradeGate(Protocol):
    """Value exchanges from one seat's information set."""

    trade_floor: float

    def gains_many(
        self, view: View, received: Sequence[Bundle], counterparties: Sequence[int]
    ) -> list[float]: ...


class TradesBy:
    """``mover``'s moves over ``trader``'s trading.

    ``choose`` is the mover's. Every other name -- each trade hook the engine
    and the arena look up (``candidates``, ``offer``, ``respond``,
    ``respond_any``, ``pick``, ``consent_gain``, ``gains_many``,
    ``estimate_many``, ``observe_trade``, ``trade_params``, the offer budget)
    -- is the trader's, read, written and deleted through, so
    ``hexset.trading.retune`` on the seat retunes the trader and reinstalls
    its protocol.
    Neither side knows about the other: any bot can move and any gate can
    trade. The trader is only ever asked about exchanges, so a search bot used
    as one never searches a move.
    """

    __slots__ = ("mover", "trader")

    def __init__(self, mover: Bot, trader: object) -> None:
        object.__setattr__(self, "mover", mover)
        object.__setattr__(self, "trader", trader)

    def choose(self, game: Game) -> Action:
        return self.mover.choose(game)

    def __getattr__(self, name: str):
        return getattr(object.__getattribute__(self, "trader"), name)

    def __setattr__(self, name: str, value) -> None:
        setattr(self.trader, name, value)

    def __delattr__(self, name: str) -> None:
        delattr(self.trader, name)

    def __reduce__(self):
        return (TradesBy, (self.mover, self.trader))


def handed_off(game: Game, at: int) -> bool:
    """Whether `game` is past round `at`: every playing seat has had `at`
    main-phase turns. Read off `Game.turns`, which counts turns the playing
    seats take, so the answer is the same for every seat and for any copy of
    the game (a search's imagined one, a rejoined table's). A seat retired
    mid-game shortens the rounds after it."""
    playing = game.num_players - len(game.locked)
    return game.turns >= at * max(playing, 1)


class Handoff:
    """``first``'s moves through round ``at``, ``second``'s after it.

    A seat's own turns 1..``at`` and every decision it makes in those rounds
    (discards, robber moves on another's seven) are ``first``'s; from round
    ``at + 1`` on they are ``second``'s (`handed_off`). Trading is
    ``first``'s for the whole game: every other name -- each trade hook, the
    trade params, the offer budget -- is read, written and deleted through to
    ``first``, as `TradesBy` does to its trader, so the seat bargains one way
    from deal to finish. Seat it over another bot's trading with
    `TradesBy(Handoff(...), trader)` (``~<trader>`` in a lineup).
    """

    __slots__ = ("first", "second", "at")

    def __init__(self, first: Bot, second: Bot, at: int) -> None:
        if at < 0:
            raise ValueError(f"a handoff round is a count of rounds, not {at}")
        object.__setattr__(self, "first", first)
        object.__setattr__(self, "second", second)
        object.__setattr__(self, "at", at)

    def playing(self, game: Game) -> Bot:
        """The part that moves in `game` now."""
        return self.second if handed_off(game, self.at) else self.first

    def choose(self, game: Game) -> Action:
        return self.playing(game).choose(game)

    def __getattr__(self, name: str):
        return getattr(object.__getattribute__(self, "first"), name)

    def __setattr__(self, name: str, value) -> None:
        setattr(self.first, name, value)

    def __delattr__(self, name: str) -> None:
        delattr(self.first, name)

    def __reduce__(self):
        return (Handoff, (self.first, self.second, self.at))


def seat_at(bot: object, game: Game) -> None:
    """`seat_at(game)` on every part of `bot` that has one: a network bot's
    gate, a search's, both halves of a `TradesBy` seat and both bots of a
    `Handoff`, so a gate asked about a trade before its first move prices it
    on `game`. Other bots have nothing to seat."""
    if isinstance(bot, TradesBy):
        seat_at(bot.mover, game)
        seat_at(bot.trader, game)
        return
    if isinstance(bot, Handoff):
        seat_at(bot.first, game)
        seat_at(bot.second, game)
        return
    method = getattr(bot, "seat_at", None)
    if method is not None:
        method(game)


class PlaysAgainst(Protocol):
    """Optional: a bot that can play as one side against named seats.

    ``play_against(seats)`` names the seats it plays against; every other
    seat is then a partner, whose win counts as its own. Empty, the default,
    every seat plays for itself. How a bot reads that is its own: a search
    ranks positions by the chance none of ``seats`` wins
    (``hexset.mcts.Search.play_against``). A ``Coalition`` seat calls it with
    the table's targets; a bot without it plays there under the coalition's
    table rules alone."""

    def play_against(self, seats: Collection[int]) -> None: ...


def play_against(bot: object, seats: Collection[int]) -> None:
    """`play_against(seats)` on every part of `bot` that implements
    `PlaysAgainst`: the bot, or both halves of a `TradesBy` seat, mover
    first, and both bots of a `Handoff`. A part without one, or with it set
    to `None`, is left as it is."""
    if isinstance(bot, TradesBy):
        play_against(bot.mover, seats)
        play_against(bot.trader, seats)
        return
    if isinstance(bot, Handoff):
        play_against(bot.first, seats)
        play_against(bot.second, seats)
        return
    method = getattr(bot, "play_against", None)
    if method is not None:
        method(seats)


@dataclass
class RandomBot:
    """Sample uniformly from legal actions; decline player trades."""

    rng: random.Random = field(default_factory=random.Random)
    trade_floor: float = 0.0

    def choose(self, game: Game) -> Action:
        return self.rng.choice(options_for(game))
