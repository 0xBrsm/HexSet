# SPDX-License-Identifier: GPL-3.0-only
"""Bot interfaces and the uniform random policy.

A bot only needs ``choose(game)``. Trading is optional: a gate implements
``gains_many(view, received, counterparties)`` and must declare a
nonnegative ``trade_floor`` when it returns positive gains. ``hexset.trading``
also accepts boolean ``accepts``/``accepts_many`` gates. ``TradesBy`` seats
one bot's moves over another's trading; ``Handoff`` seats one bot's moves up
to a round and another's after it; ``Closes`` plays a closer's move wherever
it has one and the bot's own everywhere else. ``PlaysAgainst`` is the optional hook
a bot implements to play as one side against named seats
(``hexset.bots.coalition``).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Collection, Protocol, Sequence

from ..actions import Action, options_for
from ..game import Game
from ..trading import (
    RESPONSE_ACCEPT, RESPONSE_COUNTER, RESPONSE_PASS, Bundle, Response, clears_floor, valued_many,
)
from ..view import View

__all__ = [
    "Bot",
    "TradeGate",
    "TradesBy",
    "Handoff",
    "handed_off",
    "Closes",
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


class Vetoed:
    """``proposer``'s trading, thinned to the exchanges ``veto`` also gains
    from: its offers (``candidates``), its answers (``respond``,
    ``respond_any``: an acceptance or counter the veto refuses is a pass) and
    its picks (an answer the veto refuses reaches ``proposer`` as a pass).
    The veto reads each exchange with its own ``gains_many`` and clears it
    above its own ``trade_floor`` (`hexset.trading.clears_floor`), so a
    network seat vetoing its trader holds every trade to the floor its
    checkpoint declares. ``gains_many`` is the proposer's, ``-1.0`` where the
    veto refuses; every other name is the proposer's.

    The veto never adds or changes an exchange: it only removes ones the
    proposer would make. Nor does it choose one, unless ``picks``: then of
    the answers to this seat's own offer that the proposer would take, each
    on its own, and the veto clears, the pick is the one the veto gains most
    from (the proposer's own pick among equals). Seat it with
    ``TradesBy(mover, Vetoed(trader, mover))`` (`hexset.arena.traded` with
    ``veto=True``) for a mover that vetoes its own trader.
    """

    __slots__ = ("proposer", "veto", "picks")

    def __init__(self, proposer: object, veto: object, picks: bool = False) -> None:
        object.__setattr__(self, "proposer", proposer)
        object.__setattr__(self, "veto", veto)
        object.__setattr__(self, "picks", picks)

    def __getattr__(self, name: str):
        return getattr(object.__getattribute__(self, "proposer"), name)

    def __setattr__(self, name: str, value) -> None:
        setattr(self.proposer, name, value)

    def __delattr__(self, name: str) -> None:
        delattr(self.proposer, name)

    def __reduce__(self):
        return (Vetoed, (self.proposer, self.veto, self.picks))

    def seat_at(self, game: Game) -> None:
        seat_at(self.proposer, game)
        seat_at(self.veto, game)

    def _gains(self, view: View, received: Sequence[Bundle], counterparties: Sequence[int]) -> list[float]:
        """The veto's gain from each of ``received`` (signed towards this
        seat) with the matching counterparty, in one batch."""
        if not received:
            return []
        return valued_many(self.veto, view, list(received), list(counterparties))

    def _clears(self, view: View, received: Sequence[Bundle], counterparties: Sequence[int]) -> list[bool]:
        """Whether the veto gains from each of ``received`` above its floor."""
        return [clears_floor(g, self.veto, view) for g in self._gains(view, received, counterparties)]

    def candidates(self, view: View, counterparties: Sequence[int], *, turn=None, already_offered=()):
        menu = list(self.proposer.candidates(view, counterparties, turn=turn, already_offered=already_offered))
        ok = self._clears(view, [b for _, b in menu], [them for them, _ in menu])
        return [c for c, keep in zip(menu, ok) if keep]

    def _answer(self, view: View, actor: int, response: Response) -> Response:
        """``response`` to ``actor``, or a pass where the veto refuses what it
        would give this seat (bundles are signed towards ``actor``)."""
        if response.kind not in (RESPONSE_ACCEPT, RESPONSE_COUNTER) or response.bundle is None:
            return response
        if self._clears(view, [tuple(-n for n in response.bundle)], [actor])[0]:
            return response
        return Response(response.seat, RESPONSE_PASS, None)

    def respond(self, view: View, offer) -> Response:
        return self._answer(view, offer.actor, self.proposer.respond(view, offer))

    def respond_any(self, view: View, offer) -> Response:
        return self._answer(view, offer.actor, self.proposer.respond_any(view, offer))

    def pick(self, view: View, responses: Sequence[Response]) -> int | None:
        live = [i for i, r in enumerate(responses)
                if r.kind in (RESPONSE_ACCEPT, RESPONSE_COUNTER) and r.bundle is not None]
        gains = dict(zip(live, self._gains(view, [responses[i].bundle for i in live],
                                           [responses[i].seat for i in live])))
        ok = {i: clears_floor(g, self.veto, view) for i, g in gains.items()}
        passed = [Response(r.seat, RESPONSE_PASS, None) for r in responses]
        thinned = [r if ok.get(i, True) else passed[i] for i, r in enumerate(responses)]
        if not self.picks:
            return self.proposer.pick(view, thinned)
        # Each answer the veto clears, put to the proposer alone: those it
        # would take are the field, ranked by the veto's own gain.
        field = [i for i in live if ok[i]
                 and self.proposer.pick(view, [thinned[i]]) == 0]
        if not field:
            return None
        best = max(gains[i] for i in field)
        top = [i for i in field if gains[i] == best]
        if len(top) == 1:
            return top[0]
        return self.proposer.pick(view, [r if i in top else passed[i] for i, r in enumerate(responses)])

    def gains_many(self, view: View, received: Sequence[Bundle], counterparties: Sequence[int]) -> list[float]:
        own = valued_many(self.proposer, view, list(received), list(counterparties))
        ok = self._clears(view, received, counterparties)
        return [g if keep else -1.0 for g, keep in zip(own, ok)]


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


class Closes:
    """``mover``'s moves, except where ``closer`` has one of its own.

    At each decision ``closer.closing(game)`` is asked first: an action it
    names is played, whatever the mover would choose; ``None`` leaves the
    decision to ``mover``. A closer's move that is not legal at the position
    is not played either: the mover chooses. Every other name -- each trade
    hook, the trade params, the offer budget, a network gate's values -- is
    the mover's, read, written and deleted through, as `TradesBy` does to its
    trader, so the seat trades and is vetoed exactly as the mover alone
    would. Seat it over another bot's trading with
    ``TradesBy(Closes(mover, closer), trader)``.

    A closer is meant to be narrow and deterministic: a rule that sees
    something the mover may miss -- a finish this turn its hand holds -- and
    stays out of everything else. A mover that plays a decision a move at a
    time (`hexset.decisions.Plans`) checks each planned move against the
    legal moves, so a decision the closer steps into part-way is finished
    from the position as it then stands.
    """

    __slots__ = ("mover", "closer")

    def __init__(self, mover: Bot, closer: object) -> None:
        object.__setattr__(self, "mover", mover)
        object.__setattr__(self, "closer", closer)

    def choose(self, game: Game) -> Action:
        move = self.closer.closing(game)
        if move is not None and move in options_for(game):
            return move
        return self.mover.choose(game)

    def __getattr__(self, name: str):
        return getattr(object.__getattribute__(self, "mover"), name)

    def __setattr__(self, name: str, value) -> None:
        setattr(self.mover, name, value)

    def __delattr__(self, name: str) -> None:
        delattr(self.mover, name)

    def __reduce__(self):
        return (Closes, (self.mover, self.closer))


def seat_at(bot: object, game: Game) -> None:
    """`seat_at(game)` on every part of `bot` that has one: a network bot's
    gate, a search's, both halves of a `TradesBy` seat and both bots of a
    `Handoff`, and the mover and closer of a `Closes` seat, so a gate asked
    about a trade before its first move prices it on `game`. Other bots have
    nothing to seat."""
    if isinstance(bot, Closes):
        seat_at(bot.mover, game)
        seat_at(bot.closer, game)
        return
    if isinstance(bot, TradesBy):
        seat_at(bot.mover, game)
        seat_at(bot.trader, game)
        return
    if isinstance(bot, Vetoed):
        bot.seat_at(game)
        return
    if isinstance(bot, Handoff):
        seat_at(bot.first, game)
        seat_at(bot.second, game)
        return
    method = getattr(bot, "seat_at", None)
    if method is not None:
        method(game)


def watch(bot: object, game: Game, actor: int, action: Action, viewer: int) -> None:
    """`seen(game, actor, action, viewer)` on every part of `bot` that has
    one, walked as `seat_at` walks it: `actor`'s `action` at `game`, before it
    is applied, as seen from `viewer`'s seat. What a bot that models the
    other seats from their moves (`hexset.clients.conditioned.Opponents`)
    learns from; other bots see nothing."""
    if isinstance(bot, Closes):
        watch(bot.mover, game, actor, action, viewer)
        return
    if isinstance(bot, TradesBy):
        watch(bot.mover, game, actor, action, viewer)
        return
    if isinstance(bot, Handoff):
        watch(bot.first, game, actor, action, viewer)
        watch(bot.second, game, actor, action, viewer)
        return
    if bot is None:
        return
    method = getattr(type(bot), "seen", None)
    if method is not None:
        method(bot, game, actor, action, viewer)


def watches(bot: object) -> bool:
    """Whether any part of `bot` has a `seen` (`watch`): a table asks no one else."""
    if isinstance(bot, (Closes, TradesBy)):
        return watches(bot.mover)
    if isinstance(bot, Handoff):
        return watches(bot.first) or watches(bot.second)
    return getattr(type(bot), "seen", None) is not None


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
    first, both bots of a `Handoff`, and the mover of a `Closes` seat. A part
    without one, or with it set to `None`, is left as it is."""
    if isinstance(bot, Closes):
        play_against(bot.mover, seats)
        return
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
