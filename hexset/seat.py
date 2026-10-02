# SPDX-License-Identifier: GPL-3.0-only
"""One bot seat at a table somebody else hosts.

The engine's own tables -- the arena, the served web table -- hold every
seat's gate and drive the trade round themselves: they ask each gate in turn
and move the cards. A seat at a table hosted elsewhere has only its own gate.
The other players answer on the host's schedule, through the host's wire, and
the host moves the cards. `Seat` is the engine's side of such a seat:

* **the game**, observed from this seat (`game.observe`) and kept current by
  the table's events -- every seat's actions through the ordinary
  `actions.apply` (`play`), every trade the host executes (`traded`), with
  the host supplying what this seat cannot draw for itself;
* **the decisions**, every one asked of the bot exactly as the engine's own
  tables ask it: its move (`choose`, `discard`), whether and what to offer
  (`offer`, through `trading.offer` under the gate's own budget, menu,
  repeat rule and `trade_now`), which answer to take (`pick`), how its round
  went (`close_round`), how to answer another seat's offer (`answer`), how
  to answer a counter to a counter of its own or to its offer (`counter`,
  a `trading.CounterThread` of at most `counter_steps` counters), and what
  the table's trading looked like (`observe_trade`, once a turn).

The client also reports each offer and answer the table shows, and a chat ask
for a card, through `shown`, which puts it on the public ledger.

A client mirroring a foreign table therefore carries decisions and never
makes one: it translates the host's events into `play`/`traded` calls, puts
what this seat decides on the host's wire, and handles the host's own timing.
Which cards this seat offers, how many offers a turn, when in the turn, whom
it deals with -- all of that is the gate's, here, as it is at every other
table.

**An agreement binds** (`docs/trading.md`): an answer this seat gives is its
consent, and the answer it picks to its own offer is executed, never
re-priced. `pick` prices each acceptance by who took the offer.
"""

from __future__ import annotations

import random
from typing import Iterable, Sequence

from .actions import Action, ActionType, apply
from .board.board import Board
from .chance import UNSEEN, ChanceError, Hosted
from .game import (
    NO_TURN_CAP,
    Game,
    Phase,
    event_hand_sizes,
    imagine,
    may_act,
    observe,
    publish_trade_event,
    start,
)
from .rules import STANDARD_GAME, GameType
from .state import HiddenRead
from . import trading
from .trading import (
    RESPONSE_COUNTER,
    RESPONSE_PASS,
    Bundle,
    CounterThread,
    Offer,
    Response,
    Trade,
    execute_agreed,
    params_of,
)


class OutOfStep(ValueError):
    """The table's events and this seat's game disagree: an outcome the host
    revealed that the engine never asked for, or an event it cannot apply.
    The game this seat holds no longer describes the table; the client should
    rebuild it from the host's own statement of the position (`resync`)."""


class Seat:
    """The game one seat observes at a hosted table, and that seat's bot.

    `bot` answers `choose(game)`; `gate` is what prices trades
    (`gains_many`), the bot itself by default when it has one. A seat with no
    gate trades nothing: it offers nothing and passes every offer.
    `counter_steps` is how many counters a thread with the mover may carry
    (`counter`); one, the default, is a counter the actor takes or leaves.
    """

    def __init__(self, game: Game, seat: int, bot: object, gate: object | None = None,
                 *, counter_steps: int = 1) -> None:
        if not isinstance(game.chance, Hosted):
            raise ValueError("a hosted seat's game takes its outcomes from the host: "
                             "build it with game.observe or Seat.sit")
        self.game = game
        self.seat = seat
        self.bot = bot
        if gate is None and hasattr(bot, "gains_many"):
            gate = bot
        self.gate = gate
        self.counter_steps = counter_steps
        self._turn: tuple[int, int] | None = None
        self._reset_turn()

    @classmethod
    def sit(
        cls,
        board: Board,
        num_players: int,
        seat: int,
        bot: object,
        gate: object | None = None,
        *,
        game_type: GameType = STANDARD_GAME,
        first: int = 0,
        locked: Iterable[int] = (),
        turn_cap: int = NO_TURN_CAP,
        counter_steps: int = 1,
    ) -> "Seat":
        """Sit down at the deal: a fresh game of `game_type`, as `seat` sees it.
        No turn cap by default -- the host ends the game."""
        dealt = start(board, num_players, random.Random(0), first=first,
                      game_type=game_type, locked=locked, turn_cap=turn_cap)
        return cls(observe(dealt, seat), seat, bot, gate, counter_steps=counter_steps)

    # -- the table's events ----------------------------------------------

    def play(
        self,
        actor: int,
        action: Action,
        *,
        roll: int | None = None,
        stolen: int | None = None,
        drawn: int | None = None,
        surrendered: Sequence[int] = (),
    ) -> None:
        """Apply `actor`'s `action`, as the table took it, with what the host
        revealed of it: the dice total of a `ROLL`; the card a steal moved,
        or `chance.UNSEEN` when this seat was not party to it; the card a
        purchase drew for this seat (another seat's is unseen, and need not be
        said); what each hidden seat surrendered to a Monopoly, in seat
        order. Raises `OutOfStep` for an outcome the engine did not take."""
        self._sync()
        game = self.game
        host = game.chance
        if roll is not None:
            host.expect("roll", roll)
        if stolen is not None:
            host.expect("steal", stolen)
        if action.type is ActionType.BUY_DEV_CARD:
            if drawn is None and actor == self.seat:
                raise ValueError("this seat's own purchase: which card did it draw?")
            host.expect("draw", UNSEEN if drawn is None else drawn)
        for count in surrendered:
            host.expect("surrender", count)
        was_main = game.phase is Phase.MAIN and game.current_player == actor
        if action.type is ActionType.END_TURN:
            self._turn_over()
        try:
            apply(game, action, seat=actor)
        except (ValueError, IndexError, ChanceError, HiddenRead) as exc:
            host.drain()
            raise OutOfStep(f"{action} by seat {actor} does not apply: {exc}") from exc
        left = host.drain()
        if left:
            raise OutOfStep(f"the table revealed {left} that {action} did not take")
        if was_main and actor == self.seat and action.type is not ActionType.END_TURN:
            self._moves += 1
        self._sync()

    def traded(self, a: int, b: int, received: Bundle) -> Trade:
        """An exchange the host executed between `a` and `b`, `received`
        signed towards `a` -- this seat's own or anybody else's. Nobody is
        asked: it has happened. Applied by name, since a trade is public."""
        self._sync()
        try:
            trade = execute_agreed(self.game, a, b, tuple(received),
                                   ask_actor=False, ask_counterparty=False)
        except ValueError as exc:
            raise OutOfStep(f"the table's trade {a}<->{b} {received} does not apply: "
                            f"{exc}") from exc
        self._trades.append(trade)
        return trade

    def resync(self, game: Game) -> None:
        """Replace the game with one rebuilt from the host's own statement of
        the position -- after a reconnect, or an `OutOfStep`. Its `chance`
        must be `Hosted`. This turn's trade bookkeeping survives if the
        rebuilt game is still in the same turn."""
        if not isinstance(game.chance, Hosted):
            raise ValueError("a hosted seat's game takes its outcomes from the host")
        self.game = game
        self._sync()

    # -- this seat's moves -----------------------------------------------

    def choose(self) -> Action:
        """The bot's move, asked with the game as every table asks it."""
        if not may_act(self.game, self.seat):
            raise ValueError(f"seat {self.seat} has no move in {self.game.phase.name}")
        return self.bot.choose(self.game)

    def discard(self) -> list[int] | None:
        """The whole discard this seat owes, as resource indices, one per card.

        The engine asks for a discard one card at a time; a host that takes
        the selection in one piece gets every card the bot chose, each asked
        with the ones before it already gone -- on a copy, since nothing has
        left the hand until the host takes it. Other seats' discards are
        theirs, and do not stand between this seat and its own: on the copy
        only this seat still owes.

        `None` when the bot has no answer yet (its first is `None`, as a seat
        driven by a person's calls says before they have made them); a bot
        that stops after it has started raises instead, since what it chose
        so far is not a discard."""
        game = self.game
        owed = game.discard_quota[self.seat] if game.phase is Phase.DISCARD else 0
        if not owed:
            raise ValueError(f"seat {self.seat} owes no discard")
        copy = imagine(game, random.Random(0), randomize_deck=False)
        copy.discard_quota = [owed if s == self.seat else 0 for s in range(game.num_players)]
        cards: list[int] = []
        while copy.phase is Phase.DISCARD and copy.discard_quota[self.seat]:
            action = self.bot.choose(copy)
            if action is None and not cards:
                return None
            if action is None or action.type is not ActionType.DISCARD:
                raise ValueError(f"the bot stopped discarding with "
                                 f"{copy.discard_quota[self.seat]} still owed")
            cards.append(action.a)
            apply(copy, action, seat=self.seat)
        return cards

    def restrict_robber(self, hexes: Iterable[int] | None) -> None:
        """The host's rule for the robber move at hand: the hexes it will
        take, or `None` for the rulebook's. The engine honours it for this
        move only (`Game.robber_allowed`)."""
        self.game.robber_allowed = None if hexes is None else frozenset(hexes)

    # -- this seat's trading ---------------------------------------------

    @property
    def open_offer(self) -> Offer | None:
        """This seat's offer still awaiting `close_round`, or `None`."""
        return self._round

    def offer(self) -> Offer | None:
        """This seat's next offer, or `None`.

        Asked at every decision point of this seat's own main phase, before
        its move. The turn's one trade event opens the way it does at the
        engine's own tables: on entering the main phase, or -- for a gate
        with `trade_now` -- at the first decision point it says yes, never
        between a Road Building card's free roads. Once open it offers
        through `trading.offer`: the gate's menu, its pick, its repeat rule,
        against the bundles already put to the table this turn, until its
        own `trade_offer_budget` is spent or it passes -- which closes the
        event for the turn. The engine runs a turn's rounds back to back at
        the one decision point the event opened at, so a move of this seat's
        after that closes it too. `None` while a round of this seat's is
        still open.
        """
        game, gate = self.game, self.gate
        self._sync()
        if (gate is None or game.phase is not Phase.MAIN or game.current_player != self.seat
                or game.free_roads > 0 or self._round is not None or self._closed):
            return None
        budget = params_of(gate).trade_offer_budget
        if not self._opened:
            if budget == 0:
                self._closed = True
                return None
            ask = getattr(gate, "trade_now", None)
            if ask is None and self._moves:
                self._closed = True     # the event belonged to the main phase's entry
                return None
            if ask is not None and not ask(game):
                return None             # the window stays open
            self._opened = True
            self._opened_at = self._moves
            self._hand_sizes = event_hand_sizes(game)
        elif self._moves > self._opened_at:
            return self._close_event()  # a move since: the event's decision point is past
        if 0 <= budget <= self._offers_made:
            return self._close_event()
        offered = trading.offer(game, self._gates(), already_offered=self._already_offered)
        if offered is None:
            return self._close_event()
        self._offers_made += 1
        self._round = offered
        self._responses = []
        return offered

    def pick(self, responses: Sequence[Response]) -> Response | None:
        """Which answer to this seat's open offer to execute, among what the
        other seats said (an acceptance carries the offer's own bundle, a
        counter its own; both signed towards this seat). The gate's `pick`,
        else `default_pick`, which prices an acceptance by its taker. `None`
        executes nothing. The answers are kept for `close_round`."""
        if self._round is None:
            raise ValueError("this seat has no open offer")
        self._responses = list(responses)
        return trading.pick(self.game, self.gate, self.seat, self._responses)

    def close_round(self, trade: Trade | None = None) -> None:
        """This seat's open round is over, with `trade` the exchange the host
        executed on it (already applied through `traded`), or `None`. The
        gate hears how it went (`trade_round_finished`), which is how a plan
        spread over several offers learns whether its fragment cleared."""
        if self._round is None:
            return
        offer, self._round = self._round, None
        finished = getattr(self.gate, "trade_round_finished", None)
        if finished is not None:
            finished(self.game.state(self.seat), offer, list(self._responses), trade,
                     turn=self.game.turns)
        self._responses = []

    def answer(self, offer: Offer) -> Response:
        """This seat's answer to another seat's offer (`offer.received` signed
        towards that seat): the gate's `respond`, else `default_respond` --
        accept, counter, or pass. An open offer (`offer.any`) is answered by a
        counter naming its any cards, or a pass (`trading.respond`). A seat
        with no gate passes."""
        if self.gate is None:
            return Response(self.seat, RESPONSE_PASS)
        self._sync()
        answer = trading.respond(self.game, self.gate, self.seat, offer)
        if answer.kind == RESPONSE_COUNTER and answer.bundle is not None:
            self._countered[offer.actor] = (tuple(offer.received), tuple(answer.bundle))
        return answer

    def counter(self, other: int, received: Sequence[int], any: int = 0) -> Response:
        """This seat's answer to `other`'s counter, `received` signed towards
        `other` (`any` its "any" cards, as an `Offer`'s): a counter to this
        seat's open offer, or -- `other` on turn -- to the counter this seat
        answered its offer with (`answer`). The two are one
        `trading.CounterThread` for the turn, whose rules bind both sides:
        a counter of `other`'s past `counter_steps`, or repeating a bundle
        already put, is a pass, and so is one of this seat's. Otherwise the
        gate answers it as `answer` would -- accept, counter, or pass -- and
        a counter back is the thread's next. The client puts what comes back
        on the host's wire; an acceptance of a counter to this seat's offer
        is this seat's to execute."""
        self._sync()
        game, me = self.game, self.seat
        passed = Response(me, RESPONSE_PASS)
        if self.gate is None:
            return passed
        mover = game.current_player
        if other != mover and me != mover:
            return passed                   # not a thread with the mover
        thread = self._threads.get(other)
        towards_me = tuple(-int(n) for n in received)   # `other`'s counter, as this seat receives it
        towards_mover = tuple(int(n) for n in received)
        if me == mover:
            towards_mover = tuple(-n for n in towards_mover)
            if thread is None:
                seen = () if self._round is None else (tuple(self._round.received),)
                thread = self._threads[other] = CounterThread(me, other, towards_mover,
                                                              self.counter_steps, seen)
            elif thread.take(Response(other, RESPONSE_COUNTER, towards_me)).kind == RESPONSE_PASS:
                return passed
        else:
            if thread is None:
                opened = self._countered.get(other)
                if opened is None:
                    return passed           # this seat never countered the mover
                offered, mine = opened
                thread = self._threads[other] = CounterThread(other, me, mine, self.counter_steps,
                                                              (offered,))
            if thread.take(Response(other, RESPONSE_COUNTER, towards_me)).kind == RESPONSE_PASS:
                return passed
        if thread.answerer != me:
            return passed
        return thread.take(trading.respond(game, self.gate, me, Offer(other, tuple(received), any)))

    def shown(self, seat: int, received: Sequence[int]) -> None:
        """What the table showed of `seat`, `received` signed towards `seat`:
        an offer or an answer it put to the table -- this seat's own
        included -- or an ask with nothing named in return ("need wood" is
        `received` with one Wood and nothing given). It goes on the public
        ledger as what `seat` wants and will give up (`PublicLedger.show`).
        The client reports each as the table shows it, in the table's order:
        `offer`, `answer` and `pick` are decisions and note nothing, since a
        client may ask `answer` of an offer nobody made."""
        trading.show(self.game, seat, received)

    def gains(self, received: Sequence[Bundle], counterparty: int) -> list[float]:
        """The gate's own gain on each exchange (signed towards this seat),
        for a client that must choose among answers the host's own protocol
        asks for and the engine has no verb for."""
        if self.gate is None:
            return [-1.0] * len(received)
        return trading.valued_many(self.gate, self.game.state(self.seat), list(received),
                                   [counterparty] * len(received))

    # -- bookkeeping -------------------------------------------------------

    def _gates(self) -> list[object | None]:
        return [self.gate if s == self.seat else None for s in range(self.game.num_players)]

    def _reset_turn(self) -> None:
        self._opened = False        # the turn's trade event has started
        self._closed = False        # ... and is over
        self._moves = 0             # this seat's main-phase moves this turn
        self._opened_at = 0         # ... when the event opened
        self._already_offered: set[Bundle] = set()
        self._offers_made = 0
        self._round: Offer | None = None
        self._responses: list[Response] = []
        self._hand_sizes: tuple[int, ...] | None = None
        self._published = False
        self._trades: list[Trade] = []
        self._countered: dict[int, tuple[Bundle, Bundle]] = {}   # actor -> (its offer, this seat's counter)
        self._threads: dict[int, CounterThread] = {}             # the other side -> its thread

    def _sync(self) -> None:
        """Start a new turn's bookkeeping when the game has moved to one, and
        note the hand sizes the turn's trading opened with."""
        game = self.game
        key = (game.turns, game.current_player)
        if key != self._turn:
            self._turn = key
            self._reset_turn()
        if self._hand_sizes is None and game.phase is Phase.MAIN:
            self._hand_sizes = event_hand_sizes(game)

    def _close_event(self) -> None:
        self._closed = True
        self._publish()
        return None

    def _turn_over(self) -> None:
        """Before an `END_TURN` applies: publish what the turn's trading did."""
        self._sync()
        self._publish()

    def _publish(self) -> None:
        """Tell the gate what this turn's trading did -- once a turn, the way
        the engine's own driver publishes a turn's event. This seat's own turn
        publishes when its event closes, another seat's when it ends: which
        exchanges it made is known only then."""
        if self._published or self.gate is None or self._hand_sizes is None:
            return
        if self.game.current_player == self.seat and not self._opened:
            return                      # no event ran; the engine publishes none
        self._published = True
        publish_trade_event(self.game, [self.gate], self._hand_sizes, tuple(self._trades))


__all__ = ["OutOfStep", "Seat"]
