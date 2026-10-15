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
for a card, through `shown`, which puts it on the public ledger. A robber
mover's ask for a card against no block or no steal is put to the seat
through `deal_answer` (`hexset.deals`). A promise binds like any other
answer: once the robber has moved and spared this seat, `answer` accepts the
mover's offer that collects the promised card, without pricing it.

As the robber's mover, a seat whose gate deals (`DealParams(ask=True)`,
`hexset.deals.Dealing`) asks before it moves the robber (`deal_ask`); the
client puts the ask to the table, hears who promised, and hands them back
(`deal_promised`), which decides whom the move spares. Once the robber has
moved and the turn is open to trading, its next offer (`offer`) is the
deal's exchange, put to the whole table like any other; whoever takes it,
promiser or not, is the gate's pick, and a spared promiser that does not is
blacklisted for the game (`deals.blacklist`). The seat keeps its deals
(`deal_log`) across a rebuilt game.

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

from .actions import Action, ActionType, apply, victim_of
from .board.board import Board
from .bots.base import seat_at
from .chance import UNSEEN, ChanceError, Hosted
from .deals import (
    ANSWERING, NO_BLOCK, Ask, Deal, DealParams, blacklist, closed, deal_params_of, default_accept, default_ask,
    default_return, robber_model,
)
from .deals import answer as deal_answer
from .deals import targeted as deal_targeted
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
from .robber import occupants
from .rules import STANDARD_GAME, GameType
from .state import HiddenRead
from . import trading
from .trading import (
    RESPONSE_ACCEPT,
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
    `deals` is how the seat answers a robber deal (`deal_answer`): by
    default it answers (`hexset.deals.ANSWERING`); `DealParams()` promises
    nothing. A promise it made is paid when the robber spares it (`answer`).
    """

    def __init__(self, game: Game, seat: int, bot: object, gate: object | None = None,
                 *, counter_steps: int = 1, deals: DealParams | None = None) -> None:
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
        self.deals = ANSWERING if deals is None else deals
        self._deal_log: list[Deal] = []
        self._turn: tuple[int, int] | None = None
        self._reset_turn()
        self._seat_bots()

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
        deals: DealParams | None = None,
    ) -> "Seat":
        """Sit down at the deal: a fresh game of `game_type`, as `seat` sees it.
        No turn cap by default -- the host ends the game."""
        dealt = start(board, num_players, random.Random(0), first=first,
                      game_type=game_type, locked=locked, turn_cap=turn_cap)
        return cls(observe(dealt, seat), seat, bot, gate, counter_steps=counter_steps, deals=deals)

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
        robbed = victim_of(game, action.b) if action.type is ActionType.MOVE_ROBBER else None
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
        if action.type is ActionType.MOVE_ROBBER:
            self._robber_moved = True
            self._settle_promise(actor, action.a, robbed)
            if actor == self.seat:
                self._close_deal(action.a, robbed)
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
        if self._owed is not None:
            mover, resource = self._owed
            towards_mover = received if a == mover else tuple(-n for n in received)
            if {a, b} == {mover, self.seat} and towards_mover[resource] > 0:
                self._owed = None
        return trade

    def resync(self, game: Game) -> None:
        """Replace the game with one rebuilt from the host's own statement of
        the position -- after a reconnect, or an `OutOfStep`. Its `chance`
        must be `Hosted`. This turn's trade bookkeeping survives if the
        rebuilt game is still in the same turn."""
        if not isinstance(game.chance, Hosted):
            raise ValueError("a hosted seat's game takes its outcomes from the host")
        self.game = game
        self._seat_bots()
        self._sync()
        deal = self._deal
        if deal is not None and deal.hex is None and game.phase is Phase.ROBBER:
            game.robber_promised = tuple((s, deal.kind) for s in deal.accepted)

    def _seat_bots(self) -> None:
        # The gate prices offers on this seat's game from the first one, not
        # from the seat's first move (`hexset.bots.seat_at`).
        seat_at(self.bot, self.game)
        if self.gate is not None and self.gate is not self.bot:
            seat_at(self.gate, self.game)
        self.game.deal_log = self._deal_log     # what `deals.blacklist` reads

    # -- this seat's moves -----------------------------------------------

    def choose(self) -> Action:
        """The bot's move, asked with the game as every table asks it. During
        a seven's discards it is this seat's next card, asked as `discard`
        asks it: on a copy where only this seat still owes, since the game
        itself has the lowest owing seat to move."""
        if not may_act(self.game, self.seat):
            raise ValueError(f"seat {self.seat} has no move in {self.game.phase.name}")
        if self.game.phase is Phase.DISCARD:
            return self.bot.choose(self._owing_alone())
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
        copy = self._owing_alone()
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

    def _owing_alone(self) -> Game:
        """A copy of the game where this seat's discard is the only one owed."""
        game = self.game
        owed = game.discard_quota[self.seat] if game.phase is Phase.DISCARD else 0
        if not owed:
            raise ValueError(f"seat {self.seat} owes no discard")
        copy = imagine(game, random.Random(0), randomize_deck=False, seat=self.seat)
        copy.discard_quota = [owed if s == self.seat else 0 for s in range(game.num_players)]
        return copy

    def restrict_robber(self, hexes: Iterable[int] | None) -> None:
        """The host's rule for the robber move at hand: the hexes it will
        take, or `None` for the rulebook's. The engine honours it for this
        move only (`Game.robber_allowed`)."""
        self.game.robber_allowed = None if hexes is None else frozenset(hexes)

    def set_embargoes(self, pairs: Iterable[tuple[int, int]]) -> None:
        """The table's trade embargoes as the host states them now: `(seat,
        against)` pairs, each a seat refusing every trade with another
        (`Game.embargoes`). An offer of this seat's is then built for the
        seats it can still trade with, and none goes up when there are none
        (`trading.offer`); and a seat under an embargo with it is out of its
        robber deals both ways: never asked or spared for a promise, never
        promised (`deals.closed`)."""
        self.game.embargoes = frozenset((int(a), int(b)) for a, b in pairs)

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

        Before any of that, a robber deal this seat made this turn: when its
        robber move spared a promiser, the first offer is the deal's exchange
        (`_deal_offer`), outside the gate's budget, as the engine's own tables
        put it before the turn's trade event (`deals.settle_deals`).
        """
        game, gate = self.game, self.gate
        self._sync()
        if (gate is None or game.phase is not Phase.MAIN or game.current_player != self.seat
                or game.free_roads > 0 or self._round is not None):
            return None
        owed = self._deal_offer()
        if owed is not None:
            return owed
        if self._closed:
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
        offered = trading.offer(game, self._gates(), already_offered=self._already_offered,
                                slots_left=None if budget < 0 else budget - self._offers_made)
        if offered is None:
            return self._close_event()
        self._offers_made += trading.offer_slots(game, offered)
        self._round = offered
        self._responses = []
        return offered

    def pick(self, responses: Sequence[Response], waiting: Iterable[int] = ()) -> Response | None:
        """Which answer to this seat's open offer to execute, among what the
        other seats said (an acceptance carries the offer's own bundle, a
        counter its own; both signed towards this seat). The gate's `pick`,
        else `default_pick`, which prices an acceptance by its taker. `None`
        executes nothing. The answers are kept for `close_round`.

        At a table whose seats answer over time, `waiting` names the seats
        that have not answered yet. The answer picked is then executed only
        if none of them would be picked over it by accepting the offer as
        put -- the gate's own pick between the two decides -- so the round
        takes its preferred taker the moment no better one can still come,
        and otherwise `None`: wait for the rest, with `held` saying what is
        waited for. An open offer (`Offer.any`) cannot be accepted as put, so
        it never waits."""
        if self._round is None:
            raise ValueError("this seat has no open offer")
        self._responses = list(responses)
        self.held = None
        chosen = trading.pick(self.game, self.gate, self.seat, self._responses)
        if chosen is None or self._round.any:
            return chosen
        rivals = [Response(s, RESPONSE_ACCEPT, tuple(self._round.received))
                  for s in waiting if s != chosen.seat]
        if not rivals:
            return chosen
        best = trading.pick(self.game, self.gate, self.seat, [chosen, *rivals])
        if best is chosen:
            return chosen
        self.held = (chosen, best.seat)
        return None

    def close_round(self, trade: Trade | None = None) -> None:
        """This seat's open round is over, with `trade` the exchange the host
        executed on it (already applied through `traded`), or `None`. The
        gate hears how it went (`trade_round_finished`), which is how a plan
        spread over several offers learns whether its fragment cleared."""
        if self._round is None:
            return
        if self._deal_round:
            self._settle_deal(trade)
        offer, self._round = self._round, None
        trading.clear_repeat(self.gate, self._already_offered, offer, self._responses, trade)
        finished = getattr(self.gate, "trade_round_finished", None)
        if finished is not None:
            finished(self.game.state(self.seat), offer, list(self._responses), trade,
                     turn=self.game.turns)
        self._responses = []
        self.held = None

    def answer(self, offer: Offer) -> Response:
        """This seat's answer to another seat's offer (`offer.received` signed
        towards that seat): the gate's `respond`, else `default_respond` --
        accept, counter, or pass. An open offer (`offer.any`) is answered by a
        counter naming its any cards, or a pass (`trading.respond`). A seat
        with no gate passes."""
        if self.gate is None:
            return Response(self.seat, RESPONSE_PASS)
        self._sync()
        if self._pays(offer):
            return Response(self.seat, RESPONSE_ACCEPT, tuple(offer.received))
        named = self._pays_named(offer)
        if named is not None:
            self._countered[offer.actor] = (tuple(offer.received), named)
            return Response(self.seat, RESPONSE_COUNTER, named)
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

    def deal_answer(self, mover: int, kind: str, resource: int, promised: Sequence[int] = ()) -> bool:
        """Whether this seat promises `mover` a card of `resource` for a deal
        of `kind` (`hexset.deals.NO_BLOCK`, `NO_STEAL`), `promised` the seats
        that already have, in the order they did (`hexset.deals.answer`).
        A promise is the seat's consent: the client pays the card if the
        robber spares it. Once the robber has moved this turn, the seat on
        turn is promised nothing."""
        if self.gate is None:
            return False                    # a seat with no gate trades nothing, a promise included
        self._sync()
        if self._robber_moved and mover == self.game.current_player:
            return False                    # this turn's robber has moved: nothing left to promise against
        promise = deal_answer(self.game, self.seat, self.gate, self.deals, Ask(mover, kind, resource), promised)
        if promise:
            self._promise = (mover, kind, resource)
        return promise

    def deal_targeted(self, mover: int, kind: str, resource: int, promised: Sequence[int] = ()) -> bool:
        """Whether this seat is the one `mover` is likeliest to hit, for a
        deal of `kind` asking a card of `resource` (`hexset.deals.targeted`),
        read with the seat's own robber model -- whether the seat holds the
        card or not. A seat that answers no deals, or asked once the robber
        has moved this turn, is targeted by nothing. What lets a client say
        a true "none" to an ask aimed at it while holding none to pay."""
        if self.deals is None or not self.deals.answer or kind not in self.deals.kinds:
            return False
        self._sync()
        if self._robber_moved and mover == self.game.current_player:
            return False
        return deal_targeted(self.game, self.seat, Ask(mover, kind, resource), tuple(promised),
                             robber_model(self.deals.model))

    def _settle_promise(self, mover: int, hex_index: int, robbed: int | None) -> None:
        """After `mover`'s robber move to `hex_index`, robbing `robbed`: a
        promise this seat made for it is owed when the move spared the seat
        -- no building of its on the hex for a no-block deal, not the one
        robbed for a no-steal one -- and void when it did not."""
        if self._promise is None or self._promise[0] != mover:
            return
        _, kind, resource = self._promise
        self._promise = None
        state = self.game.state(self.seat).state
        if kind == NO_BLOCK:
            spared = self.seat not in occupants(state, hex_index)
        else:
            spared = robbed != self.seat
        if spared:
            self._owed = (mover, resource)

    def _pays_named(self, offer: Offer) -> Bundle | None:
        """The counter that pays a debt the mover collects with an open offer
        ("the card for any of yours", `any == -1`): the promised card for the
        one other card this seat values most (`gains`), signed towards the
        mover; None for any other offer."""
        if self._owed is None or offer.any != -1 or offer.actor != self._owed[0]:
            return None
        resource = self._owed[1]
        unit = tuple(1 if r == resource else 0 for r in range(len(offer.received)))
        if tuple(offer.received) != unit or self.game.state(self.seat).state.hands[self.seat][resource] < 1:
            return None
        options = [tuple(n - (r == back) for r, n in enumerate(unit))
                   for back in range(len(unit)) if back != resource]
        gains = self.gains([tuple(-n for n in b) for b in options], offer.actor)
        return options[max(range(len(options)), key=lambda i: gains[i])]

    def _pays(self, offer: Offer) -> bool:
        """Whether `offer` collects the card this seat owes: from the mover
        it promised, one card of the promised resource, for at most one card
        of another resource, which the mover chooses (`received` signed
        towards the mover), and while the seat still holds the card. Every
        such exchange is a payment, so a client asked to name the card back
        (an offer with "any card" in return) may take the one it values most."""
        if self._owed is None or offer.any or offer.actor != self._owed[0]:
            return False
        resource = self._owed[1]
        received = tuple(offer.received)
        back = [r for r, n in enumerate(received) if r != resource and n]
        if received[resource] != 1 or len(back) > 1 or any(received[r] != -1 for r in back):
            return False
        return self.game.state(self.seat).state.hands[self.seat][resource] >= 1

    def gains(self, received: Sequence[Bundle], counterparty: int) -> list[float]:
        """The gate's own gain on each exchange (signed towards this seat),
        for a client that must choose among answers the host's own protocol
        asks for and the engine has no verb for."""
        if self.gate is None:
            return [-1.0] * len(received)
        return trading.valued_many(self.gate, self.game.state(self.seat), list(received),
                                   [counterparty] * len(received))

    # -- this seat's robber deals, as the mover -----------------------------

    @property
    def deal_log(self) -> list[Deal]:
        """Every deal this seat asked this game, with what came of it."""
        return list(self._deal_log)

    def _dealing(self) -> tuple[DealParams | None, object]:
        """The gate's deal params where they ask, and its value dealer, if any."""
        params = deal_params_of(self.gate) if self.gate is not None else None
        if params is None or not params.ask:
            return None, None
        return params, getattr(self.gate, "deal_dealer", None)

    def deal_ask(self) -> Ask | None:
        """At this seat's own robber move, before it is chosen: whether to ask
        the table for a card against no block, and which -- the gate's value
        dealer (`Dealing.deal_dealer`), else its own `deal_ask`, else
        `deals.default_ask`. Decided once a robber move; asked again, the same
        answer. `None` for a gate that does not ask, or nothing worth asking.
        An ask is this seat's `Deal`, kept in `deal_log`."""
        self._sync()
        game = self.game
        if game.phase is not Phase.ROBBER or game.current_player != self.seat:
            return None
        key = (game.turns, game._state.robber)
        if self._deal_key == key:
            return self._ask
        params, dealer = self._dealing()
        if params is None:
            return None
        hook = dealer.ask if dealer is not None else getattr(self.gate, "deal_ask", None)
        ask = hook(game, self.seat) if hook is not None else default_ask(game, self.seat, self.gate, params)
        self._deal_key, self._ask = key, ask
        if ask is not None:
            self._deal = Deal(game.turns, self.seat, ask.kind, ask.resource)
            self._deal_log.append(self._deal)
        return ask

    def deal_promised(self, promised: Sequence[int]) -> tuple[int, ...]:
        """The seats that promised this seat's ask, in the order they did.
        Of them it counts every one but a seat its own view says cannot hold
        the card (`Deal.void`) and one it has blacklisted or is under an
        embargo with (`Deal.ignored`, `deals.blacklist`, `deals.closed`), and spares for those its dealer picks
        (`deal_accept`, else `deals.default_accept`): the robber move it then
        chooses spares them (`Game.robber_promised`). Returns the seats spared
        for; `()` without an open ask."""
        deal, game = self._deal, self.game
        if deal is None or deal.hex is not None or game.phase is not Phase.ROBBER or self._ask is None:
            return ()
        view = game.state(self.seat)
        card = tuple(int(r == deal.resource) for r in range(len(view.known[self.seat])))
        banned = blacklist(game, self.seat) | closed(game, self.seat)
        promised = tuple(dict.fromkeys(s for s in promised if s != self.seat))
        deal.promised = promised
        deal.void = tuple(s for s in promised if not trading.has_room(view, s, card))
        deal.ignored = tuple(s for s in promised if s in banned and s not in deal.void)
        counted = deal.counted
        deal.accepted = ()
        if counted:
            _, dealer = self._dealing()
            accept = dealer.accept if dealer is not None else getattr(self.gate, "deal_accept", None)
            taken = (accept(game, self.seat, self._ask, counted) if accept is not None
                     else default_accept(game, self.seat, self.gate, self._ask, counted))
            deal.accepted = tuple(s for s in counted if s in set(taken))
        game.robber_promised = tuple((s, deal.kind) for s in deal.accepted)
        return deal.accepted

    def _close_deal(self, hex_index: int, robbed: int | None) -> None:
        """This seat's robber move, as the table took it: where it went, whom it
        robbed, and which counted promisers it spared (`kept`) and hit."""
        deal = self._deal
        if deal is None or deal.hex is not None:
            return
        deal.hex, deal.victim = hex_index, robbed
        on = occupants(self.game.state(self.seat).state, hex_index)
        hit = (lambda s: s in on) if deal.kind == NO_BLOCK else (lambda s: s == robbed)
        deal.kept = tuple(s for s in deal.counted if not hit(s))
        deal.broken = tuple(s for s in deal.counted if hit(s))

    def _deal_offer(self) -> Offer | None:
        """The deal's exchange, once, when the robber spared a promiser: the
        asked card for the one the dealer offers back (`give_back`, else the
        gate's `deal_return`, else `deals.default_return`). None owed, or no
        card to give, no offer (`Deal.unpaid`)."""
        deal = self._deal
        if deal is None or not deal.kept or deal.offered is not None or deal.unpaid:
            return None
        game, gate = self.game, self.gate
        _, dealer = self._dealing()
        hook = dealer.give_back if dealer is not None else getattr(gate, "deal_return", None)
        back = (hook(game, self.seat, deal.kept, deal.resource) if hook is not None
                else default_return(game, self.seat, gate, deal.kept, deal.resource))
        hand = game.state(self.seat).state.hands[self.seat]
        if back is None or back == deal.resource or hand[back] < 1:
            deal.unpaid = deal.kept
            return None
        deal.offered = back
        received = [0] * len(hand)
        received[deal.resource], received[back] = 1, -1
        self._round = Offer(self.seat, tuple(received))
        self._responses = []
        self._deal_round = True
        return self._round

    def _settle_deal(self, trade: Trade | None) -> None:
        """The deal's round is over: who took it, and which spared promisers
        did not (`Deal.rejected`, blacklisted from here on)."""
        deal, self._deal_round = self._deal, False
        if deal is None:
            return
        if trade is not None:
            deal.taker = trade.b if trade.a == self.seat else trade.a
            if deal.taker in deal.kept:
                deal.paid = ((deal.taker, deal.offered),)
        took = {r.seat for r in self._responses if r.kind == RESPONSE_ACCEPT}
        deal.rejected = tuple(s for s in deal.kept if s not in took and s != deal.taker)

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
        # Why the last `pick` returned None to wait: (the answer it would
        # take now, the waiting seat the gate picks over it); else None.
        self.held: tuple[Response, int] | None = None
        self._hand_sizes: tuple[int, ...] | None = None
        self._published = False
        self._trades: list[Trade] = []
        self._countered: dict[int, tuple[Bundle, Bundle]] = {}   # actor -> (its offer, this seat's counter)
        self._threads: dict[int, CounterThread] = {}             # the other side -> its thread
        self._promise: tuple[int, str, int] | None = None   # (mover, kind, resource) this seat promised
        self._owed: tuple[int, int] | None = None           # (mover, resource) the robber spared it for
        self._robber_moved = False                          # the robber has moved this turn
        deal = getattr(self, "_deal", None)
        if deal is not None and deal.kept and deal.offered is None and not deal.unpaid:
            deal.unpaid = deal.kept                         # the turn ended before the deal's offer
        self._deal: Deal | None = None                      # this turn's ask of this seat's
        self._deal_key: tuple[int, int | None] | None = None   # the robber move it was decided for
        self._ask: Ask | None = None
        self._deal_round = False                            # the open round is the deal's

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
