# SPDX-License-Identifier: GPL-3.0-only
"""Robber deals at the engine's own tables (`hexset.deals`): the ask, the
promises, a robber move that keeps them, the payment, and a table where
nobody asks playing exactly as before."""
from __future__ import annotations

import random

import numpy as np

import pytest

from hexset import deals as D
from hexset.actions import ActionType, apply, legal_actions
from hexset.arena import Entrant, entrant_from_name, lineup_from_names, run_seated, seat_bots, spawn
from hexset.board.board import random_base_board
from hexset.bots import RandomBot
from hexset.cards import DevCard
from hexset.game import Phase, move_robber_to, roll_dice, start
from hexset.robber import occupants, victims

from helpers import clear_hand, deal
from tests.test_game import run_setup

WOOD, BRICK, SHEEP, WHEAT, ORE = range(5)


class Fixed:
    """An `Answerer` that promises as told and pays unless told not to."""

    def __init__(self, promise: bool, pays: bool = True):
        self.answer, self.paying = promise, pays

    def promise(self, game, seat, ask, promised):
        return self.answer

    def pays(self, game, seat, deal):
        return self.paying


class Gate:
    """A gate that takes any exchange (every one gains it 1), deals under
    `params`, and answers with fixed hooks where the test gives them (an
    answer through a fixed `Answerer`, so it promises whether or not it is
    the mover's likeliest target). It trades but proposes nothing of its
    own, so the only offer after the robber is the deal's."""

    trade_floor = 0.0

    def candidates(self, view, counterparties, **kw):
        return []

    def __init__(self, params=None, *, ask=None, answer=None, accept=None, pays=True):
        self.deal_params = params
        if ask is not None:
            self.deal_ask = lambda game, seat: ask
        if answer is not None:
            self.deal_answerer = Fixed(answer, pays)
        if accept is not None:
            self.deal_accept = lambda game, seat, a, promised: accept(promised)

    def gains_many(self, view, received, counterparties):
        return [1.0 for _ in received]


def a_table(seed: int = 0):
    game = run_setup(start(random_base_board(random.Random(seed)), 4, random.Random(seed)))
    for s in range(4):
        clear_hand(game._state, s)
        game.ledger.seats[s].known = [0] * 5     # the record agrees: nobody holds a card
        game.ledger.seats[s].unknown = 0
    game.phase = Phase.ROLL
    game.current_player = 0
    return game


def seat(game, gates):
    game.gates = tuple(gates)
    game.trade_mode = "round"


def robber_hexes(game) -> set[int]:
    return {a.a for a in legal_actions(game) if a.type is ActionType.MOVE_ROBBER}


def hexes_of(game, s) -> set[int]:
    state = game._state
    return {h for h in range(state.board.num_hexes) if s in occupants(state, h)}


ASKER = D.DealParams(ask=True)
ANSWERS = D.DealParams(answer=True)


# --- params and names ------------------------------------------------------


def test_params_parse_back_from_their_spec():
    p = D.DealParams.parse("answer,ask,nosteal,model=corpus")
    assert (p.answer, p.ask, p.kinds, p.model) == (True, True, (D.NO_STEAL,), "corpus")
    assert D.DealParams.parse(p.spec()) == p
    assert D.DealParams.parse("answer").kinds == D.KINDS
    with pytest.raises(ValueError, match="unknown deal setting"):
        D.DealParams.parse("answer,bribe")
    with pytest.raises(ValueError, match="deal kinds"):
        D.DealParams(kinds=("bribe",))


def test_a_lineup_names_how_a_seat_deals():
    e = entrant_from_name("random^answer,ask")
    assert e.name == "random^answer,ask"
    assert e.deals == D.DealParams(answer=True, ask=True)
    assert [x.deals for x in lineup_from_names(["random^answer", "random"])] == [ANSWERS, None]
    bot = spawn(e, random_base_board(random.Random(0)), random.Random(0))
    assert isinstance(bot, D.Dealing) and bot.deal_params == e.deals
    assert spawn(Entrant("random", kind="random"), random_base_board(random.Random(0)),
                 random.Random(0)).__class__ is RandomBot


def test_a_deal_survives_the_journal():
    d = D.Deal(3, 1, D.NO_BLOCK, WOOD, promised=(2, 0), accepted=(2,), hex=4, victim=0,
               kept=(2,), paid=((2, ORE),))
    assert D.Deal.from_json(d.to_json()) == d


# --- nobody dealing --------------------------------------------------------


def play_out(dealing: bool, seed: int = 1, actions: int = 3000):
    game = start(random_base_board(random.Random(seed)), 4, random.Random(seed))
    bots = [RandomBot(random.Random(seed * 10 + s)) for s in range(4)]
    if dealing:      # dealing params that ask nothing and answer nothing
        bots = [D.Dealing(b, D.DealParams()) for b in bots]
    seat_bots(game, bots)
    run_seated(game, bots, actions)
    s = game._state
    return game.turns, [list(h) for h in s.hands], s.robber, list(s.vertex_owner), game.deal_log


def test_a_table_where_nobody_asks_plays_as_it_did():
    assert play_out(True) == play_out(False)
    assert play_out(False)[-1] == []


# --- the robber move keeps the promises -------------------------------------


def test_no_block_takes_the_payers_hexes_away():
    game = a_table()
    state = game._state
    allowed, spared = D.keep_promises(state, 0, None, {1: D.NO_BLOCK})
    assert allowed and not (allowed & hexes_of(game, 1)) and spared == frozenset()


def test_no_steal_leaves_the_hexes_but_spares_the_payer():
    game = a_table()
    for s in (1, 2, 3):
        deal(game, s, WOOD)
    game.phase = Phase.ROBBER
    game.robber_promised = ((1, D.NO_STEAL),)
    moves = [a for a in legal_actions(game) if a.type is ActionType.MOVE_ROBBER]
    assert all(a.b != 1 for a in moves)
    shared = [h for h in hexes_of(game, 1) if set(victims(game._state, h, 0)) - {1}]
    assert shared and shared[0] in robber_hexes(game)
    alone = [h for h in hexes_of(game, 1) if set(victims(game._state, h, 0)) == {1}]
    for h in alone:
        assert h not in robber_hexes(game)
    with pytest.raises(ValueError, match="promised"):
        move_robber_to(game, shared[0], 1)


def test_promises_that_cannot_all_be_kept_bind_nothing():
    game = a_table()
    everyone = {s: D.NO_BLOCK for s in (1, 2, 3)}
    state = game._state
    stand = [h for h in D.candidate_hexes(state, None)
             if not any(s in occupants(state, h) for s in (1, 2, 3))]
    allowed = frozenset(D.candidate_hexes(state, None)) - frozenset(stand)
    assert D.keep_promises(state, 0, allowed, everyone) == (allowed, frozenset())


# --- the event, end to end ------------------------------------------------


def a_deal_on_a_seven(promise=(True, True, False), kind=D.NO_BLOCK):
    game = a_table()
    deal(game, 0, ORE)
    for s in (1, 2, 3):
        deal(game, s, WOOD)
    ask = D.Ask(0, kind, WOOD)
    seat(game, [Gate(ASKER, ask=ask, accept=lambda promised: promised)]
         + [Gate(ANSWERS, answer=yes) for yes in promise])
    roll_dice(game, 7)
    return game


def test_the_table_promises_in_turn_and_the_mover_takes_them():
    game = a_deal_on_a_seven()
    assert game.phase is Phase.ROBBER
    (d,) = game.deal_log
    assert (d.mover, d.kind, d.resource, d.promised, d.accepted) == (0, D.NO_BLOCK, WOOD, (1, 2), (1, 2))
    assert game.robber_promised == ((1, D.NO_BLOCK), (2, D.NO_BLOCK))
    assert not robber_hexes(game) & (hexes_of(game, 1) | hexes_of(game, 2))


def test_a_spared_payer_pays_once_the_robber_has_moved():
    game = a_deal_on_a_seven(promise=(True, False, False))
    move = next(a for a in legal_actions(game) if a.type is ActionType.MOVE_ROBBER)
    apply(game, move)
    (d,) = game.deal_log
    assert game.phase is Phase.MAIN and game.robber_promised == ()
    assert d.hex == move.a and d.kept == (1,) and d.paid == ((1, ORE),)
    hands = game._state.hands
    assert hands[1][WOOD] == 0 and hands[1][ORE] == 1 and hands[0][WOOD] >= 1


def test_a_payer_hit_anyway_pays_nothing():
    game = a_deal_on_a_seven(promise=(True, False, False))
    game.robber_promised = ()           # as if no move could have kept it
    target = min(hexes_of(game, 1) - {game._state.robber})
    move_robber_to(game, target, 1)
    (d,) = game.deal_log
    assert d.broken == (1,) and d.kept == () and d.paid == ()


def test_a_knight_before_the_roll_is_paid_after_it():
    game = a_table()
    state = game._state
    deal(game, 0, ORE)
    deal(game, 1, WOOD)
    state.dev_cards[0][DevCard.KNIGHT] += 1
    seat(game, [Gate(ASKER, ask=D.Ask(0, D.NO_BLOCK, WOOD), accept=lambda p: p),
                Gate(ANSWERS, answer=True), Gate(), Gate()])
    apply(game, next(a for a in legal_actions(game) if a.type is ActionType.PLAY_KNIGHT))
    apply(game, next(a for a in legal_actions(game) if a.type is ActionType.MOVE_ROBBER))
    (d,) = game.deal_log
    assert game.phase is Phase.ROLL and d.kept == (1,) and d.paid == ()
    roll_dice(game, 6 if game._state.robber is None else 8)
    assert d.paid == ((1, ORE),)


def test_a_seat_without_the_card_or_the_params_is_not_asked():
    game = a_table()
    deal(game, 0, ORE)
    deal(game, 2, WOOD)
    asked = []
    class Counting(Gate):
        class deal_answerer:
            @staticmethod
            def promise(game, s, ask, promised):
                asked.append(s)
                return False
    seat(game, [Gate(ASKER, ask=D.Ask(0, D.NO_BLOCK, WOOD)), Counting(ANSWERS), Counting(ANSWERS), Counting()])
    roll_dice(game, 7)
    assert asked == [2]
    assert game.deal_log[0].promised == () and game.robber_promised == ()


# --- params for the value mover and answerers -----------------------------


def test_mover_answerer_and_pay_parse_back_and_are_checked():
    p = D.DealParams.parse("answer,ask,noblock,mover=value,answerer=fixed,pay=0.8")
    assert (p.mover, p.answerer, p.pay) == ("value", "fixed", 0.8)
    assert D.DealParams.parse(p.spec()) == p
    with pytest.raises(ValueError, match="deal mover"):
        D.DealParams(mover="guess")
    with pytest.raises(ValueError, match="payment chance"):
        D.DealParams(pay=0.0)
    with pytest.raises(ValueError, match="answers only"):
        D.DealParams(answerer="fixed")
    with pytest.raises(ValueError, match="no answerer is registered"):
        D.answerer_for("nobody-registered-this", random.Random(0))


def test_a_registered_answerer_answers_for_a_dealing_seat():
    D.register_answerer("test-always", lambda rng: Fixed(True))
    bot = D.Dealing(RandomBot(random.Random(0)), D.DealParams(answer=True, answerer="test-always"))
    assert isinstance(bot.deal_answerer, Fixed)
    game = a_table()
    deal(game, 0, ORE)
    deal(game, 3, WOOD)
    seat(game, [Gate(ASKER, ask=D.Ask(0, D.NO_BLOCK, WOOD), accept=lambda p: p), Gate(), Gate(), bot])
    roll_dice(game, 7)
    assert game.deal_log[0].promised == (3,)


class Dislikes(Gate):
    """A gate whose own pricing refuses every exchange."""

    def gains_many(self, view, received, counterparties):
        return [-1.0 for _ in received]


def test_a_spared_payer_may_refuse_to_pay():
    game = a_table()
    deal(game, 0, ORE)
    deal(game, 1, WOOD)
    seat(game, [Gate(ASKER, ask=D.Ask(0, D.NO_BLOCK, WOOD), accept=lambda p: p),
                Dislikes(ANSWERS, answer=True, pays=False), Dislikes(), Dislikes()])
    roll_dice(game, 7)
    apply(game, next(a for a in legal_actions(game) if a.type is ActionType.MOVE_ROBBER))
    (d,) = game.deal_log
    assert d.kept == (1,) and d.paid == () and d.rejected == (1,) and d.taker is None
    assert game._state.hands[1][WOOD] == 1
    assert D.Deal.from_json(d.to_json()) == d
    assert D.blacklist(game, 0) == {1}


# --- the value mover --------------------------------------------------------


class Net:
    """A stand-in network: the mover's win chance is `high` with the robber
    on `target` and `low` elsewhere; `gains_many` prices `payer` giving
    `resource` at `gain` and anything else as uncoverable."""

    def __init__(self, target, payer, resource, gain, high=0.6, low=0.5, back=None):
        self.target, self.payer, self.resource, self.gain = target, payer, resource, gain
        self.high, self.low, self.back = high, low, back or {}
        self.policy = self

    calls = 0
    after_gain = None      # the gain once the robber has moved, where it differs

    def value_rows(self, rows):
        self.calls += 1
        return [tuple(self.high if g._state.robber == self.target else self.low for _ in range(4))
                for g, seat in rows]

    def seat_at(self, game):
        self.seated = game

    def score_rows(self, rows):
        """The robber ranking: `target` first, then the hexes in order."""
        out = []
        for game, seat, options in rows:
            out.append((np.array([2.0 if o.a == self.target else 1.0 / (1 + o.a) for o in options]),
                        (0.0,) * 4))
        return out

    def gains_many(self, view, received, counterparties):
        moved = getattr(self, "seated", None) is not None and self.seated.phase is not Phase.ROBBER
        if moved and self.after_gain is not None:
            return [self.after_gain if c == self.payer and b[self.resource] == 1 else -1.0
                    for b, c in zip(received, counterparties)]
        out = []
        for bundle, c in zip(received, counterparties):
            if c == self.payer and bundle[self.resource] == 1:
                give = next(i for i, n in enumerate(bundle) if n == -1)
                out.append(self.gain + self.back.get(give, 0.0))
            else:
                out.append(-1.0)
        return out


class Trader:
    """A stand-in for the mover's trading: it asks `payer` for `resource`, one
    for one, giving `give`, and takes any exchange (or, with `refuse_after`,
    none once the robber has moved)."""

    def __init__(self, payer, resource, give, refuse_after=False):
        self.payer, self.resource, self.give, self.refuse_after = payer, resource, give, refuse_after

    def candidates(self, view, counterparties, *, turn=None, already_offered=()):
        if view.state.hands[view.perspective][self.give] <= 0:
            return []                      # nothing to give: no ask, as a real trader
        bundle = tuple(1 if r == self.resource else -1 if r == self.give else 0 for r in range(5))
        return [(s, bundle) for s in counterparties if s == self.payer]

    def pick(self, view, responses):
        moved = view.state.robber != self.robber_at
        return None if (self.refuse_after and moved) else 0


def a_value_mover(gain, pay=1.0, back=None, refuse_after=False):
    """Seat 0 to move the robber, holding ore and brick; seat 1 holding wood;
    the robber best on a hex of seat 1's that seat 2 is not on."""
    game = a_table()
    deal(game, 0, ORE)
    deal(game, 0, BRICK)
    deal(game, 1, WOOD)
    state = game._state
    target = min(h for h in hexes_of(game, 1) - hexes_of(game, 2) if h != state.robber)
    net = Net(target, 1, WOOD, gain, back=back)
    trader = Trader(1, WOOD, ORE, refuse_after)
    trader.robber_at = state.robber
    dealer = D.ValueDealer(net, D.DealParams(ask=True, mover="value", pay=pay), random.Random(0), gate=trader)
    game.phase = Phase.ROBBER
    return game, dealer


def test_the_value_mover_stays_silent_when_no_deal_beats_its_best_move():
    game, dealer = a_value_mover(gain=0.05)     # sparing seat 1 costs 0.1, its card is worth 0.05
    assert dealer.ask(game, 0) is None


def test_the_value_mover_asks_for_the_card_its_trading_wants_most():
    game, dealer = a_value_mover(gain=0.2)
    assert dealer.ask(game, 0) == D.Ask(0, D.NO_BLOCK, WOOD)
    game2, dealer2 = a_value_mover(gain=0.2, pay=0.4)   # 0.4 x 0.2 no longer covers the 0.1
    assert dealer2.ask(game2, 0) is None


def test_the_value_mover_takes_at_most_the_one_promise_worth_most():
    game, dealer = a_value_mover(gain=0.2)
    ask = dealer.ask(game, 0)
    assert dealer.accept(game, 0, ask, (2, 1, 3)) == (1,)
    assert dealer.accept(game, 0, ask, (2, 3)) == ()


def test_the_value_mover_gives_back_the_card_it_minds_least():
    game, dealer = a_value_mover(gain=0.2, back={ORE: -0.05, BRICK: 0.0})
    assert dealer.give_back(game, 0, (1,), WOOD) == BRICK


class Promises:
    """A stand-in promise model: every holder promises at `chance`, every spared promiser pays at `pays`."""

    def __init__(self, chance, pays=1.0):
        self.chance, self.pays = chance, pays

    def promise_chance(self, game, seat, ask):
        return self.chance

    def pay_chance(self, game, mover):
        return self.pays


def a_reading_mover(gain, chance, cost=0.0, pays=1.0):
    """`a_value_mover` reading the table through a registered promise model."""
    D.register_promise_model("test-promises", Promises(chance, pays))
    game, dealer = a_value_mover(gain=gain)
    dealer.params = D.DealParams(ask=True, mover="value", promise="test-promises", ask_cost=cost)
    return game, dealer


def test_promise_and_cost_parse_back_and_are_checked():
    p = D.DealParams.parse("answer,ask,noblock,mover=value,promise=corpus,cost=0.01")
    assert p.promise == "corpus" and p.ask_cost == 0.01
    assert D.DealParams.parse(p.spec()) == p
    with pytest.raises(ValueError, match="cost"):
        D.DealParams(ask=True, ask_cost=-0.1)
    with pytest.raises(ValueError, match="no promise model"):
        D.promise_model("nobody-registered-this")


def test_the_value_mover_asks_where_the_ask_is_worth_its_cost():
    # Sparing seat 1 costs 0.1 and its card is worth 0.2: the deal is worth 0.1, the ask 0.1 x chance.
    game, dealer = a_reading_mover(gain=0.2, chance=0.3, cost=0.02)
    assert dealer.ask(game, 0) == D.Ask(0, D.NO_BLOCK, WOOD)        # 0.03 beats 0.02
    game, dealer = a_reading_mover(gain=0.2, chance=0.1, cost=0.02)
    assert dealer.ask(game, 0) is None                              # 0.01 does not
    game, dealer = a_reading_mover(gain=0.2, chance=0.1, cost=0.0)
    assert dealer.ask(game, 0) == D.Ask(0, D.NO_BLOCK, WOOD)        # free, any deal is worth asking


def test_the_promise_model_s_payment_chance_discounts_the_deal():
    game, dealer = a_reading_mover(gain=0.2, chance=1.0, pays=0.4)  # 0.4 x 0.2 no longer covers the 0.1
    assert dealer.ask(game, 0) is None


def test_a_value_mover_needs_a_network():
    with pytest.raises(ValueError, match="value head"):
        D.Dealing(RandomBot(random.Random(0)), D.DealParams(ask=True, mover="value"))
    net = Net(0, 1, WOOD, 0.1)
    from hexset.bots import TradesBy
    bot = D.Dealing(TradesBy(net, Gate()), D.DealParams(ask=True, mover="value"))
    assert bot.deal_dealer.net is net


def test_a_spared_promiser_not_honouring_may_still_take_the_offer_as_any_seat_would():
    game = a_table()
    deal(game, 0, ORE)
    deal(game, 1, WOOD)
    seat(game, [Gate(ASKER, ask=D.Ask(0, D.NO_BLOCK, WOOD), accept=lambda p: p),
                Gate(ANSWERS, answer=True, pays=False), Dislikes(), Dislikes()])
    roll_dice(game, 7)
    apply(game, next(a for a in legal_actions(game) if a.type is ActionType.MOVE_ROBBER))
    (d,) = game.deal_log
    assert d.taker == 1 and d.paid == ((1, ORE),) and d.rejected == ()


def test_any_seat_may_take_the_deal_offer_and_the_mover_picks():
    game = a_table()
    deal(game, 0, ORE)
    for s in (1, 2):
        deal(game, s, WOOD)
    seat(game, [Gate(ASKER, ask=D.Ask(0, D.NO_BLOCK, WOOD), accept=lambda p: p),
                Dislikes(ANSWERS, answer=True, pays=False), Gate(), Dislikes()])
    roll_dice(game, 7)
    apply(game, next(a for a in legal_actions(game) if a.type is ActionType.MOVE_ROBBER))
    (d,) = game.deal_log
    assert d.promised == (1,) and d.taker == 2 and d.paid == () and d.rejected == (1,)
    assert game._state.hands[2][ORE] == 1 and game._state.hands[0][WOOD] >= 1


def test_a_promise_of_a_card_the_view_rules_out_is_void_and_blacklists():
    game = a_table()
    deal(game, 0, ORE)
    game._state.hands[2][WOOD] += 1                  # held, but the record says its one card is wheat
    game.ledger.seats[2].known = [0, 0, 0, 1, 0]
    game.ledger.seats[2].unknown = 0
    seat(game, [Gate(ASKER, ask=D.Ask(0, D.NO_BLOCK, WOOD), accept=lambda p: p),
                Gate(), Gate(ANSWERS, answer=True), Gate()])
    roll_dice(game, 7)
    (d,) = game.deal_log
    assert d.promised == (2,) and d.void == (2,) and d.counted == () and d.accepted == ()
    assert D.blacklist(game, 0) == {2}


def test_a_blacklisted_seat_s_promise_is_ignored():
    game = a_table()
    deal(game, 0, ORE)
    deal(game, 1, WOOD)
    game.deal_log.append(D.Deal(0, 0, D.NO_BLOCK, WOOD, promised=(1,), kept=(1,), rejected=(1,)))
    seat(game, [Gate(ASKER, ask=D.Ask(0, D.NO_BLOCK, WOOD), accept=lambda p: p),
                Gate(ANSWERS, answer=True), Gate(), Gate()])
    roll_dice(game, 7)
    d = game.deal_log[-1]
    assert d.promised == (1,) and d.ignored == (1,) and d.accepted == () and game.robber_promised == ()


def test_the_value_mover_keeps_one_plan_from_the_ask_to_whom_it_spares():
    game, dealer = a_value_mover(gain=0.2)
    ask = dealer.ask(game, 0)
    game.deal_log.append(D.Deal(game.turns, 0, D.NO_BLOCK, ask.resource, promised=(1,)))
    plan = dealer._plan
    assert dealer.accept(game, 0, ask, (1,)) == (1,)
    assert dealer._plan is plan and dealer.net.calls == 1      # one batch of robber values, kept


def test_the_value_mover_does_not_spare_for_a_deal_it_would_not_take_after_the_steal():
    game, dealer = a_value_mover(gain=0.2, refuse_after=True)   # its trading takes it before the move, not after
    assert dealer.accept(game, 0, D.Ask(0, D.NO_BLOCK, WOOD), (1,)) == ()


def test_the_value_mover_asks_only_where_it_would_spare_the_seat_that_promised():
    game, dealer = a_value_mover(gain=0.2, refuse_after=True)   # worth asking before the move, not after it
    assert dealer.ask(game, 0) is None
    game, dealer = a_value_mover(gain=0.2)
    dealer.net.after_gain = 0.05                                 # the steal leaves the card worth too little
    assert dealer.ask(game, 0) is None
    game, dealer = a_value_mover(gain=0.2)
    ask = dealer.ask(game, 0)
    assert ask is not None and dealer._plan.worth[(1, WOOD)] > 0
    assert dealer.accept(game, 0, ask, (1,)) == (1,)             # a seat it asked for, priced as it asked


def test_the_value_mover_with_an_empty_hand_asks_nothing():
    game, dealer = a_value_mover(gain=0.2)
    clear_hand(game._state, 0)
    game.ledger.seats[0].known = [0] * 5
    assert dealer.ask(game, 0) is None
    assert dealer.accept(game, 0, D.Ask(0, D.NO_BLOCK, WOOD), (1,)) == ()


def test_a_value_mover_finds_the_network_behind_a_search():
    class Search:
        def __init__(self, gate):
            self.gate = gate
    net = Net(0, 1, WOOD, 0.1)
    from hexset.bots import TradesBy
    bot = D.Dealing(TradesBy(Search(net), Gate()), D.DealParams(ask=True, mover="value"))
    assert bot.deal_dealer.net is net


def test_a_seat_answers_unless_it_says_otherwise():
    """Answering is the default: a gate naming no deal parameters answers and
    never asks; no gate deals nothing; `^none` opts a seat out."""
    from hexset.deals import ANSWERING, DealParams, deal_params_of

    class Plain:
        pass

    assert deal_params_of(Plain()) is ANSWERING
    assert ANSWERING.answer and not ANSWERING.ask
    assert deal_params_of(None) is None
    none = DealParams.parse("none")
    assert not none.answer and not none.ask


# --- the answer, off a value head -------------------------------------------


class AnswerNet:
    """A stand-in network for the answering seat 1: its win chance is `low`
    with the robber on `target` and `high` elsewhere; it prices giving wood
    to the mover for any card at `gain`."""

    def __init__(self, target, gain, high=0.5, low=0.4):
        self.target, self.gain, self.high, self.low = target, gain, high, low
        self.policy = self

    def value_rows(self, rows):
        return [tuple(self.low if g._state.robber == self.target else self.high for _ in range(4))
                for g, seat in rows]

    def seat_at(self, game):
        pass

    def gains_many(self, view, received, counterparties):
        return [self.gain if c == 0 and b[WOOD] == -1 else -1.0 for b, c in zip(received, counterparties)]


class Reading:
    """A robber model: the mover puts the robber on `target` with chance
    `silent`, or `sure` once seat 1 has promised, else on `other`; it robs
    any seat there alike."""

    stay = 5.0

    def __init__(self, target, other, silent=0.9, sure=0.3):
        self.target, self.other, self.silent, self.sure = target, other, silent, sure

    def hexes(self, state, mover, allowed, promised):
        p = self.sure if 1 in promised else self.silent
        return {self.target: p, self.other: 1.0 - p}

    def victims(self, state, mover, hex_index, promised):
        seats = victims(state, hex_index, mover)
        return {s: 1.0 / len(seats) for s in seats}


_READINGS: dict = {}


def a_value_answer(gain, silent=0.9, sure=0.3):
    """Mover 0 asks seat 1 for wood; the mover's reading puts the robber on
    a hex of seat 1's alone, or on a hex nobody is on."""
    game = a_table()
    deal(game, 0, ORE)
    deal(game, 1, WOOD)
    deal(game, 1, SHEEP)
    state = game._state
    game.phase = Phase.ROBBER
    target = min(h for h in hexes_of(game, 1) - hexes_of(game, 2) - hexes_of(game, 3) if h != state.robber)
    other = min(h for h in robber_hexes_free(game) if h != state.robber)
    name = f"test-reading-{target}-{other}-{silent}-{sure}"
    if name not in _READINGS:
        reading = Reading(target, other, silent, sure)
        _READINGS[name] = lambda: reading
        D.register_robber_model(name, _READINGS[name])
    params = D.DealParams(answer=True, model=name, answerer="value")
    return game, D.ValueDealer(AnswerNet(target, gain), params, random.Random(0)), D.Ask(0, D.NO_BLOCK, WOOD)


def robber_hexes_free(game) -> set[int]:
    state = game._state
    return {h for h in range(state.board.num_hexes) if not occupants(state, h)} & set(robber_hexes(game))


def test_the_value_answer_prices_the_promise_in_win_chance():
    # Silent: 0.9 x 0.4 + 0.1 x 0.5 = 0.41. Promising: 0.3 x 0.4 + 0.7 x (0.5 + gain).
    game, dealer, ask = a_value_answer(gain=-0.05)
    assert dealer.promise_worth(game, 1, ask, ()) == pytest.approx(0.06 + 0.7 * -0.05)
    assert dealer.promise(game, 1, ask, ())
    game, dealer, ask = a_value_answer(gain=-0.2)
    assert dealer.promise_worth(game, 1, ask, ()) == pytest.approx(0.06 + 0.7 * -0.2)
    assert not dealer.promise(game, 1, ask, ())


def test_the_value_answer_promises_nothing_the_promise_does_not_move():
    game, dealer, ask = a_value_answer(gain=0.01, silent=0.9, sure=0.9)
    assert dealer.promise_worth(game, 1, ask, ()) == pytest.approx(0.1 * 0.01)
    game, dealer, ask = a_value_answer(gain=-0.01, silent=0.9, sure=0.9)
    assert not dealer.promise(game, 1, ask, ())


def test_the_value_answer_speaks_only_as_the_likeliest_target():
    game, dealer, ask = a_value_answer(gain=0.2)
    assert not dealer.promise(game, 2, D.Ask(0, D.NO_BLOCK, WOOD), ())     # seat 2 is not on the hex


def test_a_dealing_seat_answers_off_its_value_head():
    net = AnswerNet(0, 0.0)
    bot = D.Dealing(net, D.DealParams(answer=True, answerer="value"))
    assert isinstance(bot.deal_answerer, D.ValueDealer) and bot.deal_answerer.net is net
    assert bot.deal_dealer is None
    both = D.Dealing(net, D.DealParams(answer=True, answerer="value", ask=True, mover="value"))
    assert both.deal_answerer is both.deal_dealer
    p = D.DealParams.parse("answer,answerer=value,model=reading")
    assert p.answerer == D.VALUE_ANSWERER and D.DealParams.parse(p.spec()) == p


def test_the_value_answer_reads_nothing_hidden_whatever_the_phase():
    """Asked before the roll or during discards, on the answerer's own game
    (the mover's hand and cards hidden), it prices as at the robber move."""
    from hexset.game import observe

    game, dealer, ask = a_value_answer(gain=-0.05)
    deal(game, 0, WHEAT)
    game._state.dev_cards[0][DevCard.KNIGHT] += 1
    want = dealer.promise_worth(game, 1, ask, ())
    for phase in (Phase.ROLL, Phase.DISCARD, Phase.ROBBER):
        game.phase = phase
        assert dealer.promise_worth(observe(game, 1), 1, ask, ()) == pytest.approx(want)


# --- trade embargoes --------------------------------------------------------


def test_closed_reads_an_embargo_either_way():
    game = a_table()
    game.embargoes = frozenset({(1, 0), (2, 3)})
    assert D.closed(game, 0) == {1} and D.closed(game, 1) == {0}
    assert D.closed(game, 2) == {3} and D.closed(game, 3) == {2}


@pytest.mark.parametrize("pair", [(1, 0), (0, 1)])
def test_a_seat_under_an_embargo_with_the_mover_promises_nothing(pair):
    game = a_table()
    deal(game, 0, ORE)
    for s in (1, 2):
        deal(game, s, WOOD)
    game.embargoes = frozenset({pair})
    seat(game, [Gate(ASKER, ask=D.Ask(0, D.NO_BLOCK, WOOD), accept=lambda p: p),
                Gate(ANSWERS, answer=True), Gate(ANSWERS, answer=True), Gate()])
    roll_dice(game, 7)
    (d,) = game.deal_log
    assert d.promised == (2,) and d.accepted == (2,) and game.robber_promised == ((2, D.NO_BLOCK),)


def test_the_plain_mover_asks_nobody_it_is_under_an_embargo_with():
    game = a_table()
    deal(game, 0, ORE)
    gate = Gate(ASKER)
    assert D.default_ask(game, 0, gate, ASKER) is not None
    game.embargoes = frozenset({(1, 0), (0, 2), (3, 0)})
    assert D.default_ask(game, 0, gate, ASKER) is None


@pytest.mark.parametrize("pair", [(1, 0), (0, 1)])
def test_the_value_mover_neither_asks_for_nor_spares_a_seat_under_an_embargo(pair):
    game, dealer = a_value_mover(gain=0.2)
    game.embargoes = frozenset({pair})
    assert dealer.ask(game, 0) is None
    assert dealer.accept(game, 0, D.Ask(0, D.NO_BLOCK, WOOD), (1,)) == ()


def test_a_promise_from_a_seat_under_an_embargo_is_ignored(monkeypatch):
    game = a_table()
    deal(game, 0, ORE)
    deal(game, 1, WOOD)
    game.embargoes = frozenset({(1, 0)})
    # Seat 1's promise reaching the mover all the same, as a client might hand it in.
    monkeypatch.setattr(D, "answer", lambda game, s, *a, **k: s == 1)
    seat(game, [Gate(ASKER, ask=D.Ask(0, D.NO_BLOCK, WOOD), accept=lambda p: p),
                Gate(ANSWERS, answer=True), Gate(), Gate()])
    roll_dice(game, 7)
    d = game.deal_log[-1]
    assert d.promised == (1,) and d.ignored == (1,) and d.accepted == () and game.robber_promised == ()
