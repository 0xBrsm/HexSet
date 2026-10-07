# SPDX-License-Identifier: GPL-3.0-only
"""`hexset.bots.Coalition`: any bot seated to play with its partners against
the rest of the table, and the `coalition:<entrant>` spelling that seats it."""
from __future__ import annotations

import pickle
import random

import pytest

from hexset.actions import ActionType as A, apply, legal_actions
from hexset.arena import (
    Entrant, RetiredSeat, compete, deal_board, deal_game, entrant_from_name, lineup_from_names,
    register_entrant_kind, register_preset, register_spec, spawn, unregister_preset,
)
from hexset.bots import Coalition, RandomBot, TradesBy, play_against, targets_at
from hexset.economy import hand_size
from hexset.game import is_over, to_move
from hexset.robber import occupants
from hexset.trading import RESPONSE_ACCEPT, RESPONSE_PASS, Offer, Response
from traders import ScarcityTrader

TARGET = 0


class Told:
    """A bot that only records what `play_against` told it."""

    def __init__(self):
        self.told = []

    def choose(self, game):  # pragma: no cover - never seated to move
        raise AssertionError("never asked to move")

    def play_against(self, seats):
        self.told.append(frozenset(seats))


class Hooks:
    """A bot with every trade hook, each recording what it was asked and
    answering for everyone it was offered."""

    trade_floor = 0.0

    def __init__(self):
        self.asked = {}

    def choose(self, game):  # pragma: no cover - never seated to move
        raise AssertionError("never asked to move")

    def _ask(self, name, *args):
        self.asked[name] = args
        return args

    def candidates(self, view, counterparties, **kwargs):
        self._ask("candidates", counterparties)
        return [(c, (1, -1, 0, 0, 0)) for c in (0, 1, 2, 3)]

    def offer(self, view, candidates):
        self._ask("offer", candidates)
        return len(candidates) - 1

    def respond(self, view, offer):
        self._ask("respond", offer)
        return Response(view.perspective, RESPONSE_ACCEPT, None)

    respond_any = respond

    def pick(self, view, responses):
        self._ask("pick", responses)
        return 0

    def gains_many(self, view, received, counterparties):
        self._ask("gains_many", received, counterparties)
        return [1.0] * len(received)

    def accepts_many(self, view, received, counterparties):
        self._ask("accepts_many", received, counterparties)
        return [True] * len(received)

    def accepts(self, view, received, counterparty):
        return True

    def consent_gain(self, view, received, counterparty, **kwargs):
        return 1.0


def _table(seed: int, target_seat: int = TARGET, member=lambda s: ScarcityTrader(random.Random(s))):
    board = deal_board(seed, 0)
    game = deal_game(seed, 0, 4, board=board)
    bots = [Coalition(member(s)) for s in range(4)]
    bots[target_seat] = ScarcityTrader(random.Random(99))
    game.gates, game.trade_mode = tuple(bots), "round"   # what the arena seats
    return game, bots


def _play(game, bots, watch=None, cap: int = 6000):
    """`hexset.arena`'s loop, with `watch(game, seat, action)` before each
    step; the trades it cleared (`Game.trades` holds only the turn's)."""
    cleared = []
    for _ in range(cap):
        if is_over(game):
            break
        seat = to_move(game)
        action = bots[seat].choose(game)
        if watch is not None:
            watch(game, seat, action)
        before = len(game.trades)
        apply(game, action)
        cleared += game.trades[before:]
    return cleared


def _robber_watch(target, moves):
    def watch(game, seat, action):
        if action.type is A.MOVE_ROBBER and seat != target:
            st = game.state(seat).state
            possible = any(target in occupants(st, a.a) for a in legal_actions(game, seat))
            # A steal from the target where it holds a card to steal.
            steal = action.b == target or not hand_size(st, target)
            moves.append((possible, target in occupants(st, action.a), steal))
    return watch


def test_the_targets_are_the_seats_not_in_the_coalition():
    c = lambda: Coalition(RandomBot())
    assert targets_at([RandomBot(), c(), c(), c()]) == {0}
    assert targets_at([c(), c(), RandomBot(), c()]) == {2}
    assert targets_at([c(), RetiredSeat(), RandomBot(), c()]) == {2}


def test_a_seat_off_a_table_or_unseated_refuses_to_guess():
    game, bots = _table(3)
    game.gates = None
    with pytest.raises(ValueError):
        bots[1].choose(game)
    with pytest.raises(RuntimeError):
        Coalition(ScarcityTrader()).gains_many(game.state(1), [(1, -1, 0, 0, 0)], [0])


@pytest.mark.parametrize("seed,target", [(11, 0), (12, 2), (13, 3)])
def test_no_trade_ever_clears_with_the_target_and_the_robber_goes_on_it(seed, target):
    game, bots = _table(seed, target)
    moves = []
    cleared = _play(game, bots, _robber_watch(target, moves))
    assert cleared, "the members trade among themselves"
    assert all(target not in (t.a, t.b) for t in cleared)
    assert moves, "a member moved the robber"
    assert all(hit and steal for possible, hit, steal in moves if possible)


def test_a_bot_with_no_hooks_still_robs_the_target():
    """The table layer alone: `RandomBot` has no `play_against` and no trade
    hooks, and its robber still goes on the target every time it can."""
    game, bots = _table(21, member=lambda s: RandomBot(random.Random(s)))
    moves = []
    _play(game, bots, _robber_watch(TARGET, moves))
    assert any(possible for possible, _, _ in moves)
    assert all(hit for possible, hit, _ in moves if possible)


def test_every_trade_hook_takes_the_target_out():
    """Each hook the bot has is asked about partners only, and answers for
    the target are refusals; a hook it lacks stays absent."""
    game, _ = _table(5)
    hooks = Hooks()
    member = Coalition(hooks)
    game.gates = (RandomBot(), member, Coalition(RandomBot()), Coalition(RandomBot()))
    member.seat_at(game)
    view = game.state(1)
    bundle = (1, -1, 0, 0, 0)

    assert member.candidates(view, [0, 2, 3], turn=0) == [(1, bundle), (2, bundle), (3, bundle)]
    assert hooks.asked["candidates"] == ([2, 3],)
    assert member.offer(view, [(0, bundle), (2, bundle)]) == 1
    assert hooks.asked["offer"] == ([(2, bundle)],)
    assert member.offer(view, [(0, bundle)]) is None
    for respond in (member.respond, member.respond_any):
        assert respond(view, Offer(actor=TARGET, received=bundle)).kind == RESPONSE_PASS
        assert respond(view, Offer(actor=2, received=bundle)).kind == RESPONSE_ACCEPT
    answers = [Response(TARGET, RESPONSE_ACCEPT, None), Response(3, RESPONSE_ACCEPT, None)]
    assert member.pick(view, answers) == 1
    assert member.pick(view, answers[:1]) is None
    assert member.gains_many(view, [bundle, bundle], [TARGET, 2]) == [-1.0, 1.0]
    assert hooks.asked["gains_many"] == ([bundle], [2])
    assert member.accepts_many(view, [bundle, bundle], [2, TARGET]) == [True, False]
    assert member.accepts(view, bundle, TARGET) is False and member.accepts(view, bundle, 2) is True
    assert member.consent_gain(view, bundle, TARGET) == -1.0 and member.consent_gain(view, bundle, 2) == 1.0

    bare = Coalition(RandomBot())
    assert not hasattr(bare, "respond") and not hasattr(bare, "gains_many")


def test_a_member_prices_the_target_out_of_a_real_gate():
    game, bots = _table(5)
    _play(game, bots, cap=400)     # well past setup, some cards in hand
    member, view = bots[1], game.state(1)
    bundle = (-1, 1, 0, 0, 0)
    gains = member.gains_many(view, [bundle, bundle], [TARGET, 2])
    assert gains[0] < 0 and gains[1] == member.bot.gains_many(view, [bundle], [2])[0]


def test_every_part_that_plays_against_is_told_the_targets_once():
    """Both halves of a `TradesBy` seat are told; a part with no hook, or
    with it set to `None`, is skipped; a table seen again tells no one."""
    board = deal_board(4, 0)
    mover, trader = Told(), Told()
    game = deal_game(4, 0, 4, board=board)
    game.gates = (RandomBot(), Coalition(TradesBy(mover, trader)), Coalition(RandomBot()),
                  Coalition(RandomBot()))
    game.gates[1].seat_at(game)
    game.gates[1].seat_at(game)
    assert mover.told == trader.told == [{0}]
    assert game.gates[1].targets == {0}

    off = Told()
    off.play_against = None
    play_against(TradesBy(off, RandomBot()), {2})


def test_a_coalition_seat_pickles_and_writes_through_to_its_bot():
    seat = Coalition(ScarcityTrader(random.Random(1)))
    seat.trade_floor = 0.5
    assert seat.bot.trade_floor == 0.5
    again = pickle.loads(pickle.dumps(seat))
    assert isinstance(again, Coalition) and again.targets is None
    assert again.bot.trade_floor == 0.5


def test_the_spec_names_any_entrant_and_plays_the_arena_rotation():
    """`coalition:<entrant>` seats any entrant, a shipped one included; a
    1-vs-3 lineup plays in the arena, the target in a different seat each
    game, and no trade touches it."""
    entrant = entrant_from_name("coalition:random")
    assert (entrant.name, entrant.kind, entrant.option("inner")) == ("coalition:random", "coalition", "random")
    assert pickle.loads(pickle.dumps(entrant)) == entrant
    assert isinstance(spawn(entrant, deal_board(1, 0), random.Random(0)).bot, RandomBot)

    entrants = lineup_from_names(["test-trader", "coalition:test-trader", "coalition:test-trader",
                                  "coalition:random"])
    t = compete(entrants, 4, seed=404, records=True, action_cap=1500, turn_cap=1000)
    assert len({seating[0] for seating in t.seating}) > 1
    assert any(t.cleared)
    for seating, cleared in zip(t.seating, t.cleared):
        assert all(seating[0] not in (trade.a, trade.b) for trade in cleared)


def test_the_spec_refuses_an_unknown_entrant_and_a_trader():
    with pytest.raises(ValueError, match="unknown bot"):
        lineup_from_names(["coalition:nobody"])
    with pytest.raises(ValueError, match="name the entrant"):
        entrant_from_name("coalition:")
    with pytest.raises(ValueError, match="trades through its own bot"):
        lineup_from_names(["coalition:random~test-trader"])
    built = Entrant("c", kind="coalition", options=(("inner", "random"),), trader="test-trader")
    with pytest.raises(ValueError, match="trades through its own bot"):
        spawn(built, deal_board(1, 0), random.Random(0))


def test_the_kind_and_prefix_are_shipped_and_the_preset_name_is_free():
    """A runtime cannot register over the `coalition` kind or the
    `coalition:` prefix, and may name a preset `coalition` for a coalition of
    its own bot."""
    with pytest.raises(ValueError, match="ships"):
        register_entrant_kind("coalition", lambda entrant, board, rng: None)
    with pytest.raises(ValueError, match="parses itself"):
        register_spec("coalition:", entrant_from_name)
    from hexset.arena import coalition

    register_preset("coalition", coalition("test-trader", "coalition"))
    try:
        entrant = entrant_from_name("coalition")
        assert (entrant.name, entrant.kind, entrant.option("inner")) == ("coalition", "coalition", "test-trader")
        assert isinstance(spawn(entrant, deal_board(1, 0), random.Random(0)).bot, ScarcityTrader)
    finally:
        unregister_preset("coalition")
