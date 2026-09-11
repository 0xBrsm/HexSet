# SPDX-License-Identifier: GPL-3.0-only
import random

import pytest

from hexset.arena import PRESETS, spawn
from hexset.bots.evaluate import TERM_NAMES
from hexset.bots.heximax import BALANCED_WEIGHTS, NO_TRADE_WEIGHTS, TRADING_WEIGHTS, heximax
from hexset.bots.heximax.adaptive import TradeActivity, trading_profile
from hexset.game import Phase, imagine, run_trade_event
from hexset.trading import one_for_one
from test_heximax import after_setup, set_known_hand


def event(turn, traded, **overrides):
    return dict(turn=turn, actor=0, hand_sizes=(3, 3, 3, 3),
                trade_participants=((0, 1),) if traded else (), **overrides)


def test_slider_has_exact_current_endpoints_and_linear_intermediates():
    assert trading_profile(0) == (NO_TRADE_WEIGHTS, .25)
    assert trading_profile(1) == (BALANCED_WEIGHTS, .125)
    weights, expansion = trading_profile(.5)
    assert expansion == .1875
    for name in TERM_NAMES:
        assert getattr(weights, name) == (getattr(NO_TRADE_WEIGHTS, name) + getattr(BALANCED_WEIGHTS, name)) / 2
    for invalid in (-.01, 1.01, float('nan')):
        with pytest.raises(ValueError):
            trading_profile(invalid)


def test_activity_reaches_both_endpoints_and_recovers_in_both_directions():
    activity = TradeActivity()
    assert activity.mean == 0
    for turn in range(8):
        activity.observe(**event(turn, True))
        assert activity.mean == (turn + 1) / 8
    assert activity.mean == 1
    for turn in range(8, 16):
        activity.observe(**event(turn, False))
    assert activity.mean == 0
    for turn in range(16, 24):
        activity.observe(**event(turn, True))
    assert activity.mean == 1


def test_one_public_turn_is_one_vote_and_no_opportunity_is_not_a_vote():
    activity = TradeActivity()
    public = event(0, True)
    public['trade_participants'] *= 20
    activity.observe(**public)
    activity.observe(**event(0, False))
    assert activity.mean == 1 / 8
    for turn, sizes in [(1, (1, 3, 3, 3)), (2, (3, 1, 1, 1))]:
        public = event(turn, False); public['hand_sizes'] = sizes
        activity.observe(**public)
    assert activity.mean == 1 / 8


@pytest.mark.parametrize('successes', [0, 4, 8])
def test_adaptive_search_matches_the_selected_fixed_profile_and_trade_gate(successes):
    game = after_setup(26)
    game.phase = Phase.MAIN
    for seat in range(4):
        set_known_hand(game, seat, [2, 2, 2, 2, 2])
    board = game._state.board
    adaptive = spawn(PRESETS['heximax'], board, random.Random(81))
    for turn in range(8):
        adaptive.observe_trade(**event(turn, turn < successes))
    weights, expansion = trading_profile(successes / 8)
    fixed = heximax(board, random.Random(81), weights=weights, expansion_value=expansion,
                    trade_weights=TRADING_WEIGHTS, trade_floor=0)
    assert adaptive.choose(game) == fixed.choose(game)
    assert adaptive.evaluator.weights == weights
    assert adaptive.expansion_value == expansion
    assert adaptive.rng.getstate() == fixed.rng.getstate()
    received = [one_for_one(0, 4)]
    assert adaptive.gains_many(game.state(0), received, [1]) == fixed.gains_many(game.state(0), received, [1])
    assert adaptive.trade_floor == 0 and adaptive.max_trades is None
    game.max_trades = 0
    adaptive.choose(game)
    assert adaptive.evaluator.weights == NO_TRADE_WEIGHTS


def test_native_event_publishes_only_public_inputs_once_and_ignores_search_copies(monkeypatch):
    import hexset.game as engine
    game = after_setup(26)
    game.phase = Phase.MAIN
    for seat in range(4):
        set_known_hand(game, seat, [2, 2, 2, 2, 2])
    bot = heximax(game._state.board)
    game.gates = (bot,) * 4
    seen = []
    original = bot.observe_trade
    def observe(**public):
        seen.append(public)
        original(**public)
    monkeypatch.setattr(bot, 'observe_trade', observe)
    monkeypatch.setattr(engine, 'trade_event', lambda *args: ())
    run_trade_event(game)
    assert len(seen) == 4  # one callback per seated gate
    assert set(seen[0]) == {'turn', 'actor', 'hand_sizes', 'trade_participants'}
    assert seen[0]['hand_sizes'] == (10, 10, 10, 10)
    assert bot.activity.last_turn == game.turns
    run_trade_event(game)
    assert len(seen) == 4
    hypothetical = imagine(game, random.Random(4))
    hypothetical.turns += 1
    run_trade_event(hypothetical)
    assert len(seen) == 4


@pytest.mark.parametrize('pin', [0, 1])
def test_pin_holds_weights_and_expansion_despite_activity_and_trade_switches(pin):
    game = after_setup(26)
    game.phase = Phase.MAIN
    for seat in range(4):
        set_known_hand(game, seat, [2, 2, 2, 2, 2])
    weights, expansion = trading_profile(pin)
    bot = heximax(game._state.board, random.Random(81), pin_weights=pin)
    fixed = heximax(game._state.board, random.Random(81), weights=weights,
                    expansion_value=expansion)
    assert bot.max_trades is None  # pinning zero still permits exchanges
    for turn in range(8):
        bot.observe_trade(**event(turn, not bool(pin)))
    assert bot.choose(game) == fixed.choose(game)
    assert bot.evaluator.weights == weights
    assert bot.expansion_value == expansion
    assert bot.rng.getstate() == fixed.rng.getstate()
    received = [one_for_one(0, 4)]
    assert bot.gains_many(game.state(0), received, [1]) == fixed.gains_many(game.state(0), received, [1])
    for switch in ('game', 'bot'):
        game.max_trades = 0 if switch == 'game' else None
        bot.max_trades = 0 if switch == 'bot' else None
        bot.choose(game)
        assert bot.evaluator.weights == weights
        assert bot.expansion_value == expansion
    assert bot.gains_many(game.state(0), received, [1]) == [-1.0]


@pytest.mark.parametrize('pin', ['trade', 'notrade', .5, -1, 2, float('nan')])
def test_only_numeric_endpoint_pins_are_accepted(pin):
    with pytest.raises(ValueError, match='pin_weights'):
        heximax(after_setup(26)._state.board, pin_weights=pin)


def test_custom_experiments_cannot_silently_override_a_pin():
    board = after_setup(26)._state.board
    with pytest.raises(ValueError, match='custom weights'):
        heximax(board, weights=NO_TRADE_WEIGHTS, pin_weights=0)
    with pytest.raises(ValueError, match='endpoint bonuses'):
        heximax(board, pin_weights=1, expansion_value=.25)
