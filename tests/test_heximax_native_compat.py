import random

from hexset.actions import ActionType, apply, legal_actions
from hexset.arena import deal_game, entrant_from_name, spawn
from hexset.game import Phase


def _position_with_road_and_trade():
    for seed in range(100):
        game = deal_game(seed, 0, 4)
        rng = random.Random(seed + 99)
        for _ in range(200):
            if game.phase is Phase.MAIN:
                options = legal_actions(game)
                if (any(a.type is ActionType.BUILD_ROAD for a in options)
                        and any(a.type is ActionType.BANK_TRADE for a in options)):
                    return game
            options = legal_actions(game)
            if not options:
                break
            apply(game, rng.choice(options))
    raise AssertionError("no suitable deterministic HexSet position")


def test_native_compat_filters_only_remaining_free_roads_and_then_releases_main():
    game = _position_with_road_and_trade()
    bot = spawn(entrant_from_name("heximax-notrade"), game._state.board, random.Random(3))
    bot.native_action_compat = True

    game.free_roads = 2
    assert all(a.type is ActionType.BUILD_ROAD for a in bot._options_in(game, 0))
    game.free_roads = 1
    road = next(a for a in legal_actions(game) if a.type is ActionType.BUILD_ROAD)
    apply(game, road)
    assert game.free_roads == 0
    assert any(a.type is ActionType.BANK_TRADE for a in bot._options_in(game, 0))


def test_native_compat_does_not_change_pre_roll_or_default_legality():
    game = _position_with_road_and_trade()
    ordinary = spawn(entrant_from_name("heximax-notrade"), game._state.board, random.Random(4))
    expected = legal_actions(game)
    assert ordinary._options_in(game, 0) == expected

    game.phase = Phase.ROLL
    game.free_roads = 0
    native = spawn(entrant_from_name("heximax-notrade"), game._state.board, random.Random(5))
    native.native_action_compat = True
    options = native._options_in(game, 0)
    assert any(a.type is ActionType.ROLL for a in options)


def test_native_compat_has_no_nonroad_fallback_when_no_free_road_can_be_placed():
    game = _position_with_road_and_trade()
    game._state.edge_owner = [game.current_player] * game._state.board.topology.num_edges
    game.free_roads = 1
    bot = spawn(entrant_from_name("heximax-notrade"), game._state.board, random.Random(6))
    bot.native_action_compat = True
    assert not any(a.type is ActionType.BUILD_ROAD for a in legal_actions(game))
    assert bot._options_in(game, game.current_player) == []


def test_native_compat_root_does_not_reopen_main_actions_without_road_placement():
    game = _position_with_road_and_trade()
    game._state.edge_owner = [game.current_player] * game._state.board.topology.num_edges
    game.free_roads = 1
    bot = spawn(entrant_from_name("heximax-notrade"), game._state.board, random.Random(7))
    bot.native_action_compat = True
    import pytest
    with pytest.raises(ValueError, match="no legal road placement"):
        bot.root_options(game)
