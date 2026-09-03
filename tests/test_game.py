# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import random

import pytest
from helpers import clear_hand, give

from hexset.board.board import random_base_board
from hexset.board.terrain import Resource
from hexset.cards import DevCard
from hexset.economy import COSTS, Purchase, expected_total, total_in_play
from hexset.game import (
    Phase,
    build_city,
    build_road,
    buy_development_card,
    end_turn,
    legal_initial_roads,
    move_robber_to,
    place_initial_road,
    place_initial_settlement,
    play_knight_card,
    play_monopoly_card,
    play_road_building_card,
    play_year_of_plenty_card,
    players_owing_discards,
    roll_dice,
    start,
    submit_discard,
    trade_with_bank,
)
from hexset.state import NO_OWNER, can_place_settlement


def a_game(players: int = 3, seed: int = 0):
    return start(random_base_board(random.Random(seed)), players, random.Random(seed))


def free_vertex(game):
    return next(
        v
        for v in range(game._state.board.topology.num_vertices)
        if can_place_settlement(game._state, game.current_player, v, connected=False)
    )


def run_setup(game):
    while game.phase in (Phase.SETUP_SETTLEMENT, Phase.SETUP_ROAD):
        if game.phase is Phase.SETUP_SETTLEMENT:
            place_initial_settlement(game, free_vertex(game))
        else:
            place_initial_road(game, legal_initial_roads(game)[0])
    return game


def fund(state, player, purchase):
    for resource, count in enumerate(COSTS[purchase]):
        give(state, player, resource, count)


def test_setup_uses_snake_order():
    game = a_game(players=3)
    assert game.setup_queue == [0, 1, 2, 2, 1, 0]


def test_setup_places_two_settlements_and_roads_each():
    game = run_setup(a_game(players=3))

    assert game.phase is Phase.ROLL
    assert game.current_player == 0
    for player in range(3):
        assert game._state.vertex_owner.count(player) == 2
        assert game._state.edge_owner.count(player) == 2


def test_only_the_first_round_is_unpaid():
    game = a_game(players=2)
    place_initial_settlement(game, free_vertex(game))
    assert game._state.hands[0] == [0] * 5

    run_setup(game)
    # Every player is given the yield of their second settlement.
    assert any(sum(hand) > 0 for hand in game._state.hands)
    assert total_in_play(game._state) == expected_total()


def test_opening_road_must_touch_the_new_settlement():
    game = a_game()
    place_initial_settlement(game, free_vertex(game))
    illegal = next(
        e
        for e in range(game._state.board.topology.num_edges)
        if e not in legal_initial_roads(game)
    )
    with pytest.raises(ValueError):
        place_initial_road(game, illegal)


def test_actions_are_rejected_in_the_wrong_phase():
    game = a_game()
    with pytest.raises(ValueError):
        roll_dice(game)
    with pytest.raises(ValueError):
        end_turn(game)


def test_rolling_seven_goes_to_the_robber():
    game = run_setup(a_game())
    game.rng = random.Random()
    while True:
        roll = roll_dice(game)
        if roll == 7:
            break
        game.phase = Phase.ROLL
    assert game.phase in (Phase.DISCARD, Phase.ROBBER)


def test_a_big_hand_must_discard_on_seven():
    game = run_setup(a_game())
    clear_hand(game._state, 0)
    for resource in Resource:
        give(game._state, 0, resource, 2)
    assert sum(game._state.hands[0]) == 10

    game.last_roll = 7
    game.phase = Phase.ROLL
    game.rng = random.Random(1)
    while roll_dice(game) != 7:
        game.phase = Phase.ROLL

    assert game.phase is Phase.DISCARD
    assert 0 in players_owing_discards(game)
    assert game.discard_quota[0] == 5

    submit_discard(game, 0, [1, 1, 1, 1, 1])
    assert sum(game._state.hands[0]) == 5
    assert game.discard_quota[0] == 0


def test_robber_moves_then_play_resumes():
    game = run_setup(a_game())
    game.phase = Phase.ROBBER
    target = (game._state.robber + 1) % game._state.board.num_hexes

    move_robber_to(game, target)

    assert game._state.robber == target
    assert game.phase is Phase.MAIN


def test_building_costs_resources_and_advances_the_road():
    game = run_setup(a_game())
    game.phase = Phase.MAIN
    clear_hand(game._state, 0)
    fund(game._state, 0, Purchase.ROAD)
    topology = game._state.board.topology
    mine = game._state.edge_owner.index(0)
    junction = topology.edges[mine][0]
    edge = next(
        e
        for e in topology.vertex_edges[junction]
        if game._state.edge_owner[e] == NO_OWNER
    )

    build_road(game, edge)

    assert game._state.edge_owner[edge] == 0
    assert game._state.hands[0] == [0] * 5
    assert total_in_play(game._state) == expected_total()


def test_only_one_development_card_per_turn():
    game = run_setup(a_game())
    game.phase = Phase.MAIN
    game._state.dev_cards[0][DevCard.KNIGHT] = 2

    play_knight_card(game, (game._state.robber + 1) % game._state.board.num_hexes)
    with pytest.raises(ValueError):
        play_knight_card(game, (game._state.robber + 2) % game._state.board.num_hexes)


def test_ending_a_turn_matures_cards_and_passes_play():
    game = run_setup(a_game(players=3))
    game.phase = Phase.MAIN
    game._state.new_dev_cards[0][DevCard.MONOPOLY] = 1

    end_turn(game)

    assert game._state.dev_cards[0][DevCard.MONOPOLY] == 1
    assert game._state.new_dev_cards[0][DevCard.MONOPOLY] == 0
    assert game.current_player == 1
    assert game.phase is Phase.ROLL


def test_the_card_allowance_resets_each_turn():
    game = run_setup(a_game(players=2))
    game.phase = Phase.MAIN
    game._state.dev_cards[0][DevCard.MONOPOLY] = 1
    play_monopoly_card(game, Resource.ORE)
    assert game.dev_card_played

    end_turn(game)
    assert not game.dev_card_played


def test_reaching_ten_points_ends_the_game():
    game = run_setup(a_game(players=2))
    game.phase = Phase.MAIN
    fund(game._state, 0, Purchase.CITY)

    # Two opening settlements plus seven cards stands the player at nine, so
    # upgrading one of them is the winning point.
    game._state.dev_cards[0][DevCard.VICTORY_POINT] = 7
    settlement = game._state.vertex_owner.index(0)

    build_city(game, settlement)

    assert game.phase is Phase.GAME_OVER
    assert game.won_by == 0


# --- trade/build interleaving (owner review against the rulebook, 2026-09-03) -


def _spy_on_trade_event(monkeypatch):
    """Count calls to `trade_event` as `hexset.game` itself sees it -- the
    name `run_trade_event` calls -- recording the phase at each call. A
    patch on `hexset.trading.trade_event` would not be seen here: `game.py`
    imported the name directly (`from .trading import ... trade_event`), so
    it holds its own reference, same as any other `from x import y`."""
    import hexset.game as gamemod

    calls: list[Phase] = []
    real = gamemod.trade_event

    def spy(game, gate):
        calls.append(game.phase)
        return real(game, gate)

    monkeypatch.setattr(gamemod, "trade_event", spy)
    return calls


def _seated(game):
    """The minimum for `run_trade_event` to reach `trade_event` rather than
    short-circuiting on `gates is None`. An object with neither `valuation`
    nor `accepts` trades nothing -- these tests count the *call*, not
    whether anything clears."""
    n = game._state.num_players
    game.gates = tuple(object() for _ in range(n))
    return game


@pytest.mark.parametrize(
    "setup, act",
    [
        (
            lambda g: (clear_hand(g._state, 0), fund(g._state, 0, Purchase.ROAD)),
            lambda g: build_road(
                g,
                next(
                    e
                    for e in g._state.board.topology.vertex_edges[
                        g._state.board.topology.edges[g._state.edge_owner.index(0)][0]
                    ]
                    if g._state.edge_owner[e] == NO_OWNER
                ),
            ),
        ),
        (
            lambda g: fund(g._state, 0, Purchase.CITY),
            lambda g: build_city(g, g._state.vertex_owner.index(0)),
        ),
        (
            lambda g: fund(g._state, 0, Purchase.DEV_CARD),
            lambda g: buy_development_card(g),
        ),
        (
            lambda g: give(g._state, 0, Resource.WOOD, 4),
            lambda g: trade_with_bank(g, Resource.WOOD, Resource.ORE),
        ),
        (
            lambda g: g._state.dev_cards[0].__setitem__(DevCard.MONOPOLY, 1),
            lambda g: play_monopoly_card(g, Resource.ORE),
        ),
        (
            lambda g: g._state.dev_cards[0].__setitem__(DevCard.ROAD_BUILDING, 1),
            lambda g: play_road_building_card(g),
        ),
        (
            lambda g: g._state.dev_cards[0].__setitem__(DevCard.YEAR_OF_PLENTY, 1),
            lambda g: play_year_of_plenty_card(g, [Resource.WOOD, Resource.BRICK]),
        ),
    ],
    ids=["build_road", "build_city", "buy_dev_card", "bank_trade", "monopoly", "road_building", "year_of_plenty"],
)
def test_trade_event_runs_again_after_every_main_action(monkeypatch, setup, act):
    game = _seated(run_setup(a_game(players=3)))
    game.phase = Phase.MAIN
    setup(game)
    calls = _spy_on_trade_event(monkeypatch)

    act(game)

    assert calls == [Phase.MAIN]


def test_trade_event_runs_again_after_a_knight_in_main_but_not_in_roll(monkeypatch):
    """`play_knight_card` is the one action legal in both `ROLL` and `MAIN`
    (playing it before the roll, to move the robber pre-emptively). The
    interleaving only ever applies to `MAIN`."""
    game = _seated(run_setup(a_game(players=3)))
    game._state.dev_cards[0][DevCard.KNIGHT] = 2
    target = (game._state.robber + 1) % game._state.board.num_hexes
    other_target = (game._state.robber + 2) % game._state.board.num_hexes

    game.phase = Phase.ROLL
    calls = _spy_on_trade_event(monkeypatch)
    play_knight_card(game, target)
    assert calls == []

    game.dev_card_played = False
    game.phase = Phase.MAIN
    play_knight_card(game, other_target)
    assert calls == [Phase.MAIN]


def test_trade_event_never_runs_after_end_turn(monkeypatch):
    game = _seated(run_setup(a_game(players=3)))
    game.phase = Phase.MAIN
    calls = _spy_on_trade_event(monkeypatch)

    end_turn(game)

    assert calls == []
    assert game.phase is Phase.ROLL


def test_trade_event_never_runs_during_setup(monkeypatch):
    game = _seated(a_game(players=3))
    calls = _spy_on_trade_event(monkeypatch)

    run_setup(game)

    assert calls == []
    assert game.phase is Phase.ROLL


def test_trade_event_never_runs_during_discard_resolution(monkeypatch):
    game = _seated(run_setup(a_game()))
    clear_hand(game._state, 0)
    for resource in Resource:
        give(game._state, 0, resource, 2)
    game.last_roll = 7
    game.phase = Phase.ROLL
    game.rng = random.Random(1)
    calls = _spy_on_trade_event(monkeypatch)

    while roll_dice(game) != 7:
        game.phase = Phase.ROLL
        calls.clear()  # only the seven that sticks matters
    assert game.phase is Phase.DISCARD
    assert calls == []  # rolling a 7 with discards owed never enters MAIN

    submit_discard(game, 0, [1, 1, 1, 1, 1])
    assert game.phase is Phase.ROBBER
    assert calls == []  # finishing discards moves to ROBBER, not MAIN

    move_robber_to(game, (game._state.robber + 1) % game._state.board.num_hexes)
    assert calls == [Phase.MAIN]  # the one legitimate trigger: entering MAIN
