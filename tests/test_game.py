# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import random

import pytest
from helpers import clear_hand, give, independent_vertices, mini_board

from hexset.actions import ActionType, legal_actions
from hexset.board.board import random_base_board
from hexset.board.terrain import Resource
from hexset.board.topology import coastal_rings
from hexset.cards import DevCard
from hexset.economy import COSTS, Purchase, expected_total, total_in_play
from hexset.game import (
    Phase,
    build_city,
    build_road,
    build_settlement,
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
from hexset.state import NO_OWNER, Building, can_place_settlement
from hexset.victory import WINNING_POINTS, update_longest_road, victory_points


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
    # `enter_main` only arms `event_pending` now -- it does not call
    # `trade_event` itself any more (the PI amendment "publish points and
    # the event trigger"). The event still runs exactly once, the first
    # time anything observes the game for the current player.
    assert game.phase is Phase.MAIN
    assert calls == []
    legal_actions(game)
    assert calls == [Phase.MAIN]  # the one legitimate trigger: entering MAIN


# --- every playable card before the roll, not only the knight (rulebook,
# --- Production Phase: "you may play one of them before rolling the dice") -


def test_road_building_is_legal_before_the_roll():
    game = run_setup(a_game(players=2))
    game._state.dev_cards[0][DevCard.ROAD_BUILDING] = 1

    play_road_building_card(game)

    assert game.phase is Phase.ROLL  # playing the card does not itself roll
    assert game.free_roads == 2
    assert game.dev_card_played


def test_monopoly_is_legal_before_the_roll():
    game = run_setup(a_game(players=3))
    for player in range(3):
        clear_hand(game._state, player)  # setup's own opening resources would confound the count
    game._state.dev_cards[0][DevCard.MONOPOLY] = 1
    give(game._state, 1, Resource.SHEEP, 2)

    taken = play_monopoly_card(game, Resource.SHEEP)

    assert taken == 2
    assert game.phase is Phase.ROLL
    assert game.dev_card_played


def test_year_of_plenty_is_legal_before_the_roll():
    game = run_setup(a_game(players=2))
    clear_hand(game._state, 0)  # setup's own opening resources would confound the count
    game._state.dev_cards[0][DevCard.YEAR_OF_PLENTY] = 1

    play_year_of_plenty_card(game, [Resource.ORE, Resource.ORE])

    assert game._state.hands[0][Resource.ORE] == 2
    assert game.phase is Phase.ROLL
    assert game.dev_card_played


def test_only_one_card_total_across_roll_and_main():
    """One card a turn is a turn-wide allowance, not one per phase: playing
    a knight before rolling must still block a second card afterwards."""
    game = run_setup(a_game(players=2))
    game._state.dev_cards[0][DevCard.KNIGHT] = 1
    game._state.dev_cards[0][DevCard.MONOPOLY] = 1

    play_knight_card(game, (game._state.robber + 1) % game._state.board.num_hexes)
    roll_dice(game, roll=6)  # deterministic: a producing roll, never the seven

    with pytest.raises(ValueError):
        play_monopoly_card(game, Resource.ORE)


def test_a_card_bought_this_turn_still_cannot_be_played_before_the_roll():
    """Relaxing the phase requirement to `ROLL` must not relax "not one
    built this turn": a card sitting in `new_dev_cards` (not yet matured)
    stays unplayable regardless of which phase asks."""
    game = run_setup(a_game(players=2))
    game.phase = Phase.ROLL
    game._state.new_dev_cards[0][DevCard.MONOPOLY] = 1

    with pytest.raises(ValueError):
        play_monopoly_card(game, Resource.ORE)


def test_legal_actions_before_the_roll_offer_every_playable_card():
    game = run_setup(a_game(players=2))
    game._state.dev_cards[0][DevCard.ROAD_BUILDING] = 1
    game._state.dev_cards[0][DevCard.MONOPOLY] = 1
    game._state.dev_cards[0][DevCard.YEAR_OF_PLENTY] = 1

    kinds = {a.type for a in legal_actions(game)}

    assert ActionType.ROLL in kinds
    assert ActionType.PLAY_ROAD_BUILDING in kinds
    assert ActionType.PLAY_MONOPOLY in kinds
    assert ActionType.PLAY_YEAR_OF_PLENTY in kinds
    # Building, buying and trading stay Action-phase only.
    assert ActionType.BUILD_ROAD not in kinds
    assert ActionType.BUY_DEV_CARD not in kinds
    assert ActionType.BANK_TRADE not in kinds


# --- winning only on your own turn (rulebook, "Winning the Game": "If you
# --- have 10 or more VPs at any point during YOUR turn") ------------------


def test_a_tile_transfer_does_not_win_for_a_seat_off_the_move():
    """Player 0's own settlement breaks player 1's Longest Road and hands the
    tile to player 2, who is not on the move and was already sitting on 8
    visible VP from cities. Player 2 must not win here -- only player 0's
    own total is checked, since it is player 0's turn."""
    game = start(mini_board(), 3, random.Random(0))
    state = game._state
    topology = state.board.topology
    game.phase = Phase.MAIN
    game.current_player = 0

    # Player 1 holds Longest Road with a plain 5-road chain.
    ring = coastal_rings(topology)[0]
    p1_path = ring[0:5]
    for e in p1_path:
        state.edge_owner[e] = 1
    update_longest_road(state)
    assert state.longest_road_holder == 1

    # The junction where player 0 is about to settle sits inside that chain
    # and will split it into a 2-segment and a 3-segment piece.
    shared = set(topology.edges[p1_path[1]]) & set(topology.edges[p1_path[2]])
    break_vertex = shared.pop()

    # Player 2's own, disjoint 6-road chain is already longer than either
    # half of player 1's chain, so it takes the tile the instant it breaks.
    p2_path = ring[10:16]
    for e in p2_path:
        state.edge_owner[e] = 2
    cities = [v for v in independent_vertices(state.board, 6)
              if v not in topology.vertex_neighbors[break_vertex] and v != break_vertex]
    for v in cities[:4]:
        state.vertex_owner[v] = 2
        state.vertex_building[v] = Building.CITY
    assert victory_points(state, 2) == 8  # four cities, tile not yet transferred

    # Player 0 builds the breaking settlement for real, through the public
    # API, so `_check_win` runs exactly as it would in a played game.
    third_edge = next(e for e in topology.vertex_edges[break_vertex] if e not in p1_path)
    state.edge_owner[third_edge] = 0
    fund(state, 0, Purchase.SETTLEMENT)

    build_settlement(game, break_vertex)

    assert state.longest_road_holder == 2  # the tile did transfer
    assert victory_points(state, 2) >= WINNING_POINTS  # player 2 is over 10 on paper
    assert game.won_by is None  # but it is not their turn
    assert game.phase is Phase.MAIN  # so the game keeps going


def test_a_seat_that_crosses_ten_off_turn_wins_at_the_start_of_its_own_turn():
    """Winning the Game (rulebook): "the first player to reach 10 or more
    VPs on their turn wins" -- the FAQ reading is that this is announced at
    the *start* of a seat's own turn, before it takes any action, not only
    as a consequence of something it does itself. Player 1 crosses ten
    off-turn here (a Longest Road transfer during player 0's turn, same
    shape as the test above) and must not win then -- but must win the
    instant `end_turn` hands play to them, before `to_move`/`legal_actions`
    ever asks them for a move."""
    game = start(mini_board(), 3, random.Random(0))
    state = game._state
    topology = state.board.topology
    game.phase = Phase.MAIN
    game.current_player = 0

    # Player 2 holds Longest Road with a plain 5-road chain.
    ring = coastal_rings(topology)[0]
    p2_path = ring[0:5]
    for e in p2_path:
        state.edge_owner[e] = 2
    update_longest_road(state)
    assert state.longest_road_holder == 2

    # The junction where player 0 is about to settle sits inside that chain
    # and will split it into a 2-segment and a 3-segment piece.
    shared = set(topology.edges[p2_path[1]]) & set(topology.edges[p2_path[2]])
    break_vertex = shared.pop()

    # Player 1 -- who takes the very next turn once player 0 ends theirs --
    # holds a disjoint 6-road chain, already longer than either half of
    # player 2's chain once it breaks, and sits at 9 VP from buildings
    # alone (four cities and a settlement): one tile short of ten.
    p1_path = ring[10:16]
    for e in p1_path:
        state.edge_owner[e] = 1
    spots = [
        v
        for v in independent_vertices(state.board, 8)
        if v not in topology.vertex_neighbors[break_vertex] and v != break_vertex
    ]
    for v in spots[:4]:
        state.vertex_owner[v] = 1
        state.vertex_building[v] = Building.CITY
    state.vertex_owner[spots[4]] = 1
    state.vertex_building[spots[4]] = Building.SETTLEMENT
    assert victory_points(state, 1) == 9

    # Player 0 builds the breaking settlement for real.
    third_edge = next(e for e in topology.vertex_edges[break_vertex] if e not in p2_path)
    state.edge_owner[third_edge] = 0
    fund(state, 0, Purchase.SETTLEMENT)

    build_settlement(game, break_vertex)

    assert state.longest_road_holder == 1
    assert victory_points(state, 1) >= WINNING_POINTS
    assert game.won_by is None  # still player 0's turn
    assert game.phase is Phase.MAIN

    end_turn(game)

    assert game.current_player == 1  # turn order hands play straight to them
    assert game.won_by == 1  # ... and they win before taking any action
    assert game.phase is Phase.GAME_OVER
