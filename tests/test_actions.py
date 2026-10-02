# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import random

import pytest

from hexset.actions import (
    Action,
    ActionType,
    apply,
    build_space,
    legal_actions,
    legal_mask,
    space_for,
)
from hexset.board.board import random_base_board
from hexset.board.terrain import NUM_RESOURCES, Resource
from hexset.economy import expected_total, total_in_play
from hexset.game import (
    Phase,
    is_over,
    may_act,
    pending_free_roads,
    players_owing_discards,
    start,
    to_move,
)
from hexset.play import play_random_game, step_randomly
from hexset.state import MAX_ROADS, NO_OWNER, road_count
from hexset.victory import WINNING_POINTS, victory_points


def a_game(players: int = 4, seed: int = 0):
    rng = random.Random(seed)
    return start(random_base_board(rng), players, rng)


def test_the_flat_action_space_size_is_pinned():
    """Pinned so a change to the layout has to own the new number deliberately."""
    space = build_space(54, 72, 19, 4)
    assert space.size == 456
    assert space.sizes[ActionType.PLAY_KNIGHT] == 1


def test_setup_offers_only_setup_actions():
    game = a_game()
    assert {a.type for a in legal_actions(game)} == {ActionType.SETUP_SETTLEMENT}

    apply(game, legal_actions(game)[0])
    assert {a.type for a in legal_actions(game)} == {ActionType.SETUP_ROAD}


def test_the_mask_agrees_with_the_action_list():
    game = a_game()
    rng = random.Random(1)
    space = space_for(game)
    for _ in range(200):
        if is_over(game):
            break
        mask = legal_mask(game, space)
        expected = {space.index(a) for a in legal_actions(game)}
        assert {i for i, ok in enumerate(mask) if ok} == expected
        step_randomly(game, rng)


def test_a_finished_game_offers_nothing():
    game = play_random_game(num_players=3, rng=random.Random(5))
    assert is_over(game)
    assert legal_actions(game) == []


def test_a_random_game_finishes_with_a_legal_winner():
    game = play_random_game(num_players=4, rng=random.Random(0))

    assert game.won_by is not None
    assert victory_points(game._state, game.won_by) >= WINNING_POINTS


def test_resources_survive_a_whole_game():
    game = play_random_game(num_players=4, rng=random.Random(1))
    assert total_in_play(game._state) == expected_total()
    assert all(n >= 0 for n in game._state.bank)
    assert all(n >= 0 for hand in game._state.hands for n in hand)


def test_every_action_survives_a_round_trip_through_its_index():
    """Trading is an engine event rather than an action, so no slot is lossy."""
    space = space_for(a_game())
    for index in range(space.size):
        assert space.index(space.decode(index)) == index


def test_free_roads_come_before_everything_else_in_main():
    game = a_game(players=2)
    rng = random.Random(2)
    while game.phase is not Phase.MAIN:
        step_randomly(game, rng)

    game.free_roads = 2
    kinds = {a.type for a in legal_actions(game)}
    # Asserted, not guarded on: `if BUILD_ROAD in kinds` would make the check
    # conditional on the fixture. The stranded case has its own test below.
    assert pending_free_roads(game), "fixture left no road placeable: tests nothing"
    assert kinds == {ActionType.BUILD_ROAD}


def test_a_seat_out_of_roads_owes_nothing_placeable():
    """`pending_free_roads` hoists the piece-limit read out of `road_placeable`.
    Get it wrong and, because owed roads suppress everything else in MAIN, the
    turn has no legal move at all.
    """
    game = a_game(players=2)
    rng = random.Random(3)
    while game.phase is not Phase.MAIN:
        step_randomly(game, rng)

    state = game._state
    seat = game.current_player
    for edge in range(state.board.topology.num_edges):
        if road_count(state, seat) >= MAX_ROADS:
            break
        if state.edge_owner[edge] == NO_OWNER:
            state.edge_owner[edge] = seat
    assert road_count(state, seat) == MAX_ROADS

    game.free_roads = 2
    assert pending_free_roads(game) == []
    kinds = {a.type for a in legal_actions(game)}
    assert ActionType.BUILD_ROAD not in kinds
    assert ActionType.END_TURN in kinds


#
# Not a turn: every seat over the limit discards at the same instant, bounded
# only by its own hand and quota. `to_move` names one of them for callers that
# want a single actor; the seat argument is what a live table uses instead.


def a_game_owing(seed: int = 7):
    """Seats 0 and 3 owing two cards each; seat 1, who rolled, owing none."""
    game = a_game(seed=seed)
    game.phase = Phase.DISCARD
    game.current_player = 1
    for seat in range(4):
        game._state.hands[seat] = [0] * NUM_RESOURCES
    game._state.hands[0] = [4, 0, 0, 0, 0]
    game._state.hands[3] = [0, 0, 0, 0, 4]
    game.discard_quota = [2, 0, 0, 2]
    return game


def test_every_owing_seat_may_discard_not_just_the_lowest():
    game = a_game_owing()
    assert players_owing_discards(game) == [0, 3]
    assert may_act(game, 0) and may_act(game, 3)
    assert not may_act(game, 1) and not may_act(game, 2)
    assert to_move(game) == 0


def test_a_seat_is_offered_its_own_cards_not_the_lowest_owing_seats():
    game = a_game_owing()
    assert [a.a for a in legal_actions(game, 3)] == [Resource.ORE]
    assert [a.a for a in legal_actions(game, 0)] == [Resource.WOOD]
    assert [a.a for a in legal_actions(game)] == [Resource.WOOD]
    assert legal_actions(game, 1) == []


def test_a_discard_round_is_order_invariant():
    def discard(game, seat):
        apply(game, Action(ActionType.DISCARD, legal_actions(game, seat)[0].a), seat=seat)

    interleaved = a_game_owing()
    for seat in (3, 0, 3, 0):
        discard(interleaved, seat)

    serialized = a_game_owing()
    for seat in (0, 0, 3, 3):
        discard(serialized, seat)

    assert interleaved._state.hands == serialized._state.hands
    assert interleaved._state.bank == serialized._state.bank
    assert interleaved.discard_quota == serialized.discard_quota == [0, 0, 0, 0]
    assert interleaved.phase is serialized.phase is Phase.ROBBER
    assert interleaved.current_player == 1


def test_the_mask_answers_for_the_seat_it_is_asked_about():
    game = a_game_owing()
    space = space_for(game)
    for seat in (0, 3):
        mask = legal_mask(game, space, seat)
        expected = {space.index(a) for a in legal_actions(game, seat)}
        assert {i for i, ok in enumerate(mask) if ok} == expected
    assert legal_mask(game, space) == legal_mask(game, space, 0)


def test_a_table_can_refuse_road_building_on_the_last_road_piece():
    """`Rules.road_building_min_roads`: one piece left plays the card under
    the printed rule and does not under a table asking for two -- offered
    and applied alike."""
    import dataclasses
    import random

    import pytest

    from hexset.actions import Action, ActionType, apply, legal_actions
    from hexset.board.board import random_base_board
    from hexset.cards import DevCard
    from hexset.game import Phase, start
    from hexset.rules import Rules
    from hexset.state import MAX_ROADS, NO_OWNER

    rng = random.Random(3)
    game = start(random_base_board(rng), 2, rng)
    state = game._state
    seat = game.current_player
    game.phase = Phase.MAIN
    state.dev_cards[seat][DevCard.ROAD_BUILDING] = 1
    free = [e for e in range(state.board.topology.num_edges) if state.edge_owner[e] == NO_OWNER]
    owned = sum(1 for e in range(state.board.topology.num_edges) if state.edge_owner[e] == seat)
    for e in free[: MAX_ROADS - owned - 1]:
        state.edge_owner[e] = seat
    card = Action(ActionType.PLAY_ROAD_BUILDING)
    assert card in legal_actions(game)                       # printed rule: one piece is enough

    state.rules = dataclasses.replace(state.rules, road_building_min_roads=2)
    assert card not in legal_actions(game)
    with pytest.raises(ValueError, match="road pieces left"):
        apply(game, card)
    with pytest.raises(ValueError):
        Rules(road_building_min_roads=3)
