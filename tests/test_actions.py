# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import random

import pytest

from hexset.actions import (
    YEAR_OF_PLENTY_PAIRS,
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
    players_owing_discards,
    start,
    to_move,
)
from hexset.play import play_random_game, step_randomly
from hexset.victory import WINNING_POINTS, victory_points


def a_game(players: int = 4, seed: int = 0):
    rng = random.Random(seed)
    return start(random_base_board(rng), players, rng)


def test_space_size_is_the_sum_of_its_blocks():
    space = build_space(54, 72, 19, 4)
    assert space.size == sum(space.sizes)
    assert space.offsets[0] == 0


def test_every_index_round_trips():
    space = build_space(54, 72, 19, 4)
    for index in range(space.size):
        assert space.index(space.decode(index)) == index


def test_every_action_round_trips():
    space = build_space(54, 72, 19, 4)
    samples = [
        Action(ActionType.ROLL),
        Action(ActionType.END_TURN),
        Action(ActionType.BUILD_ROAD, 71),
        Action(ActionType.BUILD_SETTLEMENT, 53),
        Action(ActionType.BUILD_CITY, 0),
        Action(ActionType.MOVE_ROBBER, 18, 3),
        Action(ActionType.PLAY_KNIGHT),
        Action(ActionType.BANK_TRADE, 4, 0),
        Action(ActionType.PLAY_YEAR_OF_PLENTY, len(YEAR_OF_PLENTY_PAIRS) - 1),
        Action(ActionType.DISCARD, 2),
    ]
    for action in samples:
        assert space.decode(space.index(action)) == action


def test_the_space_grows_with_the_board():
    small = build_space(24, 30, 7, 4)
    large = build_space(54, 72, 19, 4)
    assert large.size > small.size


def test_more_players_widen_only_the_robber_block():
    """`PLAY_KNIGHT` no longer widens with the table: it carries no operand
    of its own any more (the knight two-step fix), so only `MOVE_ROBBER`'s
    (hex, victim) block grows with the player count."""
    three = build_space(54, 72, 19, 3)
    four = build_space(54, 72, 19, 4)
    grew = [
        kind
        for kind in ActionType
        if four.sizes[kind] != three.sizes[kind]
    ]
    assert grew == [ActionType.MOVE_ROBBER]


def test_the_flat_action_space_size_is_pinned():
    """The knight two-step fix shrinks the flat space from 550 to 456 for the
    4-player base board (`PLAY_KNIGHT`'s block drops from `robber` (95) to 1):
    pinned so a future change to the layout has to own the new number
    deliberately."""
    space = build_space(54, 72, 19, 4)
    assert space.size == 456
    assert space.sizes[ActionType.PLAY_KNIGHT] == 1


def test_setup_offers_only_setup_actions():
    game = a_game()
    assert {a.type for a in legal_actions(game)} == {ActionType.SETUP_SETTLEMENT}

    apply(game, legal_actions(game)[0])
    assert {a.type for a in legal_actions(game)} == {ActionType.SETUP_ROAD}


def test_the_opening_road_must_touch_the_new_settlement():
    game = a_game()
    apply(game, legal_actions(game)[0])
    offered = {a.a for a in legal_actions(game)}
    topology = game._state.board.topology
    assert offered == set(topology.vertex_edges[game.last_settlement])


def test_rolling_is_the_only_option_without_cards():
    game = a_game()
    while game.phase is not Phase.ROLL:
        apply(game, legal_actions(game)[0])
    assert legal_actions(game) == [Action(ActionType.ROLL)]


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


@pytest.mark.parametrize("seed", range(3))
def test_random_games_finish_with_a_legal_winner(seed):
    game = play_random_game(num_players=4, rng=random.Random(seed))

    assert game.won_by is not None
    assert victory_points(game._state, game.won_by) >= WINNING_POINTS


@pytest.mark.parametrize("seed", range(2))
def test_resources_survive_a_whole_game(seed):
    game = play_random_game(num_players=4, rng=random.Random(seed))
    assert total_in_play(game._state) == expected_total()
    assert all(n >= 0 for n in game._state.bank)
    assert all(n >= 0 for hand in game._state.hands for n in hand)


@pytest.mark.parametrize("players", [2, 3, 4])
def test_every_supported_player_count_plays(players):
    game = play_random_game(num_players=players, rng=random.Random(11))
    assert is_over(game)


def test_a_live_game_always_offers_something():
    rng = random.Random(7)
    game = a_game(players=4, seed=7)
    while not is_over(game):
        assert legal_actions(game), f"stuck in {game.phase.name}"
        step_randomly(game, rng)


def test_every_action_survives_a_round_trip_through_its_index():
    """The offer slot was the one lossy entry in the flat space (an offer is
    ten numbers and was never in an index). With trading an engine event
    rather than an action, `decode` is exactly `index`'s inverse."""
    space = space_for(a_game())
    for index in range(space.size):
        assert space.index(space.decode(index)) == index


def test_free_roads_are_placed_before_the_turn_can_end():
    game = a_game(players=2)
    rng = random.Random(2)
    while game.phase is not Phase.MAIN:
        step_randomly(game, rng)

    game.free_roads = 2
    kinds = {a.type for a in legal_actions(game)}
    if ActionType.BUILD_ROAD in kinds:
        assert ActionType.END_TURN not in kinds


def test_free_roads_come_before_everything_else_in_main():
    """A Road Building card resolves when played, in MAIN exactly as before
    the roll: while roads are owed no other action is legal. Colonist's
    server takes nothing else in that state -- two live games were lost
    sending a settlement and a dev-card buy past owed roads."""
    game = a_game(players=2)
    rng = random.Random(2)
    while game.phase is not Phase.MAIN:
        step_randomly(game, rng)

    game.free_roads = 2
    kinds = {a.type for a in legal_actions(game)}
    if ActionType.BUILD_ROAD in kinds:
        assert kinds == {ActionType.BUILD_ROAD}


def test_a_stranded_free_road_does_not_deadlock():
    game = a_game(players=2)
    rng = random.Random(3)
    while game.phase is not Phase.MAIN:
        step_randomly(game, rng)

    # Nowhere legal to build, so the turn must still be endable.
    game.free_roads = 2
    game._state.edge_owner = [0] * len(game._state.edge_owner)
    kinds = {a.type for a in legal_actions(game)}
    assert ActionType.END_TURN in kinds


# --- a seven's discards are simultaneous ------------------------------------
#
# Not a turn: every seat over the limit discards at the same instant, each
# bounded only by its own hand and its own quota (see `hexset.game.to_move`).
# `to_move` still names one of them, because the arena, the AEC environment
# and a bot runner all want a single actor and any of them will do; the seat
# argument below is what a live table uses instead.


def a_game_owing(seed: int = 7):
    """A game parked in `Phase.DISCARD` with seats 0 and 3 both owing two
    cards, and seat 1 (who rolled) owing none."""
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
    # The single-actor answer is unchanged, for the callers that want one.
    assert to_move(game) == 0


def test_a_seat_is_offered_its_own_cards_not_the_lowest_owing_seats():
    game = a_game_owing()
    assert [a.a for a in legal_actions(game, 3)] == [Resource.ORE]
    assert [a.a for a in legal_actions(game, 0)] == [Resource.WOOD]
    # No seat named: the engine's own serialization, the lowest owing seat.
    assert [a.a for a in legal_actions(game)] == [Resource.WOOD]
    # A seat that owes nothing has nothing to play, whoever else does.
    assert legal_actions(game, 1) == []


def test_a_higher_seat_discards_without_waiting_for_a_lower_one():
    """The bug: seat 3 could not act until seat 0 had, and its discard landed
    on seat 0's hand if it forced one through."""
    game = a_game_owing()

    apply(game, Action(ActionType.DISCARD, Resource.ORE), seat=3)

    assert game.discard_quota == [2, 0, 0, 1]
    assert game._state.hands[3][Resource.ORE] == 3
    assert game._state.hands[0][Resource.WOOD] == 4  # untouched
    assert game.phase is Phase.DISCARD


def test_a_discard_round_is_order_invariant():
    """Why the AEC environment may go on resolving this one seat at a time:
    interleaved or serialized, the same position."""
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
    # The round closing is what moves the phase on, not any one seat's turn.
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


def test_naming_no_seat_is_exactly_the_old_behaviour():
    """Every offline caller (arena, bots, `hexset.gym`) passes no seat, so
    the whole of this change has to be invisible to them."""
    game = a_game(seed=3)
    rng = random.Random(3)
    for _ in range(300):
        if is_over(game):
            break
        assert legal_actions(game) == legal_actions(game, to_move(game))
        step_randomly(game, rng)


def test_road_building_is_not_offered_with_no_road_pieces_left():
    """The card places roads; with fifteen on the board there is nothing to
    place, and colonist refuses the play outright."""
    import random

    from hexset.actions import Action, ActionType, legal_actions
    from hexset.board.board import random_base_board
    from hexset.cards import DevCard
    from hexset.game import Phase, start
    from hexset.state import MAX_ROADS, NO_OWNER

    rng = random.Random(3)
    game = start(random_base_board(rng), 2, rng)
    state = game._state
    seat = game.current_player
    game.phase = Phase.MAIN
    state.dev_cards[seat][DevCard.ROAD_BUILDING] = 1
    assert Action(ActionType.PLAY_ROAD_BUILDING) in legal_actions(game)
    free = [e for e in range(state.board.topology.num_edges) if state.edge_owner[e] == NO_OWNER]
    owned = sum(1 for e in range(state.board.topology.num_edges) if state.edge_owner[e] == seat)
    for e in free[: MAX_ROADS - owned]:
        state.edge_owner[e] = seat
    assert Action(ActionType.PLAY_ROAD_BUILDING) not in legal_actions(game)
