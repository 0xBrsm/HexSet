"""The served table's seat rules on the engine's own `start(first=)` and
`lock_seat`: the snake opens at the creator and skips retired seats, and a
seat closed before the first move can reopen (`seating.unlock_seat`).
"""

from __future__ import annotations

import random

from hexset.actions import ActionType, apply, options_for
from hexset.board.board import random_base_board
from hexset.game import Phase, _in_second_setup_round, end_turn, lock_seat, roll_dice, start, to_move
from hexset.server._seating import unlock_seat


def _board():
    return random_base_board(random.Random(0))


def _place(game):
    settlement = next(a for a in options_for(game) if a.type is ActionType.SETUP_SETTLEMENT)
    apply(game, settlement)
    road = next(a for a in options_for(game) if a.type is ActionType.SETUP_ROAD)
    apply(game, road)


def test_a_creator_not_seated_at_zero_does_not_deadlock():
    game = start(_board(), 4, random.Random(1), first=2)
    assert game.setup_queue[0] == 2
    assert game.current_player == 2
    assert to_move(game) == 2
    _place(game)


def test_the_worked_example_from_the_plan():
    """creator = 0, seat 2 joins mid-setup, seats 1 and 3 never do, so the queue
    is [0,1,2,3,3,2,1,0], including the double-skip at indices 3 and 4.
    """
    game = start(_board(), 4, random.Random(1), first=0)
    assert game.setup_queue == [0, 1, 2, 3, 3, 2, 1, 0]

    _place(game)
    assert (game.setup_step, game.current_player, game.phase) == (1, 1, Phase.SETUP_SETTLEMENT)

    lock_seat(game, 1)  # step 1 locks, skip -> step 2: seat 2
    assert (game.setup_step, game.current_player, game.phase) == (2, 2, Phase.SETUP_SETTLEMENT)

    _place(game)
    assert not _in_second_setup_round(game)
    assert (game.setup_step, game.current_player, game.phase) == (3, 3, Phase.SETUP_SETTLEMENT)

    lock_seat(game, 3)  # steps 3 and 4 both lock (seat 3 twice), skip -> step 5: seat 2
    assert (game.setup_step, game.current_player, game.phase) == (5, 2, Phase.SETUP_SETTLEMENT)
    assert _in_second_setup_round(game)

    before = sum(game._state.hands[2])
    _place(game)
    assert sum(game._state.hands[2]) > before
    assert (game.setup_step, game.current_player, game.phase) == (7, 0, Phase.SETUP_SETTLEMENT)

    before = sum(game._state.hands[0])
    _place(game)
    assert sum(game._state.hands[0]) > before
    assert game.setup_step == 8
    assert game.phase is Phase.ROLL
    assert game.current_player == 0  # queue[0], the creator, never locked


def test_setup_ending_hands_the_first_roll_to_the_first_seat_still_in_the_game():
    """`setup_queue[0]` is a retired seat here: the first roll must go to the
    first seat still in the game, or the table hangs.
    """
    game = start(_board(), 4, random.Random(1), first=0)
    lock_seat(game, 0)
    lock_seat(game, 3)
    assert game.current_player == 1
    for _ in range(4):  # 1, 2, then 2, 1
        _place(game)
    assert game.phase is Phase.ROLL
    assert game.current_player == 1
    assert to_move(game) == 1


def test_end_turn_never_lands_on_a_locked_seat():
    game = start(_board(), 4, random.Random(1), first=0)
    _place(game)
    lock_seat(game, 1)
    _place(game)
    lock_seat(game, 3)
    _place(game)
    _place(game)

    assert game.locked == {1, 3}
    for _ in range(6):
        roll_dice(game, roll=8)
        if game.phase is Phase.MAIN:
            end_turn(game)
        assert game.current_player not in game.locked
        assert game.current_player in (0, 2)


def test_a_seat_reopened_before_the_first_move_takes_its_place_in_the_snake_back():
    """Closing the snake's opening seat moves the snake past it; reopening
    it before anything is placed points the snake back at it, as though it
    had never closed."""
    game = start(_board(), 4, random.Random(1), first=0)
    lock_seat(game, 0)
    lock_seat(game, 1)
    assert (game.setup_step, game.current_player) == (2, 2)
    unlock_seat(game, 0)
    assert game.locked == {1}
    assert (game.setup_step, game.current_player, game.phase) == (0, 0, Phase.SETUP_SETTLEMENT)
    unlock_seat(game, 0)  # not retired: nothing to do
    assert (game.setup_step, game.current_player) == (0, 0)


def test_reopening_a_seat_behind_the_snakes_head_leaves_the_head_where_it_is():
    game = start(_board(), 4, random.Random(1), first=0)
    lock_seat(game, 2)
    unlock_seat(game, 2)
    assert game.locked == frozenset()
    assert (game.setup_step, game.current_player) == (0, 0)


