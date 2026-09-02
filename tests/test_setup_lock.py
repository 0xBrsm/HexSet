from __future__ import annotations

import random

from hexset.actions import ActionType, apply
from hexset.board.board import random_base_board
from hexset.game import Phase, _in_second_setup_round, end_turn, roll_dice, to_move

from hexset_ui.rules import options_for
from hexset_ui.seating import (
    advance_setup,
    lock_seat,
    locked_of,
    next_unlocked,
    settle,
    snapshot,
    start_at,
)


def _board():
    return random_base_board(random.Random(0))


def _apply(game, action):
    """`apply`, plus the seating correction `webplay.GameSession._apply` makes
    on every action -- the snake starts at the creator and skips retired
    seats, neither of which `hexset.game` knows about (`hexset_ui.seating`)."""
    before = snapshot(game)
    apply(game, action)
    settle(game, before)


def _place(game):
    """Play the seat on move's whole setup turn (settlement, then road) with
    whatever the engine offers first -- geometry is not the point here."""
    settlement = next(a for a in options_for(game) if a.type is ActionType.SETUP_SETTLEMENT)
    _apply(game, settlement)
    road = next(a for a in options_for(game) if a.type is ActionType.SETUP_ROAD)
    _apply(game, road)


def test_a_creator_not_seated_at_zero_does_not_deadlock():
    """The snake starts at `first`, not always seat 0 -- otherwise a creator
    randomly seated anywhere else would find it seat 0's turn, seat 0 empty,
    and nothing able to ever advance."""
    game = start_at(_board(), 4, random.Random(1), first=2)
    assert game.setup_queue[0] == 2
    assert game.current_player == 2
    assert to_move(game) == 2
    _place(game)  # would raise (wrong seat) if this weren't seat 2's turn


def test_the_worked_example_from_the_plan():
    """creator = 0, seat 2 joins mid-setup, seats 1 and 3 never do.
    queue = [0,1,2,3,3,2,1,0]. Traced step by step against the plan's own
    worked example, including the double-skip when a seat appears twice in
    a row (seat 3, at indices 3 and 4)."""
    game = start_at(_board(), 4, random.Random(1), first=0)
    assert game.setup_queue == [0, 1, 2, 3, 3, 2, 1, 0]

    _place(game)  # step 0: seat 0
    assert (game.setup_step, game.current_player, game.phase) == (1, 1, Phase.SETUP_SETTLEMENT)

    lock_seat(game, 1)  # step 1 locks, skip -> step 2: seat 2
    assert (game.setup_step, game.current_player, game.phase) == (2, 2, Phase.SETUP_SETTLEMENT)

    _place(game)  # step 2: seat 2 (first placement, no resources yet)
    assert not _in_second_setup_round(game)
    assert (game.setup_step, game.current_player, game.phase) == (3, 3, Phase.SETUP_SETTLEMENT)

    lock_seat(game, 3)  # steps 3 and 4 both lock (seat 3 twice), skip -> step 5: seat 2
    assert (game.setup_step, game.current_player, game.phase) == (5, 2, Phase.SETUP_SETTLEMENT)
    assert _in_second_setup_round(game)

    before = sum(game.state.hands[2])
    _place(game)  # step 5: seat 2's second placement -- gets resources
    assert sum(game.state.hands[2]) > before
    assert (game.setup_step, game.current_player, game.phase) == (7, 0, Phase.SETUP_SETTLEMENT)

    before = sum(game.state.hands[0])
    _place(game)  # step 7: seat 0's second placement -- gets resources
    assert sum(game.state.hands[0]) > before
    assert game.setup_step == 8
    assert game.phase is Phase.ROLL
    assert game.current_player == 0  # queue[0], the creator, never locked


def test_locking_every_other_seat_still_ends_setup_on_the_creator():
    """The all-lock (solo) case: every seat but the creator locks before
    setup ever reaches them. The creator still owes their own second
    placement (queue index 7, `[0,1,2,3,3,2,1,0]`) -- locking the other
    three seats skips straight to it, not straight past it."""
    game = start_at(_board(), 4, random.Random(1), first=0)
    _place(game)  # seat 0's first placement
    assert game.current_player == 1

    lock_seat(game, 1)
    lock_seat(game, 2)
    lock_seat(game, 3)
    assert locked_of(game) == {1, 2, 3}
    assert (game.setup_step, game.current_player, game.phase) == (7, 0, Phase.SETUP_SETTLEMENT)

    _place(game)  # seat 0's own second placement -- setup only ends now
    assert game.setup_step == 8
    assert game.phase is Phase.ROLL
    assert game.current_player == 0
    assert to_move(game) == 0


def test_end_turn_never_lands_on_a_locked_seat():
    game = start_at(_board(), 4, random.Random(1), first=0)
    _place(game)
    lock_seat(game, 1)
    _place(game)  # seat 2
    lock_seat(game, 3)
    _place(game)  # seat 2's second placement (step 5)
    _place(game)  # seat 0's second placement (step 7) -> Phase.ROLL, seat 0

    assert locked_of(game) == {1, 3}
    for _ in range(6):
        roll_dice(game, roll=8)
        if game.phase is Phase.MAIN:
            before = snapshot(game)
            end_turn(game)
            settle(game, before)
        assert game.current_player not in locked_of(game)
        assert game.current_player in (0, 2)


def test_next_unlocked_raises_only_when_every_seat_is_locked():
    game = start_at(_board(), 4, random.Random(1), first=0)
    game.locked = frozenset({1, 2})
    assert next_unlocked(game, 0) == 3
    game.locked = frozenset({1, 2, 3})
    assert next_unlocked(game, 0) == 0  # wraps back to the only unlocked seat


def test_advance_setup_is_idempotent_once_resolved():
    game = start_at(_board(), 3, random.Random(1), first=0)
    game.locked = frozenset({1})
    advance_setup(game)
    resolved = (game.setup_step, game.current_player, game.phase)
    advance_setup(game)
    assert (game.setup_step, game.current_player, game.phase) == resolved
