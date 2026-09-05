# SPDX-License-Identifier: GPL-3.0-only
"""`Game.locked`: the per-seat setup lock / seat retirement, upstreamed.

`hexset.server.seating` (the gym's interface layer) implemented this as a
post-apply correction bolted onto a live `Game`, because `hexset.game` had no
notion of a locked seat: after every applied action, `settle()` re-pointed the
setup snake or the turn rotation past a seat the table had retired. That is
correct for the game actually being played, but `hexset.game.imagine` did not
know to copy the correction's plain, undeclared `game.locked` attribute, so an
embedded bot searching forward from a table with a retired seat simulated
turns for it as though it still played
(`docs/engine-divergence-2026-09-02.md`, request R2).

This module tests the primitive that closes that gap: `Game.locked` is now a
real dataclass field, `imagine` copies it, and `lock_seat`/`start(first=)`
give the engine what `hexset.server.seating.lock_seat`/`start_at` used to bolt on
from outside. The tests below are organised the way the request was: setup-
phase skipping, main-phase skipping, trade offers, `imagine`, and finally the
byte-identity guard that nothing about an *unlocked* game changed at all.
"""

from __future__ import annotations

import random

from hexset.actions import apply
from hexset.board.board import random_base_board
from hexset.board.terrain import NUM_RESOURCES, Resource
from hexset.game import (
    Phase,
    end_turn,
    imagine,
    legal_initial_roads,
    lock_seat,
    place_initial_road,
    place_initial_settlement,
    players_owing_discards,
    roll_dice,
    start,
    to_move,
)
from hexset.state import can_place_settlement
from hexset.trading import trade_event
from helpers import clear_hand, give


def a_game(players: int = 3, seed: int = 0, *, first: int = 0):
    rng = random.Random(seed)
    return start(random_base_board(rng), players, rng, first=first)


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


def a_trade_ready_game(players: int = 3):
    """Setup done, every hand emptied, so a test's own `give` calls are the
    only cards on the table -- setup's initial yield is otherwise random per
    board and would make an "only seat X can cover this" trade flaky."""
    game = run_setup(a_game(players=players))
    game.phase = Phase.MAIN
    for player in range(players):
        clear_hand(game._state, player)
    return game


# --- start(first=) ------------------------------------------------------


def test_start_default_first_matches_todays_snake():
    game = a_game(players=3)
    assert game.setup_queue == [0, 1, 2, 2, 1, 0]
    assert game.current_player == 0


def test_start_first_rotates_the_snake_keeping_its_compensating_property():
    """Whoever places first in round one places last in round two, from
    whichever seat the snake starts at -- `hexset.server.seating.start_at`'s
    argument, upstreamed."""
    game = a_game(players=3, first=2)
    assert game.setup_queue == [2, 0, 1, 1, 0, 2]
    assert game.current_player == 2
    assert game.setup_queue[0] == game.setup_queue[-1] == 2


# --- setup-phase lock -----------------------------------------------------


def test_locked_seat_is_skipped_by_the_setup_snake():
    game = a_game(players=3)
    lock_seat(game, 1)
    run_setup(game)

    assert game.phase is Phase.ROLL
    assert game._state.vertex_owner.count(1) == 0
    assert game._state.edge_owner.count(1) == 0
    assert game._state.vertex_owner.count(0) == 2
    assert game._state.vertex_owner.count(2) == 2
    # Whoever placed first takes the first real turn, and seat 0 (never
    # locked) is `setup_queue[0]`.
    assert game.current_player == 0


def test_locking_the_seat_currently_up_moves_the_snake_off_it_at_once():
    game = a_game(players=3)
    place_initial_settlement(game, free_vertex(game))
    place_initial_road(game, legal_initial_roads(game)[0])
    assert game.current_player == 1

    lock_seat(game, 1)

    assert game.current_player == 2
    assert game.phase is Phase.SETUP_SETTLEMENT


def test_lock_seat_is_a_noop_if_already_locked():
    game = a_game(players=3)
    lock_seat(game, 1)
    before = (game.current_player, game.phase, game.setup_step, game.locked)

    lock_seat(game, 1)

    assert (game.current_player, game.phase, game.setup_step, game.locked) == before


# --- main-phase turn rotation ----------------------------------------------


def test_locked_seat_is_skipped_by_end_turn():
    game = run_setup(a_game(players=3))
    assert game.current_player == 0
    lock_seat(game, 1)

    roll_dice(game, roll=8)
    end_turn(game)
    assert game.current_player == 2

    roll_dice(game, roll=8)
    end_turn(game)
    assert game.current_player == 0


def test_locked_seat_never_owes_a_discard():
    game = run_setup(a_game(players=3))
    lock_seat(game, 1)
    give(game._state, 1, Resource.WOOD, 8)

    roll_dice(game, roll=7)

    assert 1 not in players_owing_discards(game)
    assert game.discard_quota[1] == 0


def test_to_move_never_returns_a_locked_seat():
    game = run_setup(a_game(players=3))
    lock_seat(game, 1)
    seen = set()
    for _ in range(12):
        seen.add(to_move(game))
        roll_dice(game, roll=8)
        end_turn(game)
    assert 1 not in seen


# --- trading ------------------------------------------------------------


def test_a_locked_seat_is_never_a_trade_counterparty():
    """`lock_seat` used to have to reach into a standing offer. Trading is one
    engine event now (`hexset.trading`), and the event skips a retired seat
    outright -- one check, in the candidate enumeration, rather than a
    correction applied afterwards."""
    game = a_trade_ready_game()
    give(game._state, 0, Resource.WOOD, 2)
    give(game._state, 1, Resource.ORE, 1)
    give(game._state, 2, Resource.ORE, 1)
    lock_seat(game, 1)

    # Seat 0 wants ore, everyone else wants wood back -- direction-aware, so
    # the reverse of whichever trade clears never also prices positively
    # (a blanket "always yes" gate would ping-pong forever between seat 0
    # and seat 2 once the one ore has moved).
    def gate(seat, view, received, other):
        wanted = Resource.ORE if seat == 0 else Resource.WOOD
        return 1.0 if received[wanted] > 0 else -1.0

    done = trade_event(game, gate)
    assert done
    assert all(trade.b == 2 for trade in done)


# --- imagine ------------------------------------------------------------


def test_imagine_carries_the_lock():
    game = run_setup(a_game(players=3))
    lock_seat(game, 1)

    copy = imagine(game, random.Random(99))

    assert copy.locked == frozenset({1})
    # And the copy actually honours it: a search forward from here must not
    # simulate seat 1's turn, which is exactly what R2 was filed to fix.
    roll_dice(copy, roll=8)
    end_turn(copy)
    assert copy.current_player == 2
    # The original is untouched by mutating the copy.
    assert game.locked == frozenset({1})


def test_imagine_without_any_lock_carries_an_empty_one():
    game = run_setup(a_game(players=3))
    copy = imagine(game, random.Random(1))
    assert copy.locked == frozenset()
