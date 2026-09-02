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

import hashlib
import random

from hexset.actions import apply
from hexset.arena import PRESETS, spawn
from hexset.board.board import random_base_board
from hexset.board.terrain import Resource
from hexset.game import (
    Phase,
    end_turn,
    imagine,
    legal_initial_roads,
    lock_seat,
    place_initial_road,
    place_initial_settlement,
    players_owing_discards,
    propose_trade,
    roll_dice,
    start,
    to_move,
)
from hexset.state import can_place_settlement
from hexset.trading import bundle
from helpers import clear_hand, give


def a_game(players: int = 3, seed: int = 0, *, first: int = 0):
    rng = random.Random(seed)
    return start(random_base_board(rng), players, rng, first=first)


def free_vertex(game):
    return next(
        v
        for v in range(game.state.board.topology.num_vertices)
        if can_place_settlement(game.state, game.current_player, v, connected=False)
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
        clear_hand(game.state, player)
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
    assert game.state.vertex_owner.count(1) == 0
    assert game.state.edge_owner.count(1) == 0
    assert game.state.vertex_owner.count(0) == 2
    assert game.state.vertex_owner.count(2) == 2
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
    give(game.state, 1, Resource.WOOD, 8)

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


# --- trade offers ------------------------------------------------------


def test_trade_offer_skips_a_locked_responder():
    game = a_trade_ready_game()
    give(game.state, 0, Resource.WOOD, 2)
    give(game.state, 1, Resource.ORE, 1)
    give(game.state, 2, Resource.ORE, 1)
    lock_seat(game, 1)

    propose_trade(game, bundle(wood=2), bundle(ore=1))

    assert game.pending_responders == [2]


def test_locking_the_head_responder_drops_it_from_the_offer():
    game = a_trade_ready_game()
    give(game.state, 0, Resource.WOOD, 2)
    give(game.state, 1, Resource.ORE, 1)
    give(game.state, 2, Resource.ORE, 1)
    propose_trade(game, bundle(wood=2), bundle(ore=1), ask=(1, 2))
    assert game.pending_responders == [1, 2]

    lock_seat(game, 1)

    assert game.pending_responders == [2]
    assert game.phase is Phase.TRADE_RESPOND


def test_locking_the_last_responder_ends_the_offer():
    game = a_trade_ready_game()
    give(game.state, 0, Resource.WOOD, 2)
    give(game.state, 1, Resource.ORE, 1)
    propose_trade(game, bundle(wood=2), bundle(ore=1))
    assert game.pending_responders == [1]

    lock_seat(game, 1)

    assert game.phase is Phase.MAIN
    assert game.offer is None
    assert game.pending_responders == []


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


# --- byte identity for an unlocked game -------------------------------------

# Computed at `b1763e6` (this branch's base, before `Game.locked` existed) by
# playing five seeded 4-player games with `PRESETS["greedy"]` in every seat
# and hashing the full `(seat, action)` trace -- the same method
# `test_heximax.test_choices_are_byte_identical_to_the_recorded_census` uses,
# borrowed here because no locking test may change what an unlocked game
# does: not the decision order, not the rng draws, not who moves next. A
# change to `hexset.game` that flips even one of these hashes has changed
# something about default play, which every requirement for this change
# forbids.
BYTE_IDENTITY_TRACES = {
    "200": "bf55b52e75832cf5ec7e1555a403e1ac19857e64dd6531bfa76857599806b44d",
    "201": "1776ab375483782d00c8b587785d5b52005ef74cdd3a8b47e111b478c60258db",
    "202": "291805f82d65c8deeca519a3105571c0ef3a0019af549947fbf38db0dc528ba7",
    "203": "64d4dea5e3348c831a348eba6bbebd3efe2c2bf8dc321c2c9a1f879f031cb08f",
    "204": "c3d81461909f6e0c7098f20499d55b3b450de184ee9f8f741846477873f62a6b",
}


def _played_trace(seed: int, players: int = 4) -> str:
    rng = random.Random(seed)
    board = random_base_board(rng)
    game = start(board, players, rng)
    bots = [
        spawn(PRESETS["greedy"], board, random.Random(f"{seed}:{seat}"))
        for seat in range(players)
    ]
    trace = []
    moves = 0
    while game.phase is not Phase.GAME_OVER:
        seat = to_move(game)
        action = bots[seat].choose(game)
        trace.append(
            (
                seat,
                int(action.type),
                action.a,
                action.b,
                list(action.give),
                list(action.want),
                list(action.ask),
            )
        )
        apply(game, action)
        moves += 1
        if moves > 60000:
            raise AssertionError(f"seed {seed} did not finish")
    return hashlib.sha256(repr(trace).encode()).hexdigest()


def test_an_unlocked_game_is_byte_identical_to_before_the_lock_existed():
    for seed_str, expected in BYTE_IDENTITY_TRACES.items():
        assert _played_trace(int(seed_str)) == expected, f"seed {seed_str} diverged"
