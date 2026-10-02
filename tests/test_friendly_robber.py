# SPDX-License-Identifier: GPL-3.0-only
"""The friendly robber, as `Rules.friendly_robber`.

The rule: the robber may not take a hex occupied by a seat at or below two
*public* points, so a seat that is visibly behind can be neither blocked nor
robbed. It is one of the four settings `rules.DUEL_VARIANT` spells.
"""
from __future__ import annotations

import random

import pytest

from hexset.actions import ActionType, legal_actions
from hexset.board.board import random_base_board
from hexset.cards import DevCard
from hexset.game import Phase, imagine, move_robber_to, start
from hexset.robber import (
    FRIENDLY_ROBBER_POINTS,
    allowed_targets,
    occupants,
)
from hexset.rules import (
    DUEL_VARIANT_GAME,
    DUEL_VARIANT,
    STANDARD,
    STANDARD_GAME,
    GameType,
    Rules,
)
from hexset.state import Building
from hexset.victory import public_victory_points

from helpers import victim_on
from tests.test_game import run_setup


def a_duel(seed: int = 0, game_type: GameType = DUEL_VARIANT_GAME):
    board = random_base_board(random.Random(seed))
    return run_setup(start(board, 2, random.Random(seed), game_type=game_type))


def robber_hexes(game) -> set[int]:
    return {a.a for a in legal_actions(game) if a.type is ActionType.MOVE_ROBBER}


def promote(game, seat: int, target: int = 3) -> None:
    """Raise a seat's *public* points by upgrading its settlements to cities."""
    state = game._state
    for v, owner in enumerate(state.vertex_owner):
        if owner == seat and state.vertex_building[v] == Building.SETTLEMENT:
            state.vertex_building[v] = Building.CITY
        if public_victory_points(state, seat) >= target:
            return
    raise AssertionError(f"could not lift seat {seat} to {target} public points")


def test_duel_variant_spells_all_four_settings():
    assert DUEL_VARIANT == Rules(
        winning_points=15, discard_limit=9, friendly_robber=True, balanced_dice=True
    )
    assert STANDARD.friendly_robber is False


def test_a_seat_at_two_public_points_can_be_neither_blocked_nor_robbed():
    game = a_duel()
    game.phase = Phase.ROBBER
    state = game._state
    assert public_victory_points(state, 1) <= FRIENDLY_ROBBER_POINTS

    theirs = {h for h in range(state.board.num_hexes) if 1 in occupants(state, h)}
    assert theirs, "the fixture must seat the shielded player somewhere"
    assert not (robber_hexes(game) & theirs)

    with pytest.raises(ValueError, match="does not allow"):
        move_robber_to(game, min(theirs))


def test_the_shield_lifts_once_the_seat_is_visibly_ahead():
    game = a_duel()
    game.phase = Phase.ROBBER
    state = game._state
    theirs = {h for h in range(state.board.num_hexes) if 1 in occupants(state, h)}
    assert not (robber_hexes(game) & theirs)

    promote(game, 1, FRIENDLY_ROBBER_POINTS + 1)
    assert robber_hexes(game) & theirs
    move_robber_to(game, min(theirs), victim_on(game, min(theirs)))
    assert state.robber == min(theirs)


def test_a_victory_point_card_does_not_lift_the_shield():
    """The shield is a public read: a hidden VP card wins games but does not
    expose its holder to the robber."""
    game = a_duel()
    game.phase = Phase.ROBBER
    state = game._state
    theirs = {h for h in range(state.board.num_hexes) if 1 in occupants(state, h)}

    state.dev_cards[1][DevCard.VICTORY_POINT] += 3
    assert public_victory_points(state, 1) <= FRIENDLY_ROBBER_POINTS
    assert not (robber_hexes(game) & theirs)


def test_standard_rules_keep_the_rulebook_robber():
    game = a_duel(game_type=STANDARD_GAME)
    game.phase = Phase.ROBBER
    state = game._state
    assert len(robber_hexes(game)) == state.board.num_hexes - 1
    assert allowed_targets(state, 0, None) is None


def test_a_host_rule_still_wins_over_the_game_type():
    """A mirror of a real table has already applied that table's rules; the
    engine must not narrow its list further, or widen it."""
    game = a_duel()
    game.phase = Phase.ROBBER
    state = game._state
    theirs = sorted(h for h in range(state.board.num_hexes) if 1 in occupants(state, h))

    game.robber_allowed = frozenset(theirs[:1])
    assert robber_hexes(game) == frozenset(theirs[:1])
    assert imagine(game, random.Random(1)).robber_allowed == frozenset(theirs[:1])
    move_robber_to(game, theirs[0], victim_on(game, theirs[0]))
    assert state.robber == theirs[0]
    assert game.robber_allowed is None
