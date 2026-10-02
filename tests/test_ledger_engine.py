# SPDX-License-Identifier: GPL-3.0-only
"""`hexset.ledger`: the public-knowledge reconstruction of each seat's hand. The
spec is `test_the_ledger_never_overclaims_over_long_playouts`; the rest pins
one convention at a time.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from hexset.actions import ActionType, apply, legal_actions, victim_of
from hexset.board.board import random_base_board
from hexset.board.terrain import NUM_RESOURCES, Resource
from hexset.cards import DevCard
from hexset.encoding import encode
from hexset.game import (
    Phase,
    discard_one,
    imagine,
    move_robber_to,
    play_monopoly_card,
    start,
)
from hexset.ledger import PublicLedger, SeatLedger
from hexset.play import step_randomly


def a_game(players: int = 4, seed: int = 0):
    rng = random.Random(seed)
    return start(random_base_board(rng), players, rng)


def after_setup(seed: int = 0, players: int = 4):
    game = a_game(players, seed)
    while game.phase in (Phase.SETUP_SETTLEMENT, Phase.SETUP_ROAD):
        apply(game, legal_actions(game)[0])
    return game


def _set_known_hand(game, player: int, counts: list[int]) -> None:
    """Unlike `helpers.give`/`clear_hand`, keeps `game.ledger` in sync."""
    state = game._state
    for r, n in enumerate(state.hands[player]):
        if n:
            state.bank[r] += n
            state.hands[player][r] = 0
    game.ledger.seats[player] = SeatLedger()
    for r, n in enumerate(counts):
        if n:
            state.bank[r] -= n
            state.hands[player][r] += n
            game.ledger.receive(player, r, n)


def _assert_invariant(game) -> None:
    for seat, seat_ledger in enumerate(game.ledger.seats):
        true_hand = game._state.hands[seat]
        assert seat_ledger.total() == sum(true_hand), (
            f"seat {seat}: sum(known)+unknown={seat_ledger.total()} "
            f"!= true hand total {sum(true_hand)}"
        )
        for r, k in enumerate(seat_ledger.known):
            assert k <= true_hand[r], (
                f"seat {seat} resource {r}: known={k} > true={true_hand[r]}"
            )


def _steal_action(game, thief: int, victim: int):
    """The hex for a `MOVE_ROBBER` that pairs `thief` with `victim`."""
    game.current_player = thief
    for action in legal_actions(game):
        if action.type is ActionType.MOVE_ROBBER and victim_of(game, action.b) == victim:
            return action.a
    raise AssertionError(f"no MOVE_ROBBER pairs {thief} -> {victim} on this board")


def _ledger_block(obs, players: int = 4) -> np.ndarray:
    """Each opponent's `known[5]` then `unknown`, seat-relative, own seat excluded."""
    width = (players - 1) * (NUM_RESOURCES + 1)
    return obs.globals[-width:]




@pytest.mark.parametrize("players, seed", [(3, 0), (4, 1)])
def test_the_ledger_never_overclaims_over_long_playouts(players, seed):
    rng = random.Random(seed)
    game = start(random_base_board(rng), players, rng)
    for _ in range(500):
        if game.phase is Phase.GAME_OVER:
            break
        step_randomly(game, rng)
        _assert_invariant(game)


def test_setup_grants_are_public():
    game = after_setup()
    _assert_invariant(game)
    for seat_ledger in game.ledger.seats:
        assert seat_ledger.unknown == 0


def test_a_steal_credits_the_thief_with_exactly_one_unknown_card():
    game = after_setup()
    game.phase = Phase.ROBBER
    for player in range(game._state.num_players):
        _set_known_hand(game, player, [0] * NUM_RESOURCES)
    _set_known_hand(game, 1, [2, 0, 3, 0, 0])
    target = _steal_action(game, thief=0, victim=1)

    move_robber_to(game, target, 1)

    assert game.ledger.seats[0].unknown == 1
    assert game.ledger.seats[0].known == [0] * NUM_RESOURCES
    _assert_invariant(game)


def test_a_steal_floors_every_known_resource_by_one():
    """Mixed hand, so the floor visibly touches more than one resource."""
    ledger = PublicLedger.new(2)
    ledger.receive(1, int(Resource.WOOD), 2)
    ledger.receive(1, int(Resource.SHEEP), 3)

    ledger.steal(thief=0, victim=1)

    assert ledger.seats[1].known == [1, 0, 2, 0, 0]
    assert ledger.seats[1].unknown == 1
    assert ledger.seats[1].total() == 4
    assert ledger.seats[0].unknown == 1
    assert ledger.seats[0].known == [0] * NUM_RESOURCES


def test_a_steal_from_a_fully_uncertain_hand_only_grows_unknown():
    ledger = PublicLedger.new(2)
    ledger.gain_unknown(1, 3)

    ledger.steal(thief=0, victim=1)

    assert ledger.seats[1].known == [0] * NUM_RESOURCES
    assert ledger.seats[1].unknown == 2
    assert ledger.seats[0].unknown == 1


def test_a_steal_is_identity_independent_in_the_encoding():
    """Two worlds differing only in which resource the victim holds. The thief's
    own hand block is the one place they may differ.
    """
    thief, victim = 0, 1

    def a_steal_world(single_resource: int, seed: int = 42):
        game = after_setup(seed)
        game.phase = Phase.ROBBER
        for player in range(game._state.num_players):
            _set_known_hand(game, player, [0] * NUM_RESOURCES)
        hand = [0] * NUM_RESOURCES
        hand[single_resource] = 1
        _set_known_hand(game, victim, hand)
        target = _steal_action(game, thief, victim)
        move_robber_to(game, target, victim)
        return game

    world_a = a_steal_world(int(Resource.WOOD))
    world_b = a_steal_world(int(Resource.ORE))
    players = world_a._state.num_players

    for perspective in range(players):
        block_a = _ledger_block(encode(world_a, perspective), players)
        block_b = _ledger_block(encode(world_b, perspective), players)
        assert np.array_equal(block_a, block_b), (
            f"perspective {perspective} leaked the stolen identity"
        )

    hand_a = encode(world_a, thief).globals[:NUM_RESOURCES]
    hand_b = encode(world_b, thief).globals[:NUM_RESOURCES]
    assert not np.array_equal(hand_a, hand_b)




def test_an_over_draw_spend_resolves_unknown_cards():
    ledger = PublicLedger.new(1)
    ledger.receive(0, int(Resource.WOOD), 1)
    ledger.gain_unknown(0, 2)  # true wood count is really 3

    ledger.spend(0, int(Resource.WOOD), 3)

    assert ledger.seats[0].known[Resource.WOOD] == 0
    assert ledger.seats[0].unknown == 0


def test_monopoly_re_pins_the_announced_resource():
    game = after_setup()
    game.phase = Phase.MAIN
    game.current_player = 0
    game._state.dev_cards[0][DevCard.MONOPOLY] = 1
    for player in range(game._state.num_players):
        _set_known_hand(game, player, [0] * NUM_RESOURCES)
    game._state.bank[Resource.SHEEP] -= 2
    game._state.hands[1][Resource.SHEEP] += 2
    game.ledger.gain_unknown(1, 2)
    _set_known_hand(game, 2, [0, 0, 1, 0, 0])

    play_monopoly_card(game, Resource.SHEEP)

    assert game._state.hands[1][Resource.SHEEP] == 0
    assert game._state.hands[2][Resource.SHEEP] == 0
    assert game.ledger.seats[1].known[Resource.SHEEP] == 0
    assert game.ledger.seats[1].unknown == 0
    assert game.ledger.seats[2].known[Resource.SHEEP] == 0
    assert game.ledger.seats[0].known[Resource.SHEEP] == 3
    assert game.ledger.seats[0].unknown == 0
    _assert_invariant(game)


def test_a_discard_reveals_the_resource_it_names():
    game = after_setup()
    game.discard_quota = [0] * game._state.num_players
    game.phase = Phase.DISCARD
    _set_known_hand(game, 0, [0] * NUM_RESOURCES)
    game._state.bank[Resource.ORE] -= 1
    game._state.hands[0][Resource.ORE] += 1
    game.ledger.gain_unknown(0, 1)
    game.discard_quota[0] = 1

    discard_one(game, 0, Resource.ORE)

    assert game.ledger.seats[0].known == [0] * NUM_RESOURCES
    assert game.ledger.seats[0].unknown == 0
    _assert_invariant(game)


def test_imagine_copies_the_ledger_independently():
    game = after_setup()
    _set_known_hand(game, 0, [1, 0, 0, 0, 0])
    clone = imagine(game, random.Random(5))

    assert clone.ledger.seats[0].known == game.ledger.seats[0].known
    assert clone.ledger.seats[0] is not game.ledger.seats[0]

    clone.ledger.receive(0, int(Resource.ORE), 3)
    assert game.ledger.seats[0].known[Resource.ORE] == 0
