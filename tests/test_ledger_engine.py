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
from hexset.chance import Live
from hexset.encoding import encode, global_columns
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
    game.ledger.seats[player] = SeatLedger(most=[0] * NUM_RESOURCES)
    for r, n in enumerate(counts):
        if n:
            state.bank[r] -= n
            state.hands[player][r] += n
            game.ledger.receive(player, r, n)


def _assert_invariant(game) -> None:
    """Over the common rows and every seat's own."""
    for knower in range(game._state.num_players):
        _assert_rows(game, game.ledger.for_seat(knower).seats)


def _assert_rows(game, rows) -> None:
    for seat, seat_ledger in enumerate(rows):
        true_hand = game._state.hands[seat]
        assert seat_ledger.total() == sum(true_hand), (
            f"seat {seat}: sum(known)+unknown={seat_ledger.total()} "
            f"!= true hand total {sum(true_hand)}"
        )
        for r, k in enumerate(seat_ledger.known):
            assert k <= true_hand[r], (
                f"seat {seat} resource {r}: known={k} > true={true_hand[r]}"
            )
            assert true_hand[r] <= seat_ledger.most[r], (
                f"seat {seat} resource {r}: true={true_hand[r]} > most={seat_ledger.most[r]}"
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
    return obs.globals[global_columns(players)["ledger"]]




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


class _Takes(Live):
    """Dice off `rng`, every steal the one resource named."""

    def __init__(self, resource: int) -> None:
        super().__init__(random.Random(0))
        self.resource = resource

    def steal(self, hand):
        return self.resource if sum(hand) else None


def _steal_world(taken: int, thief: int = 0, victim: int = 1, seed: int = 42):
    """A steal of `taken` from a victim certified to hold one wood and one
    ore: the table cannot tell which card moved."""
    game = after_setup(seed)
    game.phase = Phase.ROBBER
    for player in range(game._state.num_players):
        _set_known_hand(game, player, [0] * NUM_RESOURCES)
    hand = [0] * NUM_RESOURCES
    hand[Resource.WOOD] = hand[Resource.ORE] = 1
    _set_known_hand(game, victim, hand)
    target = _steal_action(game, thief, victim)
    game.chance = _Takes(taken)
    move_robber_to(game, target, victim)
    return game


def test_a_steal_reaches_the_encoding_of_its_thief_and_victim_alone():
    """Two worlds differing only in which card the steal took. Every seat but
    the two it moved between encodes them alike; those two see the card."""
    thief, victim = 0, 1
    world_a = _steal_world(int(Resource.WOOD), thief, victim)
    world_b = _steal_world(int(Resource.ORE), thief, victim)
    players = world_a._state.num_players

    for perspective in range(players):
        block_a = _ledger_block(encode(world_a, perspective), players)
        block_b = _ledger_block(encode(world_b, perspective), players)
        if perspective in (thief, victim):
            assert not np.array_equal(block_a, block_b), (
                f"seat {perspective} saw the card and forgot it"
            )
        else:
            assert np.array_equal(block_a, block_b), (
                f"perspective {perspective} leaked the stolen identity"
            )

    hand_a = encode(world_a, thief).globals[:NUM_RESOURCES]
    hand_b = encode(world_b, thief).globals[:NUM_RESOURCES]
    assert not np.array_equal(hand_a, hand_b)


def test_the_thief_and_victim_type_the_card_and_the_table_does_not():
    game = _steal_world(int(Resource.ORE))
    ledger = game.ledger

    assert ledger.seats[0].unknown == 1 and ledger.seats[0].known == [0] * NUM_RESOURCES
    assert ledger.seats[1].known == [0] * NUM_RESOURCES and ledger.seats[1].unknown == 1
    for seat in (0, 1):
        rows = ledger.for_seat(seat).seats
        assert rows[0].known[Resource.ORE] == 1 and rows[0].unknown == 0
        assert rows[1].known[Resource.WOOD] == 1 and rows[1].unknown == 0
    for seat in (2, 3):
        assert ledger.for_seat(seat) is ledger
    _assert_invariant(game)


def test_the_table_types_a_steal_from_a_hand_of_one_resource():
    """The victim's whole hand certified, all sheep: everyone knows the card."""
    ledger = PublicLedger.new(3)
    ledger.receive(1, int(Resource.SHEEP), 3)

    ledger.steal(thief=0, victim=1)

    assert ledger.seats[1].known == [0, 0, 2, 0, 0] and ledger.seats[1].unknown == 0
    assert ledger.seats[0].known == [0, 0, 1, 0, 0] and ledger.seats[0].unknown == 0


def test_a_seat_keeps_its_own_rows_through_every_later_public_change():
    ledger = PublicLedger.new(3)
    ledger.receive(1, int(Resource.WOOD), 1)
    ledger.receive(1, int(Resource.ORE), 1)
    ledger.steal(thief=0, victim=1, resource=int(Resource.WOOD))
    common, own = ledger, ledger.for_seat(0)
    assert own is not common and own is ledger.for_seat(0)
    assert ledger.for_seat(1) is not own

    ledger.receive(0, int(Resource.BRICK), 2)
    ledger.spend(0, int(Resource.WOOD), 1)
    ledger.show(2, [0, 0, 1, 0, 0])
    ledger.gain_unknown(2, 1)

    assert own.seats[0].known == [0, 2, 0, 0, 0] and own.seats[0].unknown == 0
    # The table could not type the stolen card, so the wood came from it.
    assert common.seats[0].known == [0, 2, 0, 0, 0] and common.seats[0].unknown == 0
    assert own.seats[2].want == common.seats[2].want == [0, 0, 1, 0, 0]
    assert own.seats[2].unknown == common.seats[2].unknown == 1
    assert own.seats[1].known == [0, 0, 0, 0, 1] and common.seats[1].unknown == 1


def test_an_unseen_steal_applies_untyped_to_a_seat_that_saw_another():
    ledger = PublicLedger.new(3)
    ledger.receive(1, int(Resource.WOOD), 1)
    ledger.receive(1, int(Resource.ORE), 1)
    ledger.steal(thief=0, victim=1, resource=int(Resource.WOOD))
    ledger.receive(2, int(Resource.WHEAT), 1)
    ledger.receive(2, int(Resource.SHEEP), 1)

    ledger.steal(thief=1, victim=2, resource=int(Resource.WHEAT))

    seat0 = ledger.for_seat(0).seats
    assert seat0[1].unknown == 1 and seat0[1].known[Resource.ORE] == 1
    assert ledger.for_seat(2).seats[1].known[Resource.WHEAT] == 1


def test_a_seat_that_saw_nothing_shares_the_common_rows():
    ledger = PublicLedger.new(4)
    assert ledger.private is None
    assert all(ledger.for_seat(s) is ledger for s in range(4))
    ledger.steal(thief=0, victim=1)
    assert ledger.private is None


def test_scrubbed_keeps_one_seats_own_rows():
    ledger = PublicLedger.new(3)
    ledger.receive(1, int(Resource.WOOD), 1)
    ledger.receive(1, int(Resource.ORE), 1)
    ledger.steal(thief=0, victim=1, resource=int(Resource.WOOD))

    kept = ledger.scrubbed(0)
    assert kept.for_seat(0).seats == ledger.for_seat(0).seats
    assert kept.for_seat(0) is not ledger.for_seat(0)
    assert kept.for_seat(1) is kept and kept.seats == ledger.seats
    assert ledger.scrubbed(2).private is None
    assert ledger.scrubbed(None).private is None

    alone = ledger.knowledge(1)
    assert alone.private is None and alone.seats == ledger.for_seat(1).seats


def test_imagine_for_a_seat_keeps_that_seats_rows_alone():
    game = _steal_world(int(Resource.ORE))
    whole = imagine(game, random.Random(1), randomize_deck=False)
    assert whole.ledger == game.ledger and whole.ledger is not game.ledger

    for seat in range(game._state.num_players):
        world = imagine(game, random.Random(1), randomize_deck=False, seat=seat)
        assert world.ledger.for_seat(seat).seats == game.ledger.for_seat(seat).seats
        for other in range(game._state.num_players):
            if other != seat:
                assert world.ledger.for_seat(other) is world.ledger
                assert world.ledger.seats == game.ledger.seats


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


def test_a_monopoly_leaves_every_other_seat_holding_none_of_it():
    game = after_setup()
    game.phase = Phase.MAIN
    game.current_player = 0
    game._state.dev_cards[0][DevCard.MONOPOLY] = 1
    for player in range(game._state.num_players):
        _set_known_hand(game, player, [0] * NUM_RESOURCES)
    _set_known_hand(game, 1, [1, 0, 0, 0, 1])
    game.ledger.steal(thief=2, victim=1)  # wood or ore, nobody says which
    game._state.hands[1][Resource.WOOD] -= 1
    game._state.hands[2][Resource.WOOD] += 1
    assert game.ledger.seats[2].unknown == 1

    play_monopoly_card(game, Resource.ORE)

    for seat in (1, 2, 3):
        assert game.ledger.seats[seat].most[Resource.ORE] == 0
    # Seat 2's card was wood or ore, and it holds no ore.
    assert game.ledger.seats[2].known == [1, 0, 0, 0, 0]
    assert game.ledger.seats[2].unknown == 0
    _assert_invariant(game)


def test_a_thiefs_card_is_one_the_victim_could_hold():
    ledger = PublicLedger.new(3)
    ledger.receive(1, int(Resource.WOOD), 1)
    ledger.receive(1, int(Resource.BRICK), 1)

    ledger.steal(thief=0, victim=1)

    row = ledger.seats[0]
    assert row.unknown == 1
    assert row.room() == [1, 1, 0, 0, 0]


def test_settle_types_the_cards_a_resources_room_must_take():
    row = SeatLedger(known=[0, 0, 0, 0, 0], unknown=3, most=[1, 0, 0, 0, 3])
    row.settle()
    assert row.known == [0, 0, 0, 0, 2] and row.unknown == 1
    assert row.room() == [1, 0, 0, 0, 1]


def test_settle_drops_bounds_a_hand_written_row_contradicts():
    row = SeatLedger(known=[0] * NUM_RESOURCES, unknown=3, most=[1, 0, 0, 0, 0])
    row.settle()
    assert row.known == [0] * NUM_RESOURCES and row.room() == [3] * NUM_RESOURCES


def test_a_row_written_by_hand_claims_no_bound():
    row = SeatLedger(known=[1, 0, 0, 0, 0], unknown=2)
    assert row.room() == [2] * NUM_RESOURCES
