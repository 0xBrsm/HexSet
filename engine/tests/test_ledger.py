# SPDX-License-Identifier: GPL-3.0-only
"""`hexset.ledger`: the public-knowledge reconstruction of each seat's hand.

The property test (`test_the_ledger_never_overclaims_over_long_playouts`) is
the spec this module exists to satisfy: over long randomized playouts, every
seat's ledger must always account for exactly the true hand size and never
certify more of a resource than the seat truly holds. Everything else here
pins one documented convention at a time — a steal, an over-draw, monopoly,
discard — either directly against `PublicLedger` or through the `hexset.game`
wiring that calls it.
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
    play_knight_card,
    play_monopoly_card,
    roll_dice,
    start,
    submit_discard,
)
from hexset.ledger import PublicLedger, SeatLedger
from hexset.play import step_randomly


def a_game(players: int = 4, seed: int = 0):
    rng = random.Random(seed)
    return start(random_base_board(rng), players, rng)


def after_setup(seed: int = 0, players: int = 4):
    """The opening placements played out through `apply`, so the ledger has
    already tracked the second-round settlement grants like any other public
    gain, and hexes have occupants for a robber move to steal from."""
    game = a_game(players, seed)
    while game.phase in (Phase.SETUP_SETTLEMENT, Phase.SETUP_ROAD):
        apply(game, legal_actions(game)[0])
    return game


def _set_known_hand(game, player: int, counts: list[int]) -> None:
    """Give `player` exactly `counts` from the bank, keeping `game.ledger` in
    sync. Unlike `helpers.give`/`clear_hand`, which poke `state.hands`
    directly, this is safe to use in a test that goes on to inspect the
    ledger -- `give`/`clear_hand` alone would desync it."""
    state = game.state
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
        true_hand = game.state.hands[seat]
        assert seat_ledger.total() == sum(true_hand), (
            f"seat {seat}: sum(known)+unknown={seat_ledger.total()} "
            f"!= true hand total {sum(true_hand)}"
        )
        for r, k in enumerate(seat_ledger.known):
            assert k <= true_hand[r], (
                f"seat {seat} resource {r}: known={k} > true={true_hand[r]}"
            )


def _steal_action(game, thief: int, victim: int, kind=ActionType.MOVE_ROBBER):
    """The `(hex, victim)` pair `move_robber_to`/`play_knight_card` want, for
    a target hex that actually pairs `thief` moving the robber with `victim`
    occupying it -- `victim_of` resolves the action's opaque victim slot the
    same way `actions.apply` does."""
    game.current_player = thief
    for action in legal_actions(game):
        if action.type is kind and victim_of(game, action.b) == victim:
            return action.a
    raise AssertionError(f"no {kind.name} pairs {thief} -> {victim} on this board")


def _ledger_block(obs, players: int = 4) -> np.ndarray:
    """The tail of `obs.globals` the ledger owns -- `(players - 1) *
    (NUM_RESOURCES + 1)` floats, each opponent's `known[5]` then `unknown`,
    seat-relative, own seat excluded. Mirrors `encoding._ledger_parts`'s
    width without importing a private helper across modules."""
    width = (players - 1) * (NUM_RESOURCES + 1)
    return obs.globals[-width:]


# --- the property test: the spec ---


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("players", [3, 4])
def test_the_ledger_never_overclaims_over_long_playouts(players, seed):
    rng = random.Random(seed)
    game = start(random_base_board(rng), players, rng)
    for _ in range(500):
        if game.phase is Phase.GAME_OVER:
            break
        step_randomly(game, rng)
        _assert_invariant(game)


def test_a_fresh_game_has_an_empty_ledger():
    game = a_game()
    for seat_ledger in game.ledger.seats:
        assert seat_ledger.known == [0] * NUM_RESOURCES
        assert seat_ledger.unknown == 0


def test_setup_grants_are_public():
    """The second-round settlement grant is exact and public, so a fresh
    ledger tracks it with zero `unknown` throughout setup."""
    game = after_setup()
    _assert_invariant(game)
    for seat_ledger in game.ledger.seats:
        assert seat_ledger.unknown == 0


def test_production_is_exact_and_public():
    game = after_setup()
    game.phase = Phase.ROLL
    game.rng = random.Random(3)
    for _ in range(20):
        if game.phase is Phase.GAME_OVER:
            break
        if game.phase is Phase.ROLL:
            roll_dice(game)
        else:
            step_randomly(game, game.rng)
    _assert_invariant(game)


# --- the steal convention ---


def test_a_steal_credits_the_thief_with_exactly_one_unknown_card():
    game = after_setup()
    game.phase = Phase.ROBBER
    for player in range(game.state.num_players):
        _set_known_hand(game, player, [0] * NUM_RESOURCES)
    _set_known_hand(game, 1, [2, 0, 3, 0, 0])  # wood + sheep, known exactly
    target = _steal_action(game, thief=0, victim=1)

    move_robber_to(game, target, 1)

    assert game.ledger.seats[0].unknown == 1
    assert game.ledger.seats[0].known == [0] * NUM_RESOURCES
    _assert_invariant(game)


def test_a_victimless_robber_move_touches_no_ledger():
    game = after_setup()
    game.phase = Phase.ROBBER
    before = [s.copy() for s in game.ledger.seats]
    # A target with nobody on it, or nobody holding cards, resolves to no
    # victim -- legal_actions would offer it as a plain MOVE_ROBBER with
    # `victim_of` returning None.
    target = next(
        a.a for a in legal_actions(game) if a.type is ActionType.MOVE_ROBBER and victim_of(game, a.b) is None
    )
    move_robber_to(game, target, None)
    for prior, seat_ledger in zip(before, game.ledger.seats):
        assert seat_ledger.known == prior.known
        assert seat_ledger.unknown == prior.unknown


def test_a_knight_steal_updates_the_ledger_the_same_way_as_the_robber():
    game = after_setup()
    game.phase = Phase.MAIN
    game.current_player = 0
    game.state.dev_cards[0][DevCard.KNIGHT] = 1
    for player in range(game.state.num_players):
        _set_known_hand(game, player, [0] * NUM_RESOURCES)
    _set_known_hand(game, 1, [1, 1, 0, 0, 0])
    target = _steal_action(game, thief=0, victim=1, kind=ActionType.PLAY_KNIGHT)

    play_knight_card(game, target, 1)

    assert game.ledger.seats[0].unknown == 1
    _assert_invariant(game)


def test_a_steal_floors_every_known_resource_by_one():
    """`PublicLedger.steal` never reads which resource was actually taken --
    it floors *every* `known[r]` by one (never below zero) and re-solves
    `unknown` from the seat's own previously tracked total, rather than
    decrementing one specific entry. Mixed hand, so the floor visibly
    touches more than one resource: known=[2,0,3,0,0] (5 total, no
    unknown) -> known=[1,0,2,0,0] (3 total), and the one card that left
    plus the two units the floor gave up become `unknown`."""
    ledger = PublicLedger.new(2)
    ledger.receive(1, int(Resource.WOOD), 2)
    ledger.receive(1, int(Resource.SHEEP), 3)

    ledger.steal(thief=0, victim=1)

    assert ledger.seats[1].known == [1, 0, 2, 0, 0]
    assert ledger.seats[1].unknown == 1
    assert ledger.seats[1].total() == 4  # 5 true cards, one stolen
    assert ledger.seats[0].unknown == 1
    assert ledger.seats[0].known == [0] * NUM_RESOURCES


def test_a_steal_from_a_fully_uncertain_hand_only_grows_unknown():
    """When the ledger already has zero certified cards of every resource
    for the victim (the whole hand is already `unknown`), flooring `known`
    is a no-op and the entire loss lands on `unknown`."""
    ledger = PublicLedger.new(2)
    ledger.gain_unknown(1, 3)  # victim's hand is 3 cards, none typed

    ledger.steal(thief=0, victim=1)

    assert ledger.seats[1].known == [0] * NUM_RESOURCES
    assert ledger.seats[1].unknown == 2
    assert ledger.seats[0].unknown == 1


def test_a_steal_can_balloon_uncertainty_by_up_to_four():
    """The honest cost of identity-independence: a victim certified for one
    of every resource (`known` sums to `NUM_RESOURCES`, `unknown == 0`)
    loses only one true card, but every entry floors to zero and `unknown`
    absorbs the whole prior total minus one -- a jump of `NUM_RESOURCES - 1`
    cards' worth of certainty in a single steal."""
    ledger = PublicLedger.new(2)
    for resource in range(NUM_RESOURCES):
        ledger.receive(1, resource, 1)

    ledger.steal(thief=0, victim=1)

    assert ledger.seats[1].known == [0] * NUM_RESOURCES
    assert ledger.seats[1].unknown == NUM_RESOURCES - 1 == 4
    assert ledger.seats[1].total() == NUM_RESOURCES - 1


def test_a_steal_is_identity_independent_in_the_encoding():
    """The regression test for the leak an earlier version of this module
    had: resolving a steal's loss against the *true* stolen resource made
    `known[r]` for that specific resource visibly drop, which told every
    seat watching the encoded ledger block exactly which resource was
    taken -- exactly the private half of a steal `encoding`'s
    information-set rule says must never reach the observation (style per
    `test_encoding.test_opponent_hand_contents_do_not_leak`).

    Two worlds, identical except which single resource the victim holds
    (so the steal is deterministic -- one card, one type -- and differs
    only in identity). Every seat's encoded ledger block must be
    byte-identical between them. The thief's own hand (a different globals
    block entirely, exact by design) is the one place the worlds are
    allowed -- and expected -- to differ: the thief genuinely knows what
    they took."""
    thief, victim = 0, 1

    def a_steal_world(single_resource: int, seed: int = 42):
        game = after_setup(seed)
        game.phase = Phase.ROBBER
        for player in range(game.state.num_players):
            _set_known_hand(game, player, [0] * NUM_RESOURCES)
        hand = [0] * NUM_RESOURCES
        hand[single_resource] = 1
        _set_known_hand(game, victim, hand)
        target = _steal_action(game, thief, victim)
        move_robber_to(game, target, victim)
        return game

    world_a = a_steal_world(int(Resource.WOOD))
    world_b = a_steal_world(int(Resource.ORE))
    players = world_a.state.num_players

    for perspective in range(players):
        block_a = _ledger_block(encode(world_a, perspective), players)
        block_b = _ledger_block(encode(world_b, perspective), players)
        assert np.array_equal(block_a, block_b), (
            f"perspective {perspective} leaked the stolen identity"
        )

    # Sanity: the two worlds really are different, and the difference is
    # visible exactly where it should be -- the thief's own hand, not the
    # ledger block just checked above.
    hand_a = encode(world_a, thief).globals[:NUM_RESOURCES]
    hand_b = encode(world_b, thief).globals[:NUM_RESOURCES]
    assert not np.array_equal(hand_a, hand_b)


# --- resolving uncertainty: over-draw, monopoly, discard ---


def test_an_over_draw_spend_resolves_unknown_cards():
    """`known[resource]` is short of what a public spend takes: the deficit
    can only have come from `unknown`, so it moves out of there instead of
    driving `known[resource]` negative."""
    ledger = PublicLedger.new(1)
    ledger.receive(0, int(Resource.WOOD), 1)
    ledger.gain_unknown(0, 2)  # true wood count is really 3

    ledger.spend(0, int(Resource.WOOD), 3)

    assert ledger.seats[0].known[Resource.WOOD] == 0
    assert ledger.seats[0].unknown == 0


def test_a_spend_beyond_the_seats_total_clamps_rather_than_going_negative():
    """A desynced ledger (this fixture never credited the second card) is
    exactly what a test built directly on `state.hands` looks like to
    `spend` -- `unknown` bottoms out at zero instead of raising, since the
    proof in `spend`'s docstring only promises safety when the ledger
    started in sync, which most of the suite has no reason to keep true."""
    ledger = PublicLedger.new(1)
    ledger.receive(0, int(Resource.WOOD), 1)
    ledger.spend(0, int(Resource.WOOD), 2)
    assert ledger.seats[0].known[Resource.WOOD] == 0
    assert ledger.seats[0].unknown == 0


def test_monopoly_re_pins_the_announced_resource():
    """Monopoly forces every other seat to publicly hand over every card of
    one resource -- even a share that was sitting in `unknown` -- so it
    resolves ambiguity rather than creating it."""
    game = after_setup()
    game.phase = Phase.MAIN
    game.current_player = 0
    game.state.dev_cards[0][DevCard.MONOPOLY] = 1
    for player in range(game.state.num_players):
        _set_known_hand(game, player, [0] * NUM_RESOURCES)
    # Seat 1 holds 3 sheep, but the ledger only certifies 1 of them -- the
    # other 2 are folded into `unknown`, standing in for a resource whose
    # exact composition the ledger could not otherwise pin.
    game.state.bank[Resource.SHEEP] -= 2
    game.state.hands[1][Resource.SHEEP] += 2
    game.ledger.gain_unknown(1, 2)
    _set_known_hand(game, 2, [0, 0, 1, 0, 0])

    play_monopoly_card(game, Resource.SHEEP)

    assert game.state.hands[1][Resource.SHEEP] == 0
    assert game.state.hands[2][Resource.SHEEP] == 0
    assert game.ledger.seats[1].known[Resource.SHEEP] == 0
    assert game.ledger.seats[1].unknown == 0
    assert game.ledger.seats[2].known[Resource.SHEEP] == 0
    # The thief's gain is fully public (monopoly announces the count), so it
    # is exact -- unlike a steal's `unknown`-only credit.
    assert game.ledger.seats[0].known[Resource.SHEEP] == 3
    assert game.ledger.seats[0].unknown == 0
    _assert_invariant(game)


def test_a_discard_reveals_the_resource_it_names():
    game = after_setup()
    game.discard_quota = [0] * game.state.num_players
    game.phase = Phase.DISCARD
    _set_known_hand(game, 0, [0] * NUM_RESOURCES)
    game.state.bank[Resource.ORE] -= 1
    game.state.hands[0][Resource.ORE] += 1
    game.ledger.gain_unknown(0, 1)
    game.discard_quota[0] = 1

    discard_one(game, 0, Resource.ORE)

    assert game.ledger.seats[0].known == [0] * NUM_RESOURCES
    assert game.ledger.seats[0].unknown == 0
    _assert_invariant(game)


def test_submit_discard_is_also_public():
    game = after_setup()
    game.discard_quota = [0] * game.state.num_players
    game.phase = Phase.DISCARD
    _set_known_hand(game, 0, [2, 2, 0, 0, 0])
    game.discard_quota[0] = 2

    submit_discard(game, 0, [1, 1, 0, 0, 0])

    assert game.ledger.seats[0].known == [1, 1, 0, 0, 0]
    _assert_invariant(game)


# --- carried through game copy ---


def test_imagine_copies_the_ledger_independently():
    game = after_setup()
    _set_known_hand(game, 0, [1, 0, 0, 0, 0])
    clone = imagine(game, random.Random(5))

    assert clone.ledger.seats[0].known == game.ledger.seats[0].known
    assert clone.ledger.seats[0] is not game.ledger.seats[0]

    clone.ledger.receive(0, int(Resource.ORE), 3)
    assert game.ledger.seats[0].known[Resource.ORE] == 0
