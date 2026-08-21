from __future__ import annotations

import random

import pytest

from catan.actions import Action, ActionType, apply, legal_actions
from catan.board.board import random_base_board
from catan.board.terrain import Resource
from catan.game import (
    MAX_OFFERS_PER_TURN,
    Phase,
    accept_trade,
    decline_trade,
    end_turn,
    imagine,
    propose_trade,
    start,
    to_move,
)
from catan.trading import (
    Offer,
    bundle,
    can_accept,
    can_propose,
    execute,
    holds,
    responders,
    well_formed,
)
from helpers import give


def a_game(players: int = 4):
    rng = random.Random(0)
    game = start(random_base_board(rng), players, rng)
    game.phase = Phase.MAIN
    game.current_player = 0
    return game


def stocked(*hands: tuple[int, Resource, int]):
    game = a_game()
    for player, resource, count in hands:
        give(game.state, player, resource, count)
    return game


def test_a_bundle_reads_by_resource_name():
    assert bundle(wood=2, ore=1) == (2, 0, 0, 0, 1)


def test_both_sides_must_be_non_empty():
    assert not well_formed(Offer(0, bundle(wood=1), bundle()))
    assert not well_formed(Offer(0, bundle(), bundle(ore=1)))
    assert well_formed(Offer(0, bundle(wood=1), bundle(ore=1)))


def test_a_resource_may_not_appear_on_both_sides():
    """Two wood for one wood and one brick is just one wood for one brick."""
    assert not well_formed(Offer(0, bundle(wood=2), bundle(wood=1, brick=1)))
    assert well_formed(Offer(0, bundle(wood=2), bundle(brick=1)))


def test_negative_amounts_are_not_an_offer():
    assert not well_formed(Offer(0, (-1, 0, 0, 0, 0), bundle(ore=1)))


def test_offers_are_uncapped_in_size():
    """The rules place no limit on how much may change hands."""
    game = stocked((0, Resource.WOOD, 12), (1, Resource.ORE, 9))
    big = Offer(0, bundle(wood=12), bundle(ore=9))
    assert can_propose(game.state, big)
    assert can_accept(game.state, big, 1)


def test_a_player_cannot_offer_what_they_do_not_hold():
    game = stocked((0, Resource.WOOD, 1))
    assert not can_propose(game.state, Offer(0, bundle(wood=2), bundle(ore=1)))


def test_a_player_cannot_accept_what_they_cannot_cover():
    game = stocked((0, Resource.WOOD, 2), (1, Resource.ORE, 1))
    offer = Offer(0, bundle(wood=2), bundle(ore=2))
    assert not can_accept(game.state, offer, 1)


def test_nobody_may_take_their_own_offer():
    game = stocked((0, Resource.WOOD, 2), (0, Resource.ORE, 2))
    assert not can_accept(game.state, Offer(0, bundle(wood=2), bundle(ore=1)), 0)


def test_only_players_who_can_cover_the_offer_are_asked():
    game = stocked((0, Resource.WOOD, 2), (1, Resource.ORE, 1), (3, Resource.ORE, 4))
    offer = Offer(0, bundle(wood=2), bundle(ore=1))
    assert responders(game.state, offer) == (1, 3)


def test_the_offer_goes_round_the_table_from_the_proposer():
    """First refusal follows the proposer, so it belongs to no seat permanently."""
    game = stocked((2, Resource.WOOD, 2), (0, Resource.ORE, 1), (3, Resource.ORE, 1))
    offer = Offer(2, bundle(wood=2), bundle(ore=1))
    assert responders(game.state, offer) == (3, 0)


def test_the_proposer_can_name_who_it_would_rather_ask():
    game = stocked((0, Resource.WOOD, 2), (1, Resource.ORE, 1), (3, Resource.ORE, 1))
    propose_trade(game, bundle(wood=2), bundle(ore=1), ask=(3, 1))
    assert game.pending_responders == [3, 1]


def test_naming_cannot_add_a_player_who_cannot_cover_the_offer():
    game = stocked((0, Resource.WOOD, 2), (1, Resource.ORE, 1))
    propose_trade(game, bundle(wood=2), bundle(ore=1), ask=(2, 3, 1))
    assert game.pending_responders == [1]


def test_players_left_out_of_the_naming_queue_behind_those_named():
    game = stocked(
        (0, Resource.WOOD, 2),
        (1, Resource.ORE, 1),
        (2, Resource.ORE, 1),
        (3, Resource.ORE, 1),
    )
    propose_trade(game, bundle(wood=2), bundle(ore=1), ask=(3,))
    assert game.pending_responders == [3, 1, 2]


def test_going_round_wraps_past_the_last_seat():
    game = stocked((3, Resource.WOOD, 2), (0, Resource.ORE, 1), (2, Resource.ORE, 1))
    offer = Offer(3, bundle(wood=2), bundle(ore=1))
    assert responders(game.state, offer) == (0, 2)


def test_executing_moves_both_sides_and_conserves_the_cards():
    game = stocked((0, Resource.WOOD, 3), (2, Resource.ORE, 2))
    before = sum(sum(h) for h in game.state.hands)

    execute(game.state, Offer(0, bundle(wood=2), bundle(ore=1)), 2)
    assert game.state.hands[0][Resource.WOOD] == 1
    assert game.state.hands[0][Resource.ORE] == 1
    assert game.state.hands[2][Resource.WOOD] == 2
    assert game.state.hands[2][Resource.ORE] == 1
    assert sum(sum(h) for h in game.state.hands) == before


def test_executing_an_offer_nobody_can_cover_is_refused():
    game = stocked((0, Resource.WOOD, 2))
    with pytest.raises(ValueError, match="cannot take"):
        execute(game.state, Offer(0, bundle(wood=2), bundle(ore=1)), 1)


def test_holds_checks_every_resource():
    game = stocked((0, Resource.WOOD, 2), (0, Resource.ORE, 1))
    assert holds(game.state, 0, bundle(wood=2, ore=1))
    assert not holds(game.state, 0, bundle(wood=2, ore=2))


def test_proposing_hands_the_decision_to_the_players_being_asked():
    game = stocked((0, Resource.WOOD, 2), (1, Resource.ORE, 1), (2, Resource.ORE, 1))
    propose_trade(game, bundle(wood=2), bundle(ore=1))

    assert game.phase is Phase.TRADE_RESPOND
    assert game.pending_responders == [1, 2]
    assert to_move(game) == 1
    assert game.current_player == 0


def test_declining_passes_the_offer_along_and_then_drops_it():
    game = stocked((0, Resource.WOOD, 2), (1, Resource.ORE, 1), (2, Resource.ORE, 1))
    propose_trade(game, bundle(wood=2), bundle(ore=1))

    decline_trade(game, 1)
    assert to_move(game) == 2
    decline_trade(game, 2)
    assert game.phase is Phase.MAIN
    assert game.offer is None
    assert game.state.hands[0][Resource.WOOD] == 2


def test_the_first_player_to_accept_takes_the_trade():
    game = stocked((0, Resource.WOOD, 2), (1, Resource.ORE, 1), (2, Resource.ORE, 1))
    propose_trade(game, bundle(wood=2), bundle(ore=1))

    accept_trade(game, 1)
    assert game.phase is Phase.MAIN
    assert game.state.hands[1][Resource.WOOD] == 2
    assert game.state.hands[2][Resource.ORE] == 1


def test_only_the_player_being_asked_may_answer():
    game = stocked((0, Resource.WOOD, 2), (1, Resource.ORE, 1), (2, Resource.ORE, 1))
    propose_trade(game, bundle(wood=2), bundle(ore=1))
    with pytest.raises(ValueError, match="not the one being asked"):
        accept_trade(game, 2)


def test_an_offer_nobody_can_cover_never_reaches_a_response():
    game = stocked((0, Resource.WOOD, 2))
    propose_trade(game, bundle(wood=2), bundle(ore=1))
    assert game.phase is Phase.MAIN
    # It still counts as a move that was made.
    assert game.offers_made == 1


def test_a_turn_may_not_negotiate_forever():
    game = stocked((0, Resource.WOOD, 8))
    for _ in range(MAX_OFFERS_PER_TURN):
        propose_trade(game, bundle(wood=1), bundle(ore=1))
    with pytest.raises(ValueError, match="offers allowed per turn"):
        propose_trade(game, bundle(wood=1), bundle(ore=1))


def test_the_allowance_resets_with_the_turn():
    game = stocked((0, Resource.WOOD, 8))
    for _ in range(MAX_OFFERS_PER_TURN):
        propose_trade(game, bundle(wood=1), bundle(ore=1))
    end_turn(game)
    game.phase = Phase.MAIN
    game.current_player = 0
    propose_trade(game, bundle(wood=1), bundle(ore=1))
    assert game.offers_made == 1


def offers(game):
    return {
        (action.give, action.want)
        for action in legal_actions(game)
        if action.type is ActionType.PROPOSE_TRADE
    }


def test_a_bundle_already_declined_is_not_offered_again_this_turn():
    game = stocked((0, Resource.WOOD, 2), (1, Resource.ORE, 1), (1, Resource.BRICK, 1))
    repeat = (bundle(wood=1), bundle(ore=1))
    assert repeat in offers(game)

    propose_trade(game, *repeat)
    decline_trade(game)

    assert repeat not in offers(game)
    # Only that bundle goes. The rest of the sample is untouched.
    assert (bundle(wood=1), bundle(brick=1)) in offers(game)
    assert game.offered == {repeat}


def test_repeating_an_offer_stays_legal_even_though_it_is_not_enumerated():
    """The sample narrows; the rules do not.

    `_offer_actions` is documented as a sample rather than the whole legal set,
    and this is what keeps that true. A stronger policy that wants to ask twice
    still can, which is also why `--max-offers` remains the thing that bounds a
    turn.
    """
    game = stocked((0, Resource.WOOD, 2), (1, Resource.ORE, 1))
    propose_trade(game, bundle(wood=1), bundle(ore=1))
    decline_trade(game)

    propose_trade(game, bundle(wood=1), bundle(ore=1))
    assert game.offers_made == 2


def test_the_offer_record_resets_with_the_turn():
    game = stocked((0, Resource.WOOD, 2), (1, Resource.ORE, 1))
    propose_trade(game, bundle(wood=1), bundle(ore=1))
    decline_trade(game)
    end_turn(game)

    game.phase = Phase.MAIN
    game.current_player = 0
    assert game.offered == set()
    assert (bundle(wood=1), bundle(ore=1)) in offers(game)


def test_an_imagined_game_carries_what_has_been_offered():
    """Without this a search reads every repeat as a fresh option.

    `imagine` is the search's only view of the position, so a field the engine
    keeps and the copy drops is worse than one that never existed.
    """
    game = stocked((0, Resource.WOOD, 2), (1, Resource.ORE, 1))
    propose_trade(game, bundle(wood=1), bundle(ore=1))
    decline_trade(game)

    copy = imagine(game, random.Random(1))
    assert copy.offered == game.offered
    assert (bundle(wood=1), bundle(ore=1)) not in offers(copy)

    propose_trade(copy, bundle(wood=1), bundle(ore=2))
    assert game.offered == {(bundle(wood=1), bundle(ore=1))}


def test_responding_is_in_the_action_space():
    game = stocked((0, Resource.WOOD, 2), (1, Resource.ORE, 1))
    propose_trade(game, bundle(wood=2), bundle(ore=1))

    options = legal_actions(game)
    assert options == [
        Action(ActionType.ACCEPT_TRADE),
        Action(ActionType.DECLINE_TRADE),
    ]
    apply(game, Action(ActionType.ACCEPT_TRADE))
    assert game.phase is Phase.MAIN
    assert game.state.hands[1][Resource.WOOD] == 2


def test_declining_through_the_action_space_ends_the_offer():
    game = stocked((0, Resource.WOOD, 2), (1, Resource.ORE, 1))
    propose_trade(game, bundle(wood=2), bundle(ore=1))
    apply(game, Action(ActionType.DECLINE_TRADE))
    assert game.phase is Phase.MAIN
    assert game.state.hands[0][Resource.WOOD] == 2


def test_an_imagined_game_carries_the_open_offer():
    game = stocked((0, Resource.WOOD, 2), (1, Resource.ORE, 1))
    propose_trade(game, bundle(wood=2), bundle(ore=1))

    copy = imagine(game, random.Random(1))
    assert copy.offer == game.offer
    assert copy.pending_responders == game.pending_responders
    assert copy.offers_made == game.offers_made

    apply(copy, Action(ActionType.ACCEPT_TRADE))
    assert copy.phase is Phase.MAIN
    assert game.phase is Phase.TRADE_RESPOND
