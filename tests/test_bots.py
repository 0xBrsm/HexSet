# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import random

import pytest

from hexset.actions import Action, ActionType, apply, legal_actions
from hexset.board.board import pips, random_base_board
from hexset.board.terrain import Resource
from hexset.bots import RandomBot, SearchBot, greedy, own, paranoid, relative
from hexset.cards import DevCard
from hexset.evaluate import Evaluator
from hexset.game import (
    MAX_OFFERS_PER_TURN,
    ROLL_ODDS,
    Phase,
    imagine,
    is_over,
    propose_trade,
    roll_dice,
    start,
    to_move,
)
from hexset.state import place_settlement, upgrade_to_city
from hexset.trading import bundle
from hexset.victory import victory_points
from helpers import clear_hand, give, independent_vertices, mini_board


def a_game(seed: int = 0, players: int = 4):
    rng = random.Random(seed)
    board = random_base_board(rng)
    return start(board, players, rng)


def snapshot(game):
    state = game._state
    return (
        game.phase,
        game.current_player,
        game.turns,
        state.vertex_owner[:],
        state.vertex_building[:],
        state.edge_owner[:],
        state.robber,
        [hand[:] for hand in state.hands],
        state.bank[:],
        state.deck[:],
        game.rng.getstate(),
    )


def play_out(game, bot, cap: int = 20000) -> int:
    moves = 0
    while not is_over(game):
        apply(game, bot.choose(game))
        moves += 1
        if moves > cap:
            raise AssertionError("bot did not finish a game")
    return moves


def nine_points_and_a_city_to_come():
    """Player 0 one build from winning, holding exactly the cost of that build."""
    board = mini_board()
    game = start(board, 4, random.Random(0))
    game.phase = Phase.MAIN
    game.current_player = 0

    # Three cities, one settlement and two victory-point cards: nine points,
    # and the one winning build is a *fourth* city on the one settlement -- the
    # supply allows four, so the build the bot must find is a legal one, and
    # there is exactly one of it to find.
    spots = independent_vertices(board, 4)
    for vertex in spots[:3]:
        place_settlement(game._state, 0, vertex, connected=False)
        upgrade_to_city(game._state, 0, vertex)
    place_settlement(game._state, 0, spots[3], connected=False)
    game._state.dev_cards[0][DevCard.VICTORY_POINT] += 2
    give(game._state, 0, Resource.WHEAT, 2)
    give(game._state, 0, Resource.ORE, 3)

    assert victory_points(game._state, 0) == 9
    return game, spots[3]


def test_roll_odds_are_the_dice():
    assert sum(weight for _, weight in ROLL_ODDS) == pytest.approx(1.0)
    assert [roll for roll, _ in ROLL_ODDS] == list(range(2, 13))
    assert all(weight == pips(roll) / 36 for roll, weight in ROLL_ODDS)


def test_an_explicit_roll_is_resolved_as_rolled():
    game = a_game()
    while game.phase is not Phase.ROLL:
        apply(game, legal_actions(game)[0])

    assert roll_dice(game, 8) == 8
    assert game.last_roll == 8


def test_imagining_leaves_the_real_game_untouched():
    game = a_game()
    for _ in range(60):
        apply(game, RandomBot(random.Random(1)).choose(game))
    before = snapshot(game)

    copy = imagine(game, random.Random(2))
    play_out(copy, RandomBot(random.Random(3)))

    assert snapshot(game) == before
    assert is_over(copy)


def test_imagining_hides_the_deck_it_copies():
    game = a_game()
    copy = imagine(game, random.Random(4))
    assert sorted(copy._state.deck) == sorted(game._state.deck)
    assert copy._state.deck != game._state.deck


def test_hidden_deck_randomization_can_be_deferred():
    game = a_game()
    copy = imagine(game, random.Random(4), randomize_deck=False)
    assert copy._state.deck == game._state.deck
    assert copy._state.deck is not game._state.deck


def test_to_move_is_the_discarding_player_not_the_roller():
    game = a_game()
    while game.phase is not Phase.ROLL:
        apply(game, legal_actions(game)[0])
    game.current_player = 0
    game._state.hands[2] = [4, 4, 0, 0, 0]
    roll_dice(game, 7)

    assert game.phase is Phase.DISCARD
    assert to_move(game) == 2
    assert all(a.type is ActionType.DISCARD for a in legal_actions(game))


def test_a_random_bot_finishes_a_game():
    game = a_game(seed=3)
    play_out(game, RandomBot(random.Random(3)))
    assert is_over(game)


def test_greedy_finishes_a_game_sooner_than_random_play():
    random_game = a_game(seed=5)
    play_out(random_game, RandomBot(random.Random(5)))

    greedy_game = a_game(seed=5)
    play_out(greedy_game, greedy(Evaluator(greedy_game._state.board), random.Random(5)))

    assert greedy_game.won_by is not None
    assert greedy_game.turns < random_game.turns


def test_greedy_takes_a_winning_build():
    game, vertex = nine_points_and_a_city_to_come()
    chosen = greedy(Evaluator(game._state.board), random.Random(0)).choose(game)
    assert chosen == Action(ActionType.BUILD_CITY, vertex)


def test_search_takes_a_winning_build():
    game, vertex = nine_points_and_a_city_to_come()
    bot = SearchBot(Evaluator(game._state.board), depth=2, width=4, rng=random.Random(0))
    assert bot.choose(game) == Action(ActionType.BUILD_CITY, vertex)


def test_choosing_does_not_disturb_the_game_or_its_random_stream():
    game = a_game(seed=6)
    for _ in range(60):
        apply(game, RandomBot(random.Random(6)).choose(game))
    before = snapshot(game)

    SearchBot(Evaluator(game._state.board), depth=2, rng=random.Random(6)).choose(game)
    assert snapshot(game) == before


def test_a_beam_of_one_is_the_greedy_choice():
    game = a_game(seed=8)
    for _ in range(80):
        apply(game, RandomBot(random.Random(8)).choose(game))
    evaluator = Evaluator(game._state.board)

    beamed = SearchBot(evaluator, depth=3, width=1, rng=random.Random(0)).choose(game)
    assert beamed == greedy(evaluator, random.Random(0)).choose(game)


def test_the_stances_read_one_vector_three_ways():
    vector = [3.0, 5.0, 1.0, 0.0]
    assert own(vector, 0) == 3.0
    assert relative(vector, 0) == 3.0 - 2.0
    assert paranoid(vector, 0) == 3.0 - 5.0


def test_a_stance_only_matters_relative_to_the_table():
    """Lifting every seat alike is worth nothing to a bot playing relatively."""
    before = [3.0, 5.0, 1.0, 0.0]
    after = [value + 10 for value in before]
    assert own(after, 0) > own(before, 0)
    assert relative(after, 0) == pytest.approx(relative(before, 0))
    assert paranoid(after, 0) == pytest.approx(paranoid(before, 0))


def test_an_unknown_stance_is_refused():
    game = a_game()
    with pytest.raises(ValueError, match="unknown stance"):
        SearchBot(Evaluator(game._state.board), stance="spiteful")


def test_a_relative_bot_still_takes_a_winning_build():
    game, vertex = nine_points_and_a_city_to_come()
    chosen = greedy(
        Evaluator(game._state.board), random.Random(0), stance="relative"
    ).choose(game)
    assert chosen == Action(ActionType.BUILD_CITY, vertex)


def a_trade_that_wins_the_game_for_the_proposer():
    """Player 0 is nine points and one ore short of a winning city, and asks.

    The offer is good for player 1 in isolation — it moves their hand closer to
    a settlement — and fatal in context, because taking it ends the game.
    """
    board = mini_board()
    game = start(board, 4, random.Random(0))
    game.phase = Phase.MAIN
    game.current_player = 0

    spots = independent_vertices(board, 4)
    for vertex in spots[:3]:
        place_settlement(game._state, 0, vertex, connected=False)
        upgrade_to_city(game._state, 0, vertex)
    place_settlement(game._state, 0, spots[3], connected=False)
    game._state.dev_cards[0][DevCard.VICTORY_POINT] += 2
    assert victory_points(game._state, 0) == 9

    clear_hand(game._state, 0)
    give(game._state, 0, Resource.WHEAT, 2)
    give(game._state, 0, Resource.ORE, 2)
    give(game._state, 0, Resource.WOOD, 1)

    clear_hand(game._state, 1)
    give(game._state, 1, Resource.ORE, 1)
    give(game._state, 1, Resource.BRICK, 1)
    give(game._state, 1, Resource.WHEAT, 1)

    propose_trade(game, bundle(wood=1), bundle(ore=1))
    assert game.pending_responders == [1]
    return game


def test_a_relative_bot_will_not_trade_the_leader_into_a_win():
    game = a_trade_that_wins_the_game_for_the_proposer()
    evaluator = Evaluator(game._state.board)

    def responds_with(stance: str) -> ActionType:
        bot = SearchBot(
            evaluator, depth=2, width=6, rng=random.Random(0), stance=stance
        )
        return bot.choose(game).type

    assert responds_with("own") is ActionType.ACCEPT_TRADE
    assert responds_with("relative") is ActionType.DECLINE_TRADE


def test_only_subtracting_the_max_can_tell_opponents_apart():
    """Why partner choice needs `paranoid`, and gets nothing from `relative`.

    A trade hands the same value to whoever takes it. Subtracting the mean of
    the others moves by that value whichever opponent received it, so the choice
    is a tie by construction. Subtracting the largest moves only when the
    recipient was the one in front.
    """
    feed_the_leader = [10.0, 21.0, 5.0, 5.0]
    feed_a_trailer = [10.0, 20.0, 6.0, 5.0]

    assert relative(feed_the_leader, 0) == pytest.approx(relative(feed_a_trailer, 0))
    assert paranoid(feed_a_trailer, 0) > paranoid(feed_the_leader, 0)


def a_table_where_one_seat_is_far_ahead():
    """Seats 1 and 3 can both cover the same offer. Seat 1 has eight points."""
    board = mini_board()
    game = start(board, 4, random.Random(0))
    game.phase = Phase.MAIN
    game.current_player = 0

    spots = independent_vertices(board, 6)
    for vertex in spots[:4]:
        place_settlement(game._state, 1, vertex, connected=False)
        upgrade_to_city(game._state, 1, vertex)
    place_settlement(game._state, 3, spots[4], connected=False)

    for player in range(4):
        clear_hand(game._state, player)
    give(game._state, 0, Resource.WOOD, 2)
    give(game._state, 1, Resource.ORE, 1)
    give(game._state, 3, Resource.ORE, 1)
    return game, board


def test_a_choosing_proposer_asks_the_player_it_costs_least_to_feed():
    game, board = a_table_where_one_seat_is_far_ahead()
    action = Action(
        ActionType.PROPOSE_TRADE, give=bundle(wood=2), want=bundle(ore=1)
    )
    bot = greedy(
        Evaluator(board), random.Random(0), stance="paranoid", partner_choice=True
    )
    assert bot._addressed(game, action, 0).ask == (3, 1)


def test_a_proposer_that_does_not_choose_names_nobody():
    game, board = a_table_where_one_seat_is_far_ahead()
    action = Action(
        ActionType.PROPOSE_TRADE, give=bundle(wood=2), want=bundle(ore=1)
    )
    plain = greedy(Evaluator(board), random.Random(0), stance="paranoid")
    assert plain._addressed(game, action, 0).ask == ()


def _offers_per_turn(max_offers: int | None, actions: int = 1200) -> int:
    """Most offers any one turn saw, with every seat sharing the same budget."""
    game = a_game(seed=3)
    bots = [
        greedy(Evaluator(game._state.board), random.Random(seat), max_offers=max_offers)
        for seat in range(4)
    ]
    peak = 0
    for _ in range(actions):
        if is_over(game):
            break
        apply(game, bots[to_move(game)].choose(game))
        peak = max(peak, game.offers_made)
    return peak


def test_a_bot_can_hold_itself_below_the_engines_offer_cap():
    """The cap has to be the bot's own, not the engine's.

    An engine-wide limit lands on both entrants, and a duel cannot see a
    capability everyone receives. So this pins that the budget is spent by the
    bot while `MAX_OFFERS_PER_TURN` still permits more, which is the only
    arrangement that can measure what the extra offers are worth.
    """
    assert MAX_OFFERS_PER_TURN > 2
    assert _offers_per_turn(2) == 2
    assert _offers_per_turn(None) > 2


def test_the_same_seed_plays_the_same_game():
    def run():
        game = a_game(seed=9)
        bot = SearchBot(
            Evaluator(game._state.board), depth=2, width=4, rng=random.Random(9)
        )
        chosen = []
        for _ in range(30):
            action = bot.choose(game)
            chosen.append(action)
            apply(game, action)
        return chosen, snapshot(game)

    assert run() == run()
