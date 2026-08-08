from __future__ import annotations

import random

import pytest

from catan.arena import (
    FACTORIES,
    MAX_ACTIONS,
    Standing,
    compete,
    lineup_from_names,
    play,
    seat_of,
    wilson,
)
from catan.board.board import random_base_board
from catan.bots import RandomBot
from catan.game import is_over


@pytest.mark.parametrize("seats", [2, 3, 4])
def test_rotation_gives_every_entrant_every_seat_once(seats):
    for entrant in range(seats):
        taken = [seat_of(entrant, game, seats) for game in range(seats)]
        assert sorted(taken) == list(range(seats))
    for game in range(seats):
        lineup = [seat_of(entrant, game, seats) for entrant in range(seats)]
        assert sorted(lineup) == list(range(seats))


def test_wilson_brackets_the_observed_rate():
    low, high = wilson(50, 100)
    assert low < 0.5 < high


def test_wilson_narrows_as_games_accumulate():
    narrow = wilson(500, 1000)
    wide = wilson(50, 100)
    assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])


def test_wilson_stays_inside_the_unit_interval_at_the_extremes():
    assert wilson(0, 100)[0] == 0.0
    assert wilson(100, 100)[1] == 1.0
    assert wilson(0, 100)[1] > 0.0
    assert wilson(100, 100)[0] < 1.0


def test_wilson_says_nothing_without_games():
    assert wilson(0, 0) == (0.0, 1.0)


def test_a_standing_reports_its_rate():
    assert Standing("greedy", 3, 4).win_rate == 0.75


def test_play_seats_each_bot_at_its_own_index():
    rng = random.Random(0)
    board = random_base_board(rng)
    bots = [RandomBot(random.Random(i)) for i in range(4)]
    game = play(bots, board, rng)
    assert is_over(game)
    assert game.state.num_players == 4


def test_the_action_cap_stops_a_game_that_will_not_end():
    rng = random.Random(0)
    board = random_base_board(rng)
    bots = [RandomBot(random.Random(i)) for i in range(4)]
    game = play(bots, board, rng, action_cap=5)
    assert not is_over(game)


def test_every_game_is_won_or_counted_unfinished():
    result = compete(lineup_from_names(["random"] * 4), 4, seed=1)
    assert sum(s.wins for s in result.standings) + result.unfinished == result.games
    assert all(s.games == 4 for s in result.standings)
    assert result.mean_turns > 0


def test_the_same_seed_runs_the_same_tournament():
    lineup = lineup_from_names(["random", "greedy", "random", "greedy"])
    first = compete(lineup, 4, seed=2)
    second = compete(lineup, 4, seed=2)
    assert first.standings == second.standings


def test_a_lineup_the_rotation_cannot_balance_is_refused():
    with pytest.raises(ValueError, match="divide evenly"):
        compete(lineup_from_names(["random"] * 4), 6)


def test_a_tournament_needs_opponents():
    with pytest.raises(ValueError, match="at least two"):
        compete(lineup_from_names(["random"]), 4)


def test_repeated_bots_are_numbered_and_unknown_ones_rejected():
    named = [name for name, _ in lineup_from_names(["greedy", "random", "greedy"])]
    assert named == ["greedy#0", "random", "greedy#1"]
    with pytest.raises(ValueError, match="unknown bots: mcts"):
        lineup_from_names(["mcts", "random"])


def test_every_registered_bot_can_be_built():
    board = random_base_board(random.Random(0))
    for factory in FACTORIES.values():
        assert factory(board, random.Random(0)) is not None


def test_greedy_beats_random_play():
    result = compete(lineup_from_names(["greedy", "greedy", "random", "random"]), 4)
    by_name = {s.name: s.wins for s in result.standings}
    assert by_name["greedy#0"] + by_name["greedy#1"] == result.games - result.unfinished
    assert by_name["random#0"] == by_name["random#1"] == 0


def test_the_default_action_cap_is_not_reached_by_ordinary_play():
    result = compete(lineup_from_names(["greedy"] * 4), 4)
    assert result.unfinished == 0
    assert MAX_ACTIONS > result.mean_turns
