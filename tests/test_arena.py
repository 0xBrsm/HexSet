from __future__ import annotations

import pickle
import random

import pytest

from catan.arena import (
    MAX_ACTIONS,
    PRESETS,
    Entrant,
    Estimate,
    Standing,
    Tournament,
    compete,
    entrant_from_name,
    lineup_from_names,
    mean_interval,
    play,
    pooled,
    seat_of,
    spawn,
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
    named = [e.name for e in lineup_from_names(["greedy", "random", "greedy"])]
    assert named == ["greedy#0", "random", "greedy#1"]
    with pytest.raises(ValueError, match="unknown bots: mcts"):
        lineup_from_names(["mcts", "random"])


def test_a_checkpoint_path_names_an_entrant_wherever_a_preset_would():
    lineup = lineup_from_names(
        ["network:/tmp/latest.pt", "network:/tmp/latest.pt", "greedy", "greedy"]
    )
    assert [e.name for e in lineup] == [
        "network#0",
        "network#1",
        "greedy#0",
        "greedy#1",
    ]
    assert lineup[0].kind == "network"
    assert lineup[0].weights == "/tmp/latest.pt"
    # The whole reason entrants are descriptions: this has to reach a worker.
    assert pickle.loads(pickle.dumps(lineup)) == lineup


def test_a_network_entrant_leaves_its_offer_budget_to_the_checkpoint():
    """`None` here means what it trained under, not the engine's eight.

    Scoring a policy on a horizon it never played is the mistake this default
    exists to make hard, so it is pinned here rather than left to `netbot`.
    """
    assert entrant_from_name("network:/tmp/latest.pt").max_offers is None


def test_pooling_adds_up_the_seats_a_side_took():
    standings = (
        Standing("network#0", 30, 100),
        Standing("network#1", 25, 100),
        Standing("greedy#0", 22, 100),
        Standing("greedy#1", 23, 100),
    )
    assert pooled(standings, 100) == [
        Standing("network", 55, 100),
        Standing("greedy", 45, 100),
    ]


def test_every_preset_can_be_built():
    board = random_base_board(random.Random(0))
    for entrant in PRESETS.values():
        assert spawn(entrant, board, random.Random(0)) is not None


def test_an_unknown_bot_kind_is_refused():
    board = random_base_board(random.Random(0))
    with pytest.raises(ValueError, match="unknown bot kind"):
        spawn(Entrant("bogus", kind="oracle"), board, random.Random(0))


def test_entrants_are_picklable_so_they_can_cross_a_process():
    """The reason entrants are data and not closures."""
    lineup = lineup_from_names(["greedy", "search2"])
    assert pickle.loads(pickle.dumps(lineup)) == lineup


def test_workers_do_not_change_the_result():
    """Parallelism may change the clock and nothing else."""
    lineup = lineup_from_names(["greedy", "random", "greedy", "random"])
    serial = compete(lineup, 8, seed=11, workers=1)
    parallel = compete(lineup, 8, seed=11, workers=4)
    assert serial.standings == parallel.standings
    assert serial.unfinished == parallel.unfinished
    assert serial.mean_turns == parallel.mean_turns


def test_greedy_beats_random_play():
    result = compete(lineup_from_names(["greedy", "greedy", "random", "random"]), 4)
    by_name = {s.name: s.wins for s in result.standings}
    assert by_name["greedy#0"] + by_name["greedy#1"] == result.games - result.unfinished
    assert by_name["random#0"] == by_name["random#1"] == 0


def test_the_default_action_cap_is_not_reached_by_ordinary_play():
    result = compete(lineup_from_names(["greedy"] * 4), 4)
    assert result.unfinished == 0
    assert MAX_ACTIONS > result.mean_turns


def test_the_winning_entrants_points_reach_the_victory_threshold():
    """Points are indexed by entrant, so the winner's row must show a win.

    Seat-order points would put an arbitrary seat's score at the winner's
    index, which is the one way this mapping can silently go wrong.
    """
    result = compete(lineup_from_names(["greedy"] * 4), 8, seed=3)

    assert result.unfinished == 0
    for winner, row in zip(result.winners, result.points):
        assert row[winner] >= 10
        assert row[winner] == max(row)


def test_stronger_entrants_score_more_points_than_the_seats_they_beat():
    result = compete(
        lineup_from_names(["greedy", "greedy", "random", "random"]), 8, seed=5
    )

    differences = [row[0] - row[2] for _, row in result.decided()]
    assert mean_interval(differences).lower > 0


def test_unfinished_games_are_left_out_of_the_decided_rows():
    result = Tournament(
        standings=(),
        games=3,
        unfinished=1,
        mean_turns=0.0,
        seconds=0.0,
        winners=(0, None, 0),
        points=((10, 6), (0, 99), (9, 5)),
    )

    assert result.decided() == [(0, (10, 6)), (0, (9, 5))]


def test_a_constant_paired_advantage_has_no_sampling_error():
    assert mean_interval([2.0] * 20) == Estimate(2.0, 2.0, 2.0, 20)


def test_too_few_point_samples_say_nothing_either_way():
    estimate = mean_interval([2.0])
    assert estimate.mean == 2.0
    assert estimate.lower == float("-inf")
    assert estimate.upper == float("inf")

    empty = mean_interval([])
    assert empty.samples == 0
    assert empty.lower == float("-inf")
    assert empty.upper == float("inf")
