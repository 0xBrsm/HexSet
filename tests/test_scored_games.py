from __future__ import annotations

import pytest

from catan.arena import Entrant
from catan.scored_games import Estimate, ScoredTournament, mean_interval, scored_compete


def test_a_paired_interval_uses_the_difference_within_each_game():
    result = ScoredTournament(
        standings=(),
        winners=(0, 1, 0, 1),
        points=((10, 6), (3, 8), (9, 5), (4, 7)),
        unfinished=0,
        seconds=0.0,
    )

    estimate = result.paired_points(0, 1)

    assert estimate.mean == 0.0
    assert estimate.lower < 0.0 < estimate.upper


def test_unfinished_games_are_left_out_of_a_paired_interval():
    result = ScoredTournament(
        standings=(),
        winners=(0, None, 0),
        points=((10, 6), (0, 99), (9, 5)),
        unfinished=1,
        seconds=0.0,
    )

    assert result.paired_points(0, 1).samples == 2


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


def test_a_scored_population_requires_four_seats_and_balanced_rotation():
    entrants = [Entrant(str(i), kind="random") for i in range(4)]
    with pytest.raises(ValueError, match="exactly four"):
        scored_compete(entrants[:3], 4)
    with pytest.raises(ValueError, match="divide evenly"):
        scored_compete(entrants, 6)


def test_population_games_keep_points_from_losing_seats():
    entrants = [Entrant(str(i), kind="random") for i in range(4)]
    result = scored_compete(entrants, 4, seed=7, action_cap=1)

    assert result.games == 4
    assert result.unfinished == 4
    assert len(result.points) == 4
    assert all(len(row) == 4 for row in result.points)


def test_scored_games_are_worker_independent():
    entrants = [Entrant(str(i), kind="random") for i in range(4)]
    serial = scored_compete(entrants, 4, seed=9, action_cap=40)
    parallel = scored_compete(entrants, 4, seed=9, action_cap=40, workers=2)

    assert serial.winners == parallel.winners
    assert serial.points == parallel.points
