from __future__ import annotations

import random

from catan.arena import PRESETS, spawn
from catan.behaviour import (
    Play,
    Walk,
    by_count,
    by_timing,
    per_game,
    seat_win_rates,
    walk,
    walks,
)
from catan.board.board import random_base_board
from catan.record import record_game


def some_records(count: int = 6, bot: str = "greedy"):
    out = []
    for seed in range(count):
        board = random_base_board(random.Random(2000 + seed))
        bots = [
            spawn(PRESETS[bot], board, random.Random(seed * 16 + seat))
            for seat in range(4)
        ]
        out.append(record_game(bots, board, seed))
    return out


def a_walk(plays, winner=0, game=0, players=4):
    return Walk(game=game, winner=winner, players=players, turns=50, plays=tuple(plays))


def test_a_walk_attributes_actions_to_the_seat_that_took_them():
    records = some_records(2)
    walked = walk(records[0], 0)
    assert walked is not None
    assert walked.winner == records[0].winner
    assert all(0 <= p.seat < 4 for p in walked.plays)
    assert all(0.0 <= p.progress <= 1.0 for p in walked.plays)
    # Every game builds settlements, since setup alone places two per seat.
    assert sum(walked.count(s, "settlement") for s in range(4)) > 0


def test_undecided_games_are_skipped():
    board = random_base_board(random.Random(0))
    bots = [spawn(PRESETS["random"], board, random.Random(s)) for s in range(4)]
    assert walk(record_game(bots, board, 0, action_cap=10), 0) is None
    assert walks([record_game(bots, board, 0, action_cap=10)]) == []


def test_counting_by_seat_and_kind():
    walked = a_walk([
        Play(0, "knight", 0.1),
        Play(0, "knight", 0.5),
        Play(1, "knight", 0.2),
        Play(0, "monopoly", 0.9),
    ])
    assert walked.count(0, "knight") == 2
    assert walked.count(1, "knight") == 1
    assert walked.count(2, "knight") == 0
    assert walked.count(0, "monopoly") == 1


def test_per_game_averages_over_seats_not_games():
    walked = [a_walk([Play(0, "knight", 0.5), Play(1, "knight", 0.5)])]
    # Two knights across four seats in one game.
    assert per_game(walked, "knight") == 0.5
    assert per_game([], "knight") == 0.0


def test_by_count_groups_every_seat_and_pools_the_tail():
    walked = [
        a_walk([Play(0, "knight", 0.1)] * 6, winner=0),
        a_walk([Play(1, "knight", 0.1)], winner=1, game=1),
    ]
    grouped = by_count(walked, "knight", cap=4)
    # Seat 0 played six, pooled into the 4+ bucket and won.
    assert grouped[4] == (1, 1)
    # Seat 1 played one and won.
    assert grouped[1] == (1, 1)
    # The remaining six seats played none and none of them won.
    assert grouped[0] == (6, 0)
    assert sum(games for games, _ in grouped.values()) == 8


def test_by_timing_uses_only_the_first_play_per_seat():
    walked = [a_walk([
        Play(0, "road_building", 0.05),
        Play(0, "road_building", 0.95),
    ], winner=0)]
    grouped = by_timing(walked, "road_building", buckets=5)
    assert grouped == {0: (1, 1)}


def test_by_timing_puts_the_very_end_in_the_last_bucket():
    walked = [a_walk([Play(0, "monopoly", 1.0)], winner=1)]
    assert by_timing(walked, "monopoly", buckets=5) == {4: (1, 0)}


def test_seat_win_rates_count_every_seat_once_per_game():
    walked = [a_walk([], winner=0), a_walk([], winner=0, game=1)]
    rates = seat_win_rates(walked)
    assert rates[0] == (2, 2)
    assert rates[1] == (2, 0)
    assert sorted(rates) == [0, 1, 2, 3]


def test_real_games_produce_sane_aggregates():
    walked = walks(some_records(6))
    assert walked
    seats = seat_win_rates(walked)
    assert sum(wins for _, wins in seats.values()) == len(walked)
    assert per_game(walked, "settlement") > 0
    knights = by_count(walked, "knight")
    assert sum(games for games, _ in knights.values()) == 4 * len(walked)
