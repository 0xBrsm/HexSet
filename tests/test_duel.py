# SPDX-License-Identifier: GPL-3.0-only
"""The duel's side split and seat geometry, which decide every verdict.

Torch-free on purpose, so all of this tests on a box without torch. The arena
is driven through a fake `compete` returning a fixed points table.
"""

from __future__ import annotations

import statistics
from types import SimpleNamespace

import pytest

from hexset.bench import duel
from hexset.bench.duel import (
    _via_arena,
    arena_lineup,
    sides,
)
from hexset.arena import Standing, Tournament, base_name, lineup_from_names, pooled
from hexset.rules import STANDARD_GAME
from hexset.game import MAX_TURNS


def test_naming_the_sides_separates_two_checkpoints():
    lineup = sides(
        lineup_from_names(
            ["network:/runs/a.pt", "network:/runs/a.pt", "network:/runs/b.pt", "network:/runs/b.pt"]
        ),
        "ppo6-655",
        "ppo4-585",
    )

    assert [base_name(entrant.name) for entrant in lineup] == [
        "ppo6-655",
        "ppo6-655",
        "ppo4-585",
        "ppo4-585",
    ]
    grouped = pooled([Standing(entrant.name, 1, 1.0) for entrant in lineup], 4)
    assert len(grouped) == 2


def test_a_checkpoint_duelled_against_itself_still_has_two_sides():
    lineup = sides(
        lineup_from_names(["network:/runs/a.pt"] * 4), "ppo4-585", "ppo4-585"
    )

    assert [base_name(entrant.name) for entrant in lineup] == [
        "ppo4-585-a",
        "ppo4-585-a",
        "ppo4-585-b",
        "ppo4-585-b",
    ]


A, B = "network:/runs/a.pt", "network:/runs/b.pt"


def test_a_seating_is_any_pattern_of_a_and_b_slots():
    assert arena_lineup(A, B, "aab") == ([A, A, B], [0, 1], [2])
    assert arena_lineup(A, B, "abbba") == ([A, B, B, B, A], [0, 4], [1, 2, 3])


def test_a_comma_separated_seating_can_name_third_party_entrants():
    assert arena_lineup(A, B, "a,b,test-trader,random") == (
        [A, B, "test-trader", "random"], [0], [1]
    )


def _fake_compete(seen: dict, points, turns=None, winners=None):
    """Stand in for `arena.compete`, returning one fixed points row per game.
    `turns`/`winners` default to ordinary finishes.
    """

    def compete(lineup, games, *, seed, workers, records=False,
                worker_initializer=None, worker_initargs=(),
                trade_mode="round", game_type=STANDARD_GAME,
                turn_cap=MAX_TURNS, complement=None, progress=None, journal=None,
                resume=False):
        seen["progress"] = progress
        seen["complement"] = complement
        seen["game_type"] = game_type
        seen["worker_initializer"] = worker_initializer
        seen["worker_initargs"] = worker_initargs
        seen["trade_mode"] = trade_mode
        seen["lineup"] = [entrant.weights for entrant in lineup]
        seen["names"] = [entrant.name for entrant in lineup]
        seen["records"] = records
        seen["workers"] = workers
        game_turns = tuple(turns) if turns is not None else tuple(80 for _ in range(games))
        game_winners = (
            tuple(winners) if winners is not None else tuple(0 for _ in range(games))
        )
        if progress is not None:
            from hexset.arena import Outcome
            for n, (winner, t) in enumerate(zip(game_winners, game_turns), start=1):
                progress(n, games, Outcome(
                    winner=winner, seat=winner, turns=t,
                    seating=tuple(range(len(lineup))), points=tuple(points),
                    roads=(), settlements=(), cities=(), cleared=(), record=None,
                ))
        return Tournament(
            standings=tuple(Standing(e.name, game_winners.count(i), games)
                            for i, e in enumerate(lineup)),
            games=games,
            unfinished=sum(1 for w in game_winners if w is None),
            mean_turns=statistics.mean(game_turns) if game_turns else 0.0,
            seconds=0.0,
            winners=game_winners,
            seating=tuple(tuple(range(len(lineup))) for _ in range(games)),
            points=tuple(points for _ in range(games)),
            turns=game_turns,
        )

    return compete


def _arena_args(**overrides):
    base = dict(a=A, b=B, games=4, duel_seed=20_000, workers=2, records=None, runtime=None,
                trade_mode="round", game_type="standard", turn_cap=MAX_TURNS)
    base.update(overrides)
    return SimpleNamespace(**base)


# One points row in *entrant* order, chosen so the two geometries disagree:
# blocked [0,1] vs [2,3] is +2.5, interleaved [0,2] vs [1,3] is +3.5.
POINTS = (10, 2, 3, 4)


def test_arena_verdict_defaults_to_blocked_and_records_it(monkeypatch):
    seen: dict = {}
    monkeypatch.setattr("hexset.arena.compete", _fake_compete(seen, POINTS))

    verdict = _via_arena(_arena_args(), "a", "b")

    assert seen["lineup"] == ["/runs/a.pt", "/runs/a.pt", "/runs/b.pt", "/runs/b.pt"]
    assert verdict["geometry"] == "aabb"
    assert verdict["via"] == "arena.compete"
    assert verdict["paired_vp"] == pytest.approx(2.5)


def test_arena_verdict_reports_game_length_and_unfinished_games(monkeypatch):
    """A game the action cap cut short is `unfinished`; one that reached the
    turn cap never reaches the verdict (`compete` raises `Exhausted`), so the
    verdict carries no count of them."""
    turns = (120, 340, 200, 210)
    winners = (0, 2, None, None)
    monkeypatch.setattr(
        "hexset.arena.compete",
        _fake_compete({}, POINTS, turns=turns, winners=winners),
    )

    verdict = _via_arena(_arena_args(), "a", "b")

    assert verdict["turns_mean"] == pytest.approx(sum(turns) / len(turns))
    assert verdict["turns_median"] == pytest.approx(statistics.median(turns))
    assert verdict["turns_max"] == max(turns)
    assert verdict["unfinished"] == 2
    assert "exhausted" not in verdict and "truncated" not in verdict



@pytest.mark.parametrize("geometry,swap", [
    ("aabb", (2, 3, 0, 1)),
    ("ab", (1, 0)),
    ("abab", (1, 0, 3, 2)),
    ("a,b,random,random", (1, 0, 2, 3)),
    ("a,b,retired,retired", (1, 0, 2, 3)),
    ("random,a,b,random", (0, 2, 1, 3)),
    ("aab", None),
])
def test_the_pair_complement_swaps_the_sides_whatever_the_geometry(monkeypatch, geometry, swap):
    """The second game of a board pair trades the sides' seats and leaves
    every third party where it was; sides of unequal size cannot be traded,
    and keep the arena's own half-turn."""
    names, mine, theirs = arena_lineup(A, B, geometry)
    seats = len(names)
    assert duel.side_swap(mine, theirs, seats) == swap
    seen: dict = {}
    monkeypatch.setattr(
        "hexset.arena.compete", _fake_compete(seen, (10, 2, 3, 4)[:seats]),
    )
    _via_arena(_arena_args(games=2 * seats), "a", "b", geometry)
    assert seen["complement"] == swap


def test_duel_counts_side_a_by_slot_even_when_labels_collide(monkeypatch):
    """Side a is slot 1 alone here; the two third-party randoms share its label."""
    monkeypatch.setattr("hexset.arena.compete", _fake_compete({}, POINTS, winners=(0, 1, 2, None)))
    verdict = _via_arena(_arena_args(), "random", "random", "random,a,b,random")
    assert verdict["wins"] == 1
    assert verdict["win_rate"] == 1 / 4


def test_duel_intervals_use_board_pairs(monkeypatch):
    compete = _fake_compete({}, POINTS)
    def paired(*args, **kwargs):
        from dataclasses import replace
        return replace(compete(*args, **kwargs), points=((10, 10, 0, 0), (0, 0, 10, 10)) * 2)
    monkeypatch.setattr("hexset.arena.compete", paired)
    verdict = _via_arena(_arena_args(), "a", "b")
    assert verdict["boards"] == 2
    assert verdict["paired_vp_low"] == verdict["paired_vp_high"] == 0


def test_one_board_verdict_serializes_unestimable_vp_bounds_as_null(monkeypatch):
    import json

    monkeypatch.setattr("hexset.arena.compete", _fake_compete({}, (10, 3)))
    verdict = _via_arena(_arena_args(games=2), "a", "b", "ab")
    assert verdict["paired_vp_low"] is None
    assert verdict["paired_vp_high"] is None
    json.dumps(verdict, allow_nan=False)


def test_a_named_runtime_reaches_the_workers_that_spawn_entrants(monkeypatch):
    """Under spawn or forkserver a worker never ran this process's imports."""
    from hexset.arena import load_runtime

    seen = {}
    monkeypatch.setattr("hexset.arena.compete", _fake_compete(seen, POINTS))
    _via_arena(_arena_args(runtime=["json"]), "a", "b")

    assert seen["worker_initializer"] is load_runtime
    assert seen["worker_initargs"] == ("json",)
