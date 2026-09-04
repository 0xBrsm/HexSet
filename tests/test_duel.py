# SPDX-License-Identifier: GPL-3.0-only
"""The duel's side split and seat geometry, which decide every verdict it reports.

Torch-free: `hexset.bench.duel` imports its torch-bound modules where they are
used, so the lineup, the seating and the arena path all test on a box without
it. The arena is driven through a fake `compete` that hands back a fixed points
table, so the paired split can be checked slot by slot.
"""

from __future__ import annotations

import statistics
from types import SimpleNamespace

import pytest

from hexset.bench import duel
from hexset.bench.duel import (
    ARENA_GEOMETRY,
    GEOMETRIES,
    _default_workers,
    _via_arena,
    arena_lineup,
    sides,
)
from hexset.arena import Standing, Tournament, base_name, lineup_from_names, pooled
from hexset.game import MAX_TURNS


def test_two_checkpoints_arrive_sharing_one_name():
    """The reason `sides` exists, pinned so the fix cannot look gratuitous.

    `entrant_from_name` labels every `network:` spec "network" whatever
    checkpoint it carries, so `lineup_from_names` sees four repeats of one name
    rather than two of each side.
    """
    lineup = lineup_from_names(
        ["network:/runs/a.pt", "network:/runs/a.pt", "network:/runs/b.pt", "network:/runs/b.pt"]
    )

    assert {base_name(entrant.name) for entrant in lineup} == {"network"}


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
    # What the rename is for: two sides to pool, not one.
    grouped = pooled([Standing(entrant.name, 1, 1.0) for entrant in lineup], 4)
    assert len(grouped) == 2


def test_a_checkpoint_duelled_against_itself_still_has_two_sides():
    """The harness check that must read 50%, not a single pooled side."""
    lineup = sides(
        lineup_from_names(["network:/runs/a.pt"] * 4), "ppo4-585", "ppo4-585"
    )

    assert [base_name(entrant.name) for entrant in lineup] == [
        "ppo4-585-a",
        "ppo4-585-a",
        "ppo4-585-b",
        "ppo4-585-b",
    ]


def test_default_workers_is_one_only_for_two_bare_network_checkpoints(tmp_path):
    """`workers=1` batches two checkpoints in one process; anything else
    (a scripted opponent, a search wrapper) cannot batch and needs the
    26-worker default to finish in reasonable wall clock."""
    a = tmp_path / "a.pt"
    b = tmp_path / "b.pt"
    a.touch()
    b.touch()
    assert _default_workers(str(a), str(b)) == 1
    assert _default_workers("network:/runs/a.pt", "network:/runs/b.pt") == 1
    assert _default_workers(str(a), "search2-offers3") == 26
    assert _default_workers("search2-offers3", "random") == 26


# --------------------------------------------------------------------------
# seat geometry
# --------------------------------------------------------------------------
A, B = "network:/runs/a.pt", "network:/runs/b.pt"


def test_the_default_arena_lineup_is_the_recorded_one():
    """`[a, a, b, b]` with sides on `[0, 1]` / `[2, 3]` is what every recorded
    arena verdict played; the default must keep reproducing it exactly."""
    assert ARENA_GEOMETRY == "blocked"
    assert arena_lineup(A, B, ARENA_GEOMETRY) == ([A, A, B, B], [0, 1], [2, 3])


def test_the_interleaved_lineup_matches_the_seat_geometry_probe():
    """The lineup and slots `tmp/seat_geometry.py` used for the archived rows."""
    assert arena_lineup(A, B, "interleaved") == ([A, B, A, B], [0, 2], [1, 3])


def test_each_geometry_partitions_the_four_slots_between_the_sides():
    for order, mine, theirs in GEOMETRIES.values():
        assert len(order) == 4
        assert sorted(mine + theirs) == [0, 1, 2, 3]
        assert [order[i] for i in mine] == ["a", "a"]
        assert [order[i] for i in theirs] == ["b", "b"]


def _fake_compete(seen: dict, points, turns=None, winners=None):
    """Stand in for `arena.compete`: records the lineup it was handed and
    returns a tournament with one fixed points row per game, in entrant order.

    `turns` and `winners` default to every game running 80 turns and being won
    by seat 0 -- ordinary finishes, so callers that do not care about game
    length or exhaustion see `exhausted == 0` as before this stand-in grew
    those fields.
    """

    def compete(lineup, games, *, seed, workers, records=False):
        seen["lineup"] = [entrant.weights for entrant in lineup]
        seen["names"] = [entrant.name for entrant in lineup]
        seen["records"] = records
        game_turns = tuple(turns) if turns is not None else tuple(80 for _ in range(games))
        game_winners = (
            tuple(winners) if winners is not None else tuple(0 for _ in range(games))
        )
        return Tournament(
            standings=tuple(Standing(e.name, 1, games) for e in lineup),
            games=games,
            unfinished=sum(1 for w in game_winners if w is None),
            mean_turns=statistics.mean(game_turns) if game_turns else 0.0,
            seconds=0.0,
            winners=game_winners,
            points=tuple(points for _ in range(games)),
            turns=game_turns,
        )

    return compete


def _arena_args(**overrides):
    base = dict(a=A, b=B, games=4, duel_seed=20_000, workers=2, records=None)
    base.update(overrides)
    return SimpleNamespace(**base)


# One points row in *entrant* order. Blocked reads slots [0,1] against [2,3]:
# (10+2)/2 - (3+4)/2 = +2.5. Interleaved reads [0,2] against [1,3]:
# (10+3)/2 - (2+4)/2 = +3.5. Same row, different verdict -- which is the point.
POINTS = (10, 2, 3, 4)


def test_arena_verdict_defaults_to_blocked_and_records_it(monkeypatch):
    seen: dict = {}
    monkeypatch.setattr("hexset.arena.compete", _fake_compete(seen, POINTS))

    verdict = _via_arena(_arena_args(), "a", "b")

    assert seen["lineup"] == ["/runs/a.pt", "/runs/a.pt", "/runs/b.pt", "/runs/b.pt"]
    assert verdict["geometry"] == "blocked"
    assert verdict["via"] == "arena.compete"
    assert verdict["paired_vp"] == pytest.approx(2.5)


def test_arena_verdict_can_play_interleaved_and_says_so(monkeypatch):
    seen: dict = {}
    monkeypatch.setattr("hexset.arena.compete", _fake_compete(seen, POINTS))

    verdict = _via_arena(_arena_args(), "a", "b", "interleaved")

    assert seen["lineup"] == ["/runs/a.pt", "/runs/b.pt", "/runs/a.pt", "/runs/b.pt"]
    assert seen["names"] == ["a#0", "b#0", "a#1", "b#1"]
    assert verdict["geometry"] == "interleaved"
    assert verdict["paired_vp"] == pytest.approx(3.5)


def test_arena_verdict_reports_game_length_and_exhaustion(monkeypatch):
    """The P1half gap this closes: a verdict with nothing to read game length
    from. Two games run out `MAX_TURNS` without a winner -- exhausted -- and
    two finish normally, one of them short."""
    turns = (120, 340, MAX_TURNS, MAX_TURNS)
    winners = (0, 2, None, None)
    monkeypatch.setattr(
        "hexset.arena.compete",
        _fake_compete({}, POINTS, turns=turns, winners=winners),
    )

    verdict = _via_arena(_arena_args(), "a", "b")

    assert verdict["turns_mean"] == pytest.approx(sum(turns) / len(turns))
    assert verdict["turns_median"] == pytest.approx(statistics.median(turns))
    assert verdict["turns_max"] == max(turns)
    assert verdict["exhausted"] == 2
    # Existing fields keep their name and meaning: `unfinished` still counts
    # every game with no winner, `exhausted` is the subset that ran out the
    # turn clock rather than the action cap.
    assert verdict["unfinished"] == 2


def test_the_versus_path_refuses_a_blocked_geometry(capsys):
    """`collect.alternating` is the interleaving; asking for anything else at
    workers=1 must fail loudly rather than play interleaved under a wrong label."""
    code = duel.main([A, B, "--workers", "1", "--geometry", "blocked", "--no-json"])
    assert code == 2
    _, err = capsys.readouterr()
    assert "--geometry blocked is not available at --workers 1" in err
    assert "--workers 2" in err
