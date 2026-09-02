# SPDX-License-Identifier: GPL-3.0-only
"""The duel's side split and seat geometry, which decide every verdict it reports.

Torch-free: `benchmarks.duel` imports its torch-bound modules where they are
used, so the lineup, the seating and the arena path all test on a box without
it. The arena is driven through a fake `compete` that hands back a fixed points
table, so the paired split can be checked slot by slot.
"""

from __future__ import annotations

import json
import statistics
from types import SimpleNamespace

import pytest

from benchmarks import duel
from benchmarks.duel import (
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


def test_the_weights_survive_the_rename():
    """`spawn` reads `kind` and `weights`; renaming must not touch either."""
    lineup = sides(
        lineup_from_names(
            ["network:/runs/a.pt", "network:/runs/a.pt", "network:/runs/b.pt", "network:/runs/b.pt"]
        ),
        "a",
        "b",
    )

    assert [entrant.weights for entrant in lineup] == [
        "/runs/a.pt",
        "/runs/a.pt",
        "/runs/b.pt",
        "/runs/b.pt",
    ]
    assert all(entrant.kind == "network" for entrant in lineup)


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


def test_the_arena_path_can_be_called_at_all():
    """A regression guard for the call site, not the helper.

    `sides` was extracted while `_via_arena` still bound a local of the same
    name further down, which makes the name local for the whole function and
    raises `UnboundLocalError` on the first line that uses it. The unit tests
    above all passed while the duel could not run.
    """
    import ast
    from pathlib import Path

    import benchmarks.duel as module

    tree = ast.parse(Path(module.__file__).read_text())
    functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        assigned = {
            target.id
            for inner in ast.walk(node)
            if isinstance(inner, ast.Assign)
            for target in inner.targets
            if isinstance(target, ast.Name)
        }
        assert not assigned & functions, (
            f"{node.name} assigns a local shadowing {sorted(assigned & functions)}"
        )


def test_default_workers_is_one_for_two_bare_checkpoint_paths(tmp_path):
    a = tmp_path / "a.pt"
    b = tmp_path / "b.pt"
    a.touch()
    b.touch()
    assert _default_workers(str(a), str(b)) == 1


def test_default_workers_is_one_for_two_network_prefixed_checkpoints():
    assert _default_workers("network:/runs/a.pt", "network:/runs/b.pt") == 1


def test_default_workers_is_26_against_a_preset_bot(tmp_path):
    """A scripted opponent cannot batch, and workers=1 left it unfinished."""
    a = tmp_path / "a.pt"
    a.touch()
    assert _default_workers(str(a), "search2-offers3") == 26


def test_default_workers_is_26_for_a_search_wrapped_checkpoint(tmp_path):
    """`netsearch:`/`netgreedy:`/`mcts:` still run a per-lane search."""
    a = tmp_path / "a.pt"
    a.touch()
    assert _default_workers(str(a), "netsearch:/runs/a.pt") == 26


def test_default_workers_is_26_when_neither_side_is_a_bare_network():
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


def test_sides_labels_an_interleaved_lineup_by_slot_not_by_position():
    names, mine, _ = arena_lineup(A, B, "interleaved")
    lineup = sides(lineup_from_names(names), "lam095-805", "ppo4-585", mine)

    assert [base_name(entrant.name) for entrant in lineup] == [
        "lam095-805",
        "ppo4-585",
        "lam095-805",
        "ppo4-585",
    ]
    assert [entrant.weights for entrant in lineup] == [
        "/runs/a.pt",
        "/runs/b.pt",
        "/runs/a.pt",
        "/runs/b.pt",
    ]
    # Repeat numbers still run 0, 1 within a side, and side A pools first.
    assert [entrant.name for entrant in lineup] == [
        "lam095-805#0",
        "ppo4-585#0",
        "lam095-805#1",
        "ppo4-585#1",
    ]
    grouped = pooled([Standing(entrant.name, 1, 4) for entrant in lineup], 4)
    assert [g.name for g in grouped] == ["lam095-805", "ppo4-585"]


def _fake_compete(seen: dict, points, turns=None, winners=None):
    """Stand in for `arena.compete`: records the lineup it was handed and
    returns a tournament with one fixed points row per game, in entrant order.

    `turns` and `winners` default to every game running 80 turns and being won
    by seat 0 -- ordinary finishes, so callers that do not care about game
    length or exhaustion see `exhausted == 0` as before this stand-in grew
    those fields.
    """

    def compete(lineup, games, *, seed, workers):
        seen["lineup"] = [entrant.weights for entrant in lineup]
        seen["names"] = [entrant.name for entrant in lineup]
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
    base = dict(a=A, b=B, games=4, duel_seed=20_000, workers=2)
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


def test_arena_verdict_turns_are_sane_when_nothing_is_exhausted(monkeypatch):
    monkeypatch.setattr("hexset.arena.compete", _fake_compete({}, POINTS))

    verdict = _via_arena(_arena_args(), "a", "b")

    assert 1 <= verdict["turns_mean"] <= MAX_TURNS
    assert verdict["exhausted"] == 0
    assert verdict["exhausted"] <= verdict["games"]


def test_main_prints_the_geometry_beside_the_worker_count(monkeypatch, capsys):
    monkeypatch.setattr("hexset.arena.compete", _fake_compete({}, POINTS))

    assert duel.main([A, B, "--workers", "2", "--games", "4", "--no-json"]) == 0
    out, err = capsys.readouterr()
    assert "--workers 2" in err
    assert "--geometry not given; defaulting to blocked" in err
    # stdout carries the verdict alone; everything else goes to stderr.
    assert json.loads(out)["geometry"] == "blocked"

    assert duel.main(
        [A, B, "--workers", "2", "--games", "4", "--no-json", "--geometry", "interleaved"]
    ) == 0
    _, err = capsys.readouterr()
    assert "--geometry interleaved" in err


def test_the_versus_path_refuses_a_blocked_geometry(capsys):
    """`collect.alternating` is the interleaving; asking for anything else at
    workers=1 must fail loudly rather than play interleaved under a wrong label."""
    code = duel.main([A, B, "--workers", "1", "--geometry", "blocked", "--no-json"])
    assert code == 2
    _, err = capsys.readouterr()
    assert "--geometry blocked is not available at --workers 1" in err
    assert "--workers 2" in err


def test_the_versus_path_accepts_interleaved_by_name(capsys, monkeypatch):
    """Naming the seating it plays anyway is allowed and is printed as such.

    `_via_versus` itself (the network-backed --workers 1 runner) now lives in
    `hexnet.duel` and is exercised there
    (`tests/hexnet/test_duel_versus.py::test_the_versus_verdict_records_interleaved`);
    this only pins that `benchmarks.duel.main` calls whatever registered
    itself as `_VERSUS_BACKEND`.
    """
    monkeypatch.setattr(duel, "_VERSUS_BACKEND", lambda args, la, lb: {
        "a": la, "b": lb, "games": 0, "win_rate": 0.5, "wilson_low": 0.0,
        "wilson_high": 1.0, "paired_vp": 0.0, "geometry": "interleaved",
    })
    code = duel.main([A, B, "--workers", "1", "--geometry", "interleaved", "--no-json"])
    assert code == 0
    _, err = capsys.readouterr()
    assert "--geometry interleaved (the only seating `train.versus` plays)" in err


def test_main_writes_the_new_fields_to_the_verdict_json(monkeypatch, tmp_path):
    monkeypatch.setattr("hexset.arena.compete", _fake_compete({}, POINTS))

    destination = tmp_path / "verdict.json"
    code = duel.main(
        [A, B, "--workers", "2", "--games", "4", "--json", str(destination)]
    )
    assert code == 0

    written = json.loads(destination.read_text())
    assert {"turns_mean", "turns_median", "turns_max", "exhausted"} <= written.keys()
    assert 1 <= written["turns_mean"] <= MAX_TURNS
    assert written["exhausted"] <= written["games"]


def test_no_json_still_writes_nothing(monkeypatch, tmp_path):
    monkeypatch.setattr("hexset.arena.compete", _fake_compete({}, POINTS))

    verdicts = tmp_path / "verdicts"
    verdicts.mkdir()
    code = duel.main(
        [A, B, "--workers", "2", "--games", "4", "--no-json", "--verdicts", str(verdicts)]
    )
    assert code == 0
    assert list(verdicts.iterdir()) == []
