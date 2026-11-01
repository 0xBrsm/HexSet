# SPDX-License-Identifier: GPL-3.0-only
"""A long run keeps every game as it finishes (`compete(journal=...)`), and a
run that stopped part-way resumes to the result it would have had."""
from __future__ import annotations

import ast
import inspect
import json
import random
from pathlib import Path

import pytest

from hexset import gamelog
from hexset.arena import (
    Exhausted, board_key, compete, deal_board, journal_tournament, lineup_from_names, read_journal,
)
from hexset.board.board import spiral_base_board
from hexset.game import UNSTRUCTURED_TURN_CAP
from hexset.record import read, to_json
from journals import fresh

LINEUP = ["random"] * 2
QUICK = dict(action_cap=40)


def lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def test_every_game_is_in_the_journal_with_its_record(tmp_path):
    journal = tmp_path / "run.games.jsonl"
    tournament = compete(lineup_from_names(LINEUP), 6, workers=1,
                         journal=journal, **QUICK)
    entries = lines(journal)
    assert entries[0]["kind"] == "header" and entries[0]["games"] == 6
    games = [e for e in entries[1:] if e["kind"] == "game"]
    assert sorted(e["index"] for e in games) == list(range(6))
    header, outcomes = read_journal(journal)
    assert [outcomes[i].winner for i in range(6)] == list(tournament.winners)
    assert [outcomes[i].turns for i in range(6)] == list(tournament.turns)
    assert [to_json(outcomes[i].record) for i in range(6)] == [to_json(r) for r in tournament.records]


def test_a_parallel_run_writes_every_game_as_it_finishes(tmp_path):
    journal = tmp_path / "run.games.jsonl"
    tournament = compete(lineup_from_names(LINEUP), 8, workers=2, journal=journal, **QUICK)
    _, outcomes = read_journal(journal)
    assert sorted(outcomes) == list(range(8))
    assert [outcomes[i].points for i in range(8)] == list(tournament.points)


def test_a_run_stopped_part_way_resumes_to_its_own_result(tmp_path):
    lineup = lineup_from_names(LINEUP)
    whole = compete(lineup, 8, workers=1, **QUICK, journal=fresh())
    journal = tmp_path / "run.games.jsonl"
    compete(lineup, 8, workers=1, journal=journal, **QUICK)
    # Keep the header, the source line and three games, and tear the fourth as
    # a dying writer would.
    kept = journal.read_text().splitlines(keepends=True)
    assert [json.loads(k)["kind"] for k in kept[:2]] == ["header", "source"]
    journal.write_text("".join(kept[:5]) + kept[5][: len(kept[5]) // 2])
    assert journal_tournament(journal).games == 3
    played = []
    resumed = compete(lineup, 8, workers=1, journal=journal, resume=True,
                      progress=lambda done, games, outcome: played.append(done), **QUICK)
    assert played == list(range(1, 9))
    for field in ("winners", "points", "turns", "seating"):
        assert getattr(resumed, field) == getattr(whole, field)
    assert [to_json(r) for r in resumed.records] == [to_json(r) for r in whole.records]
    _, outcomes = read_journal(journal)
    assert sorted(outcomes) == list(range(8))



def test_a_run_resumed_with_more_games_plays_only_the_indices_after_them(tmp_path):
    """An extension: the journal keeps the first games, the run plays the
    next indices on the same seed, and the whole equals a run planned that
    long from the start. Fewer games than planned is a different run."""
    lineup = lineup_from_names(LINEUP)
    whole = compete(lineup, 8, workers=1, **QUICK, journal=fresh())
    journal = tmp_path / "run.games.jsonl"
    compete(lineup, 4, workers=1, journal=journal, **QUICK)
    played = []
    extended = compete(lineup, 8, workers=1, journal=journal, resume=True,
                       progress=lambda done, games, outcome: played.append(outcome), **QUICK)
    entries = lines(journal)
    assert entries[0]["games"] == 4                       # the header keeps the first plan
    sources = [e for e in entries if e["kind"] == "source"]
    assert [(s["games"], s.get("extended_to")) for s in sources] == [(0, None), (4, 8)]
    assert sorted(e["index"] for e in entries if e["kind"] == "game") == list(range(8))
    for field in ("winners", "points", "turns", "seating"):
        assert getattr(extended, field) == getattr(whole, field)
    header, outcomes = read_journal(journal)
    assert header["games"] == 8 and sorted(outcomes) == list(range(8))
    compete(lineup, 8, workers=1, journal=journal, resume=True, **QUICK)   # the extended plan resumes
    with pytest.raises(ValueError, match="different run"):
        compete(lineup, 4, workers=1, journal=journal, resume=True, **QUICK)

def test_each_opening_of_a_journal_says_which_code_writes_the_games_after_it(tmp_path, monkeypatch):
    from hexset import arena
    lineup = lineup_from_names(LINEUP)
    journal = tmp_path / "run.games.jsonl"
    monkeypatch.setattr(arena, "_source", lambda _file: {"commit": "a" * 40, "dirty": False})
    compete(lineup, 2, workers=1, journal=journal, **QUICK)
    # Stop after the first game, then resume on other code: the same run,
    # and the journal says which code played which games.
    kept = lines(journal)[:3]
    assert [e["kind"] for e in kept] == ["header", "source", "game"]
    journal.write_text("".join(json.dumps(e) + "\n" for e in kept))
    monkeypatch.setattr(arena, "_source", lambda _file: {"commit": "b" * 40, "dirty": True})
    compete(lineup, 2, workers=1, journal=journal, resume=True, **QUICK)
    sources = [e for e in lines(journal) if e["kind"] == "source"]
    assert [(s["games"], s["commit"], s["dirty"]) for s in sources] == [
        (0, "a" * 40, False), (1, "b" * 40, True)]
    assert all(s["hexset"] for s in sources)
    _, outcomes = read_journal(journal)
    assert sorted(outcomes) == [0, 1]


def test_a_journal_that_holds_a_run_is_not_overwritten(tmp_path):
    journal = tmp_path / "run.games.jsonl"
    lineup = lineup_from_names(LINEUP)
    compete(lineup, 2, workers=1, journal=journal, **QUICK)
    with pytest.raises(FileExistsError, match="resume"):
        compete(lineup, 2, workers=1, journal=journal, **QUICK)
    with pytest.raises(ValueError, match="different run"):
        compete(lineup, 2, workers=1, seed=1, journal=journal, resume=True, **QUICK)
    with pytest.raises(TypeError, match="journal"):
        compete(lineup, 2, workers=1, **QUICK)
    with pytest.raises(ValueError, match="journal"):
        compete(lineup, 2, workers=1, journal="", **QUICK)


def test_a_stuck_game_is_kept_before_the_run_stops(tmp_path):
    journal = tmp_path / "run.games.jsonl"
    with pytest.raises(Exhausted):
        compete(lineup_from_names(LINEUP), 4, workers=1, turn_cap=2, journal=journal)
    assert lines(journal)[-1]["kind"] == "exhausted"


def test_a_torn_last_line_is_dropped_and_anything_else_refused(tmp_path):
    path = tmp_path / "log.jsonl"
    gamelog.append(path, {"kind": "header"})
    gamelog.append(path, {"n": 1})
    with open(path, "a") as handle:
        handle.write('{"n": 2, "par')
    assert gamelog.read_lines(path) == [{"kind": "header"}, {"n": 1}]
    path.write_text('{"a": 1}\nnot json\n{"b": 2}\n')
    with pytest.raises(ValueError, match=":2"):
        gamelog.read_lines(path)


def test_records_read_past_a_torn_last_record(tmp_path):
    tournament = compete(lineup_from_names(LINEUP), 2, workers=1, **QUICK, journal=fresh())
    path = tmp_path / "records.jsonl"
    body = "".join(to_json(r) + "\n" for r in tournament.records)
    path.write_text(body + to_json(tournament.records[0])[:50])
    assert len(list(read(str(path)))) == 2


# Runs short enough to lose nothing by: throughput is a timing benchmark of
# random bots, measured in seconds.
def _required(function, name: str) -> bool:
    """Whether `function` takes `name` with no default."""
    parameter = inspect.signature(function).parameters.get(name)
    return parameter is not None and parameter.default is inspect.Parameter.empty


def test_no_game_is_played_without_being_kept():
    """Every way the package plays whole games writes each one, record
    included, to a journal it requires: none takes a default that keeps
    nothing, none has a switch that skips the record, and the engine's loop
    files every step on a tape."""
    from hexset import arena, play, record
    from hexset.bench import throughput, trade_census, versus
    from hexset.gym.lanes import LaneEnv

    for function in (arena.compete, arena.compete_lanes, versus.compete_batched,
                     trade_census.run_census, throughput.run, record.record_game, play.play_random_game,
                     arena._play_one, arena._play_dealt):
        assert _required(function, "journal"), f"{function.__qualname__} plays games without a journal"
        assert "records" not in inspect.signature(function).parameters, function.__qualname__
    assert _required(arena.run_seated, "tape"), "the engine's loop runs without recording"
    assert "records" not in inspect.signature(LaneEnv).parameters
    assert not hasattr(arena, "play") and not hasattr(arena, "play_game")


def test_only_the_arena_reaches_its_dealt_player():
    """`arena._play_dealt` is the arena's one player of a dealt game, and it
    writes every game it plays; outside the arena a game is played through
    `compete`, `compete_lanes` or `_play_one`."""
    src = Path(__file__).resolve().parents[1]
    offenders = []
    for path in sorted((src / "hexset").rglob("*.py")):
        rel = str(path.relative_to(src))
        for node in ast.walk(ast.parse(path.read_text())):
            name = getattr(node, "id", getattr(node, "attr", None))
            if name == "_play_dealt" and rel != "hexset/arena.py":
                offenders.append(f"{rel}:{node.lineno}")
    assert not offenders, f"_play_dealt outside the arena: {offenders}"


def test_the_dealt_player_writes_its_game_and_refuses_no_journal(tmp_path):
    """There is no way to play a dealt game without naming where it goes: the
    player refuses an empty journal before dealing anything, and writes the
    game's line itself before returning."""
    from hexset.arena import _play_dealt
    from hexset.rules import STANDARD_GAME

    job = (tuple(lineup_from_names(LINEUP)), 5, 13, 40, True, "round", STANDARD_GAME, 300, "random")
    for empty in ("", None):
        with pytest.raises(ValueError, match="journal"):
            _play_dealt(job, empty)
    journal = tmp_path / "dealt.games.jsonl"
    outcome = _play_dealt(job, journal)
    (line,) = lines(journal)
    assert line["index"] == 5 and line["record"] == json.loads(json.dumps(line["record"]))
    assert outcome.index == 5 and outcome.record is not None


def test_a_game_played_one_at_a_time_is_kept_whole(tmp_path):
    """`_play_one` writes its game before it returns: a header naming the
    engine, then the game's line with its record, entrants and seed."""
    from hexset.arena import _play_one, read_game_journal
    from hexset.rules import STANDARD_GAME

    journal = tmp_path / "one.games.jsonl"
    lineup = tuple(lineup_from_names(LINEUP))
    outcome = _play_one((lineup, 3, 11, 40, True, "round", STANDARD_GAME, 300, "random"), journal)
    header, games = read_game_journal(journal)
    assert header["kind"] == "header" and header["hexset"]
    ((line, read),) = games
    assert line["key"] == 3 and line["seed"] == 11 and line["entrants"] == [e.name for e in lineup]
    assert read.record == outcome.record and read.record is not None
    _play_one((lineup, 4, 11, 40, True, "round", STANDARD_GAME, 300, "random"), journal)
    header_again, games = read_game_journal(journal)
    assert header_again == header and [line["key"] for line, _ in games] == [3, 4]


def test_a_spiral_run_deals_spiral_boards_and_says_so_in_its_header(tmp_path):
    random_journal, spiral_journal = tmp_path / "random.games.jsonl", tmp_path / "spiral.games.jsonl"
    compete(lineup_from_names(LINEUP), 2, workers=1, journal=random_journal, **QUICK)
    # Not antithetic, so game `index` is dealt board `index`.
    tournament = compete(lineup_from_names(LINEUP), 2, workers=1, antithetic=False,
                         board_mode="spiral", journal=spiral_journal, **QUICK)
    assert "board_mode" not in lines(random_journal)[0]
    assert lines(spiral_journal)[0]["board_mode"] == "spiral"
    for index, record in enumerate(tournament.records):
        dealt = spiral_base_board(random.Random(board_key(0, index)))
        assert deal_board(0, index, "spiral") == dealt
        assert tuple(record.tokens) == dealt.tokens
    with pytest.raises(ValueError, match=r"\(board_mode differ\)"):
        compete(lineup_from_names(LINEUP), 2, workers=1, antithetic=False, journal=spiral_journal,
                resume=True, **QUICK)


def test_an_unknown_board_mode_is_refused_before_any_game():
    with pytest.raises(ValueError, match="unknown board mode"):
        compete(lineup_from_names(LINEUP), 2, board_mode="hexagonal", **QUICK, journal=fresh())
