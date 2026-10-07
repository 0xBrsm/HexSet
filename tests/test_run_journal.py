# SPDX-License-Identifier: GPL-3.0-only
"""A long run keeps every game as it finishes (`compete(journal=...)`), and a
run that stopped part-way resumes to the result it would have had."""
from __future__ import annotations

import ast
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

LINEUP = ["random"] * 2
QUICK = dict(action_cap=40)


def lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def test_every_game_is_in_the_journal_with_its_record(tmp_path):
    journal = tmp_path / "run.games.jsonl"
    tournament = compete(lineup_from_names(LINEUP), 6, workers=1, records=True,
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
    whole = compete(lineup, 8, workers=1, records=True, **QUICK)
    journal = tmp_path / "run.games.jsonl"
    compete(lineup, 8, workers=1, records=True, journal=journal, **QUICK)
    # Keep the header and three games, and tear the fourth as a dying writer would.
    kept = journal.read_text().splitlines(keepends=True)
    journal.write_text("".join(kept[:4]) + kept[4][: len(kept[4]) // 2])
    assert journal_tournament(journal).games == 3
    played = []
    resumed = compete(lineup, 8, workers=1, records=True, journal=journal, resume=True,
                      progress=lambda done, games, outcome: played.append(done), **QUICK)
    assert played == list(range(1, 9))
    for field in ("winners", "points", "turns", "seating"):
        assert getattr(resumed, field) == getattr(whole, field)
    assert [to_json(r) for r in resumed.records] == [to_json(r) for r in whole.records]
    _, outcomes = read_journal(journal)
    assert sorted(outcomes) == list(range(8))


def test_a_journal_that_holds_a_run_is_not_overwritten(tmp_path):
    journal = tmp_path / "run.games.jsonl"
    lineup = lineup_from_names(LINEUP)
    compete(lineup, 2, workers=1, journal=journal, **QUICK)
    with pytest.raises(FileExistsError, match="resume"):
        compete(lineup, 2, workers=1, journal=journal, **QUICK)
    with pytest.raises(ValueError, match="different run"):
        compete(lineup, 2, workers=1, seed=1, journal=journal, resume=True, **QUICK)
    with pytest.raises(ValueError, match="journal"):
        compete(lineup, 2, workers=1, resume=True, **QUICK)


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
    tournament = compete(lineup_from_names(LINEUP), 2, workers=1, records=True, **QUICK)
    path = tmp_path / "records.jsonl"
    body = "".join(to_json(r) + "\n" for r in tournament.records)
    path.write_text(body + to_json(tournament.records[0])[:50])
    assert len(list(read(str(path)))) == 2


# Runs short enough to lose nothing by: throughput is a timing benchmark of
# random bots, measured in seconds.
UNJOURNALED = {"hexset/bench/throughput.py"}


def test_every_run_in_the_package_keeps_its_games():
    """Every `compete` call in the package passes a journal: a long run that
    holds its games until it returns loses all of them to one crash."""
    src = Path(__file__).resolve().parents[1]
    offenders = []
    for path in sorted((src / "hexset").rglob("*.py")):
        rel = str(path.relative_to(src))
        if rel in UNJOURNALED or rel == "hexset/arena.py":
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if (isinstance(node, ast.Call)
                    and getattr(node.func, "id", getattr(node.func, "attr", None)) == "compete"
                    and not any(k.arg == "journal" for k in node.keywords)):
                offenders.append(f"{rel}:{node.lineno}")
    assert not offenders, f"compete() without journal=: {offenders}"


def test_a_spiral_run_deals_spiral_boards_and_says_so_in_its_header(tmp_path):
    random_journal, spiral_journal = tmp_path / "random.games.jsonl", tmp_path / "spiral.games.jsonl"
    compete(lineup_from_names(LINEUP), 2, workers=1, records=True, journal=random_journal, **QUICK)
    # Not antithetic, so game `index` is dealt board `index`.
    tournament = compete(lineup_from_names(LINEUP), 2, workers=1, records=True, antithetic=False,
                         board_mode="spiral", journal=spiral_journal, **QUICK)
    assert "board_mode" not in lines(random_journal)[0]
    assert lines(spiral_journal)[0]["board_mode"] == "spiral"
    for index, record in enumerate(tournament.records):
        dealt = spiral_base_board(random.Random(board_key(0, index)))
        assert deal_board(0, index, "spiral") == dealt
        assert tuple(record.tokens) == dealt.tokens
    with pytest.raises(ValueError, match=r"\(board_mode differ\)"):
        compete(lineup_from_names(LINEUP), 2, workers=1, records=True, antithetic=False, journal=spiral_journal,
                resume=True, **QUICK)


def test_an_unknown_board_mode_is_refused_before_any_game():
    with pytest.raises(ValueError, match="unknown board mode"):
        compete(lineup_from_names(LINEUP), 2, board_mode="hexagonal", **QUICK)
