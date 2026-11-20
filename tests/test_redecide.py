# SPDX-License-Identifier: GPL-3.0-only
"""`hexset.redecide`: a run's logged search decisions, found again in each
game's record and decided again by other seats, every arm reseeded alike at
each decision; the rows written game by game and resumed."""

from __future__ import annotations

import json

from hexset.arena import Entrant, compete, lineup_from_names, register_entrant_kind, register_preset
from hexset.mcts import Search
from hexset.redecide import Arm, matched, run

from journals import fresh
from test_mcts import Stub

register_entrant_kind("redecide-plain", lambda entrant, board, rng: Search(Stub(value=(0.0, 0.0)), simulations=8, wave=4, rng=rng))
register_preset("redecide-plain", Entrant("redecide-plain", kind="redecide-plain"), listed=False)
register_entrant_kind("redecide-first", lambda entrant, board, rng: Search(Stub(value=(0.0, 0.0), favour=1), simulations=8, wave=4, rng=rng))
register_preset("redecide-first", Entrant("redecide-first", kind="redecide-first"), listed=False)

SEED = 20260909


def nowhere(game, seat) -> bool:
    return False


def journal_of_searched_games():
    journal = fresh()
    compete(lineup_from_names(["redecide-plain", "random"]), 2, workers=1, action_cap=60, journal=journal)
    return journal


def rows(path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def games(path) -> list[dict]:
    return [r for r in rows(path) if r.get("kind") == "game"]


def test_every_logged_decision_is_found_again_in_its_record():
    journal = journal_of_searched_games()
    for line in (r for r in rows(journal) if r.get("kind") == "game"):
        logged = line["searches"][0]
        found = [d for _, _, _, d in matched(line, 0) if d is not None]
        assert logged and found == logged


def test_arms_that_read_alike_decide_alike(tmp_path):
    journal = journal_of_searched_games()
    out = tmp_path / "rows.jsonl"
    run(str(journal), str(out), [Arm("A", "redecide-plain", SEED), Arm("A2", "redecide-plain", SEED)])
    decisions = [r for r in rows(out) if r.get("kind") == "decision"]
    assert decisions
    assert all(r["arms"]["A"]["move"] == r["arms"]["A2"]["move"] for r in decisions)
    for tally in games(out):
        assert tally["matched"] == tally["logged"] and tally["kept"] == tally["searched"] > 0
    assert rows(out)[0]["kind"] == "header"


def test_another_seat_decides_some_of_them_otherwise(tmp_path):
    journal = journal_of_searched_games()
    out = tmp_path / "rows.jsonl"
    run(str(journal), str(out), [Arm("A", "redecide-plain", SEED), Arm("B", "redecide-first", SEED)])
    decisions = [r for r in rows(out) if r.get("kind") == "decision"]
    assert any(r["arms"]["A"]["move"] != r["arms"]["B"]["move"] for r in decisions)


def test_a_filter_skips_positions_and_a_rerun_resumes(tmp_path):
    journal = journal_of_searched_games()
    out = tmp_path / "rows.jsonl"
    arms = [Arm("A", "redecide-plain", SEED)]
    assert run(str(journal), str(out), arms, where="test_redecide:nowhere") == 2
    assert not [r for r in rows(out) if r.get("kind") == "decision"]
    assert all(t["kept"] == 0 and t["searched"] > 0 for t in games(out))
    before = rows(out)
    assert run(str(journal), str(out), arms, where="test_redecide:nowhere") == 0
    assert rows(out) == before
