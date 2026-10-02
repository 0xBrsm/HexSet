"""`hexset.server._journal` on the files a crash or a restart leaves: a torn
last line, a bad line mid-file, an undo against trade-round notes, and a seat
retired part-way through the game.
"""

from __future__ import annotations

import json
import random

from hexset.actions import Action, ActionType, legal_actions
from hexset.game import Phase, is_over, to_move
from hexset.server import _journal as journal
from hexset.server._journal import Journal
from hexset.server.wire import action_to_wire

from conftest import new_tables


def test_a_line_written_after_a_torn_one_starts_a_line_of_its_own(tmp_path):
    path = tmp_path / "g.jsonl"
    path.write_text('{"kind":"game"}\n{"kind":"action","st')

    Journal(directory=str(tmp_path), game_id="g").reopened(at_step=3)
    Journal(directory=str(tmp_path), game_id="g").abandoned()

    kinds = [event["kind"] for event in journal.read(path)]
    assert kinds == ["game", "reopened", "abandoned"]


def test_a_file_that_ends_on_a_line_break_gets_no_blank_line(tmp_path):
    path = tmp_path / "g.jsonl"
    path.write_text('{"kind":"game"}\n')

    Journal(directory=str(tmp_path), game_id="g").abandoned()

    assert path.read_text().count("\n") == 2
    assert "\n\n" not in path.read_text()


def test_read_skips_a_bad_line_and_keeps_the_rest(tmp_path):
    path = tmp_path / "g.jsonl"
    path.write_text('{"kind":"game"}\n{"kind":"act\n[1, 2]\n{"kind":"result"}\n')

    assert [event["kind"] for event in journal.read(path)] == ["game", "result"]


def test_an_undo_keeps_the_notes_made_before_the_step_it_returns_to():
    def note(step: int, kind: str) -> dict:
        return {"kind": "note", "step": step, "round": 1, "note": kind,
                "seat": 0, "actor": 0, "bundle": None}

    events = [note(2, "offer"), note(3, "nobody"), {"kind": "undo", "back_to": 2}]

    notes = journal.notes_of(events)

    assert sorted(notes) == [2]
    assert [n.kind for _, n in notes[2]] == ["offer"]


def _move(session, rng: random.Random) -> None:
    if session.awaiting_confirm is not None:
        session.submit(session.awaiting_confirm, action_to_wire(Action(ActionType.END_TURN)))
        return
    seat = to_move(session.game)
    session.submit(seat, action_to_wire(rng.choice(legal_actions(session.game))))


def test_a_seat_that_left_mid_game_is_retired_from_where_it_left_on_reopen(tmp_path):
    registry = new_tables(games_dir=str(tmp_path), seed=99)
    dealt = registry.handle("POST", "/api/games", {"bots": []}, None)
    code = dealt["code"]
    tokens = {registry.by_token(dealt["token"])[1]: dealt["token"]}
    while len(tokens) < 4:
        joined = registry.handle("POST", "/api/join", {"code": code}, None)
        tokens[registry.by_token(joined["token"])[1]] = joined["token"]
    session = registry.get(code).session
    rng = random.Random(4)
    for _ in range(3000):
        if session.game.turns >= 6 and session.game.phase is Phase.ROLL:
            break
        _move(session, rng)
    assert session.game.phase is Phase.ROLL
    leaver = next(s for s in tokens if s != session.game.current_player)
    registry.handle("POST", "/api/leave", {}, tokens[leaver])
    for _ in range(40):
        if is_over(session.game):
            break
        _move(session, rng)
    events = journal.read(next(tmp_path.glob("*.jsonl")))
    [at] = journal.retirements(events)
    assert at > 0, "retired after the first move, not before it"

    registry._tables.clear()  # what a restart leaves: the files, and no table
    reopened = registry.get(code).session

    assert reopened.game.locked == {leaver}
    assert reopened.game.turns == session.game.turns
    assert reopened.game.current_player == session.game.current_player
    assert reopened.game._state.hands == session.game._state.hands


def test_a_rename_is_journalled(tmp_path):
    registry = new_tables(games_dir=str(tmp_path))
    dealt = registry.handle("POST", "/api/games", {"bots": [], "name": "Ada"}, None)
    registry.handle("POST", "/api/name", {"name": "Grace"}, dealt["token"])

    lines = [json.loads(line) for line in next(tmp_path.glob("*.jsonl")).read_text().splitlines()]

    assert [line["name"] for line in lines if line["kind"] == "renamed"] == ["Grace"]
