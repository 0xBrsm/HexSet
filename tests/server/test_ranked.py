"""Ranked play: the queue deals the table, pairing the clients that have met
least and filling the rest with the server's bot, no two versioned players
meeting past the pair cap; ranked play counts against no game allowance; a ranked
table's seats are fixed and it runs on a clock; a restart closes the ranked
games it cut short; and only ranked games reach the rankings."""

from __future__ import annotations

import hashlib
import json
import random

import pytest

from conftest import new_tables
from hexset.actions import Action, ActionType, legal_actions
from hexset.game import is_over, to_move
from hexset.server import mcptools
from hexset.server.api import ApiError, Origin
from hexset.server.ranked import PAIR_CAP, entrant
from hexset.server.wire import action_to_wire

FILL = "test-trader"
API = Origin(ip="203.0.113.9")


def client(letter: str, version: str = "1") -> dict:
    return {"id": letter * 64, "name": f"bot-{letter}", "version": version, "owner": "octocat",
            "description": f"Test client {letter}."}


def queue(tables, who: dict, game_type: str = "duel-variant", origin=None) -> str:
    return tables.handle("POST", "/api/ranked", {"game_type": game_type, "client": who}, None, origin)["ticket"]


def polled(tables, ticket: str) -> dict:
    return tables.handle("GET", f"/api/ranked/{ticket}", {}, None)


def seats_of(tables, ticket: str):
    got = polled(tables, ticket)
    assert got["status"] == "matched", got
    table = tables.get(got["code"])
    return table, got


def _play_out(session, rng: random.Random) -> None:
    for _ in range(3000):
        if is_over(session.game):
            return
        if session.awaiting_confirm is not None:
            session.submit(session.awaiting_confirm, action_to_wire(Action(ActionType.END_TURN)))
            continue
        seat = to_move(session.game)
        session.submit(seat, action_to_wire(rng.choice(legal_actions(session.game))))
    raise AssertionError("the seeded game did not finish")


def test_a_lone_client_is_seated_with_the_fill_bot_at_a_ranked_table(tmp_path):
    tables = new_tables(games_dir=str(tmp_path), ranked_bot=FILL)
    ticket = queue(tables, client("a"))
    assert polled(tables, ticket)["status"] == "waiting"
    tables.queue.tick()
    table, got = seats_of(tables, ticket)
    assert table.session.ranked and got["token"] and got["game_type"] == "duel-variant"
    assert [s.kind.value for s in table.seats[:2]].count("bot") == 1
    header = json.loads(next(tmp_path.glob("*.jsonl")).read_text().splitlines()[0])
    assert header["ranked"] is True
    # The game log names the version each seat played as: the client's here, a bot's from its profile.
    assert [(c["id"], c["version"]) for c in header["clients"].values() if c] == [("a" * 64, "1")]
    # The queue's seats are fixed.
    bot_seat = next(i for i, s in enumerate(table.seats[:2]) if s.kind.value == "bot")
    for path, body in (("/api/bot", {"seat": bot_seat, "model": FILL}), ("/api/close", {"seat": bot_seat})):
        with pytest.raises(ApiError) as refused:
            tables.handle("POST", path, body, got["token"])
        assert refused.value.status == 409


def test_waiting_clients_meet_each_other_first_and_least_met_first():
    tables = new_tables(ranked_bot=FILL)
    meetings, _ = tables.queue._history()
    meetings[frozenset(("a" * 64, "b" * 64))] = 3
    a, b, c = (queue(tables, client(x)) for x in "abc")
    tables.queue.tick()
    table_a, _ = seats_of(tables, a)
    table_c, _ = seats_of(tables, c)
    table_b, _ = seats_of(tables, b)
    assert table_a is table_c and table_b is not table_a
    assert all(s.kind.value == "player" for s in table_a.seats[:2])
    assert meetings[frozenset(("a" * 64, "c" * 64))] == 1


def test_one_client_never_sits_twice_at_a_table():
    tables = new_tables(ranked_bot=FILL)
    first, second = queue(tables, client("a")), queue(tables, client("a"))
    tables.queue.tick()
    assert seats_of(tables, first)[0] is not seats_of(tables, second)[0]


def pair(a: tuple, b: tuple) -> frozenset:
    return frozenset((a, b))


def test_the_fill_bot_stops_at_the_cap_for_a_version_and_not_for_the_next():
    tables = new_tables(ranked_bot=FILL)
    _, pairs = tables.queue._history()
    pairs[pair(entrant(client("a")), tables.queue._filler(FILL))] = PAIR_CAP
    capped = queue(tables, client("a"))
    tables.queue.tick()
    assert polled(tables, capped)["status"] == "waiting"
    fresh = queue(tables, client("a", version="2"))
    tables.queue.tick()
    assert polled(tables, fresh)["status"] == "matched"
    assert polled(tables, capped)["status"] == "waiting"
    other = queue(tables, client("b"))
    tables.queue.tick()
    assert seats_of(tables, capped)[0] is seats_of(tables, other)[0]


def test_two_versions_that_met_at_the_cap_are_not_seated_together_again():
    tables = new_tables()
    _, pairs = tables.queue._history()
    pairs[pair(entrant(client("a")), entrant(client("b")))] = PAIR_CAP
    a, b = queue(tables, client("a")), queue(tables, client("b"))
    tables.queue.tick()
    assert [polled(tables, t)["status"] for t in (a, b)] == ["waiting", "waiting"]
    fresh = queue(tables, client("b", version="2"))
    tables.queue.tick()
    assert seats_of(tables, a)[0] is seats_of(tables, fresh)[0]
    assert polled(tables, b)["status"] == "waiting"
    assert pairs[pair(entrant(client("a")), entrant(client("b", version="2")))] == 1


def test_ranked_play_counts_against_no_allowance_only_the_unfinished_games_limit(tmp_path):
    def limited():
        return new_tables(games_dir=str(tmp_path), games_per_period=1, live_tables_per_ip=2, limit_exempt=())

    tables = limited()
    queue(tables, client("a"), origin=API)
    queue(tables, client("b"), origin=API)
    tables.queue.tick()
    # Two ranked games in play from one address: its unfinished-games limit is reached...
    queue(tables, client("c"), origin=API)
    queue(tables, client("d"), origin=API)
    tables.queue.tick()
    with pytest.raises(ApiError) as refused:
        queue(tables, client("e"), origin=API)
    assert refused.value.status == 429 and "live table limit" in refused.value.args[0]
    # ...but none of its one game a period went, before or after a restart.
    assert tables.allowance(API, None)["games_left"] == 1
    assert limited().allowance(API, None)["games_left"] == 1


def test_the_fill_bot_takes_one_seat_at_a_table_and_never_two(tmp_path):
    tables = new_tables(games_dir=str(tmp_path), ranked_bot=FILL)
    pair = [queue(tables, client(x), game_type="standard") for x in "ab"]
    tables.queue.tick()
    assert [polled(tables, t)["status"] for t in pair] == ["waiting", "waiting"]
    third = queue(tables, client("c"), game_type="standard")
    tables.queue.tick()
    table, _ = seats_of(tables, third)
    assert all(seats_of(tables, t)[0] is table for t in pair)
    assert [s.name for s in table.seats[:4] if s.kind.value == "bot"] == [FILL]


def test_without_a_fill_bot_a_lone_client_waits():
    tables = new_tables()
    ticket = queue(tables, client("a"))
    tables.queue.tick()
    assert polled(tables, ticket)["status"] == "waiting"
    assert tables.handle("POST", f"/api/ranked/{ticket}/leave", {}, None) == {"left": True}
    with pytest.raises(ApiError) as gone:
        polled(tables, ticket)
    assert gone.value.status == 404


def test_the_queue_is_for_api_and_mcp_clients_that_say_what_they_are():
    tables = new_tables()
    with pytest.raises(ApiError) as page:
        queue(tables, client("a"), origin=Origin(ip="203.0.113.9", page=True))
    assert page.value.status == 403
    with pytest.raises(ApiError) as bare:
        queue(tables, {"id": "a" * 64}, origin=API)
    assert bare.value.status == 400
    with pytest.raises(ApiError) as no_id:
        queue(tables, {"name": "x"})
    assert no_id.value.status == 400
    assert queue(tables, {"id": "a" * 64}, origin=Origin(ip="203.0.113.9", via="mcp"))


def test_a_ranked_table_that_waits_too_long_on_a_move_is_aborted_naming_who_it_waited_on(tmp_path):
    seen: list = []

    def rank(games):
        seen.append(games)
        return {"boards": []}

    tables = new_tables(games_dir=str(tmp_path), ranker=rank)
    tickets = [queue(tables, client(x)) for x in "ab"]
    tables.queue.tick()
    table, _ = seats_of(tables, tickets[0])
    owing = to_move(table.session.game)
    table.changed_at -= 61
    assert tables.check_clocks() == [table.code]
    closing = json.loads(next(tmp_path.glob("*.jsonl")).read_text().splitlines()[-1])
    assert closing["kind"] == "abandoned" and closing["reason"] == "timeout" and closing["seats"] == [owing]
    with pytest.raises(ApiError):
        tables.get(table.code)
    tables.rankings.wait()
    [game] = seen[-1]
    assert (game["ended"], game["timed_out"], game["winner"]) == ("timeout", [owing], None)


def test_a_restart_closes_the_ranked_games_it_cut_short(tmp_path):
    tables = new_tables(games_dir=str(tmp_path))
    tickets = [queue(tables, client(x)) for x in "ab"]
    tables.queue.tick()
    code = seats_of(tables, tickets[0])[0].code
    restarted = new_tables(games_dir=str(tmp_path))
    closing = json.loads(next(tmp_path.glob("*.jsonl")).read_text().splitlines()[-1])
    assert closing == {"kind": "abandoned", "at": closing["at"], "reason": "restart"}
    with pytest.raises(ApiError) as gone:
        restarted.get(code)
    assert gone.value.status == 404


def test_only_ranked_games_reach_the_ranker(tmp_path):
    seen: list = []

    def rank(games):
        seen.append(games)
        return {"boards": []}

    tables = new_tables(games_dir=str(tmp_path), ranker=rank, seed=99)
    casual, _ = tables.create(name="Bea", client={"id": "b" * 64, "kind": "api"})
    for name in ("Cy", "Di", "Ed"):
        casual.join(name, {"id": None, "kind": "api"})
    _play_out(casual.session, random.Random(4))
    tickets = [queue(tables, client(x), game_type="standard") for x in "abcd"]
    tables.queue.tick()
    ranked = seats_of(tables, tickets[0])[0]
    _play_out(ranked.session, random.Random(4))
    tables.rankings.wait()
    assert [g["id"] for g in seen[-1]] == [ranked.session.journal.game_id]
    assert seen[-1][0]["ranked"] and seen[-1][0]["ended"] == "finished"


def test_a_model_queues_over_mcp_and_is_seated_at_its_first_move():
    tables = new_tables(ranked_bot=FILL)
    session = mcptools.Session()
    waiting = mcptools.call_tool(tables, session, "ranked_game",
                                 {"identity": "claude-opus-5-5", "game_type": "duel-variant", "timeout": 0})
    assert waiting["status"] == "waiting" and session.ticket
    tables.queue.tick()
    seated = mcptools.call_tool(tables, session, "ranked_game", {"identity": "claude-opus-5-5", "timeout": 30})
    assert session.ticket is None and session.token
    table = tables.get(session.code)
    assert table.session.ranked
    mine = hashlib.sha256(b"claude-opus-5-5").hexdigest()
    assert any((s.client or {}).get("id") == mine for s in table.seats)
    assert seated["your_move"] in ("act", "game_over", "wait")



# --- server bots entered as players ------------------------------------------------


ENTRANT = "mynet-2.0"


@pytest.fixture
def entrant_model(monkeypatch, tmp_path):
    """A model file `mynet` 2.0 at the game types a test names. No bot runner
    starts, so every table stays in play until it is closed."""
    import sys
    import types

    from hexset.clients.onnxmeta import with_metadata
    from hexset.server import api

    directory = tmp_path / "models"
    directory.mkdir()
    monkeypatch.setattr(api, "MODELS_DIR", directory)
    monkeypatch.setattr(api, "_warned", set())
    monkeypatch.setattr(api.Tables, "_start_runners", lambda self, table, bots: None)

    def spawn(path, board, *, rng=None, device="cpu", players=None, game_type=None):
        from hexset.bots import RandomBot

        return RandomBot(random.Random(0))

    fake = types.ModuleType("hexset.clients.onnxbot")
    fake.spawn = spawn
    monkeypatch.setitem(sys.modules, "hexset.clients.onnxbot", fake)

    def model(*modes: str) -> str:
        meta = {"name": "mynet", "version": "2.0", "commit": "abc", "modes": ",".join(modes),
                "owner": "octocat", "description": "A test net."}
        (directory / "mynet.onnx").write_bytes(with_metadata(b"\x3a\x00", meta))
        return ENTRANT

    return model


def headers(directory) -> list[dict]:
    return [json.loads(p.read_text().splitlines()[0]) for p in sorted(directory.glob("*.jsonl"))]


def test_an_entrant_plays_one_ranked_game_at_a_time_at_each_type_it_is_offered_at(tmp_path, entrant_model):
    entry = entrant_model("duel-variant")
    games = tmp_path / "games"
    games.mkdir()
    tables = new_tables(games_dir=str(games), ranked_bot=FILL, ranked_entrants=(entry,))
    tables.queue.tick()
    [header] = headers(games)
    assert header["ranked"] and header["game_type"] == "duel-variant"
    [seat] = header["entrants"]
    assert header["bots"][str(seat)] == {"name": entry, "spec": header["bots"][str(seat)]["spec"], "version": "2.0"}
    assert sorted(b["name"] for b in header["bots"].values()) == sorted([entry, FILL])
    # In play: no second game of that type, and none of a type it is not offered at.
    tables.queue.tick()
    tables.queue.tick()
    assert len(headers(games)) == 1
    # Its game over (here, aborted on the clock), it queues again.
    table = tables.get(header["code"])
    table.changed_at -= 61
    tables.check_clocks()
    tables.queue.tick()
    assert len(headers(games)) == 2


def test_an_entrant_meets_clients_and_the_fill_bot_under_the_pair_cap(tmp_path, entrant_model):
    entry = entrant_model("standard", "duel-variant")
    games = tmp_path / "games"
    games.mkdir()
    tables = new_tables(games_dir=str(games), ranked_bot=FILL, ranked_entrants=(entry,))
    ticket = queue(tables, client("a"), game_type="standard")
    queue(tables, client("b"), game_type="standard")
    tables.queue.tick()
    table, _ = seats_of(tables, ticket)
    assert sorted(s.name for s in table.seats if s.kind.value == "bot") == sorted([entry, FILL])
    # The duel went to the entrant and the fill bot alone.
    assert {h["game_type"] for h in headers(games)} == {"standard", "duel-variant"}
    me, it, fill = entrant(client("a")), (f"bot:{entry}", "2.0"), tables.queue._filler(FILL)
    met = lambda pairs: (pairs[pair(me, fill)], pairs[pair(it, fill)], pairs[pair(me, it)])  # noqa: E731
    meetings, pairs = tables.queue._history()
    assert met(pairs) == (1, 2, 1) and meetings[frozenset(("a" * 64, f"bot:{entry}"))] == 1
    # Read back from the journals the same.
    read = new_tables(games_dir=str(games), ranked_bot=FILL)
    meetings, pairs = read.queue._history()
    assert met(pairs) == (1, 2, 1) and meetings[frozenset(("a" * 64, f"bot:{entry}"))] == 1


def test_a_client_seated_with_entrants_alone_is_not_at_a_fill_table(tmp_path, entrant_model):
    entry = entrant_model("duel-variant")
    games = tmp_path / "games"
    games.mkdir()
    tables = new_tables(games_dir=str(games), ranked_bot=FILL, ranked_entrants=(entry,))
    ticket = queue(tables, client("a"))
    tables.queue.tick()
    table, _ = seats_of(tables, ticket)
    assert [s.name for s in table.seats[:2] if s.kind.value == "bot"] == [entry]
    fill = tables.queue._filler(FILL)
    for history in (tables.queue._history(), new_tables(games_dir=str(games)).queue._history()):
        meetings, pairs = history
        assert pairs[pair(entrant(client("a")), fill)] == 0 and pairs[pair((f"bot:{entry}", "2.0"), fill)] == 0
        assert pairs[pair(entrant(client("a")), (f"bot:{entry}", "2.0"))] == 1
        assert meetings[frozenset(("a" * 64, f"bot:{entry}"))] == 1


def test_an_entrants_rankings_row_carries_its_models_profile(tmp_path, entrant_model):
    entry = entrant_model("duel-variant")

    def rank(games):
        rows = [{"name": entry, "id": f"bot:{entry}", "games": 12}]
        return {"boards": [{"key": "all", "title": "All", "columns": [], "rows": rows}]}

    tables = new_tables(games_dir=str(tmp_path), ranker=rank)
    tables.rankings.wait()
    [row] = tables.handle("GET", "/api/rankings", {}, None)["boards"][0]["rows"]
    assert (row["name"], row["version"], row["owner"], row["description"]) == (entry, "2.0", "octocat", "A test net.")
