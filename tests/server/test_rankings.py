"""The rankings the server serves but do not compute: finished games read
out of the journals, handed to the configured ranker at start and after each
game that ends, and its boards served as `GET /api/rankings`, the MCP
`rankings` tool and the `/rankings` page. The rankers here are fixtures
that count, not rank."""

from __future__ import annotations

import hashlib
import json
import random
import threading
import urllib.request

import pytest

from conftest import new_tables
from hexset.actions import Action, ActionType, legal_actions
from hexset.game import is_over, to_move
from hexset.server import mcptools
from hexset.server.api import ApiError
from hexset.server.rankings import check, finished_game, finished_games, load_ranker, profiles
from hexset.server.web import HexSetServer
from hexset.server.wire import action_to_wire

# A served row's profile fields when nothing filled them in.
NO_PROFILE = {"version": None, "owner": None, "url": None, "description": None}

ADA = {"id": "a" * 64, "kind": "web", "via": "http", "ip": "203.0.113.9", "ua": "test"}


def _header(**fields) -> dict:
    header = {
        "kind": "game", "id": "g1", "at": "dealt", "seed": 7,
        "board_mode": "spiral", "game_type": "standard", "locked": [], "first": 0,
        "code": "abc234", "ranked": True, "num_players": 4, "human_seats": [0],
        "player_names": {"0": "Ada"}, "clients": {"0": ADA},
        "bots": {"1": {"name": "b-1", "spec": "b"}, "2": {"name": "c-1", "spec": "c"},
                 "3": {"name": "d-1", "spec": "d"}},
    }
    return {**header, **fields}


def _action(actor: int) -> dict:
    return {"kind": "action", "step": 0, "round": 0, "actor": actor, "type": "ROLL", "a": 0, "b": 0}


def _result(winner: int, points: list[int], at: str = "finished-1") -> dict:
    return {"kind": "result", "at": at, "winner": winner, "turns": 40, "points": points}


def _write(directory, name: str, events: list[dict]) -> None:
    (directory / f"{name}.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))


def _tally(seen: list | None = None):
    """A ranker that lists every name once with its games, in the order
    first seen; `seen` collects what each call was handed."""
    def rank(games: list[dict]) -> dict:
        if seen is not None:
            seen.append(games)
        rows: dict[str, dict] = {}
        for game in games:
            for seat in game["seats"]:
                client = seat["client"] or {}
                row = rows.setdefault(seat["name"], {"name": seat["name"], "id": client.get("id"), "games": 0})
                row["games"] += 1
                # Not a declared column: never served.
                row["ip"] = client.get("ip")
        return {"boards": [{"key": "all", "title": "All games",
                            "columns": [{"key": "games", "label": "Games"}], "rows": list(rows.values())}]}
    return rank


def _play_out(session, rng: random.Random) -> None:
    for _ in range(2000):
        if is_over(session.game):
            return
        if session.awaiting_confirm is not None:
            session.submit(session.awaiting_confirm, action_to_wire(Action(ActionType.END_TURN)))
            continue
        seat = to_move(session.game)
        session.submit(seat, action_to_wire(rng.choice(legal_actions(session.game))))
    raise AssertionError("the seeded game did not finish")


# --- reading a journal --------------------------------------------------------


def test_a_finished_game_names_who_held_each_seat_and_how_it_ended():
    events = [
        _header(),
        _action(0),
        # Seat 2's bot is swapped once play is under way; seat 3 leaves.
        {"kind": "seated", "seat": 2, "name": "e-1", "spec": "e", "client": None},
        {"kind": "locked", "seat": 3, "at_step": 1},
        _action(0),
        _result(0, [10, 4, 3, 2]),
    ]
    game = finished_game(events)
    assert {k: game[k] for k in ("id", "code", "game_type", "imported", "winner", "turns")} == {
        "id": "g1", "code": "abc234", "game_type": "standard", "imported": False, "winner": 0, "turns": 40,
    }
    assert (game["ended_at"], game["ended"], game["timed_out"], game["ranked"]) == ("finished-1", "finished", [], True)
    by_seat = {s["seat"]: s for s in game["seats"]}
    assert by_seat[0] == {"seat": 0, "name": "Ada", "points": 10, "won": True, "left": False,
                          "bot": None, "swapped": False, "client": ADA}
    assert by_seat[1]["bot"] == {"name": "b-1", "spec": "b"} and not by_seat[1]["swapped"]
    assert by_seat[2]["bot"] == {"name": "e-1", "spec": "e"} and by_seat[2]["swapped"]
    assert by_seat[3]["left"] and by_seat[3]["points"] == 2


def test_seats_closed_before_play_took_no_part():
    duel = finished_game([_header(locked=[2, 3], bots={"1": {"name": "b-1", "spec": "b"}}),
                          _action(0), _result(1, [8, 15, 0, 0])])
    assert [s["seat"] for s in duel["seats"]] == [0, 1]
    closed = finished_game([_header(), {"kind": "locked", "seat": 3, "at_step": 0},
                            _action(0), _result(0, [10, 4, 3, 0])])
    assert [s["seat"] for s in closed["seats"]] == [0, 1, 2]


def test_only_a_game_played_out_is_finished():
    assert finished_game([_header(), _action(0), {"kind": "abandoned", "at": "x"}]) is None
    assert finished_game([_header(), _action(0)]) is None
    assert finished_game([]) is None


def test_a_game_filed_from_a_record_is_marked_imported():
    assert finished_game([_header(seed=None), _action(0), _result(0, [10, 0, 0, 0])])["imported"]


def test_finished_games_are_oldest_first_by_when_they_finished(tmp_path):
    _write(tmp_path, "a-later", [_header(id="late"), _action(0), _result(0, [10, 0, 0, 0], "finished-2")])
    _write(tmp_path, "b-earlier", [_header(id="early"), _action(0), _result(0, [10, 0, 0, 0], "finished-1")])
    _write(tmp_path, "c-open", [_header(id="open"), _action(0)])
    assert [g["id"] for g in finished_games(tmp_path)] == ["early", "late"]
    assert finished_games("") == []


# --- what a ranker may return -------------------------------------------------


def test_a_board_serves_only_name_id_and_its_columns():
    boards = check(_tally()([finished_game([_header(), _action(0), _result(0, [10, 0, 0, 0])])]))
    assert boards[0]["rows"][0] == {"name": "Ada", "id": "a" * 64, "games": 1, **NO_PROFILE}
    assert boards[0]["note"] is None


@pytest.mark.parametrize("value", [
    None,
    {"boards": {}},
    {"boards": [{"key": "", "title": "t", "columns": [], "rows": []}]},
    {"boards": [{"key": "k", "title": "t", "columns": [], "rows": []}] * 2},
    {"boards": [{"key": "k", "title": "t", "columns": [{"key": "x"}], "rows": []}]},
    {"boards": [{"key": "k", "title": "t", "columns": [{"key": "id", "label": "Id"}], "rows": []}]},
    {"boards": [{"key": "k", "title": "t", "columns": [], "rows": [{"id": "x"}]}]},
    {"boards": [{"key": "k", "title": "t", "columns": [{"key": "x", "label": "X"}],
                 "rows": [{"name": "n", "x": [1]}]}]},
    {"boards": [{"key": "k", "title": "t", "columns": [{"key": "x", "label": "X"}],
                 "rows": [{"name": "n", "x": float("nan")}]}]},
    {"boards": [{"key": "k", "title": "t", "columns": [], "rows": [{"name": "n", "games": -1}]}]},
    {"boards": [{"key": "k", "title": "t", "columns": [], "rows": [{"name": "n", "games": 1.5}]}]},
    {"boards": [{"key": "k", "title": "t", "columns": [], "rows": [{"name": "n", "games": True}]}]},
])
def test_anything_else_is_refused(value):
    with pytest.raises(ValueError):
        check(value)


def test_rows_under_the_minimum_games_are_not_served():
    rows = [{"name": "nine", "games": 9}, {"name": "ten", "games": 10},
            {"name": "unsaid"}]
    board = {"key": "k", "title": "t", "columns": [{"key": "games", "label": "Games"}], "rows": rows}
    served = check({"boards": [board]}, min_games=10)[0]["rows"]
    assert served == [{"name": "ten", "id": None, "games": 10, **NO_PROFILE},
                      {"name": "unsaid", "id": None, "games": None, **NO_PROFILE}]


def test_a_pinned_row_is_served_as_pinned_and_only_when_set():
    def board(**row):
        return {"boards": [{"key": "k", "title": "t", "columns": [], "rows": [{"name": "n", **row}]}]}
    assert check(board(pinned=True))[0]["rows"][0]["pinned"] is True
    assert "pinned" not in check(board(pinned=False))[0]["rows"][0]
    with pytest.raises(ValueError):
        check(board(pinned="yes"))


def test_a_ranker_is_named_module_colon_function():
    assert load_ranker("json:dumps") is json.dumps
    for spec in ("json", ":dumps", "json:", "json:no_such_function"):
        with pytest.raises(ValueError):
            load_ranker(spec)


# --- served -------------------------------------------------------------------


def test_the_server_ranks_its_journals_at_start_and_each_game_as_it_ends(tmp_path):
    _write(tmp_path, "old", [_header(id="old"), _action(0), _result(0, [10, 0, 0, 0])])
    seen: list = []
    tables = new_tables(games_dir=str(tmp_path), ranker=_tally(seen), seed=99, rankings_min_games=0)
    tables.rankings.wait()
    assert [g["id"] for g in seen[-1]] == ["old"]

    players = [{"id": x * 64, "kind": "api", "name": name} for x, name in zip("bcde", ("Bea", "Cy", "Di", "Ed"))]
    table = tables.get(tables.deal_ranked("standard", players, None, 0, [None] * 4)["code"])
    _play_out(table.session, random.Random(4))
    tables.rankings.wait()

    assert {g["id"] for g in seen[-1]} == {"old", table.session.journal.game_id}
    served = tables.handle("GET", "/api/rankings", {}, None)
    assert served["games"] == 2 and served["updated"] and served["min_games"] == 0
    rows = {row["name"]: row for row in served["boards"][0]["rows"]}
    assert rows["Bea"] == {"name": "Bea", "id": "b" * 64, "games": 1, **NO_PROFILE}
    assert set(rows) == {"Ada", "b-1", "c-1", "d-1", "Bea", "Cy", "Di", "Ed"}


def test_a_failing_ranker_leaves_the_last_good_board_up(tmp_path, capsys):
    _write(tmp_path, "one", [_header(id="one"), _action(0), _result(0, [10, 0, 0, 0])])
    calls = []

    def rank(games):
        calls.append(len(games))
        if len(calls) > 1:
            raise RuntimeError("boom")
        return _tally()(games)

    tables = new_tables(games_dir=str(tmp_path), ranker=rank)
    tables.rankings.wait()
    before = tables.handle("GET", "/api/rankings", {}, None)
    _write(tmp_path, "two", [_header(id="two"), _action(0), _result(0, [10, 0, 0, 0])])
    tables.rankings.ended(tmp_path / "two.jsonl")
    tables.rankings.wait()
    assert calls == [1, 2]
    assert tables.handle("GET", "/api/rankings", {}, None) == before
    assert "the ranker failed" in capsys.readouterr().out


def test_without_a_ranker_there_is_no_rankings():
    tables = new_tables()
    assert tables.rankings is None
    with pytest.raises(ApiError) as refused:
        tables.handle("GET", "/api/rankings", {}, None)
    assert refused.value.status == 404


def test_the_mcp_tool_marks_the_callers_own_rows(tmp_path):
    mine = hashlib.sha256(b"claude-opus-5-5").hexdigest()

    def rank(games):
        rows = [{"name": "claude-opus-5-5", "id": mine}, {"name": "someone", "id": "c" * 64}]
        return {"boards": [{"key": "all", "title": "All", "columns": [], "rows": rows}]}

    tables = new_tables(games_dir=str(tmp_path), ranker=rank)
    tables.rankings.wait()
    board = mcptools.call_tool(tables, mcptools.Session(), "rankings", {"identity": " Claude-Opus-5-5 "})
    assert [row.get("you", False) for row in board["boards"][0]["rows"]] == [True, False]
    # The served board itself is untouched.
    assert "you" not in tables.rankings.view()["boards"][0]["rows"][0]
    with pytest.raises(mcptools.ToolError):
        mcptools.call_tool(new_tables(), mcptools.Session(), "rankings", {})


def test_the_page_is_served_at_ranked(tmp_path):
    server = HexSetServer(("127.0.0.1", 0), new_tables(games_dir=str(tmp_path), ranker=_tally()))
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with urllib.request.urlopen(base + "/rankings", timeout=5) as response:
            assert response.headers["Content-Type"].startswith("text/html")
            assert b"/api/rankings" in response.read()
        server.tables.rankings.wait()
        with urllib.request.urlopen(base + "/api/rankings", timeout=5) as response:
            assert json.load(response)["boards"][0]["key"] == "all"
    finally:
        server.shutdown()
        thread.join(timeout=5)


# --- profiles -----------------------------------------------------------------


def _profile_rank(calls: list):
    def rank(games):
        calls.append(len(games))
        rows = [{"name": "as ranked", "id": "d" * 64, "games": 12, "version": "ranker's"}]
        return {"boards": [{"key": "all", "title": "All", "columns": [], "rows": rows}]}
    return rank


def test_a_row_carries_its_ids_latest_profile_without_ranking_again(tmp_path):
    calls: list = []
    tables = new_tables(games_dir=str(tmp_path), ranker=_profile_rank(calls))
    tables.rankings.wait()
    assert tables.handle("GET", "/api/rankings", {}, None)["boards"][0]["rows"][0]["name"] == "as ranked"

    first = {"id": "d" * 64, "name": "deepvine", "version": "1", "owner": "octocat",
             "url": "https://example.org/deepvine", "description": "MCTS over a small net."}
    tables.handle("POST", "/api/games", {"client": first}, None)
    tables.rankings.wait()
    row = tables.handle("GET", "/api/rankings", {}, None)["boards"][0]["rows"][0]
    assert {k: row[k] for k in ("name", "version", "owner", "url", "description")} == {
        "name": "deepvine", "version": "1", "owner": "octocat",
        "url": "https://example.org/deepvine", "description": "MCTS over a small net.",
    }

    # The latest call is the whole profile: what it leaves out is cleared.
    code = tables.handle("POST", "/api/games", {}, None)["code"]
    tables.handle("POST", "/api/join", {"code": code, "client": {"id": "d" * 64, "version": "2"}}, None)
    tables.rankings.wait()
    row = tables.handle("GET", "/api/rankings", {}, None)["boards"][0]["rows"][0]
    assert (row["name"], row["version"], row["owner"], row["description"]) == ("as ranked", "2", None, None)
    assert calls == [0]


def test_profiles_are_read_back_from_the_journals_latest_first(tmp_path):
    early = {"id": "d" * 64, "kind": "api", "name": "deepvine", "version": "1"}
    late = {"id": "d" * 64, "kind": "api", "version": "2"}
    _write(tmp_path, "one", [_header(id="one", at="t1", clients={"0": early})])
    _write(tmp_path, "two", [_header(id="two", at="t0", clients={}),
                             {"kind": "seated", "at": "t2", "seat": 1, "name": "x", "spec": "", "client": late}])
    assert profiles(tmp_path) == {"d" * 64: {"version": "2", "at": "t2"}}


def test_a_ranker_row_may_carry_a_profile_but_only_a_web_url():
    def board(**row):
        return {"boards": [{"key": "k", "title": "t", "columns": [], "rows": [{"name": "n", **row}]}]}
    assert check(board(url="https://example.org", version="3"))[0]["rows"][0]["version"] == "3"
    for bad in ({"url": "javascript:alert(1)"}, {"version": 3}, {"description": ["x"]}):
        with pytest.raises(ValueError):
            check(board(**bad))


def test_a_rename_is_the_name_its_id_goes_by_and_keeps_the_rest(tmp_path):
    calls: list = []
    tables = new_tables(games_dir=str(tmp_path), ranker=_profile_rank(calls))
    first = {"id": "d" * 64, "name": "deepvine", "version": "1", "owner": "octocat",
             "description": "MCTS over a small net."}
    token = tables.handle("POST", "/api/games", {"client": first}, None)["token"]
    tables.handle("POST", "/api/name", {"name": "Vine"}, token)
    tables.rankings.wait()
    row = tables.handle("GET", "/api/rankings", {}, None)["boards"][0]["rows"][0]
    assert (row["name"], row["version"], row["owner"]) == ("Vine", "1", "octocat")
    assert calls == [0]
    # The journal says the same after a restart.
    assert {k: v for k, v in profiles(tmp_path)["d" * 64].items() if k != "at"} == {
        "name": "Vine", "version": "1", "owner": "octocat", "description": "MCTS over a small net.",
    }


# --- a server bot's profile -----------------------------------------------------


@pytest.fixture
def profiled_bot():
    """A preset registered with a profile, as a runtime registers its bots."""
    from hexset.arena import PRESETS, register_preset, unregister_preset

    register_preset("profiled", PRESETS["test-trader"],
                    profile={"version": "2.11.1", "owner": "octocat", "description": "A rules bot."})
    yield "profiled"
    unregister_preset("profiled")


def test_a_presets_profile_is_held_to_a_clients_rules(profiled_bot):
    from hexset.arena import PRESETS, preset_profile, register_preset

    held = preset_profile(profiled_bot)
    assert held == {"version": "2.11.1", "owner": "octocat", "description": "A rules bot."}
    for profile in ({"name": "x"}, {"url": "ftp://example.org"}, {"owner": 7}, {"description": "x" * 101}):
        with pytest.raises(ValueError):
            register_preset("badly-profiled", PRESETS["test-trader"], profile=profile)
    assert preset_profile("badly-profiled") == {}


def test_a_bots_profile_is_listed_journalled_and_shown_in_its_rankings_row(tmp_path, profiled_bot):
    def rank(games):
        rows = [{"name": "rehex", "id": f"bot:{profiled_bot}", "games": 12}]
        return {"boards": [{"key": "all", "title": "All", "columns": [], "rows": rows}]}

    tables = new_tables(games_dir=str(tmp_path), ranker=rank, ranked_bot=profiled_bot)
    listed = tables.handle("GET", "/api/models", {}, None)["game_types"]["duel-variant"][profiled_bot]
    assert listed["version"] == "2.11.1" and listed["owner"] == "octocat"

    tables.deal_ranked("duel-variant", [{"id": "a" * 64, "kind": "api"}], profiled_bot, 1, [None])
    header = json.loads(next(tmp_path.glob("*.jsonl")).read_text().splitlines()[0])
    [bot] = header["bots"].values()
    assert bot == {"name": profiled_bot, "spec": profiled_bot, "version": "2.11.1"}

    tables.rankings.wait()
    [row] = tables.handle("GET", "/api/rankings", {}, None)["boards"][0]["rows"]
    assert (row["name"], row["version"], row["owner"], row["url"]) == ("rehex", "2.11.1", "octocat", None)
