"""MCP over HTTP: `POST /mcp` against a real `HexSetServer`, the way an
actual MCP client speaks to it (see `web.py`'s module docstring for the
transport, `mcptools.py` for the tool layer it calls in-process).

`test_translate_*` below are the one exception -- pure translation logic
(`mcptools._translate_action`/`_translate_view`/etc.) needs neither a socket
nor a `Tables`, so those stay unit tests of the module directly, the same as
before this module's tools moved off stdio.
"""

from __future__ import annotations

import hashlib
import json
import random
import threading
import urllib.error
import urllib.request

import pytest

from conftest import new_tables

from hexset.server import mcptools
from hexset.server.web import HexSetServer

SOLO = ["heximax", "heximax", "heximax"]
MODEL = "claude-test-model"


@pytest.fixture
def live_server():
    server = HexSetServer(("127.0.0.1", 0), new_tables())
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


@pytest.fixture(autouse=True)
def _creator_at_seat_zero(monkeypatch):
    """Pin the creator to seat 0 so no bot seat is on move before the test
    acts (same race and same fix as `test_web.py`: `Tables.create` deals
    `first=0`, and a bot seated first starts its runner thread mid-test)."""
    monkeypatch.setattr(random.SystemRandom, "randrange", lambda self, n: 0)


class MCPClient:
    """One MCP session's worth of HTTP: JSON-RPC in, the `Mcp-Session-Id`
    `initialize` hands back carried on every request after -- the way a real
    client holds it."""

    def __init__(self, base: str) -> None:
        self.url = base + "/mcp"
        self.session_id: str | None = None
        self._next_id = 0

    def request(self, method: str, params: dict | None = None, *, notification: bool = False, origin: str | None = None):
        """One JSON-RPC message -> `(status, response_headers, parsed_body)`.
        `parsed_body` is `None` for a 202 (a notification's answer)."""
        body: dict = {"jsonrpc": "2.0", "method": method}
        if not notification:
            self._next_id += 1
            body["id"] = self._next_id
        if params is not None:
            body["params"] = params
        headers = {"Content-Type": "application/json"}
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id
        if origin is not None:
            headers["Origin"] = origin
        request = urllib.request.Request(
            self.url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST"
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                status = response.status
                sid = response.headers.get("Mcp-Session-Id")
                if sid:
                    self.session_id = sid
                raw = response.read()
                return status, response.headers, (json.loads(raw) if raw else None)
        except urllib.error.HTTPError as error:
            raw = error.read()
            return error.code, error.headers, (json.loads(raw) if raw else None)

    def initialize(self, protocol_version: str = "2025-06-18"):
        status, _, data = self.request("initialize", {"protocolVersion": protocol_version})
        assert status == 200, data
        assert self.session_id, "initialize must mint an Mcp-Session-Id"
        return data["result"]

    def call_tool_raw(self, tool: str, **arguments):
        return self.request("tools/call", {"name": tool, "arguments": arguments})

    def call_tool(self, tool: str, **arguments) -> dict:
        status, _, data = self.call_tool_raw(tool, **arguments)
        assert status == 200, data
        result = data["result"]
        payload = json.loads(result["content"][0]["text"])
        if result["isError"]:
            raise AssertionError(f"{tool}({arguments}) failed: {payload}")
        return payload


def connected(base: str) -> MCPClient:
    client = MCPClient(base)
    client.initialize()
    return client


# --- Session lifecycle ---------------------------------------------------


def test_initialize_returns_a_session_id(live_server):
    _, base = live_server
    client = MCPClient(base)
    result = client.initialize()
    assert client.session_id
    assert result["serverInfo"]["name"] == "hexset"
    assert result["protocolVersion"] == "2025-06-18"


def test_initialize_echoes_a_known_protocol_version(live_server):
    _, base = live_server
    client = MCPClient(base)
    result = client.initialize(protocol_version="2024-11-05")
    assert result["protocolVersion"] == "2024-11-05"


def test_tools_call_without_a_session_id_is_400(live_server):
    _, base = live_server
    client = MCPClient(base)  # never initialized -- no session id to send
    status, _, data = client.call_tool_raw("models")
    assert status == 400
    assert "error" in data


def test_tools_call_with_an_unknown_session_id_is_404(live_server):
    _, base = live_server
    client = MCPClient(base)
    client.initialize()
    client.session_id = "not-a-real-session"
    status, _, data = client.call_tool_raw("models")
    assert status == 404


def test_tools_list_names_every_tool(live_server):
    _, base = live_server
    client = connected(base)
    status, _, data = client.request("tools/list")
    assert status == 200
    names = {tool["name"] for tool in data["result"]["tools"]}
    assert names == set(mcptools._TOOLS)


def test_notification_gets_a_202_with_no_body(live_server):
    _, base = live_server
    client = connected(base)
    status, _, data = client.request("notifications/initialized", notification=True)
    assert status == 202
    assert data is None


def test_ping(live_server):
    _, base = live_server
    client = connected(base)
    status, _, data = client.request("ping")
    assert status == 200
    assert data["result"] == {}


def test_unknown_method_is_minus_32601(live_server):
    _, base = live_server
    client = connected(base)
    status, _, data = client.request("not/a/method")
    assert status == 200
    assert data["error"]["code"] == -32601


def test_origin_from_another_host_is_403(live_server):
    _, base = live_server
    client = MCPClient(base)
    status, _, data = client.request(
        "initialize", {"protocolVersion": "2025-06-18"}, origin="http://evil.example"
    )
    assert status == 403


def test_origin_from_localhost_is_allowed(live_server):
    _, base = live_server
    client = MCPClient(base)
    status, _, _ = client.request(
        "initialize", {"protocolVersion": "2025-06-18"}, origin="http://127.0.0.1"
    )
    assert status == 200


def test_get_mcp_is_405(live_server):
    _, base = live_server
    request = urllib.request.Request(base + "/mcp", method="GET")
    with pytest.raises(urllib.error.HTTPError) as caught:
        urllib.request.urlopen(request, timeout=5)
    assert caught.value.code == 405


def test_delete_ends_the_session(live_server):
    _, base = live_server
    client = connected(base)
    request = urllib.request.Request(
        base + "/mcp", headers={"Mcp-Session-Id": client.session_id}, method="DELETE"
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        assert response.status == 200
    status, _, _ = client.call_tool_raw("models")
    assert status == 404  # the session is gone


# --- Identity: model -> client, over the real /mcp route ----------------


def test_new_game_records_the_clients_id_and_kind_mcp(live_server):
    server, base = live_server
    client = connected(base)
    data = client.call_tool("new_game", model=" Claude-Opus-5 ", opponents=SOLO)
    expected_id = hashlib.sha256(b"claude-opus-5").hexdigest()
    seat = server.tables.get(data["code"]).seats[data["seat"]]
    assert seat.client == {"id": expected_id, "kind": "mcp"}


# --- Version guard: act(index, version) -----------------------------------


def _setup_settlement_index(view: dict) -> int:
    return next(i for i, a in enumerate(view["legal_actions"]) if a["type"] == "SETUP_SETTLEMENT")


def _wait_for_turn_streamed(client: MCPClient, **arguments) -> tuple[str, dict]:
    """`wait_for_turn` the way a real client receives it: as an SSE stream.
    Returns the response's Content-Type and the tool payload out of the one
    `data:` line."""
    body = {
        "jsonrpc": "2.0",
        "id": 999,
        "method": "tools/call",
        "params": {"name": "wait_for_turn", "arguments": arguments},
    }
    request = urllib.request.Request(
        client.url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "Mcp-Session-Id": client.session_id},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        content_type = response.headers.get("Content-Type", "")
        raw = response.read().decode("utf-8")
    data_line = next(line for line in raw.splitlines() if line.startswith("data: "))
    result = json.loads(data_line[len("data: "):])["result"]
    assert result["isError"] is False, result
    return content_type, json.loads(result["content"][0]["text"])


def test_act_with_an_expect_that_no_longer_matches_is_an_error(live_server):
    """`expect` is the guard against an index that now names a different
    action. It replaced a whole-table `version`, which bumped on every other
    seat's move and every trade answer, and so was stale by design."""
    _, base = live_server
    client = connected(base)
    data = client.call_tool("new_game", model=MODEL, opponents=SOLO)
    index = _setup_settlement_index(data)
    chosen = data["legal_actions"][index]

    status, _, response = client.call_tool_raw(
        "act", index=index, expect={"type": "SETUP_SETTLEMENT", "a": chosen["a"] + 1}
    )
    assert status == 200
    result = response["result"]
    assert result["isError"] is True
    assert "moved" in result["content"][0]["text"]
    assert client.call_tool("state")["phase"] == "SETUP_SETTLEMENT"  # nothing was played


def test_act_with_a_matching_expect_acts(live_server):
    _, base = live_server
    client = connected(base)
    data = client.call_tool("new_game", model=MODEL, opponents=SOLO)
    index = _setup_settlement_index(data)

    result = client.call_tool("act", index=index, expect=data["legal_actions"][index])
    assert result["phase"] == "SETUP_ROAD"


def test_act_expect_may_name_only_the_type(live_server):
    _, base = live_server
    client = connected(base)
    data = client.call_tool("new_game", model=MODEL, opponents=SOLO)
    index = _setup_settlement_index(data)

    result = client.call_tool("act", index=index, expect={"type": "SETUP_SETTLEMENT"})
    assert result["phase"] == "SETUP_ROAD"


def test_act_no_longer_takes_a_version(live_server):
    _, base = live_server
    client = connected(base)
    data = client.call_tool("new_game", model=MODEL, opponents=SOLO)

    status, _, response = client.call_tool_raw("act", index=_setup_settlement_index(data), version=data["version"])
    assert status == 200
    assert response["result"]["isError"] is True
    assert "bad arguments" in response["result"]["content"][0]["text"]


# --- wait_for_turn: streamed as SSE, blocks through a bot's turn --------


def test_wait_for_turn_streams_and_returns_once_it_is_our_turn_again(live_server):
    server, base = live_server
    client = connected(base)
    data = client.call_tool("new_game", model=MODEL, opponents=SOLO)
    assert data["seat"] == 0

    settlement = _setup_settlement_index(data)
    after_settlement = client.call_tool("act", index=settlement)
    road = next(i for i, a in enumerate(after_settlement["legal_actions"]) if a["type"] == "SETUP_ROAD")
    after_road = client.call_tool("act", index=road)
    assert after_road["legal_actions"] == []  # a bot (seat 1) is on move now
    assert after_road["your_move"] == "wait"
    assert after_road["waiting_on"] == [1]

    content_type, payload = _wait_for_turn_streamed(client)

    assert content_type.startswith("text/event-stream")
    assert payload["legal_actions"]  # our own turn again (round 2 of setup)
    assert payload["your_move"] == "act"
    # The streamed reply is trimmed against the session's automatic cursor
    # like any other: it continues from the last line the `act` reply sent.
    assert payload["log_from"] == max(0, after_road["log_total"] - 1)
    assert payload["log_total"] > after_road["log_total"]  # the bots' placements


# --- Trade-round responses are named dicts, the same as state()/get_table() --
#
# `_translate_trades` already turns a raw signed bundle into `you_give`/
# `you_receive` for state()/get_table(); offer_trade()/answer_trade()/
# choose_trade() are pinned to the same translated shape here, against a
# `FakeTables` stand-in rather than a live server -- these are about the
# translation, not the routing.

RAW_BUNDLE = [0, -1, 0, 1, 0]  # signed towards the actor: gives one Brick, gets one Wheat


class FakeTables:
    """A `Tables`-shaped stub: `handle(method, path, payload, token)` ->
    whatever `responses` says for that `(method, path)`, ignoring `payload`/
    `token` -- enough for `mcptools.call_tool` to run its translation logic
    against canned wire responses."""

    def __init__(self, responses: dict[tuple[str, str], dict]) -> None:
        self.responses = responses

    def handle(self, method, path, payload, token):
        return self.responses[(method, path)]


def _session_for_trade() -> mcptools.Session:
    return mcptools.Session(token="tok", code="abcdef")


def test_offer_trade_translates_its_own_response():
    raw = {
        "trade_round": {
            "offer": {"actor": 2, "bundle": RAW_BUNDLE},
            "responses": [],
            "awaiting": [0, 1, 3],
        }
    }
    tables = FakeTables({("POST", "/api/games/abcdef/trade/round"): raw})
    data = mcptools.call_tool(
        tables, _session_for_trade(), "offer_trade", {"give": {"Brick": 1}, "want": {"Wheat": 1}}
    )
    assert data["trade_round"]["you_give"] == {"Brick": 1}
    assert data["trade_round"]["you_receive"] == {"Wheat": 1}


def test_answer_trade_translates_its_own_response():
    state = {"pending": [{"actor": 0, "bundle": RAW_BUNDLE}], "version": 1}
    answered = {"pending": [], "trade_round": None}
    tables = FakeTables(
        {
            ("GET", "/api/state"): state,
            ("POST", "/api/games/abcdef/trade/round/answer"): answered,
        }
    )
    data = mcptools.call_tool(
        tables, _session_for_trade(), "answer_trade", {"index": 0, "kind": "accept"}
    )
    assert data["pending"] == []
    assert data["trade_round"] is None


def test_choose_trade_translates_its_own_response():
    state = {"trade_round": None, "version": 1}
    chosen = {
        "trades": [{"a": 2, "b": 0, "gave": [0, 1, 0, 0, 0], "got": [0, 0, 0, 1, 0]}],
        "trade_round": None,
    }
    tables = FakeTables(
        {
            ("GET", "/api/state"): state,
            ("POST", "/api/games/abcdef/trade/round/choose"): chosen,
        }
    )
    data = mcptools.call_tool(tables, _session_for_trade(), "choose_trade", {"decline": True})
    assert data["trades"] == [{"a": 2, "b": 0, "a_gave": {"Brick": 1}, "a_got": {"Wheat": 1}}]
    assert data["trade_round"] is None


# --- Unlabeled resource indices on legal_actions (`_translate_action`) --------


def test_translate_action_labels_monopoly_and_discard_by_resource_name():
    assert mcptools._translate_action({"type": "PLAY_MONOPOLY", "a": 2, "b": 0})["resource"] == "Sheep"
    assert mcptools._translate_action({"type": "DISCARD", "a": 4, "b": 0})["resource"] == "Ore"


def test_translate_action_labels_bank_trade_give_and_want():
    action = mcptools._translate_action({"type": "BANK_TRADE", "a": 0, "b": 3})
    assert action["give"] == "Wood"
    assert action["want"] == "Wheat"


def test_translate_action_labels_year_of_plenty_as_a_resource_pair():
    action = mcptools._translate_action({"type": "PLAY_YEAR_OF_PLENTY", "a": 5, "b": 0})
    pair = mcptools._YEAR_OF_PLENTY_PAIRS[5]
    assert action["resources"] == [mcptools.RESOURCES[r] for r in pair]


def test_translate_action_leaves_other_action_types_untouched():
    action = {"type": "ROLL", "a": 0, "b": 0}
    assert mcptools._translate_action(action) == action


def test_translate_action_keeps_the_raw_a_b_for_act_s_own_replay():
    """act() replays `legal_actions[index]` verbatim as the action body
    (`wire_to_action` reads only `type`/`a`/`b`) -- the added `give`/`want`/
    `resource`/`resources` keys must be extra, never a replacement."""
    action = mcptools._translate_action({"type": "BANK_TRADE", "a": 1, "b": 2})
    assert action["a"] == 1
    assert action["b"] == 2


def test_translate_view_translates_every_legal_action():
    raw = {
        "legal_actions": [{"type": "PLAY_MONOPOLY", "a": 1, "b": 0}],
        "trades": [],
        "pending": [],
        "trade_round": None,
    }
    data = mcptools._translate_view(dict(raw))
    assert data["legal_actions"] == [{"type": "PLAY_MONOPOLY", "a": 1, "b": 0, "resource": "Brick"}]


# --- Board summary: per-hex/vertex resource and pip-count annotations --------


def test_board_annotates_hexes_and_vertices_with_resource_and_pips(live_server):
    _, base = live_server
    client = connected(base)
    client.call_tool("new_game", model=MODEL, opponents=SOLO)
    data = client.call_tool("board")

    for hex_ in data["hexes"]:
        assert hex_["resource"] == mcptools._TERRAIN_RESOURCE.get(hex_["terrain"])
        assert hex_["pips"] == mcptools._PIPS.get(hex_["token"], 0)
    resourced = [h for h in data["hexes"] if h["resource"] is not None]
    assert resourced, "a real board has at least one resource-paying hex"

    for vertex in data["vertices"]:
        touching = [h for h in resourced if vertex["id"] in h["vertex_ids"]]
        assert vertex["pips"] == sum(h["pips"] for h in touching)
        assert vertex["resources"] == sorted({h["resource"] for h in touching})


# --- Seat name defaults to "mcp" when the LLM doesn't give one ---------------


def test_new_game_defaults_the_seat_name_to_mcp(live_server):
    server, base = live_server
    client = connected(base)
    data = client.call_tool("new_game", model=MODEL, opponents=SOLO)
    assert server.tables.get(data["code"]).seats[data["seat"]].name == "mcp"


def test_new_game_keeps_an_explicit_name(live_server):
    server, base = live_server
    client = connected(base)
    data = client.call_tool("new_game", model=MODEL, opponents=SOLO, name="Ada")
    assert server.tables.get(data["code"]).seats[data["seat"]].name == "Ada"


def test_join_defaults_the_seat_name_to_mcp(live_server):
    server, base = live_server
    creator = connected(base)
    created = creator.call_tool("new_game", model=MODEL)
    joiner = connected(base)
    joined = joiner.call_tool("join", code=created["code"], model=MODEL)
    assert server.tables.get(created["code"]).seats[joined["seat"]].name == "mcp"


def test_new_game_always_installs_a_pending_gate_for_the_llm_seat(live_server):
    """Human and LLM seats are direct gates, unconditionally
    (`agents/reference/trading-final.md`, item 5): there is no `confirm`
    flag left anywhere on the wire, so an LLM's own seat lands in
    `confirm_seats`/`PendingGate` the same as a human's at the web page,
    with no argument required to ask for it."""
    server, base = live_server
    client = connected(base)
    data = client.call_tool("new_game", model=MODEL, opponents=SOLO)
    seat = data["seat"]
    table = server.tables.get(data["code"])
    assert seat in table.session.confirm_seats
    from hexset.server.webplay import PendingGate

    assert isinstance(table.session.game.gates[seat], PendingGate)


# --- leave_game --------------------------------------------------------------


def test_leave_game_locks_the_seat(live_server):
    _, base = live_server
    client = connected(base)
    data = client.call_tool("new_game", model=MODEL, opponents=SOLO)
    seat = data["seat"]
    result = client.call_tool("leave_game")
    assert result["locked"] == [seat]


# --- resume_game: POST /api/reclaim, no cache file left ---------------------


def test_resume_game_reclaims_the_seat_by_code_and_model(live_server):
    server, base = live_server
    creator = connected(base)
    data = creator.call_tool("new_game", model=MODEL, opponents=SOLO)
    seat, code = data["seat"], data["code"]
    creator.call_tool("act", index=_setup_settlement_index(data))  # so there is a transcript to owe

    # A fresh session -- as if the server had restarted, or this were simply
    # a new MCP connection with no seat of its own yet.
    fresh = connected(base)
    result = fresh.call_tool("resume_game", code=code, model=MODEL)
    assert result["seat"] == seat
    # A reclaimed seat knows nothing yet, so it is owed the whole transcript.
    assert result["log_from"] == 0
    assert len(result["log"]) == result["log_total"] >= 1


# --- your_move: which tool the table wants from this seat ---------------
#
# One field in place of reading `legal_actions`, `pending`, `trade_round`,
# `trade_wait` and `to_move` together (see `mcptools._your_move`).


def test_your_move_is_act_on_your_own_turn(live_server):
    _, base = live_server
    client = connected(base)
    data = client.call_tool("new_game", model=MODEL, opponents=SOLO)
    assert data["your_move"] == "act"
    assert data["waiting_on"] == []


def test_your_move_game_over_wins_over_everything():
    move, on = mcptools._your_move({"game_over": True, "legal_actions": [{"type": "END_TURN"}]})
    assert (move, on) == ("game_over", [])


def test_your_move_discard_comes_before_any_offer():
    view = {"phase": "DISCARD", "legal_actions": [{"type": "DISCARD", "a": 0}], "pending": [{"actor": 2}]}
    assert mcptools._your_move(view) == ("discard", [])


def test_your_move_answer_trade_when_an_offer_stands_against_you():
    view = {"phase": "MAIN", "legal_actions": [], "pending": [{"actor": 2}]}
    assert mcptools._your_move(view) == ("answer_trade", [])


def test_your_move_choose_trade_once_your_round_is_fully_answered():
    view = {
        "phase": "MAIN",
        "legal_actions": [{"type": "END_TURN"}],  # still your turn -- but the round comes first
        "trade_round": {"responses": [{"seat": 1, "kind": "accept"}], "awaiting": []},
    }
    assert mcptools._your_move(view) == ("choose_trade", [])


def test_your_move_act_while_your_round_still_waits_on_a_person():
    view = {
        "phase": "MAIN",
        "legal_actions": [{"type": "END_TURN"}],
        "trade_round": {"responses": [], "awaiting": [3]},
    }
    assert mcptools._your_move(view) == ("act", [])


def test_your_move_wait_names_who_is_holding_things_up():
    assert mcptools._your_move({"legal_actions": [], "trade_round": {"responses": [], "awaiting": [2, 3]}}) == (
        "wait",
        [2, 3],
    )
    assert mcptools._your_move({"legal_actions": [], "trade_wait": [3]}) == ("wait", [3])
    assert mcptools._your_move({"legal_actions": [], "waiting_for": [2]}) == ("wait", [2])
    assert mcptools._your_move({"legal_actions": [], "phase": "DISCARD", "discard_quota": [0, 3, 0, 2]}) == (
        "wait",
        [1, 3],
    )
    assert mcptools._your_move({"legal_actions": [], "to_move": 2}) == ("wait", [2])
    assert mcptools._your_move({"legal_actions": [], "to_move": None}) == ("wait", [])


def test_turn_ready_is_your_move_not_wait():
    assert mcptools._turn_ready({"legal_actions": [], "pending": [{"actor": 1}]})
    assert not mcptools._turn_ready({"legal_actions": [], "to_move": 1})


# --- The transcript cursor: log_after -----------------------------------
#
# `log` is otherwise resent whole on every tool call and grows for the
# length of the game, which is the largest single cost an LLM seat pays to
# read this API (see `mcptools._trim_log`).


def test_trim_log_without_a_cursor_returns_the_whole_transcript():
    view = mcptools._trim_log({"log": ["a", "b", "c"]}, None)
    assert view["log"] == ["a", "b", "c"]
    assert view["log_from"] == 0
    assert view["log_total"] == 3


def test_trim_log_returns_only_what_is_new_plus_one_line_of_overlap():
    """The caller holds 2 of 4 lines. It is owed the 2 new ones and, because
    `render_log` rewrites a growing run in place, the last line it already
    has -- which may have changed under it since."""
    view = mcptools._trim_log({"log": ["a", "b", "c", "d"]}, 2)
    assert view["log"] == ["b", "c", "d"]
    assert view["log_from"] == 1
    assert view["log_total"] == 4


def test_trim_log_up_to_date_caller_still_gets_the_rewritable_line():
    view = mcptools._trim_log({"log": ["a", "b", "c"]}, 3)
    assert view["log"] == ["c"]
    assert view["log_from"] == 2


def test_trim_log_clamps_a_cursor_past_the_end_of_a_log_an_undo_shrank():
    view = mcptools._trim_log({"log": ["a", "b"]}, 9)
    assert view["log"] == ["b"]
    assert view["log_from"] == 1
    assert view["log_total"] == 2


def test_trim_log_on_an_empty_transcript_is_empty_not_an_error():
    view = mcptools._trim_log({"log": []}, 4)
    assert view["log"] == []
    assert view["log_from"] == 0
    assert view["log_total"] == 0


def test_trim_log_leaves_a_view_carrying_no_log_alone():
    view = mcptools._trim_log({"phase": "ROLL"}, 2)
    assert view == {"phase": "ROLL"}


def test_state_log_after_trims_the_transcript_it_sends_back(live_server):
    _, base = live_server
    client = connected(base)
    data = client.call_tool("new_game", model=MODEL, opponents=SOLO)
    # A freshly dealt game has an empty transcript -- place something so
    # there are lines for the cursor to be about.
    client.call_tool("act", index=_setup_settlement_index(data))

    full = client.call_tool("state", full_log=True)
    assert full["log_from"] == 0
    assert full["log_total"] == len(full["log"]) > 0

    caught_up = client.call_tool("state", log_after=full["log_total"])
    assert len(caught_up["log"]) == 1
    assert caught_up["log"][0] == full["log"][-1]
    assert caught_up["log_total"] == full["log_total"]


def test_act_log_after_sends_only_the_lines_the_action_added(live_server):
    _, base = live_server
    client = connected(base)
    data = client.call_tool("new_game", model=MODEL, opponents=SOLO)
    before = client.call_tool("state")

    result = client.call_tool("act", index=_setup_settlement_index(data), log_after=before["log_total"])
    # The whole transcript is still counted, but only its tail is carried.
    assert result["log_total"] >= before["log_total"]
    assert len(result["log"]) < result["log_total"] or result["log_total"] <= 1
    assert result["log_from"] == max(0, before["log_total"] - 1)


def test_the_cursor_is_automatic_for_a_caller_that_never_sends_log_after(live_server):
    """The session remembers how much transcript it has sent, so a caller
    that forgets `log_after` (every caller, on some call) still pays only
    for what is new. The first read after taking a seat is the whole thing;
    the next plain read is the rewritable tail only."""
    _, base = live_server
    client = connected(base)
    dealt = client.call_tool("new_game", model=MODEL, opponents=SOLO)
    assert dealt["log_from"] == 0
    # Our first settlement and road, then the bots' -- then our second
    # settlement, after which the table is ours (the road is still owed) and
    # the transcript holds still for the rest of the test.
    for _ in range(2):
        data = client.call_tool("state")
        client.call_tool("act", index=next(i for i, a in enumerate(data["legal_actions"]) if a["type"].startswith("SETUP")))
    _, data = _wait_for_turn_streamed(client)
    client.call_tool("act", index=_setup_settlement_index(data))

    first = client.call_tool("state", full_log=True)
    assert first["log_from"] == 0
    assert first["log_total"] == len(first["log"]) >= 1

    again = client.call_tool("state")
    assert again["log_total"] == first["log_total"]
    assert again["log_from"] == first["log_total"] - 1
    assert again["log"] == first["log"][-1:]


def test_an_explicit_log_after_overrides_the_automatic_cursor(live_server):
    _, base = live_server
    client = connected(base)
    data = client.call_tool("new_game", model=MODEL, opponents=SOLO)
    client.call_tool("act", index=_setup_settlement_index(data))
    client.call_tool("state")  # the session now holds everything

    rewound = client.call_tool("state", log_after=1)
    assert rewound["log_from"] == 0
    assert len(rewound["log"]) == rewound["log_total"]


def test_full_log_resets_a_session_that_is_already_caught_up(live_server):
    _, base = live_server
    client = connected(base)
    data = client.call_tool("new_game", model=MODEL, opponents=SOLO)
    client.call_tool("act", index=_setup_settlement_index(data))
    client.call_tool("state")

    whole = client.call_tool("state", full_log=True)
    assert whole["log_from"] == 0
    assert len(whole["log"]) == whole["log_total"] >= 1
    # ...and the cursor carries on from there, not from before the reset.
    assert client.call_tool("state")["log_from"] == whole["log_total"] - 1


def test_a_failed_call_does_not_move_the_cursor(live_server):
    _, base = live_server
    client = connected(base)
    data = client.call_tool("new_game", model=MODEL, opponents=SOLO)
    client.call_tool("act", index=_setup_settlement_index(data))
    before = client.call_tool("state", full_log=True)

    status, _, response = client.call_tool_raw("act", index=999)
    assert status == 200 and response["result"]["isError"]
    # The error reply carried no transcript, so the next read continues from
    # where the last successful one left off.
    assert client.call_tool("state")["log_from"] == before["log_total"] - 1


def test_trim_log_ignores_the_cursor_once_the_game_is_over():
    """`state_view` re-renders the whole transcript with redaction lifted the
    moment the game ends (`omniscient or over`), so lines the caller already
    holds change wording arbitrarily far back. A spliced reply would leave it
    with a stale prefix -- the final read sends everything instead."""
    view = mcptools._trim_log({"log": ["a", "b", "c", "d"], "game_over": True}, 3)
    assert view["log"] == ["a", "b", "c", "d"]
    assert view["log_from"] == 0
    assert view["log_total"] == 4


def test_trim_log_still_trims_while_the_game_is_running():
    view = mcptools._trim_log({"log": ["a", "b", "c", "d"], "game_over": False}, 3)
    assert view["log_from"] == 2
