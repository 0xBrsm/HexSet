"""MCP over HTTP: `POST /mcp` against a real `HexSetServer`, the way an MCP
client speaks to it. `test_translate_*` are the exception: pure translation
needs neither a socket nor a `Tables`, so those are unit tests of
`mcptools` directly.
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

SOLO = ["test-trader", "test-trader", "test-trader"]
IDENTITY = "claude-test-model"


@pytest.fixture
def live_server():
    server = HexSetServer(("127.0.0.1", 0), new_tables())
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
    )
    thread.start()
    try:
        yield server, f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


@pytest.fixture(autouse=True)
def _short_settle_cap(monkeypatch):
    monkeypatch.setattr(mcptools, "_MAX_WAIT", 10.0)
    monkeypatch.setattr(mcptools, "_WAIT_TICK", 0.5)


@pytest.fixture(autouse=True)
def _creator_at_seat_zero(monkeypatch):
    """A bot seated first starts its runner thread mid-test."""
    monkeypatch.setattr(random.SystemRandom, "randrange", lambda self, n: 0)


class MCPClient:

    def __init__(self, base: str) -> None:
        self.url = base + "/mcp"
        self.session_id: str | None = None
        self._next_id = 0

    def request(self, method: str, params: dict | None = None, *, notification: bool = False, origin: str | None = None):
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
                if response.headers.get("Content-Type", "").startswith("text/event-stream"):
                    return status, response.headers, _sse_message(raw.decode("utf-8"))
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
        text = result["content"][0]["text"]
        if result["isError"]:
            raise AssertionError(f"{tool}({arguments}) failed: {text}")
        return text if tool == "board" else json.loads(text)


def _sse_message(raw: str) -> dict:
    data_line = next(line for line in raw.splitlines() if line.startswith("data: "))
    return json.loads(data_line[len("data: "):])


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


def test_tools_call_without_a_session_id_is_400(live_server):
    _, base = live_server
    client = MCPClient(base)  # never initialized -- no session id to send
    status, _, data = client.call_tool_raw("bots")
    assert status == 400
    assert "error" in data


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
    status, _, _ = client.call_tool_raw("bots")
    assert status == 404  # the session is gone


# --- Identity: identity string -> client, over the real /mcp route -------


def test_new_game_records_the_clients_id_and_kind_mcp(live_server):
    server, base = live_server
    client = connected(base)
    data = client.call_tool("new_game", identity=" Claude-Opus-5 ", opponents=SOLO)
    expected_id = hashlib.sha256(b"claude-opus-5").hexdigest()
    seat = server.tables.get(data["code"]).seats[data["seat"]]
    assert {k: seat.client[k] for k in ("id", "kind", "via")} == {"id": expected_id, "kind": "mcp", "via": "mcp"}


def test_new_game_deals_the_board_and_game_type_it_names(live_server):
    server, base = live_server
    client = connected(base)
    data = client.call_tool("new_game", identity=IDENTITY, opponents=["test-trader"],
                            board_mode="random", game_type="duel-variant")
    session = server.tables.get(data["code"]).session
    assert (session.board_mode, session.game_type.name) == ("random", "duel-variant")
    assert data["game_type"] == "duel-variant" and data["seats_fixed"] is True
    assert data["locked"] == [2, 3] and data["winning_points"] == 15
    assert "board_mode" not in data

    plain = connected(base).call_tool("new_game", identity="another-model", opponents=SOLO)
    assert server.tables.get(plain["code"]).session.board_mode == "spiral"
    assert not {"game_type", "seats_fixed", "board_mode"} & set(plain)


# --- act(index, expect): the guard against a position that moved ----------


def _num(cell: str):
    if cell == "-":
        return None
    if cell in ("True", "False"):  # `_cell(bool)` -> str(value); read it back as one
        return cell == "True"
    try:
        return int(cell)
    except ValueError:
        return cell


def rows(table: str) -> list[dict]:
    if ":(" in table:
        table = table[table.index(":(") + 1:]
    header, _, body = table.partition(":")
    keys = header.strip("()").split(",")
    out = []
    for row in body.split("|") if body else []:
        cells = row.split(",")
        entry = {}
        for k, cell in zip(keys, cells):
            entry[k] = [_num(c) for c in cell.split(";")] if k == "resources" else _num(cell)
        out.append(entry)
    return out


def legal(view: dict, kind: str) -> list[dict]:
    return rows(view["legal_actions"][kind])


def _setup_settlement_index(view: dict) -> int:
    return rows(view["summary"]["spots"])[0]["index"]


def _setup_road_index(view: dict) -> int:
    return legal(view, "SETUP_ROAD")[0]["index"]


def _next_setup_index(view: dict) -> int:
    spots = (view.get("summary") or {}).get("spots")
    if spots:
        return rows(spots)[0]["index"]
    return rows(next(iter(view["legal_actions"].values())))[0]["index"]


def test_act_with_an_expect_that_no_longer_matches_is_an_error(live_server):
    _, base = live_server
    client = connected(base)
    data = client.call_tool("new_game", identity=IDENTITY, opponents=SOLO)
    index = _setup_settlement_index(data)
    chosen = rows(data["summary"]["spots"])[0]

    status, _, response = client.call_tool_raw(
        "act", index=index, expect={"type": "SETUP_SETTLEMENT", "vertex": chosen["vertex"] + 1}
    )
    assert status == 200
    result = response["result"]
    assert result["isError"] is True
    assert "moved" in result["content"][0]["text"]
    assert client.call_tool("state")["phase"] == "SETUP_SETTLEMENT"


def test_expect_check_compares_named_operands_raw_operands_and_resources():
    road = {"type": "BUILD_ROAD", "a": 17, "b": 0}
    mcptools._expect_check(3, road, {"type": "BUILD_ROAD", "edge": 17, "index": 3}, 4)
    mcptools._expect_check(3, road, {"type": "BUILD_ROAD", "a": 17}, 4)
    with pytest.raises(mcptools.ToolError, match="moved under you"):
        mcptools._expect_check(3, road, {"type": "BUILD_ROAD", "edge": 18}, 4)
    with pytest.raises(mcptools.ToolError, match="moved under you"):
        mcptools._expect_check(3, road, {"type": "BUILD_SETTLEMENT", "vertex": 17}, 4)

    robber = {"type": "MOVE_ROBBER", "a": 5, "b": 4}
    mcptools._expect_check(0, robber, {"type": "MOVE_ROBBER", "hex": 5, "victim": None}, 4)
    with pytest.raises(mcptools.ToolError):
        mcptools._expect_check(0, robber, {"type": "MOVE_ROBBER", "hex": 5, "victim": 1}, 4)

    bank = {"type": "BANK_TRADE", "a": 0, "b": 4}
    mcptools._expect_check(0, bank, {"type": "BANK_TRADE", "give": "Wood", "want": "Ore"}, 4)
    with pytest.raises(mcptools.ToolError):
        mcptools._expect_check(0, bank, {"type": "BANK_TRADE", "give": "Brick", "want": "Ore"}, 4)

    with pytest.raises(mcptools.ToolError, match="group key"):
        mcptools._expect_check(0, road, {"edge": 17}, 4)


# --- Settling: every acting tool replies at this seat's next move ---------


def test_act_settles_through_the_bots_turns_to_our_next_move(live_server):
    server, base = live_server
    client = connected(base)
    data = client.call_tool("new_game", identity=IDENTITY, opponents=SOLO)
    assert data["seat"] == 0

    after_settlement = client.call_tool("act", index=_setup_settlement_index(data))
    assert after_settlement["your_move"] == "act"
    road = _setup_road_index(after_settlement)
    after_road = client.call_tool("act", index=road)

    assert after_road["your_move"] == "act"
    assert "SETUP_SETTLEMENT" not in after_road["legal_actions"]
    assert after_road["summary"]["spots"]
    assert after_road["waiting_on"] == []
    assert after_road["log_from"] == max(0, after_settlement["log_total"] - 1)
    assert after_road["log_total"] > after_settlement["log_total"] + 3


def test_forced_rolls_a_lone_roll_and_passes_only_with_an_empty_hand():
    empty = {"Wood": 0, "Brick": 0, "Sheep": 0, "Wheat": 0, "Ore": 0}
    # Signed towards the actor: positive is what we give.
    offers = [{"actor": 0, "bundle": [2, 0, 0, 0, -1]}, {"actor": 2, "bundle": [1, 0, 0, 0, -1]}]
    rolled = {"seat": 1, "players": [{"seat": 1, "hand": empty}], "legal_actions": [], "pending": offers}
    passed_one = {**rolled, "pending": offers[1:]}
    passed_both = {**rolled, "pending": []}
    tables = RecordingTables({
        ("POST", "/api/action"): rolled,
        ("POST", "/api/games/abcdef/trade/round/answer"): [passed_one, passed_both],
    })
    session = mcptools.Session(token="tok", code="abcdef")
    start = {"seat": 1, "players": [{"seat": 1, "hand": empty}], "legal_actions": [{"type": "ROLL", "a": 0, "b": 0}]}
    view = mcptools._forced(tables, session, start)
    assert [(m, p) for m, p, _ in tables.calls] == [
        ("POST", "/api/action"),
        ("POST", "/api/games/abcdef/trade/round/answer"),
        ("POST", "/api/games/abcdef/trade/round/answer"),
    ]
    assert tables.calls[0][2] == {"action": {"type": "ROLL", "a": 0, "b": 0}}
    assert tables.calls[1][2] == {"actor": 0, "received": [2, 0, 0, 0, -1], "kind": "pass"}
    assert view["pending"] == []


def test_forced_leaves_an_uncoverable_offer_to_the_seat_that_can_still_counter():
    hand = {"Wood": 1, "Brick": 0, "Sheep": 0, "Wheat": 0, "Ore": 0}
    start = {"seat": 1, "players": [{"seat": 1, "hand": hand}], "legal_actions": [],
             "pending": [{"actor": 0, "bundle": [2, 0, 0, 0, -1]}]}  # wants 2 Wood; we hold 1
    tables = RecordingTables({})
    assert mcptools._forced(tables, mcptools.Session(token="tok", code="abcdef"), start) is start
    assert tables.calls == []
    view = mcptools._translate(dict(start))
    assert view["pending"] == [{"actor": 0, "you_give": {"Wood": 2}, "you_receive": {"Ore": 1}, "can_accept": False}]


def test_forced_settles_an_own_round_with_nothing_to_choose():
    bundle = [1, 0, 0, 0, -1]
    def view(responses):
        return {"seat": 1, "players": [{"seat": 1, "hand": {"Wood": 1}}], "legal_actions": [{"type": "END_TURN"}],
                "trade_round": {"offer": {"actor": 1, "bundle": bundle}, "responses": responses, "awaiting": []}}
    all_pass = view([{"seat": 0, "kind": "pass", "bundle": None}, {"seat": 2, "kind": "pass", "bundle": None}])
    assert mcptools._settled_round(all_pass) == {"decline": True}
    one_accept = view([{"seat": 0, "kind": "accept", "bundle": bundle}, {"seat": 2, "kind": "pass", "bundle": None}])
    assert mcptools._settled_round(one_accept) == {"seat": 0, "bundle": bundle}
    with_counter = view([{"seat": 0, "kind": "accept", "bundle": bundle}, {"seat": 2, "kind": "counter", "bundle": [2, 0, 0, 0, -1]}])
    assert mcptools._settled_round(with_counter) is None
    two_accepts = view([{"seat": 0, "kind": "accept", "bundle": bundle}, {"seat": 2, "kind": "accept", "bundle": bundle}])
    assert mcptools._settled_round(two_accepts) is None
    # `to` ranks and filters: the best-ranked listed accepter wins.
    assert mcptools._settled_round(two_accepts, to=[2, 0]) == {"seat": 2, "bundle": bundle}
    assert mcptools._settled_round(two_accepts, to=[3]) == {"decline": True}
    assert mcptools._settled_round(one_accept, to=[0]) == {"seat": 0, "bundle": bundle}
    assert mcptools._settled_round(one_accept, to=[2]) == {"decline": True}
    assert mcptools._settled_round(with_counter, to=[0]) is None
    still_waiting = view([{"seat": 0, "kind": "pass", "bundle": None}]); still_waiting["trade_round"]["awaiting"] = [2]
    assert mcptools._settled_round(still_waiting) is None

    closed = {**all_pass, "trade_round": None}
    tables = RecordingTables({("POST", "/api/games/abcdef/trade/round/choose"): closed})
    out = mcptools._forced(tables, mcptools.Session(token="tok", code="abcdef"), all_pass)
    assert tables.calls == [("POST", "/api/games/abcdef/trade/round/choose", {"decline": True})]
    assert out is closed


def test_board_text_annotates_hexes_and_vertices(live_server):
    _, base = live_server
    client = connected(base)
    client.call_tool("new_game", identity=IDENTITY, opponents=SOLO)
    text = client.call_tool("board")
    hex_block, vertex_block = text.split("\n\n")[:2]
    hexes = {}
    for line in hex_block.splitlines()[1:]:
        hid, label, pips, verts = line.split()
        hexes[int(hid)] = (label, int(pips), [int(v) for v in verts.split(",")])
    resourced = {h: v for h, v in hexes.items() if v[0] in mcptools.RESOURCES}
    assert resourced, "a real board has at least one resource-paying hex"
    for line in vertex_block.splitlines()[1:]:
        vid, pips, resources, port, nbrs = line.split()
        touching = [v for v in resourced.values() if int(vid) in v[2]]
        assert int(pips) == sum(v[1] for v in touching)
        assert resources == (",".join(sorted({v[0] for v in touching})) or "-")
        assert nbrs and all(int(n) != int(vid) for n in nbrs.split(","))


def test_the_game_over_reply_reports_what_the_session_was_sent(live_server):
    """Ended by hand rather than played out: `usage` rides on whichever reply
    first reads `game_over`, however the game got there.
    """
    from hexset.game import Phase

    server, base = live_server
    client = connected(base)
    data = client.call_tool("new_game", identity=IDENTITY, opponents=SOLO)
    assert "usage" not in data
    after = client.call_tool("act", index=_setup_settlement_index(data))
    assert "usage" not in after

    game = server.tables.get(data["code"]).session.game
    game.won_by = 1
    game.phase = Phase.GAME_OVER

    over = client.call_tool("state")
    assert over["game_over"]
    usage = over["usage"]
    sent = sum(len(json.dumps(reply, separators=(",", ":"))) for reply in (data, after))
    assert usage["calls"] == 3 and usage["bytes"] > sent
    again = client.call_tool("state")
    assert again["usage"]["calls"] == usage["calls"] + 1


def test_a_timeout_that_runs_out_replies_with_wait(live_server):
    _, base = live_server
    client = connected(base)
    # `timeout=0`: the open seat holds up setup until `idle` takes it.
    data = client.call_tool("new_game", identity=IDENTITY, opponents=["test-trader", "test-trader"], timeout=0)
    idle = connected(base)
    idle.call_tool("join", code=data["code"], identity="idle", timeout=0)
    client.call_tool("act", index=_setup_settlement_index(data))
    road = _setup_road_index(client.call_tool("state"))
    stalled = client.call_tool("act", index=road, timeout=0.05)
    assert stalled["your_move"] == "wait"
    assert stalled["legal_actions"] == {}
    assert stalled["waiting_on"]
    waited = client.call_tool("wait_for_turn", timeout=0.05)
    assert waited["your_move"] == "wait"


def test_every_tool_call_is_streamed(live_server):
    _, base = live_server
    client = connected(base)
    for name, arguments in (("bots", {}), ("new_game", {"identity": IDENTITY, "opponents": SOLO}), ("state", {})):
        status, headers, data = client.call_tool_raw(name, **arguments)
        assert status == 200
        assert headers.get("Content-Type", "").startswith("text/event-stream"), name
        assert data["result"]["isError"] is False, name
    status, headers, data = client.call_tool_raw("act", index=10**6)
    assert headers.get("Content-Type", "").startswith("text/event-stream")
    assert data["result"]["isError"] is True


# --- Trade-round responses are named dicts, the same as state()/get_table() --

RAW_BUNDLE = [0, -1, 0, 1, 0]  # signed towards the actor: gives one Brick, gets one Wheat


class FakeTables:

    def __init__(self, responses: dict[tuple[str, str], dict]) -> None:
        self.responses = responses

    def handle(self, method, path, payload, token, origin=None):
        return self.responses[(method, path)]


class RecordingTables(FakeTables):

    def __init__(self, responses):
        super().__init__(responses)
        self.calls: list[tuple[str, str, dict]] = []

    def handle(self, method, path, payload, token, origin=None):
        self.calls.append((method, path, payload))
        answer = self.responses[(method, path)]
        return answer.pop(0) if isinstance(answer, list) else answer


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
    session = _session_for_trade()
    data = mcptools.call_tool(
        tables, session, "offer_trade", {"give": {"Brick": 1}, "want": {"Wheat": 1}, "to": [3, 1, 3], "timeout": 0}
    )
    assert session.offer_to == [3, 1]
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
        tables, _session_for_trade(), "answer_trade", {"index": 0, "kind": "accept", "timeout": 0}
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
    data = mcptools.call_tool(tables, _session_for_trade(), "choose_trade", {"decline": True, "timeout": 0})
    assert data["trades"] == [{"a": 2, "b": 0, "a_gave": {"Brick": 1}, "a_got": {"Wheat": 1}}]
    assert data["trade_round"] is None


# --- Unlabeled resource indices on legal_actions (`_translate_action`) --------


def test_translate_action_labels_bank_trade_give_and_want():
    action = mcptools._translate_action({"type": "BANK_TRADE", "a": 0, "b": 3})
    assert action["give"] == "Wood"
    assert action["want"] == "Wheat"


# --- Seat name defaults to "mcp" when the LLM doesn't give one; leave_game -


def test_join_defaults_the_seat_name_to_mcp(live_server):
    server, base = live_server
    creator = connected(base)
    # `timeout=0` both times: open seats hold up setup, so nobody can move
    # and a settling reply would wait out its timeout.
    created = creator.call_tool("new_game", identity=IDENTITY, timeout=0)
    assert created["your_move"] == "wait"
    joiner = connected(base)
    joined = joiner.call_tool("join", code=created["code"], identity=IDENTITY, timeout=0)
    assert joined["your_move"] == "wait"
    assert server.tables.get(created["code"]).seats[joined["seat"]].name == "mcp"


def test_leave_game_locks_the_seat(live_server):
    _, base = live_server
    client = connected(base)
    data = client.call_tool("new_game", identity=IDENTITY, opponents=SOLO)
    seat = data["seat"]
    result = client.call_tool("leave_game")
    assert result["locked"] == [seat]


# --- resume_game: POST /api/reclaim, no cache file left ---------------------


def test_resume_game_reclaims_the_seat_by_code_and_identity(live_server):
    server, base = live_server
    creator = connected(base)
    data = creator.call_tool("new_game", identity=IDENTITY, opponents=SOLO)
    seat, code = data["seat"], data["code"]
    creator.call_tool("act", index=_setup_settlement_index(data))

    fresh = connected(base)
    result = fresh.call_tool("resume_game", code=code, identity=IDENTITY)
    assert result["seat"] == seat
    assert result["log_from"] == 0
    assert len(result["log"]) == result["log_total"] >= 1


# --- The compact reply: grouped legal_actions, sparse occupancy -----------


def test_group_actions_names_operands_and_keeps_the_flat_index():
    legal = [
        {"type": "ROLL", "a": 0, "b": 0},
        {"type": "BUILD_ROAD", "a": 17, "b": 0},
        {"type": "MOVE_ROBBER", "a": 3, "b": 4},
        {"type": "MOVE_ROBBER", "a": 3, "b": 1},
        {"type": "BANK_TRADE", "a": 0, "b": 4, "give": "Wood", "want": "Ore"},
        {"type": "PLAY_YEAR_OF_PLENTY", "a": 1, "b": 0, "resources": ["Wood", "Brick"]},
        {"type": "DISCARD", "a": 2, "b": 0, "resource": "Sheep"},
        {"type": "SOMETHING_NEW", "a": 9, "b": 2},
    ]
    assert mcptools._group_actions(legal, 4) == {
        "ROLL": [{"index": 0}],
        "BUILD_ROAD": [{"index": 1, "edge": 17}],
        "MOVE_ROBBER": [{"index": 2, "hex": 3, "victim": None}, {"index": 3, "hex": 3, "victim": 1}],
        "BANK_TRADE": [{"index": 4, "give": "Wood", "want": "Ore"}],
        "PLAY_YEAR_OF_PLENTY": [{"index": 5, "resources": ["Wood", "Brick"]}],
        "DISCARD": [{"index": 6, "resource": "Sheep"}],
        "SOMETHING_NEW": [{"index": 7, "a": 9, "b": 2}],
    }


def test_compact_board_lists_buildings_and_roads_by_seat():
    view = {
        "players": [{"seat": 0}, {"seat": 1}, {"seat": 2}],
        "vertex_owner": [-1, 1, 2, -1],
        "vertex_building": [0, 2, 1, 0],
        "edge_owner": [1, -1, 1, 2],
    }
    out = mcptools._compact_board(view)
    assert not {"vertex_owner", "vertex_building", "edge_owner"} & set(out)
    assert out["buildings"] == [
        {"vertex": 1, "seat": 1, "kind": "city"},
        {"vertex": 2, "seat": 2, "kind": "settlement"},
    ]
    assert out["roads"] == [[], [0, 2], [3]]


# --- summary: derived tactical facts, computed once server-side -----------


TINY_BOARD = {
    "hexes": [{"id": 3, "resource": "Ore", "pips": 5, "vertex_ids": [0, 1, 2, 3, 4, 5]}],
    "vertices": [
        {"id": 0, "pips": 10, "resources": ["Ore", "Wheat"]},
        {"id": 1, "pips": 4, "resources": ["Ore"]},
        {"id": 2, "pips": 7, "resources": ["Ore", "Sheep"]},
    ],
    "ports": [{"vertices": [1, 9], "resource": None, "ratio": 3}, {"vertices": [2], "resource": "Sheep", "ratio": 2}],
}


def test_spots_joins_legal_placements_to_pips_and_ports_best_first():
    legal = [
        {"type": "BUILD_SETTLEMENT", "a": 1, "b": 0},
        {"type": "END_TURN", "a": 0, "b": 0},
        {"type": "BUILD_SETTLEMENT", "a": 0, "b": 0},
        {"type": "BUILD_CITY", "a": 2, "b": 0},
    ]
    spots = mcptools._spots(legal, TINY_BOARD)
    assert [s["vertex"] for s in spots] == [0, 2, 1]
    assert spots[0] == {"index": 2, "type": "BUILD_SETTLEMENT", "vertex": 0, "pips": 10, "resources": ["Ore", "Wheat"]}
    assert spots[1]["port"] == "Sheep 2:1" and spots[1]["type"] == "BUILD_CITY"
    assert spots[2]["port"] == "3:1"


def test_robber_names_whose_buildings_each_hex_hits_and_an_index_per_victim():
    view = {
        "players": [{"seat": s} for s in range(4)],
        "vertex_owner": [1, -1, 2, 2, -1, -1],
        "vertex_building": [2, 0, 1, 1, 0, 0],
    }
    legal = [
        {"type": "MOVE_ROBBER", "a": 3, "b": 1},
        {"type": "MOVE_ROBBER", "a": 3, "b": 2},
        {"type": "MOVE_ROBBER", "a": 7, "b": 4},  # a hex nobody is on: victim slot 4 = nobody
    ]
    robber = mcptools._robber(legal, view, TINY_BOARD)
    assert [r["hex"] for r in robber] == [3, 7]  # 5 pips before an unknown hex's 0
    assert robber[0]["resource"] == "Ore" and robber[0]["pips"] == 5
    assert robber[0]["hits"] == [
        {"seat": 1, "settlements": 0, "cities": 1},
        {"seat": 2, "settlements": 2, "cities": 0},
    ]
    assert robber[0]["options"] == [{"index": 0, "victim": 1}, {"index": 1, "victim": 2}]
    assert robber[1]["hits"] == [] and robber[1]["options"] == [{"index": 2, "victim": None}]


ROAD_BOARD = {
    "vertices": [
        {"id": 0, "pips": 5, "resources": ["Wood"]},
        {"id": 1, "pips": 8, "resources": ["Ore", "Wheat"]},
        {"id": 2, "pips": 3, "resources": ["Sheep"]},
        {"id": 3, "pips": 6, "resources": ["Brick"]},
    ],
    "edges": [
        {"id": 10, "v0": 0, "v1": 1},
        {"id": 11, "v0": 1, "v1": 2},
        {"id": 12, "v0": 2, "v1": 3},
    ],
    "ports": [{"vertices": [3], "resource": "Brick", "ratio": 2}],
}


def test_roads_joins_legal_edges_to_the_vertex_they_reach_settle_first():
    legal = [
        {"type": "BUILD_ROAD", "a": 10, "b": 0},
        {"type": "BUILD_ROAD", "a": 11, "b": 0},
        {"type": "BUILD_ROAD", "a": 12, "b": 0},
    ]
    view = {"seat": 0, "vertex_owner": [0, -1, -1, -1], "edge_owner": [-1, -1, -1]}
    roads = mcptools._roads(legal, view, ROAD_BOARD)
    by_edge = {r["edge"]: r for r in roads}
    assert by_edge[10] == {"index": 0, "edge": 10, "to": 1, "pips": 8, "resources": ["Ore", "Wheat"], "settle": False,
                           "then": 2, "then_pips": 3}
    assert by_edge[11] == {"index": 1, "edge": 11, "to": 2, "pips": 3, "resources": ["Sheep"], "settle": True,
                           "then": 3, "then_pips": 6}
    assert by_edge[12]["to"] == 3 and by_edge[12]["port"] == "Brick 2:1" and by_edge[12]["settle"] is True
    assert "then" not in by_edge[12]
    assert [r["edge"] for r in roads] == [12, 11, 10]
    assert not any("link" in r for r in roads)

    edge_owner = [-1] * 13
    edge_owner[11] = 0  # our road on 1-2 makes vertex 1 ours too
    joined = mcptools._roads(legal[:1], {"seat": 0, "vertex_owner": [0, -1, -1, -1], "edge_owner": edge_owner}, ROAD_BOARD)
    assert joined[0]["link"] is True and joined[0]["settle"] is False


def test_afford_says_why_an_affordable_build_is_not_offered():
    hand = {"Wood": 2, "Brick": 2, "Sheep": 2, "Wheat": 3, "Ore": 3}
    board = {"piece_supply": {"road": 15, "settlement": 5, "city": 4}}
    main = {"seat": 0, "phase": "MAIN", "dev_cards_remaining": 3,
            "vertex_owner": [0, 0, -1], "vertex_building": [2, 1, 0], "edge_owner": [0, -1]}
    legal = [{"type": "END_TURN"}]
    out = mcptools._afford(hand, legal, main, board)
    assert out["road"] == {"ok": True, "legal": False, "why": "spot"}
    assert out["settlement"]["why"] == "spot" and out["city"]["why"] == "spot"
    assert out["dev_card"]["why"] == "spot"
    rolling = {**main, "phase": "ROLL"}
    assert {b: e["why"] for b, e in mcptools._afford(hand, [{"type": "ROLL"}], rolling, board).items()} == dict.fromkeys(mcptools._COSTS, "phase")
    capped = {**main, "vertex_owner": [0] * 5, "vertex_building": [1] * 5, "edge_owner": [0] * 15, "dev_cards_remaining": 0}
    out = mcptools._afford(hand, legal, capped, board)
    assert out["settlement"]["why"] == "pieces" and out["road"]["why"] == "pieces"
    assert out["city"]["why"] == "spot"
    assert out["dev_card"]["why"] == "deck"
    assert "why" not in mcptools._afford(hand, [{"type": "BUILD_ROAD"}], main, board)["road"]
    assert "why" not in mcptools._afford({"Wood": 1}, legal, main, board)["road"]


def test_race_measures_the_win_the_leader_and_both_awards():
    view = {
        "winning_points": 10,
        "players": [
            {"seat": 0, "victory_points": 6, "road_length": 4, "knights_played": 1, "longest_road": False, "largest_army": False},
            {"seat": 1, "victory_points": 7, "road_length": 6, "knights_played": 0, "longest_road": True, "largest_army": False},
            {"seat": 2, "victory_points": 3, "road_length": 2, "knights_played": 3, "longest_road": False, "largest_army": True},
        ],
    }
    race = mcptools._race(view, view["players"][0])
    assert race["points"] == 6 and race["to_win"] == 4
    assert race["top_opponent"] == {"seat": 1, "points": 7}
    assert race["longest_road"] == {"yours": 4, "held": False, "holder": 1, "holder_has": 6, "need": 7}
    assert race["largest_army"] == {"yours": 1, "held": False, "holder": 2, "holder_has": 3, "need": 4}


def test_summarize_skips_a_view_that_does_not_reveal_a_hand():
    spectator = {"seat": None, "players": [{"seat": 0}], "legal_actions": []}
    assert "summary" not in mcptools._summarize(spectator, TINY_BOARD)


def test_prune_drops_what_a_reader_never_acts_on():
    view = {
        "version": 9,
        "claimed_seats": [0, 1],
        "waiting_for": [1],
        "trade_wait": [],
        "seats": [{"seat": 0, "kind": "player", "name": "mcp"}, {"seat": 1, "kind": "empty", "name": None}],
        "bank": {"Wood": 0, "Brick": 19, "Sheep": 0, "Wheat": 1, "Ore": 0},
        "discard_quota": [0, 0, 0, 0],
        "players": [
            {
                "seat": 0,
                "last_roll": 7,
                "longest_road": True,
                "largest_army": False,
                "known": {"Wood": 1, "Brick": 0, "Sheep": 0, "Wheat": 0, "Ore": 0},
                "hand": {"Wood": 1, "Brick": 0, "Sheep": 2, "Wheat": 0, "Ore": 0},
                "dev_cards": {"Knight": 1, "Victory Point": 0, "Road Building": 0, "Year Of Plenty": 0, "Monopoly": 0},
            },
            {"seat": 1, "last_roll": None, "longest_road": False, "largest_army": False,
             "known": {"Wood": 0, "Brick": 0, "Sheep": 0, "Wheat": 0, "Ore": 0}},
        ],
    }
    pruned = mcptools._prune(view)
    for gone in ("version", "claimed_seats", "waiting_for", "trade_wait", "seats", "discard_quota"):
        assert gone not in pruned
    assert pruned["bank"] == {"Brick": 19, "Wheat": 1}
    assert pruned["players"][0] == {
        "seat": 0,
        "kind": "player",
        "known": {"Wood": 1},
        "hand": {"Wood": 1, "Sheep": 2},
        "dev_cards": {"Knight": 1},
    }
    assert pruned["players"][1] == {"seat": 1, "kind": "empty", "known": {}}


def test_prune_keeps_discard_quota_while_any_seat_owes():
    view = {"discard_quota": [0, 3, 0, 0], "players": []}
    assert mcptools._prune(view)["discard_quota"] == [0, 3, 0, 0]


def test_new_game_summary_ranks_the_setup_spots_it_offers(live_server):
    server, base = live_server
    client = connected(base)
    data = client.call_tool("new_game", identity=IDENTITY, opponents=SOLO)
    summary = data["summary"]
    assert summary["race"]["to_win"] == 10 and data["winning_points"] == 10
    assert summary["afford"]["road"] == {"ok": False, "legal": False, "missing": {"Wood": 1, "Brick": 1}}
    assert summary["spots"].startswith("SETUP_SETTLEMENT:(index,vertex,pips,resources")
    spots = rows(summary["spots"])
    assert "spots_omitted" not in summary
    raw_legal = server.tables.get(data["code"]).view(data["seat"])["legal_actions"]
    assert len(spots) == len(raw_legal) > 30
    pips = [s["pips"] for s in spots]
    assert pips == sorted(pips, reverse=True) and pips[0] > 0
    assert list(data["legal_actions"]) == []
    board = client.call_tool("board")
    vertex_rows = board.split("vertices: id pips resources port neighbors\n")[1].split("\n\n")[0].splitlines()
    by_id = {int(line.split()[0]): int(line.split()[1]) for line in vertex_rows}
    assert all(by_id[s["vertex"]] == s["pips"] for s in spots)
    assert "roads" not in summary


def test_setup_road_summary_names_the_far_vertex_and_keeps_its_legal_group(live_server):
    _, base = live_server
    client = connected(base)
    data = client.call_tool("new_game", identity=IDENTITY, opponents=SOLO)
    settlement = rows(data["summary"]["spots"])[0]
    after = client.call_tool("act", index=settlement["index"])
    assert after["phase"] == "SETUP_ROAD"

    roads = rows(after["summary"]["roads"])
    assert roads
    assert list(after["legal_actions"]) == ["SETUP_ROAD"]
    assert {r["index"] for r in roads} == {r["index"] for r in legal(after, "SETUP_ROAD")}
    assert all(r["settle"] == 0 for r in roads)
    assert all(r["to"] != settlement["vertex"] for r in roads)
    assert any(r.get("then") is not None and r["then_pips"] > 0 for r in roads)

    played = client.call_tool("act", index=roads[0]["index"])
    assert played["phase"] != "SETUP_ROAD"


# --- can_offer: whether offer_trade() would be accepted --------------------


def _park_in_main(server, code: str, seat: int, hand: dict | None = None) -> None:
    """END_TURN is always legal in MAIN, so this is enough for `can_offer`
    without a real hand; `hand` fills one in for a caller that also offers.
    """
    from hexset.board.terrain import NUM_RESOURCES, Resource
    from hexset.game import Phase

    game = server.tables.get(code).session.game
    game.phase = Phase.MAIN
    game.current_player = seat
    if hand is not None:
        game._state.hands[seat] = [0] * NUM_RESOURCES
        for name, count in hand.items():
            game._state.hands[seat][Resource[name.upper()]] = count


def test_can_offer_is_true_in_main_and_false_while_your_own_round_is_open(live_server):
    server, base = live_server
    client = connected(base)
    data = client.call_tool("new_game", identity=IDENTITY, opponents=SOLO)
    seat = data["seat"]
    assert data["can_offer"] is False

    _park_in_main(server, data["code"], seat, hand={"Wood": 1})
    state = client.call_tool("state")
    assert state["can_offer"] is True

    offered = client.call_tool("offer_trade", give={"Wood": 1}, want={"Ore": 1}, timeout=0)
    if offered["trade_round"] is not None:
        assert offered["can_offer"] is False
        offered = client.call_tool("choose_trade", decline=True)
    assert offered["trade_round"] is None
    assert offered["can_offer"] is True


# --- legal_actions drops what summary already covers ----------------------


def _park_in_robber(server, code: str, seat: int) -> None:
    from hexset.game import Phase

    game = server.tables.get(code).session.game
    game.phase = Phase.ROBBER
    game.current_player = seat


def _robber_index(view: dict) -> int:
    """`summary.robber`'s `options` cell is `index:victim`, `;`-joined; `rows()`
    only splits `resources`, so it is parsed by hand.
    """
    row = rows(view["summary"]["robber"])[0]
    return int(str(row["options"]).split(";")[0].split(":")[0])


def test_act_and_expect_still_resolve_a_robber_index_from_the_summary(live_server):
    server, base = live_server
    client = connected(base)
    data = client.call_tool("new_game", identity=IDENTITY, opponents=SOLO)
    seat = data["seat"]
    _park_in_robber(server, data["code"], seat)

    state = client.call_tool("state")
    hex_id = rows(state["summary"]["robber"])[0]["hex"]
    index = _robber_index(state)

    stale = client.call_tool_raw(
        "act", index=index, expect={"type": "MOVE_ROBBER", "hex": hex_id + 1, "victim": None}
    )
    assert stale[2]["result"]["isError"] is True

    result = client.call_tool("act", index=index, expect={"type": "MOVE_ROBBER", "hex": hex_id, "victim": None})
    assert result["robber"] == hex_id


# --- discard: every DISCARD in one call ----------------------------------


def _park_in_discard(server, code: str, seat: int, hand: dict, current_player: int | None = None) -> None:
    """Discarding is not a turn, so how the phase is reached does not matter."""
    from hexset.board.terrain import NUM_RESOURCES, Resource
    from hexset.game import Phase

    game = server.tables.get(code).session.game
    game.phase = Phase.DISCARD
    if current_player is not None:
        game.current_player = current_player
    game._state.hands[seat] = [0] * NUM_RESOURCES
    total = 0
    for name, count in hand.items():
        game._state.hands[seat][Resource[name.upper()]] = count
        total += count
    game.discard_quota = [0] * game._state.num_players
    game.discard_quota[seat] = total


def test_discard_zeroes_the_quota_and_settles_to_the_robber(live_server):
    server, base = live_server
    client = connected(base)
    data = client.call_tool("new_game", identity=IDENTITY, opponents=SOLO)
    seat = data["seat"]
    _park_in_discard(server, data["code"], seat, {"Wood": 2, "Brick": 2}, current_player=seat)

    result = client.call_tool("discard", cards={"Wood": 2, "Brick": 2})
    assert "DISCARD" not in result["legal_actions"]
    assert result["phase"] == "ROBBER"
    assert result["your_move"] == "act"


def test_discard_rejects_a_total_that_does_not_match_the_quota(live_server):
    server, base = live_server
    client = connected(base)
    data = client.call_tool("new_game", identity=IDENTITY, opponents=SOLO)
    seat = data["seat"]
    _park_in_discard(server, data["code"], seat, {"Wood": 2, "Brick": 2}, current_player=seat)

    status, _, response = client.call_tool_raw("discard", cards={"Wood": 1})
    assert status == 200 and response["result"]["isError"]
    assert "discard_quota of 4" in response["result"]["content"][0]["text"]
    assert client.call_tool("state")["phase"] == "DISCARD"


def test_discard_rejects_an_unknown_resource(live_server):
    server, base = live_server
    client = connected(base)
    data = client.call_tool("new_game", identity=IDENTITY, opponents=SOLO)
    seat = data["seat"]
    _park_in_discard(server, data["code"], seat, {"Wood": 4}, current_player=seat)

    status, _, response = client.call_tool_raw("discard", cards={"Gold": 4})
    assert status == 200 and response["result"]["isError"]
    assert "not a resource name" in response["result"]["content"][0]["text"]


def test_discard_only_works_during_the_discard_phase(live_server):
    _, base = live_server
    client = connected(base)
    client.call_tool("new_game", identity=IDENTITY, opponents=SOLO)

    status, _, response = client.call_tool_raw("discard", cards={"Wood": 1})
    assert status == 200 and response["result"]["isError"]
    assert "DISCARD phase" in response["result"]["content"][0]["text"]


def test_discard_posts_one_action_per_card_re_reading_between_them():
    state = {
        "phase": "DISCARD",
        "seat": 1,
        "discard_quota": [0, 4, 0, 0],
        "players": [{"seat": 1, "hand": {"Wood": 2, "Brick": 2}}],
        "legal_actions": [{"type": "DISCARD", "a": 0, "b": 0}, {"type": "DISCARD", "a": 1, "b": 0}],
    }
    after_wood = {**state, "discard_quota": [0, 3, 0, 0], "players": [{"seat": 1, "hand": {"Wood": 1, "Brick": 2}}]}
    after_both_wood = {
        **state,
        "discard_quota": [0, 2, 0, 0],
        "players": [{"seat": 1, "hand": {"Wood": 0, "Brick": 2}}],
        "legal_actions": [{"type": "DISCARD", "a": 1, "b": 0}],
    }
    after_one_brick = {**after_both_wood, "discard_quota": [0, 1, 0, 0], "players": [{"seat": 1, "hand": {"Brick": 1}}]}
    settled = {**after_one_brick, "discard_quota": [0, 0, 0, 0], "phase": "ROBBER", "legal_actions": []}
    tables = RecordingTables(
        {
            ("GET", "/api/state"): state,
            ("POST", "/api/action"): [after_wood, after_both_wood, after_one_brick, settled],
        }
    )
    data = mcptools.call_tool(
        tables, mcptools.Session(token="tok", code="abcdef"), "discard", {"cards": {"Wood": 2, "Brick": 2}, "timeout": 0}
    )
    assert [call[2] for call in tables.calls if call[1] == "/api/action"] == [
        {"action": {"type": "DISCARD", "a": 0, "b": 0}},
        {"action": {"type": "DISCARD", "a": 0, "b": 0}},
        {"action": {"type": "DISCARD", "a": 1, "b": 0}},
        {"action": {"type": "DISCARD", "a": 1, "b": 0}},
    ]
    assert data["phase"] == "ROBBER"


# --- The transcript cursor: log_after -----------------------------------


def test_trim_log_returns_only_what_is_new_plus_one_line_of_overlap():
    view = mcptools._trim_log({"log": ["a", "b", "c", "d"]}, 2)
    assert view["log"] == ["b", "c", "d"]
    assert view["log_from"] == 1
    assert view["log_total"] == 4


def test_trim_log_clamps_a_cursor_past_the_end_of_a_log_an_undo_shrank():
    view = mcptools._trim_log({"log": ["a", "b"]}, 9)
    assert view["log"] == ["b"]
    assert view["log_from"] == 1
    assert view["log_total"] == 2


def test_state_log_after_trims_the_transcript_it_sends_back(live_server):
    _, base = live_server
    client = connected(base)
    data = client.call_tool("new_game", identity=IDENTITY, opponents=SOLO)
    client.call_tool("act", index=_setup_settlement_index(data))

    full = client.call_tool("state", full_log=True)
    assert full["log_from"] == 0
    assert full["log_total"] == len(full["log"]) > 0

    caught_up = client.call_tool("state", log_after=full["log_total"])
    assert len(caught_up["log"]) == 1
    assert caught_up["log"][0] == full["log"][-1]
    assert caught_up["log_total"] == full["log_total"]


def test_the_cursor_is_automatic_for_a_caller_that_never_sends_log_after(live_server):
    _, base = live_server
    client = connected(base)
    dealt = client.call_tool("new_game", identity=IDENTITY, opponents=SOLO)
    assert dealt["log_from"] == 0
    for _ in range(2):
        data = client.call_tool("state")
        data = client.call_tool("act", index=_next_setup_index(data))
    client.call_tool("act", index=_setup_settlement_index(data))

    first = client.call_tool("state", full_log=True)
    assert first["log_from"] == 0
    assert first["log_total"] == len(first["log"]) >= 1

    again = client.call_tool("state")
    assert again["log_total"] == first["log_total"]
    assert again["log_from"] == first["log_total"] - 1
    assert again["log"] == first["log"][-1:]


def test_a_failed_call_does_not_move_the_cursor(live_server):
    _, base = live_server
    client = connected(base)
    data = client.call_tool("new_game", identity=IDENTITY, opponents=SOLO)
    client.call_tool("act", index=_setup_settlement_index(data))
    before = client.call_tool("state", full_log=True)

    status, _, response = client.call_tool_raw("act", index=999)
    assert status == 200 and response["result"]["isError"]
    assert client.call_tool("state")["log_from"] == before["log_total"] - 1


def test_wait_for_turn_carries_the_summary_too(live_server):
    _, base = live_server
    client = connected(base)
    client.call_tool("new_game", identity=IDENTITY, opponents=SOLO)
    for _ in range(4):
        view = client.call_tool("state")
        if not view.get("legal_actions"):
            break
        client.call_tool("act", index=0)

    waited = client.call_tool("wait_for_turn")
    assert "summary" in waited
    assert "afford" in waited["summary"]
    assert waited["summary"] == client.call_tool("state", full_log=True)["summary"]




# --- a call that fails still answers; malformed arguments are refused ----------


def test_a_tool_failing_unexpectedly_answers_is_error_rather_than_hanging(live_server, monkeypatch):
    def boom(tables, session):
        raise RuntimeError("something nobody planned for")

    _, description, schema = mcptools._TOOLS["bots"]
    monkeypatch.setitem(mcptools._TOOLS, "bots", (boom, description, schema))
    _, base = live_server
    client = connected(base)

    status, _, data = client.call_tool_raw("bots")

    assert status == 200
    assert data["result"]["isError"] is True
    assert "RuntimeError" in data["result"]["content"][0]["text"]


@pytest.mark.parametrize("counts", [{"Wood": -1}, {"Wood": 1.5}, {"Wood": "2"}, ["Wood"]])
def test_resource_counts_are_non_negative_integers(counts):
    with pytest.raises(mcptools.ToolError):
        mcptools._positional(counts)


@pytest.mark.parametrize(
    "arguments", [{"timeout": "soon"}, {"timeout": float("nan")}, {"log_after": -1}, {"log_after": 2.5}]
)
def test_a_malformed_wait_argument_refuses_the_call(arguments):
    with pytest.raises(mcptools.ToolError):
        mcptools.call_tool(new_tables(), mcptools.Session(), "state", arguments)


def test_can_offer_is_false_while_road_building_roads_are_owed():
    view = {"phase": "MAIN", "to_move": 0, "seat": 0, "trade_round": None,
            "legal_actions": [{"type": "BUILD_ROAD", "a": 3, "b": 0}]}
    assert mcptools._can_offer(view) is False
    view["legal_actions"].append({"type": "END_TURN", "a": 0, "b": 0})
    assert mcptools._can_offer(view) is True
