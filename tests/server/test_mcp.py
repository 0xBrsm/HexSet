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
IDENTITY = "claude-test-model"


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
def _short_settle_cap(monkeypatch):
    """A test that stalls a table by mistake fails on a `wait` reply within
    seconds instead of holding the call for `mcptools._MAX_WAIT`."""
    monkeypatch.setattr(mcptools, "_MAX_WAIT", 10.0)
    monkeypatch.setattr(mcptools, "_WAIT_TICK", 0.5)


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
    """The one JSON-RPC message out of a `tools/call` stream, ignoring the
    `: keepalive` comment lines before it."""
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


def test_initialize_echoes_a_known_protocol_version(live_server):
    _, base = live_server
    client = MCPClient(base)
    result = client.initialize(protocol_version="2024-11-05")
    assert result["protocolVersion"] == "2024-11-05"


def test_tools_call_without_a_session_id_is_400(live_server):
    _, base = live_server
    client = MCPClient(base)  # never initialized -- no session id to send
    status, _, data = client.call_tool_raw("bots")
    assert status == 400
    assert "error" in data


def test_tools_call_with_an_unknown_session_id_is_404(live_server):
    _, base = live_server
    client = MCPClient(base)
    client.initialize()
    client.session_id = "not-a-real-session"
    status, _, data = client.call_tool_raw("bots")
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
    status, _, _ = client.call_tool_raw("bots")
    assert status == 404  # the session is gone


# --- Identity: identity string -> client, over the real /mcp route -------


def test_new_game_records_the_clients_id_and_kind_mcp(live_server):
    server, base = live_server
    client = connected(base)
    data = client.call_tool("new_game", identity=" Claude-Opus-5 ", opponents=SOLO)
    expected_id = hashlib.sha256(b"claude-opus-5").hexdigest()
    seat = server.tables.get(data["code"]).seats[data["seat"]]
    assert seat.client == {"id": expected_id, "kind": "mcp"}


# --- Version guard: act(index, version) -----------------------------------


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
    """A reply's `(keys):row|row` table back into the dicts it was built
    from (`mcptools._tabulate`), enough for the tests to pick indexes and
    compare entries. A `KIND:` prefix (summary.spots) is dropped."""
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
    """SETUP_SETTLEMENT leaves `legal_actions` once `summary.spots` covers
    it (`mcptools._drop_superseded`) -- `spots` is where its `index` lives
    now, best (not first) placement."""
    return rows(view["summary"]["spots"])[0]["index"]


def _setup_road_index(view: dict) -> int:
    return legal(view, "SETUP_ROAD")[0]["index"]


def _next_setup_index(view: dict) -> int:
    """Either half of a setup placement, whichever this reply offers: the
    settlement half now lives in `summary.spots` (its `legal_actions` group
    is dropped once spots covers it); the road half is still the lone
    `legal_actions` group."""
    spots = (view.get("summary") or {}).get("spots")
    if spots:
        return rows(spots)[0]["index"]
    return rows(next(iter(view["legal_actions"].values())))[0]["index"]


def test_act_with_an_expect_that_no_longer_matches_is_an_error(live_server):
    """`expect` is the guard against an index that now names a different
    action. It replaced a whole-table `version`, which bumped on every other
    seat's move and every trade answer, and so was stale by design."""
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
    assert client.call_tool("state")["phase"] == "SETUP_SETTLEMENT"  # nothing was played


def test_act_with_a_matching_expect_acts(live_server):
    _, base = live_server
    client = connected(base)
    data = client.call_tool("new_game", identity=IDENTITY, opponents=SOLO)
    index = _setup_settlement_index(data)

    entry = rows(data["summary"]["spots"])[0]
    result = client.call_tool("act", index=index, expect={"type": "SETUP_SETTLEMENT", **entry})
    assert result["phase"] == "SETUP_ROAD"


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


def test_act_expect_may_name_only_the_type(live_server):
    _, base = live_server
    client = connected(base)
    data = client.call_tool("new_game", identity=IDENTITY, opponents=SOLO)
    index = _setup_settlement_index(data)

    result = client.call_tool("act", index=index, expect={"type": "SETUP_SETTLEMENT"})
    assert result["phase"] == "SETUP_ROAD"


def test_act_no_longer_takes_a_version(live_server):
    _, base = live_server
    client = connected(base)
    data = client.call_tool("new_game", identity=IDENTITY, opponents=SOLO)

    status, _, response = client.call_tool_raw("act", index=_setup_settlement_index(data), version=1)
    assert status == 200
    assert response["result"]["isError"] is True
    assert "bad arguments" in response["result"]["content"][0]["text"]


# --- Settling: every acting tool replies at this seat's next move ---------


def test_act_settles_through_the_bots_turns_to_our_next_move(live_server):
    """`act` on the move that hands the table to the bots does not come
    back until it is ours again: the reply is the next decision, not the
    instant after the action, so a seat never has to decide to wait."""
    server, base = live_server
    client = connected(base)
    data = client.call_tool("new_game", identity=IDENTITY, opponents=SOLO)
    assert data["seat"] == 0

    after_settlement = client.call_tool("act", index=_setup_settlement_index(data))
    assert after_settlement["your_move"] == "act"  # still our turn: the road
    road = _setup_road_index(after_settlement)
    after_road = client.call_tool("act", index=road)

    assert after_road["your_move"] == "act"  # round 2 of setup, ours again
    assert "SETUP_SETTLEMENT" not in after_road["legal_actions"]  # summary.spots covers it
    assert after_road["summary"]["spots"]
    assert after_road["waiting_on"] == []
    # The log slice covers what happened in between: the bots' placements.
    assert after_road["log_from"] == max(0, after_settlement["log_total"] - 1)
    assert after_road["log_total"] > after_settlement["log_total"] + 3


def test_forced_rolls_a_lone_roll_and_passes_only_with_an_empty_hand():
    empty = {"Wood": 0, "Brick": 0, "Sheep": 0, "Wheat": 0, "Ore": 0}
    # Signed towards the actor: positive is what the actor gets, i.e. what we give.
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
    # `to` ranks and filters: the best-ranked listed accepter wins, an unlisted one is refused.
    assert mcptools._settled_round(two_accepts, to=[2, 0]) == {"seat": 2, "bundle": bundle}
    assert mcptools._settled_round(two_accepts, to=[3]) == {"decline": True}
    assert mcptools._settled_round(one_accept, to=[0]) == {"seat": 0, "bundle": bundle}
    assert mcptools._settled_round(one_accept, to=[2]) == {"decline": True}
    assert mcptools._settled_round(with_counter, to=[0]) is None  # a counter is always the seat's call
    still_waiting = view([{"seat": 0, "kind": "pass", "bundle": None}]); still_waiting["trade_round"]["awaiting"] = [2]
    assert mcptools._settled_round(still_waiting) is None

    closed = {**all_pass, "trade_round": None}
    tables = RecordingTables({("POST", "/api/games/abcdef/trade/round/choose"): closed})
    out = mcptools._forced(tables, mcptools.Session(token="tok", code="abcdef"), all_pass)
    assert tables.calls == [("POST", "/api/games/abcdef/trade/round/choose", {"decline": True})]
    assert out is closed


def test_the_tool_list_has_no_undo_or_get_table():
    names = {t["name"] for t in mcptools.tool_list()}
    assert "undo" not in names and "get_table" not in names
    assert {"state", "act", "discard", "offer_trade", "answer_trade", "choose_trade"} <= names


def test_prune_drops_trivial_table_fields():
    view = {"to_move": 1, "awaiting_confirm": None, "can_undo": False, "locked": [], "trades": [], "winner": None,
            "started": True, "players": []}
    pruned = mcptools._prune(view)
    assert not {"to_move", "awaiting_confirm", "can_undo", "locked", "trades", "winner", "started"} & set(pruned)
    kept = mcptools._prune({"locked": [2], "trades": [{"a": 0}], "winner": 3, "started": False, "players": []})
    assert kept == {"locked": [2], "trades": [{"a": 0}], "winner": 3, "started": False, "players": []}


def test_forced_leaves_a_roll_alone_when_a_knight_is_also_legal():
    tables = RecordingTables({})
    start = {"seat": 1, "players": [{"seat": 1, "hand": {}}], "legal_actions": [{"type": "PLAY_KNIGHT"}, {"type": "ROLL"}]}
    assert mcptools._forced(tables, mcptools.Session(token="tok", code="abcdef"), start) is start
    assert tables.calls == []


def test_a_settled_turn_arrives_rolled(live_server):
    """Setup over, the seat's first real turn comes back past its ROLL: the
    lone forced action was played inside the settle."""
    _, base = live_server
    client = connected(base)
    data = client.call_tool("new_game", identity=IDENTITY, opponents=SOLO)
    for _ in range(4):  # two settlements, two roads; each settles to our next placement
        data = client.call_tool("act", index=_next_setup_index(data))
    assert data["round"] >= 1
    assert data["phase"] != "ROLL" and "ROLL" not in data["legal_actions"]
    assert data["your_move"] in ("act", "discard", "answer_trade")
    assert any("(mcp) rolled" in line for line in client.call_tool("state", full_log=True)["log"])


def test_board_is_incidence_text_and_leaves_the_browsers_layout_alone(live_server):
    """`board()` is text: each hex with its vertices, each vertex with its
    yield, port and neighbors -- no edge list; a legal edge's destination
    comes from `summary.roads` instead. It is built from a copy -- `GET
    /api/board` is the table's own `layout`, the dict the browser draws
    from, and its `x`/`y` must survive."""
    server, base = live_server
    client = connected(base)
    data = client.call_tool("new_game", identity=IDENTITY, opponents=SOLO)
    text = client.call_tool("board")
    assert text.startswith("hexes: id resource pips vertices\n")
    assert "\nvertices: id pips resources port neighbors\n" in text
    assert "\nedges:" not in text
    assert "x" not in text.split("\n")[1] and "y" not in text.split("\n")[1]
    layout = server.tables.get(data["code"]).layout
    assert "x" in layout["hexes"][0] and "y" in layout["vertices"][0] and "size" in layout
    assert "pips" not in layout["hexes"][0]  # the annotation never touched the shared dict either


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


def test_a_timeout_that_runs_out_replies_with_wait(live_server):
    """`timeout` (and `_MAX_WAIT`) is the one way a reply says `wait`."""
    _, base = live_server
    client = connected(base)
    data = client.call_tool("new_game", identity=IDENTITY, opponents=["heximax", "heximax"])
    # A second MCP seat that never moves: the table stalls at its placement.
    idle = connected(base)
    idle.call_tool("join", code=data["code"], identity="idle", timeout=0)
    client.call_tool("act", index=_setup_settlement_index(data))
    road = _setup_road_index(client.call_tool("state"))
    stalled = client.call_tool("act", index=road, timeout=0.2)
    assert stalled["your_move"] == "wait"
    assert stalled["legal_actions"] == {}
    assert stalled["waiting_on"]
    waited = client.call_tool("wait_for_turn", timeout=0.2)
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
    assert data["result"]["isError"] is True  # a ToolError is a message on the same stream


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


class RecordingTables(FakeTables):
    """`FakeTables` that also remembers every call, and can answer a
    `(method, path)` with a queue of responses, one per call."""

    def __init__(self, responses):
        super().__init__(responses)
        self.calls: list[tuple[str, str, dict]] = []

    def handle(self, method, path, payload, token):
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
    assert session.offer_to == [3, 1]  # ranked, de-duplicated, kept for the settle
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


# --- Seat name defaults to "mcp" when the LLM doesn't give one ---------------


def test_new_game_defaults_the_seat_name_to_mcp(live_server):
    server, base = live_server
    client = connected(base)
    data = client.call_tool("new_game", identity=IDENTITY, opponents=SOLO)
    assert server.tables.get(data["code"]).seats[data["seat"]].name == "mcp"


def test_new_game_keeps_an_explicit_name(live_server):
    server, base = live_server
    client = connected(base)
    data = client.call_tool("new_game", identity=IDENTITY, opponents=SOLO, name="Ada")
    assert server.tables.get(data["code"]).seats[data["seat"]].name == "Ada"


def test_join_defaults_the_seat_name_to_mcp(live_server):
    server, base = live_server
    creator = connected(base)
    created = creator.call_tool("new_game", identity=IDENTITY)
    joiner = connected(base)
    # `timeout=0`: the creator is to move, so a settling join would wait.
    joined = joiner.call_tool("join", code=created["code"], identity=IDENTITY, timeout=0)
    assert joined["your_move"] == "wait"
    assert server.tables.get(created["code"]).seats[joined["seat"]].name == "mcp"


def test_new_game_always_installs_a_pending_gate_for_the_llm_seat(live_server):
    """Human and LLM seats are direct gates, unconditionally
    (`agents/reference/trading-final.md`, item 5): there is no `confirm`
    flag left anywhere on the wire, so an LLM's own seat lands in
    `confirm_seats`/`PendingGate` the same as a human's at the web page,
    with no argument required to ask for it."""
    server, base = live_server
    client = connected(base)
    data = client.call_tool("new_game", identity=IDENTITY, opponents=SOLO)
    seat = data["seat"]
    table = server.tables.get(data["code"])
    assert seat in table.session.confirm_seats
    from hexset.server.webplay import PendingGate

    assert isinstance(table.session.game.gates[seat], PendingGate)


# --- leave_game --------------------------------------------------------------


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
    creator.call_tool("act", index=_setup_settlement_index(data))  # so there is a transcript to owe

    # A fresh session -- as if the server had restarted, or this were simply
    # a new MCP connection with no seat of its own yet.
    fresh = connected(base)
    result = fresh.call_tool("resume_game", code=code, identity=IDENTITY)
    assert result["seat"] == seat
    # A reclaimed seat knows nothing yet, so it is owed the whole transcript.
    assert result["log_from"] == 0
    assert len(result["log"]) == result["log_total"] >= 1


# --- The compact reply: grouped legal_actions, sparse occupancy -----------
#
# The wire's flat `{type, a, b}` list stays what `act(index)` resolves
# against; the reply groups it by type with a named operand and the flat
# index on each entry, and lists what is on the board rather than three
# dense arrays (see `mcptools._compact`).


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


def test_compact_board_leaves_a_view_without_the_arrays_alone():
    assert mcptools._compact_board({"phase": "ROLL"}) == {"phase": "ROLL"}


def test_new_game_reply_is_compact(live_server):
    _, base = live_server
    client = connected(base)
    data = client.call_tool("new_game", identity=IDENTITY, opponents=SOLO)
    assert not {"vertex_owner", "vertex_building", "edge_owner"} & set(data)
    assert data["buildings"] == []
    assert data["roads"] == [[], [], [], []]
    # SETUP_SETTLEMENT itself has left legal_actions: summary.spots covers it.
    assert data["legal_actions"] == {}
    assert len(rows(data["summary"]["spots"])) > 0 and "legal_count" not in data
    assert data["summary"]["spots"].startswith("SETUP_SETTLEMENT:(index,vertex,pips,resources")

    spot = rows(data["summary"]["spots"])[0]
    after = client.call_tool("act", index=spot["index"])
    assert after["buildings"] == [{"vertex": spot["vertex"], "seat": 0, "kind": "settlement"}]
    assert list(after["legal_actions"]) == ["SETUP_ROAD"]
    assert after["legal_actions"]["SETUP_ROAD"].startswith("(index,edge):")


# --- summary: derived tactical facts, computed once server-side -----------
#
# The joins an LLM otherwise redoes by hand every turn (see
# `mcptools._summarize`): costs against the hand, legal spots against the
# board's pips, robber targets against who is built where, award distances.


def test_afford_reports_ok_missing_and_whether_it_is_legal_now():
    hand = {"Wood": 1, "Brick": 1, "Sheep": 1, "Wheat": 0, "Ore": 3}
    legal = [{"type": "BUILD_ROAD", "a": 4, "b": 0}, {"type": "END_TURN", "a": 0, "b": 0}]
    afford = mcptools._afford(hand, legal)
    assert afford["road"] == {"ok": True, "legal": True}
    assert afford["settlement"] == {"ok": False, "legal": False, "missing": {"Wheat": 1}}
    assert afford["city"] == {"ok": False, "legal": False, "missing": {"Wheat": 2}}
    assert afford["dev_card"] == {"ok": False, "legal": False, "missing": {"Wheat": 1}}


def test_afford_omits_legal_when_legal_actions_is_empty():
    """`your_move: wait` already says nothing is legal; `legal: false` on
    all four builds for the same reason is the one fact repeated."""
    hand = {"Wood": 1, "Brick": 1, "Sheep": 1, "Wheat": 0, "Ore": 3}
    afford = mcptools._afford(hand, [])
    assert afford["road"] == {"ok": True}
    assert afford["settlement"] == {"ok": False, "missing": {"Wheat": 1}}
    assert all("legal" not in entry for entry in afford.values())


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


def test_spots_lists_every_placement_uncapped():
    """Setup offers every open vertex -- fifty-odd -- and `spots` used to
    cap that list at fifteen. It doesn't any more: the raw group next to it
    that once justified trimming a repeat is gone (`_drop_superseded`), so
    the tail is the only place any of these vertices is read back from."""
    board = {"vertices": [{"id": v, "pips": v, "resources": []} for v in range(40)], "ports": []}
    legal = [{"type": "SETUP_SETTLEMENT", "a": v, "b": 0} for v in range(40)]
    spots = mcptools._spots(legal, board)
    assert len(spots) == 40
    assert spots[0]["vertex"] == 39  # the best, not the first
    assert spots[-1]["vertex"] == 0


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


def test_far_endpoint_prefers_the_end_this_seats_network_does_not_touch():
    neighbors = {0: {1}, 1: {0, 2}, 2: {1, 3}, 3: {2}}
    # One end already ours: the other is the new ground.
    assert mcptools._far_endpoint(0, 1, {0}, neighbors) == 1
    assert mcptools._far_endpoint(1, 0, {0}, neighbors) == 1
    # Both new, but 1 is a neighbor of the owned vertex 0 and 2 is not:
    # 2 is the more frontier-ish end.
    assert mcptools._far_endpoint(1, 2, {0}, neighbors) == 2
    # Both new and neither adjacent to anything owned: either end, by
    # convention the second one named.
    assert mcptools._far_endpoint(2, 3, set(), neighbors) == 3


def test_roads_joins_legal_edges_to_the_vertex_they_reach_settle_first():
    legal = [
        {"type": "BUILD_ROAD", "a": 10, "b": 0},
        {"type": "BUILD_ROAD", "a": 11, "b": 0},
        {"type": "BUILD_ROAD", "a": 12, "b": 0},
    ]
    view = {"seat": 0, "vertex_owner": [0, -1, -1, -1], "edge_owner": [-1, -1, -1]}
    roads = mcptools._roads(legal, view, ROAD_BOARD)
    by_edge = {r["edge"]: r for r in roads}
    # Edge 10 (0-1): 0 is ours, so it reaches 1 -- but 1 neighbors our own
    # vertex 0, so no settlement could go there.
    assert by_edge[10] == {"index": 0, "edge": 10, "to": 1, "pips": 8, "resources": ["Ore", "Wheat"], "settle": False,
                           "then": 2, "then_pips": 3}
    # Edge 11 (1-2): both new, but 1 is adjacent to our vertex 0 and 2 is
    # not -- 2 is the frontier, and it is open two roads clear.
    assert by_edge[11] == {"index": 1, "edge": 11, "to": 2, "pips": 3, "resources": ["Sheep"], "settle": True,
                           "then": 3, "then_pips": 6}
    # Edge 12 (2-3): both new, neither adjacent to anything owned -- picks
    # 3, which carries its port.
    assert by_edge[12]["to"] == 3 and by_edge[12]["port"] == "Brick 2:1" and by_edge[12]["settle"] is True
    assert "then" not in by_edge[12]  # 3 is the end of the line: nothing settleable beyond it
    # Sorted settle first, then pips descending within each: 12 (6 pips)
    # before 11 (3 pips), both settleable; 10 last, not settleable at all.
    assert [r["edge"] for r in roads] == [12, 11, 10]


def test_spots_say_whether_a_port_matches_what_the_vertex_yields():
    board = {
        "vertices": [
            {"id": 0, "pips": 5, "resources": ["Wheat"]},
            {"id": 1, "pips": 4, "resources": ["Ore"]},
            {"id": 2, "pips": 3, "resources": ["Wood"]},
            {"id": 3, "pips": 2, "resources": ["Brick"]},
        ],
        "ports": [
            {"vertices": [0, 1], "resource": "Wheat", "ratio": 2},
            {"vertices": [2], "resource": None, "ratio": 3},
        ],
    }
    legal = [{"type": "BUILD_SETTLEMENT", "a": v, "b": 0} for v in range(4)]
    spots = mcptools._spots(legal, board)
    by_vertex = {s["vertex"]: s for s in spots}
    assert by_vertex[0]["port"] == "Wheat 2:1" and by_vertex[0]["port_matches"] is True
    assert by_vertex[1]["port"] == "Wheat 2:1" and by_vertex[1]["port_matches"] is False
    assert by_vertex[2]["port"] == "3:1" and by_vertex[2]["port_matches"] is True
    assert "port" not in by_vertex[3] and "port_matches" not in by_vertex[3]


def test_spots_keep_their_type_column_when_settlements_and_cities_mix():
    uniform = {"spots": [{"index": 0, "type": "BUILD_CITY", "vertex": 3, "pips": 5, "resources": ["Ore"]}]}
    mcptools._tabulate_summary(uniform)
    assert uniform["spots"] == "BUILD_CITY:(index,vertex,pips,resources):0,3,5,Ore"
    mixed = {"spots": [
        {"index": 0, "type": "BUILD_CITY", "vertex": 3, "pips": 5, "resources": ["Ore"]},
        {"index": 4, "type": "BUILD_SETTLEMENT", "vertex": 9, "pips": 4, "resources": ["Wood"]},
    ]}
    mcptools._tabulate_summary(mixed)
    assert mixed["spots"] == "(index,type,vertex,pips,resources):0,BUILD_CITY,3,5,Ore|4,BUILD_SETTLEMENT,9,4,Wood"


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
    assert race["leader"] == {"seat": 1, "points": 7}
    assert race["longest_road"] == {"yours": 4, "held": False, "holder": 1, "holder_has": 6, "need": 7}
    assert race["largest_army"] == {"yours": 1, "held": False, "holder": 2, "holder_has": 3, "need": 4}


def test_race_when_you_hold_an_award_and_nobody_holds_the_other():
    view = {
        "players": [
            {"seat": 0, "victory_points": 4, "road_length": 5, "knights_played": 1, "longest_road": True, "largest_army": False},
            {"seat": 1, "victory_points": 2, "road_length": 3, "knights_played": 2, "longest_road": False, "largest_army": False},
        ],
    }
    race = mcptools._race(view, view["players"][0])
    assert race["to_win"] == 6  # the standard ten when the view does not say
    assert race["longest_road"] == {"yours": 5, "held": True, "holder": 0, "holder_has": 5}
    assert race["largest_army"] == {"yours": 1, "held": False, "need": 3}


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


def test_trade_ratios_repeat_sends_once_then_omits_the_unchanged_value():
    session = mcptools.Session()
    ratios = {"Wood": 4, "Brick": 4, "Sheep": 4, "Wheat": 4, "Ore": 4}
    first = {"trade_ratios": dict(ratios)}
    mcptools._trade_ratios_repeat(first, session)
    assert first["trade_ratios"] == ratios  # new seat: sent once
    assert session.trade_ratios_sent == ratios

    second = {"trade_ratios": dict(ratios)}
    mcptools._trade_ratios_repeat(second, session)
    assert "trade_ratios" not in second  # unchanged since last time: omitted

    changed = {"trade_ratios": {**ratios, "Wood": 2}}
    mcptools._trade_ratios_repeat(changed, session)
    assert changed["trade_ratios"]["Wood"] == 2  # a port arrived: sent again
    assert session.trade_ratios_sent["Wood"] == 2


def test_a_live_reply_is_pruned_and_still_playable(live_server):
    _, base = live_server
    client = connected(base)
    data = client.call_tool("new_game", identity=IDENTITY, opponents=SOLO)
    for gone in ("version", "claimed_seats", "waiting_for", "trade_wait", "seats"):
        assert gone not in data
    assert [p["kind"] for p in data["players"]] == ["player", "bot", "bot", "bot"]
    assert all("last_roll" not in p for p in data["players"])
    assert all("longest_road" not in p and "largest_army" not in p for p in data["players"])
    assert "last_roll" in data and "winning_points" in data
    assert "discard_quota" not in data  # nobody owes anything yet
    me = data["players"][data["seat"]]
    assert me["hand"] == {} and all(me["known"].values()) and all(data["bank"].values())
    # A summary computed against the same hand is unaffected by sparse counts.
    assert data["summary"]["afford"]["road"]["missing"] == {"Wood": 1, "Brick": 1}
    result = client.call_tool("act", index=rows(data["summary"]["spots"])[0]["index"])
    assert result["phase"] == "SETUP_ROAD" and "version" not in result


def test_trade_ratios_is_sent_once_then_omitted_while_unchanged(live_server):
    _, base = live_server
    client = connected(base)
    data = client.call_tool("new_game", identity=IDENTITY, opponents=SOLO)
    assert "trade_ratios" in data  # a new seat always gets a starting value

    again = client.call_tool("state")
    assert "trade_ratios" not in again  # nothing changed since the last reply


def test_new_game_summary_ranks_the_setup_spots_it_offers(live_server):
    server, base = live_server
    client = connected(base)
    data = client.call_tool("new_game", identity=IDENTITY, opponents=SOLO)
    summary = data["summary"]
    assert summary["race"]["to_win"] == 10 and data["winning_points"] == 10
    assert summary["afford"]["road"] == {"ok": False, "legal": False, "missing": {"Wood": 1, "Brick": 1}}
    assert summary["spots"].startswith("SETUP_SETTLEMENT:(index,vertex,pips,resources")
    spots = rows(summary["spots"])
    assert "spots_omitted" not in summary  # uncapped: every legal vertex is here
    raw_legal = server.tables.get(data["code"]).view(data["seat"])["legal_actions"]
    assert len(spots) == len(raw_legal) > 30
    pips = [s["pips"] for s in spots]
    assert pips == sorted(pips, reverse=True) and pips[0] > 0
    assert list(data["legal_actions"]) == []  # SETUP_SETTLEMENT dropped: spots covers it
    board = client.call_tool("board")
    vertex_rows = board.split("vertices: id pips resources port neighbors\n")[1].split("\n\n")[0].splitlines()
    by_id = {int(line.split()[0]): int(line.split()[1]) for line in vertex_rows}
    assert all(by_id[s["vertex"]] == s["pips"] for s in spots)
    assert "roads" not in summary  # no road is legal yet: still SETUP_SETTLEMENT


def test_setup_road_summary_names_the_far_vertex_and_keeps_its_legal_group(live_server):
    _, base = live_server
    client = connected(base)
    data = client.call_tool("new_game", identity=IDENTITY, opponents=SOLO)
    settlement = rows(data["summary"]["spots"])[0]
    after = client.call_tool("act", index=settlement["index"])
    assert after["phase"] == "SETUP_ROAD"

    roads = rows(after["summary"]["roads"])
    assert roads
    assert list(after["legal_actions"]) == ["SETUP_ROAD"]  # summary.roads does not drop this group
    assert {r["index"] for r in roads} == {r["index"] for r in legal(after, "SETUP_ROAD")}
    # Every first road reaches a vertex one edge from the settlement just
    # placed, so none of them could take a settlement of their own (the
    # standard two-road minimum distance) and none names that vertex itself.
    assert all(r["settle"] == 0 for r in roads)
    assert all(r["to"] != settlement["vertex"] for r in roads)
    assert any(r.get("then") is not None and r["then_pips"] > 0 for r in roads)  # the two-road plan is named

    played = client.call_tool("act", index=roads[0]["index"])
    assert played["phase"] != "SETUP_ROAD"  # settled to the next decision


# --- can_offer: whether offer_trade() would be accepted --------------------


def test_can_offer_true_only_on_your_own_turn_in_main_with_something_legal():
    base = {"phase": "MAIN", "to_move": 0, "seat": 0, "legal_actions": [{"type": "END_TURN"}], "trade_round": None}
    assert mcptools._can_offer(base) is True
    assert mcptools._can_offer({**base, "phase": "SETUP_ROAD"}) is False
    assert mcptools._can_offer({**base, "to_move": 1}) is False
    assert mcptools._can_offer({**base, "legal_actions": []}) is False
    assert mcptools._can_offer({**base, "trade_round": {"offer": {}, "responses": [], "awaiting": [1]}}) is False


def _park_in_main(server, code: str, seat: int, hand: dict | None = None) -> None:
    """Force a live table straight into `Phase.MAIN` on `seat`'s turn --
    the same forcing shortcut `_park_in_discard`/`_park_in_robber` use.
    END_TURN is always legal in MAIN, so this is enough for `can_offer`
    without needing a real hand; `hand` fills one in for a caller that
    also wants to offer a trade (`open_round` refuses an offer this
    seat's hand can't cover)."""
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
    assert data["can_offer"] is False  # still SETUP_SETTLEMENT

    _park_in_main(server, data["code"], seat, hand={"Wood": 1})
    state = client.call_tool("state")
    assert state["can_offer"] is True

    offered = client.call_tool("offer_trade", give={"Wood": 1}, want={"Ore": 1}, timeout=0)
    if offered["trade_round"] is not None:  # the bots answered at once
        assert offered["can_offer"] is False
        offered = client.call_tool("choose_trade", decline=True)
    assert offered["trade_round"] is None
    assert offered["can_offer"] is True  # the round is closed; still our MAIN turn


# --- legal_actions drops what summary already covers ----------------------


def _park_in_robber(server, code: str, seat: int) -> None:
    """Force a live table straight into `Phase.ROBBER` on `seat`'s move --
    the same forcing shortcut `_park_in_discard` uses, for the same reason:
    nothing about reaching the phase honestly matters to what this checks."""
    from hexset.game import Phase

    game = server.tables.get(code).session.game
    game.phase = Phase.ROBBER
    game.current_player = seat


def _robber_index(view: dict) -> int:
    """The `act()` index out of `summary.robber`'s first hex's `options`
    cell (`index:victim`, `;`-joined per victim -- see `_tabulate_summary`).
    `rows()` only splits the `resources` column, so this one cell is parsed
    by hand."""
    row = rows(view["summary"]["robber"])[0]
    return int(str(row["options"]).split(";")[0].split(":")[0])


def test_move_robber_group_is_dropped_once_summary_robber_covers_it(live_server):
    server, base = live_server
    client = connected(base)
    data = client.call_tool("new_game", identity=IDENTITY, opponents=SOLO)
    seat = data["seat"]
    _park_in_robber(server, data["code"], seat)

    state = client.call_tool("state")
    assert "MOVE_ROBBER" not in state["legal_actions"]
    assert state["summary"]["robber"]
    assert "legal_count" not in state


def test_act_and_expect_still_resolve_a_robber_index_from_the_summary(live_server):
    """Dropping the MOVE_ROBBER group from `legal_actions` must not touch
    `act(index)`/`expect`: both resolve against the raw list `_act` fetches
    fresh, never the grouped one."""
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
#
# A seven with a nine-card hand used to be four act(index) round-trips.
# `discard(cards)` posts them all, re-reading the fresh legal_actions
# between each the way `_act` resolves an index (`mcptools._discard`).


def _park_in_discard(server, code: str, seat: int, hand: dict, current_player: int | None = None) -> None:
    """Force a live table straight into `Phase.DISCARD` owing exactly half
    of `hand` from `seat`, the same shortcut `test_api.py`/`test_webplay.py`
    use to reach the phase without a real seven -- discarding is not a
    turn (`hexset.game.may_act`), so nothing about reaching it honestly
    matters to what these tests check."""
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
    assert result["your_move"] == "act"  # the robber move is ours: same seat rolled


def test_discard_rejects_a_total_that_does_not_match_the_quota(live_server):
    server, base = live_server
    client = connected(base)
    data = client.call_tool("new_game", identity=IDENTITY, opponents=SOLO)
    seat = data["seat"]
    _park_in_discard(server, data["code"], seat, {"Wood": 2, "Brick": 2}, current_player=seat)

    status, _, response = client.call_tool_raw("discard", cards={"Wood": 1})
    assert status == 200 and response["result"]["isError"]
    assert "discard_quota of 4" in response["result"]["content"][0]["text"]
    assert client.call_tool("state")["phase"] == "DISCARD"  # nothing was played


def test_discard_rejects_more_than_the_hand_holds(live_server):
    server, base = live_server
    client = connected(base)
    data = client.call_tool("new_game", identity=IDENTITY, opponents=SOLO)
    seat = data["seat"]
    _park_in_discard(server, data["code"], seat, {"Wood": 2, "Brick": 2}, current_player=seat)

    status, _, response = client.call_tool_raw("discard", cards={"Wood": 3, "Brick": 1})
    assert status == 200 and response["result"]["isError"]
    assert "short" in response["result"]["content"][0]["text"]


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
    client.call_tool("new_game", identity=IDENTITY, opponents=SOLO)  # setup phase, not DISCARD

    status, _, response = client.call_tool_raw("discard", cards={"Wood": 1})
    assert status == 200 and response["result"]["isError"]
    assert "DISCARD phase" in response["result"]["content"][0]["text"]


def test_act_still_plays_a_single_discard(live_server):
    """The bulk `discard(cards)` tool is new; `act(index)` on one DISCARD
    entry at a time -- the only way to discard before it existed -- must
    keep working."""
    server, base = live_server
    client = connected(base)
    data = client.call_tool("new_game", identity=IDENTITY, opponents=SOLO)
    seat = data["seat"]
    _park_in_discard(server, data["code"], seat, {"Wood": 2, "Brick": 2}, current_player=seat)

    state = client.call_tool("state")
    index = legal(state, "DISCARD")[0]["index"]
    after = client.call_tool("act", index=index)
    assert after["phase"] == "DISCARD"
    assert sum((after["players"][seat].get("hand") or {}).values()) == 3


def test_discard_posts_one_action_per_card_re_reading_between_them():
    """Unit-level: exactly the DISCARD wire actions land, one per card, and
    each is matched against the `legal_actions` the previous POST just
    returned -- never a stale one from the initial GET."""
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


# --- your_move: which tool the table wants from this seat ---------------
#
# One field in place of reading `legal_actions`, `pending`, `trade_round`,
# `trade_wait` and `to_move` together (see `mcptools._your_move`).


def test_your_move_is_act_on_your_own_turn(live_server):
    _, base = live_server
    client = connected(base)
    data = client.call_tool("new_game", identity=IDENTITY, opponents=SOLO)
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
    data = client.call_tool("new_game", identity=IDENTITY, opponents=SOLO)
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
    data = client.call_tool("new_game", identity=IDENTITY, opponents=SOLO)
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
    dealt = client.call_tool("new_game", identity=IDENTITY, opponents=SOLO)
    assert dealt["log_from"] == 0
    # Our first settlement and road, then the bots' -- then our second
    # settlement, after which the table is ours (the road is still owed) and
    # the transcript holds still for the rest of the test.
    for _ in range(2):
        data = client.call_tool("state")
        data = client.call_tool("act", index=_next_setup_index(data))
    client.call_tool("act", index=_setup_settlement_index(data))  # settled: ours again

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
    data = client.call_tool("new_game", identity=IDENTITY, opponents=SOLO)
    client.call_tool("act", index=_setup_settlement_index(data))
    client.call_tool("state")  # the session now holds everything

    rewound = client.call_tool("state", log_after=1)
    assert rewound["log_from"] == 0
    assert len(rewound["log"]) == rewound["log_total"]


def test_full_log_resets_a_session_that_is_already_caught_up(live_server):
    _, base = live_server
    client = connected(base)
    data = client.call_tool("new_game", identity=IDENTITY, opponents=SOLO)
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
    data = client.call_tool("new_game", identity=IDENTITY, opponents=SOLO)
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


def test_wait_for_turn_carries_the_summary_too(live_server):
    """`wait_for_turn` is the call a seat makes to reach its own turn, so the
    view it returns is exactly the one `afford`/`spots`/`robber` are for. It
    does not go through `_reply`, so the summary has to be applied at the
    yield -- it was missing here while `state` had it."""
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


def test_wait_for_turn_returns_the_same_shape_as_state(live_server):
    """Every outgoing view crosses one seam (`mcptools._finish`), whichever
    tool answers. The streamed reply drifted from `state()` twice -- first
    without `summary`, then with the flat `legal_actions` while `state()`
    grouped them -- so the whole key set is pinned equal here, not one field."""
    _, base = live_server
    client = connected(base)
    data = client.call_tool("new_game", identity=IDENTITY, opponents=SOLO)
    client.call_tool("act", index=_setup_settlement_index(data))
    waited = client.call_tool("act", index=_setup_road_index(client.call_tool("state")))
    assert waited["your_move"] == "act"  # settled through the bots' placements

    plain = client.call_tool("state")
    assert set(waited) == set(plain)
    assert isinstance(waited["legal_actions"], dict) and waited["summary"]["spots"]
    assert waited["legal_actions"] == plain["legal_actions"]
    assert "legal_count" not in waited
    assert waited["buildings"] == plain["buildings"] and waited["roads"] == plain["roads"]
    assert not {"vertex_owner", "vertex_building", "edge_owner"} & set(waited)
