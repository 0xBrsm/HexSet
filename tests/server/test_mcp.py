"""Round-trip tests for `hexset.server.mcp` — the LLM-facing tool layer over
the same HTTP API `test_api.py` exercises directly. Each tool is a thin
`urllib` call (see `mcp.py`'s module docstring), so these tests run a real
`HexSetServer` and drive the tool functions through `_call_tool`/`_dispatch`
the way an actual MCP client would, checking that the JSON that comes back
names the right seat, game and trade.
"""

from __future__ import annotations

import hashlib
import json
import random

import threading

import pytest

from conftest import new_tables

from hexset.server import mcp

from hexset.server.web import HexSetServer

SOLO = ["search2", "search2", "search2"]


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


@pytest.fixture(autouse=True)
def _reset_mcp_state(monkeypatch):
    """Each test gets its own server and its own seat: the module-global
    `_token`/`_code`/`_model` (see `mcp.py`'s docstring on why they're
    globals — one process, one seat) must not leak across tests."""
    monkeypatch.setattr(mcp, "_token", None)
    monkeypatch.setattr(mcp, "_code", None)
    monkeypatch.setattr(mcp, "_model", None)


MODEL = "claude-test-model"


def call(tool: str, **arguments) -> dict:
    """One tool call, the way `_dispatch`'s `tools/call` branch makes it:
    JSON in, JSON out, `ToolError` surfaced as `isError` rather than raised."""
    result = mcp._call_tool(tool, arguments)
    assert result["content"][0]["type"] == "text"
    payload = json.loads(result["content"][0]["text"])
    if result["isError"]:
        raise AssertionError(f"{tool}({arguments}) failed: {payload}")
    return payload


def test_new_game_seats_the_caller_and_remembers_the_code(live_server):
    _, base = live_server
    mcp.BASE_URL = base
    data = call("new_game", model=MODEL, opponents=SOLO, name="Ada")
    assert "token" not in data  # popped by _seat -- never reaches the LLM
    assert data["code"] == mcp._code
    assert mcp._token


def test_new_game_always_installs_a_pending_gate_for_the_llm_seat(live_server):
    """Human and LLM seats are direct gates, unconditionally
    (`agents/reference/trading-final.md`, item 5): there is no `confirm`
    flag left anywhere on the wire, so an LLM's own seat lands in
    `confirm_seats`/`PendingGate` the same as a human's at the web page,
    with no argument required to ask for it."""
    server, base = live_server
    mcp.BASE_URL = base
    data = call("new_game", model=MODEL, opponents=SOLO)
    seat = data["seat"]
    table = server.tables.get(data["code"])
    assert seat in table.session.confirm_seats
    from hexset.server.webplay import PendingGate

    assert isinstance(table.session.game.gates[seat], PendingGate)


# --- Trade-round responses are named dicts, the same as state()/get_table() --
#
# `_translate_trades` already turns a raw signed bundle into `you_give`/
# `you_receive` for state()/get_table(); offer_trade()/answer_trade()/
# choose_trade() used to skip it and hand back the wire's raw `bundle`
# instead, on the one response an LLM reads right after acting. These pin
# each of the three to the same translated shape, by stubbing `_request_ok`
# rather than standing up bot traders to answer for real.

RAW_BUNDLE = [0, -1, 0, 1, 0]  # signed towards the actor: gives one Brick, gets one Wheat


@pytest.fixture
def _seat_for_trade_stubs(monkeypatch):
    """The three tests below stub `_request_ok` directly, so they need a
    seat/code already on the module (normally `_seat()`'s job) without a
    live server behind it."""
    monkeypatch.setattr(mcp, "_token", "tok")
    monkeypatch.setattr(mcp, "_code", "abcdef")


def _stub_request_ok(responses: dict[tuple[str, str], dict]):
    def fake(method: str, path: str, body: dict | None = None) -> dict:
        return responses[(method, path)]

    return fake


def test_offer_trade_translates_its_own_response(monkeypatch, _seat_for_trade_stubs):
    raw = {
        "trade_round": {
            "offer": {"actor": 2, "bundle": RAW_BUNDLE},
            "responses": [],
            "awaiting": [0, 1, 3],
        }
    }
    monkeypatch.setattr(
        mcp, "_request_ok", _stub_request_ok({("POST", "/api/games/abcdef/trade/round"): raw})
    )
    data = call("offer_trade", give={"Brick": 1}, want={"Wheat": 1})
    assert data["trade_round"]["you_give"] == {"Brick": 1}
    assert data["trade_round"]["you_receive"] == {"Wheat": 1}


def test_answer_trade_translates_its_own_response(monkeypatch, _seat_for_trade_stubs):
    state = {"pending": [{"actor": 0, "bundle": RAW_BUNDLE}]}
    answered = {"pending": [], "trade_round": None}
    monkeypatch.setattr(
        mcp,
        "_request_ok",
        _stub_request_ok(
            {
                ("GET", "/api/state"): state,
                ("POST", "/api/games/abcdef/trade/round/answer"): answered,
            }
        ),
    )
    data = call("answer_trade", index=0, kind="accept")
    assert data["pending"] == []
    assert data["trade_round"] is None


def test_choose_trade_translates_its_own_response(monkeypatch, _seat_for_trade_stubs):
    chosen = {
        "trades": [{"a": 2, "b": 0, "gave": [0, 1, 0, 0, 0], "got": [0, 0, 0, 1, 0]}],
        "trade_round": None,
    }
    monkeypatch.setattr(
        mcp, "_request_ok", _stub_request_ok({("POST", "/api/games/abcdef/trade/round/choose"): chosen})
    )
    data = call("choose_trade", decline=True)
    assert data["trades"] == [{"a": 2, "b": 0, "a_gave": {"Brick": 1}, "a_got": {"Wheat": 1}}]
    assert data["trade_round"] is None


# --- Unlabeled resource indices on legal_actions (`_translate_action`) --------


def test_translate_action_labels_monopoly_and_discard_by_resource_name():
    assert mcp._translate_action({"type": "PLAY_MONOPOLY", "a": 2, "b": 0})["resource"] == "Sheep"
    assert mcp._translate_action({"type": "DISCARD", "a": 4, "b": 0})["resource"] == "Ore"


def test_translate_action_labels_bank_trade_give_and_want():
    action = mcp._translate_action({"type": "BANK_TRADE", "a": 0, "b": 3})
    assert action["give"] == "Wood"
    assert action["want"] == "Wheat"


def test_translate_action_labels_year_of_plenty_as_a_resource_pair():
    action = mcp._translate_action({"type": "PLAY_YEAR_OF_PLENTY", "a": 5, "b": 0})
    pair = mcp._YEAR_OF_PLENTY_PAIRS[5]
    assert action["resources"] == [mcp.RESOURCES[r] for r in pair]


def test_translate_action_leaves_other_action_types_untouched():
    action = {"type": "ROLL", "a": 0, "b": 0}
    assert mcp._translate_action(action) == action


def test_translate_action_keeps_the_raw_a_b_for_act_s_own_replay():
    """act() replays `legal_actions[index]` verbatim as the action body
    (`wire_to_action` reads only `type`/`a`/`b`) -- the added `give`/`want`/
    `resource`/`resources` keys must be extra, never a replacement."""
    action = mcp._translate_action({"type": "BANK_TRADE", "a": 1, "b": 2})
    assert action["a"] == 1
    assert action["b"] == 2


def test_translate_view_translates_every_legal_action(monkeypatch, _seat_for_trade_stubs):
    raw = {
        "legal_actions": [{"type": "PLAY_MONOPOLY", "a": 1, "b": 0}],
        "trades": [],
        "pending": [],
        "trade_round": None,
    }
    monkeypatch.setattr(mcp, "_request_ok", _stub_request_ok({("GET", "/api/state"): raw}))
    data = call("state")
    assert data["legal_actions"] == [{"type": "PLAY_MONOPOLY", "a": 1, "b": 0, "resource": "Brick"}]


# --- act()/undo() return the same translated shape state()/get_table() do ----


def test_act_translates_its_own_response(monkeypatch, _seat_for_trade_stubs):
    state = {"legal_actions": [{"type": "PLAY_MONOPOLY", "a": 1, "b": 0}]}
    acted = {
        "legal_actions": [{"type": "BANK_TRADE", "a": 0, "b": 3}],
        "trades": [],
        "pending": [],
        "trade_round": None,
    }
    monkeypatch.setattr(
        mcp,
        "_request_ok",
        _stub_request_ok({("GET", "/api/state"): state, ("POST", "/api/action"): acted}),
    )
    data = call("act", index=0)
    assert data["legal_actions"] == [{"type": "BANK_TRADE", "a": 0, "b": 3, "give": "Wood", "want": "Wheat"}]


def test_undo_translates_its_own_response(monkeypatch, _seat_for_trade_stubs):
    raw = {
        "legal_actions": [{"type": "DISCARD", "a": 2, "b": 0}],
        "trades": [],
        "pending": [],
        "trade_round": None,
    }
    monkeypatch.setattr(mcp, "_request_ok", _stub_request_ok({("POST", "/api/undo"): raw}))
    data = call("undo")
    assert data["legal_actions"] == [{"type": "DISCARD", "a": 2, "b": 0, "resource": "Sheep"}]


# --- Board summary: per-hex/vertex resource and pip-count annotations --------


def test_board_annotates_hexes_and_vertices_with_resource_and_pips(live_server):
    _, base = live_server
    mcp.BASE_URL = base
    call("new_game", model=MODEL, opponents=SOLO)
    data = call("board")

    for hex_ in data["hexes"]:
        assert hex_["resource"] == mcp._TERRAIN_RESOURCE.get(hex_["terrain"])
        assert hex_["pips"] == mcp._PIPS.get(hex_["token"], 0)
    resourced = [h for h in data["hexes"] if h["resource"] is not None]
    assert resourced, "a real board has at least one resource-paying hex"

    for vertex in data["vertices"]:
        touching = [h for h in resourced if vertex["id"] in h["vertex_ids"]]
        assert vertex["pips"] == sum(h["pips"] for h in touching)
        assert vertex["resources"] == sorted({h["resource"] for h in touching})


# --- Seat name defaults to "mcp" when the LLM doesn't give one ---------------


def test_new_game_defaults_the_seat_name_to_mcp(live_server):
    server, base = live_server
    mcp.BASE_URL = base
    data = call("new_game", model=MODEL, opponents=SOLO)
    assert server.tables.get(data["code"]).seats[data["seat"]].name == "mcp"


def test_new_game_keeps_an_explicit_name(live_server):
    server, base = live_server
    mcp.BASE_URL = base
    data = call("new_game", model=MODEL, opponents=SOLO, name="Ada")
    assert server.tables.get(data["code"]).seats[data["seat"]].name == "Ada"


def test_join_defaults_the_seat_name_to_mcp(live_server):
    server, base = live_server
    mcp.BASE_URL = base
    creator = call("new_game", model=MODEL)
    joiner = call("join", code=creator["code"], model=MODEL)
    assert server.tables.get(creator["code"]).seats[joiner["seat"]].name == "mcp"


# --- leave_game --------------------------------------------------------------


def test_leave_game_locks_the_seat(live_server):
    server, base = live_server
    mcp.BASE_URL = base
    data = call("new_game", model=MODEL, opponents=SOLO)
    seat = data["seat"]
    result = call("leave_game")
    assert result["locked"] == [seat]


# --- Identity: model -> client, and resume_game's reclaim fallback -----------


def test_new_game_records_the_clients_id_and_kind_from_model(live_server):
    """`secret = model.strip().lower()`, `id = sha256(secret)`, kind "mcp" --
    the exact string the LLM gives, normalised the one way `resume_game`'s
    later reclaim can reproduce it."""
    server, base = live_server
    mcp.BASE_URL = base
    data = call("new_game", model=" Claude-Opus-5 ")
    expected_id = hashlib.sha256(b"claude-opus-5").hexdigest()
    client = server.tables.get(data["code"]).seats[data["seat"]].client
    assert client == {"id": expected_id, "kind": "mcp"}


def test_resume_game_reclaims_the_seat_once_its_token_is_dead(live_server, tmp_path, monkeypatch):
    """The saved token stops working (here: blanked directly on the server,
    the same symptom a restart or a second reclaim elsewhere would leave) --
    resume_game() falls back to POST /api/reclaim with the saved model's own
    secret and comes back with the same seat."""
    server, base = live_server
    mcp.BASE_URL = base
    monkeypatch.setattr(mcp, "_SESSION_FILE", tmp_path / "session.json")

    data = call("new_game", model=MODEL, opponents=SOLO)
    seat = data["seat"]
    code = data["code"]
    dead_token = mcp._token

    server.tables.get(code).seats[seat].token = None
    monkeypatch.setattr(mcp, "_token", None)
    monkeypatch.setattr(mcp, "_code", None)
    monkeypatch.setattr(mcp, "_model", None)

    result = call("resume_game")
    assert result["seat"] == seat
    assert mcp._token is not None
    assert mcp._token != dead_token
