"""Round-trip tests for `hexset.server.mcp` — the LLM-facing tool layer over
the same HTTP API `test_api.py` exercises directly. Each tool is a thin
`urllib` call (see `mcp.py`'s module docstring), so these tests run a real
`HexSetServer` and drive the tool functions through `_call_tool`/`_dispatch`
the way an actual MCP client would, checking that the JSON that comes back
names the right seat, game and trade.
"""

from __future__ import annotations

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
    `_token`/`_code` (see `mcp.py`'s docstring on why they're globals — one
    process, one seat) must not leak across tests."""
    monkeypatch.setattr(mcp, "_token", None)
    monkeypatch.setattr(mcp, "_code", None)


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
    data = call("new_game", opponents=SOLO, name="Ada")
    assert "token" not in data  # popped by _seat -- never reaches the LLM
    assert data["code"] == mcp._code
    assert mcp._token


def test_new_game_with_no_confirm_flag_still_auto_accepts(live_server):
    """The human-default flip (`POST /api/games`/`/api/join` now default to
    confirm mode when a request omits `confirm`, so nothing auto-clears
    against a human) must not reach MCP: `_new_game`/`_join` send `confirm`
    explicitly on every call, so an LLM seat that asks for nothing keeps
    auto-accepting on its own published vector, per PI ratification
    decision 3."""
    from hexset.board.terrain import Resource
    from hexset.game import Phase, roll_dice

    server, base = live_server
    mcp.BASE_URL = base
    data = call("new_game", opponents=SOLO)  # no `confirm` kwarg at all
    seat = data["seat"]
    table = server.tables.get(data["code"])
    assert seat not in table.session.confirm_seats

    game = table.session.game
    other = next(s for s in range(game.num_players) if s != seat)
    game.phase = Phase.ROLL
    game.current_player = seat
    state = game.state(seat, hidden=False)
    for hand in state.hands:
        hand[:] = [0, 0, 0, 0, 0]
    state.hands[seat][Resource.WOOD] = 1
    state.hands[other][Resource.ORE] = 1

    wants_ore = [0.0, 0.0, 0.0, 0.0, 0.0]
    wants_ore[Resource.ORE] = 1.0
    wants_ore[Resource.WOOD] = -1.0
    call("set_valuation", vector=wants_ore)
    table.session.publish(other, [-v for v in wants_ore])

    roll_dice(game, 8)
    assert game.phase is Phase.MAIN

    view = call("get_table")
    assert view["trades"], "an LLM seat's own default must still auto-accept"
    assert state.hands[seat][Resource.ORE] == 1
    assert state.hands[other][Resource.WOOD] == 1
