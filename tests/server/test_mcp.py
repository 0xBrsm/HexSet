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


def test_new_game_always_installs_a_pending_gate_for_the_llm_seat(live_server):
    """Human and LLM seats are direct gates, unconditionally
    (`agents/reference/trading-final.md`, item 5): there is no `confirm`
    flag left anywhere on the wire, so an LLM's own seat lands in
    `confirm_seats`/`PendingGate` the same as a human's at the web page,
    with no argument required to ask for it."""
    server, base = live_server
    mcp.BASE_URL = base
    data = call("new_game", opponents=SOLO)
    seat = data["seat"]
    table = server.tables.get(data["code"])
    assert seat in table.session.confirm_seats
    from hexset.server.webplay import PendingGate

    assert isinstance(table.session.game.gates[seat], PendingGate)
