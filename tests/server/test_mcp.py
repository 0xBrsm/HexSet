"""Round-trip tests for `hexset.server.mcp` — the LLM-facing tool layer over
the same HTTP API `test_api.py` exercises directly. Each tool is a thin
`urllib` call (see `mcp.py`'s module docstring), so these tests run a real
`HexSetServer` and drive the tool functions through `_call_tool`/`_dispatch`
the way an actual MCP client would, checking that the JSON that comes back
names the right seat, game and trade.
"""

from __future__ import annotations

import json
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


def test_set_valuation_and_get_table_round_trip(live_server):
    _, base = live_server
    mcp.BASE_URL = base
    data = call("new_game", opponents=SOLO)
    seat = data["seat"]
    vector = [1.0, 0.0, 0.0, 0.0, -1.0]

    published = call("set_valuation", vector=vector)
    assert published["valuations"][seat] == vector

    table = call("get_table")
    assert table["valuations"][seat] == vector
    assert "pending" in table and "trades" in table


def test_propose_trade_confirm_and_decline_round_trip(live_server):
    from hexset.board.terrain import Resource

    _, base = live_server
    mcp.BASE_URL = base
    data = call("new_game", opponents=SOLO, confirm=False)
    seat = data["seat"]
    # Reach the live table directly to stage hands/vectors the way
    # test_api.py does -- the tools only ever compose the wire request, they
    # don't manufacture game state.
    server, _ = live_server
    table = server.tables.get(data["code"])
    other = next(s for s in range(table.session.game.num_players) if s != seat)
    from hexset.game import Phase

    game = table.session.game
    game.phase = Phase.MAIN
    game.current_player = seat
    state = game.state(seat, hidden=False)
    for hand in state.hands:
        hand[:] = [0, 0, 0, 0, 0]
    state.hands[seat][Resource.WOOD] = 1
    state.hands[other][Resource.ORE] = 1
    table.session.publish(other, [1.0, 0, 0, 0, -1.0])  # wants wood, gives ore

    view = call("propose_trade", counterparty=other, give={"Wood": 1}, receive={"Ore": 1})
    assert view["trades"][-1] == {
        "a": seat, "b": other, "gave": [1, 0, 0, 0, 0], "got": [0, 0, 0, 0, 1]
    }


def test_confirm_mode_seat_reviews_pending_offers(live_server):
    from hexset.board.terrain import Resource
    from hexset.game import Phase
    from hexset.trading import trade_event

    server, base = live_server
    mcp.BASE_URL = base
    data = call("new_game", opponents=SOLO, confirm=True)
    seat = data["seat"]
    table = server.tables.get(data["code"])
    bot = next(s for s in range(table.session.game.num_players) if s != seat)

    game = table.session.game
    game.phase = Phase.MAIN
    game.current_player = bot
    state = game.state(bot, hidden=False)
    for hand in state.hands:
        hand[:] = [0, 0, 0, 0, 0]
    state.hands[bot][Resource.ORE] = 1
    state.hands[seat][Resource.WOOD] = 1

    call("set_valuation", vector=[-1.0, 0, 0, 0, 1.0])  # gives wood, wants ore
    table.session.publish(bot, [1.0, 0, 0, 0, -1.0])  # wants wood, gives ore
    trade_event(
        game,
        lambda s, view, received, other: game.gates[s].accepts(view, received, other),
    )

    table_view = call("get_table")
    assert table_view["pending"] == [
        {"counterparty": bot, "gave": [1, 0, 0, 0, 0], "got": [0, 0, 0, 0, 1]}
    ]

    confirmed = call("confirm_trade", index=0)
    assert confirmed["pending"] == []
    assert state.hands[seat][Resource.ORE] == 1


def test_decline_trade_drops_the_offer(live_server):
    server, base = live_server
    mcp.BASE_URL = base
    data = call("new_game", opponents=SOLO, confirm=True)
    seat = data["seat"]
    table = server.tables.get(data["code"])
    bot = next(s for s in range(table.session.game.num_players) if s != seat)
    from hexset.trading import Trade

    table.session.game.pending.append(Trade(seat, bot, (-1, 0, 0, 0, 1)))

    declined = call("decline_trade", index=0)
    assert declined["pending"] == []
