"""An MCP server so an LLM can take a seat in a HexSet game, over stdio.

This is a thin client of the HTTP API `api.py` defines and `web.py` serves,
not a second game engine binding: every tool below is a `urllib` call to a
running `python -m hexset.server.web` (see `HEXSET_UI_BASE_URL`). That server can
be anywhere — this is how an LLM joins a game on a machine that actually has
ONNX Runtime while running somewhere that does not. A bot checkpoint reaches
the same server the same way — see `botclient.py` — this module and that one
are peers, not one built on the other.

Identity is the seat token the API mints (see `api.py`), held in this process
for its lifetime. One MCP connection is one seat at one game: `new_game`
deals a fresh one, dealt and playable immediately (there is no lobby to
start), `join` takes an open seat at somebody else's by its code, and either
way the token that comes back is what every later tool acts with.

Standard library only, deliberately: the official `mcp` SDK pulls in
`pydantic` (a compiled, Rust-built dependency `onnxruntime` and `numpy` don't
ask for anywhere else in this project), and the stdio wire format it would
save writing here is a handful of JSON-RPC 2.0 methods — `initialize`,
`tools/list`, `tools/call` — small enough to hand-roll directly against the
MCP spec instead, matching the same "standard library only" choice
`web.py`'s own docstring already made for the HTTP side.

Run it with (from `src/`, alongside an already-running web)::

    python -m hexset.server.mcp

stdin/stdout carry the protocol; nothing else may write to stdout, so every
log line here goes to stderr instead.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

from .constants import TOKEN_HEADER

BASE_URL = os.environ.get("HEXSET_UI_BASE_URL", "http://127.0.0.1:8770").rstrip("/")
PROTOCOL_VERSION = "2024-11-05"
SERVER_INFO = {"name": "hexset", "version": "0.1.0"}

# The seat this connection is playing, set by new_game/join and sent on every
# request after. A module global for the same reason the cookie jar it
# replaced was one: the process is the client, and there is exactly one of it.
_token: str | None = None
# The game this connection is seated at, needed to address the trade routes
# (`/api/games/<code>/...`) -- `/api/state`/`/api/action` don't need it, since
# the token alone already names one game, but the trade endpoints are
# addressed by code (`api.py`), so it is remembered here the same way.
_code: str | None = None


class ToolError(Exception):
    """Raised by a tool implementation to report the failure back to the
    LLM as a normal (not protocol-level) tool result — see `_call_tool`."""


def _request(method: str, path: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    request = urllib.request.Request(f"{BASE_URL}{path}", data=data, method=method)
    if data is not None:
        request.add_header("Content-Type", "application/json")
    if _token is not None:
        request.add_header(TOKEN_HEADER, _token)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        # The API's own refusals are still a JSON body (see web.Handler._serve)
        # — read it rather than raising past it, so a 400 ("it is not your turn
        # to act") reaches the LLM as the same message a browser would get.
        return json.loads(error.read().decode("utf-8"))
    except urllib.error.URLError as error:
        raise ToolError(
            f"could not reach the HexSet server at {BASE_URL} ({error.reason}) "
            "— is `python -m hexset.server.web` running?"
        ) from error


def _request_ok(method: str, path: str, body: dict | None = None) -> dict:
    result = _request(method, path, body)
    if "error" in result:
        raise ToolError(result["error"])
    return result


def _seated() -> None:
    if _token is None:
        raise ToolError("not at a game yet — call new_game() or join(code) first")


def _seat(result: dict) -> dict:
    """Records the token a join or a deal handed back, and hides it again.

    The LLM never needs to see it — it is sent on its behalf by `_request` —
    and a token in the transcript is a token in the context window of whatever
    reads that transcript next. Also remembers the game's code, off the same
    response (`table.view` always carries `code`), for the trade routes below.
    """
    global _token, _code
    _token = result.pop("token")
    _code = result.get("code")
    return result


def _models() -> dict:
    return _request_ok("GET", "/api/models")


def _new_game(opponents: list[str] | None = None, name: str | None = None) -> dict:
    body: dict = {}
    if opponents:
        body["bots"] = opponents
    if name:
        body["name"] = str(name).strip()[:40]
    return _seat(_request_ok("POST", "/api/games", body))


def _join(code: str, name: str | None = None) -> dict:
    if not isinstance(code, str) or not code.strip():
        raise ToolError("code must be a game's six-character code")
    body: dict = {"code": code.strip().lower()}
    if name:
        body["name"] = str(name).strip()[:40]
    return _seat(_request_ok("POST", "/api/join", body))


def _board() -> dict:
    _seated()
    return _request_ok("GET", "/api/board")


def _state() -> dict:
    _seated()
    return _request_ok("GET", "/api/state")


def _act(index: int) -> dict:
    state = _state()
    options = state.get("legal_actions") or []
    if not isinstance(index, int) or not (0 <= index < len(options)):
        raise ToolError(
            f"index {index!r} out of range — state()'s legal_actions has "
            f"{len(options)} option(s) right now (0..{len(options) - 1})"
            if options
            else "index out of range — state()'s legal_actions is empty; it is not your turn"
        )
    return _request_ok("POST", "/api/action", {"action": options[index]})


def _undo() -> dict:
    _seated()
    return _request_ok("POST", "/api/undo")


# --- Trading (docs/bot-api.md §3; the human/LLM surface, `agents/reference/
# trading-final.md` item 5) ---------------------------------------------------


def _get_table() -> dict:
    _seated()
    return _request_ok("GET", "/api/state")


def _trade_acceptable() -> dict:
    _seated()
    return _request_ok("GET", f"/api/games/{_code}/trade/acceptable")


def _propose_trade(counterparty: int, give: dict | None = None, receive: dict | None = None) -> dict:
    _seated()
    body = {"counterparty": counterparty, "give": give or {}, "receive": receive or {}}
    return _request_ok("POST", f"/api/games/{_code}/trade", body)


def _confirm_trade(index: int) -> dict:
    _seated()
    return _request_ok("POST", f"/api/games/{_code}/trade/confirm", {"index": index})


def _decline_trade(index: int) -> dict:
    _seated()
    return _request_ok("POST", f"/api/games/{_code}/trade/decline", {"index": index})


# name -> (handler, description, JSON Schema for `arguments`)
_TOOLS: dict[str, tuple] = {
    "models": (
        _models,
        "List the opponent names new_game's `opponents` argument accepts.",
        {"type": "object", "properties": {}},
    ),
    "new_game": (
        _new_game,
        "Deal a new game, playable immediately: you at one random seat, any "
        "named opponents at others, everything else open for other people (or "
        "other bots) to join by the code this returns. There is no separate "
        "start — the board is live from the first response. Your own seat's "
        "gate is always `PendingGate` (see get_table()'s `pending`): nothing a "
        "bot or another seat proposes against you ever clears on its own, "
        "confirm_trade()/decline_trade() answer it.",
        {
            "type": "object",
            "properties": {
                "opponents": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Names from models(), one per bot seat to fill at the deal. "
                        "Omit for no bots at all — every other seat stays open."
                    ),
                },
                "name": {"type": "string", "description": "Your display name, up to 40 characters."},
            },
        },
    ),
    "join": (
        _join,
        "Take a random open seat at an existing game by its six-character code. "
        "Fails if every seat is taken or has locked out (see state()'s `locked`) "
        "— a seat somebody closed outright is retired for the rest of that "
        "game, so join before that happens. Your seat is gated the same way "
        "new_game()'s is — see its description.",
        {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "The game's six-character code."},
                "name": {"type": "string", "description": "Your display name, up to 40 characters."},
            },
            "required": ["code"],
        },
    ),
    "board": (
        _board,
        "The board's fixed layout: hex positions/terrain/numbers and vertex/edge "
        "adjacency. Unlike state(), this never changes once a game is dealt, so it "
        "only needs reading once per game.",
        {"type": "object", "properties": {}},
    ),
    "state": (
        _state,
        "The full current game state: every seat's public info (hand size, and "
        "your own hand; the public resource-count ledger for everyone else — "
        "counting isn't hidden information here, only a steal's identity and "
        "dev-card types are), the board's dynamic contents, and `legal_actions` "
        "— a 0-indexed list of the actions act() currently accepts, empty when "
        "it is not your turn. Poll this while another seat is thinking: nothing "
        "plays a turn on your behalf, bot seats included.",
        {"type": "object", "properties": {}},
    ),
    "act": (
        _act,
        "Play legal_actions[index] from the most recent state() (call state() "
        "first if unsure what's legal right now). Returns the state right after "
        "that one action — ending your own turn is END_TURN, an action like any "
        "other, not something act() infers.",
        {
            "type": "object",
            "properties": {"index": {"type": "integer", "description": "Index into legal_actions."}},
            "required": ["index"],
        },
    ),
    "undo": (
        _undo,
        "Undo your own most recent build or bank trade, if state()'s can_undo is "
        "true. Anything else (another seat's move, a played development card) "
        "cannot be undone.",
        {"type": "object", "properties": {}},
    ),
    "get_table": (
        _get_table,
        "Everything the table has said about trading: this turn's cleared "
        "`trades`, and your own `pending` offers -- every candidate a bot's "
        "trade event or another seat's propose_trade() found against you, "
        "unexecuted until confirm_trade()/decline_trade() answers it (your "
        "seat's gate never clears on its own, see new_game()) — alongside "
        "everything state() already returns, since it's the same call.",
        {"type": "object", "properties": {}},
    ),
    "trade_acceptable": (
        _trade_acceptable,
        "Your own preview of what propose_trade() would be accepted right now: "
        "every coverable bundle a bot counterparty's private gate already "
        "prices above zero, grouped by counterparty and sorted by its gain, "
        "descending, capped at 12 per counterparty. Read-only — computing this "
        "moves nothing. A seat at the table who is not a bot never appears "
        "here; propose_trade() against one instead and read its answer back "
        "through get_table()'s `pending`.",
        {"type": "object", "properties": {}},
    ),
    "propose_trade": (
        _propose_trade,
        "Compose and submit a bundle against `counterparty`: on your own turn "
        "against anyone, or during another seat's turn against that seat only. "
        "Fails (with a reason) unless the counterparty's own private gate "
        "prices the exchange above zero — your own gate is never consulted, "
        "since proposing this is your consent. trade_acceptable() previews "
        "what a bot counterparty would take; a person or another LLM answers "
        "asynchronously instead, through their own get_table()'s `pending`.",
        {
            "type": "object",
            "properties": {
                "counterparty": {"type": "integer", "description": "The seat to trade with."},
                "give": {
                    "type": "object",
                    "description": "Resource name -> count you give, e.g. {\"Wood\": 2}.",
                },
                "receive": {
                    "type": "object",
                    "description": "Resource name -> count you receive.",
                },
            },
            "required": ["counterparty"],
        },
    ),
    "confirm_trade": (
        _confirm_trade,
        "Execute one of your pending offers (get_table()'s `pending`) exactly "
        "as the table found it. May still fail if hands moved since.",
        {
            "type": "object",
            "properties": {"index": {"type": "integer", "description": "Index into your `pending`."}},
            "required": ["index"],
        },
    ),
    "decline_trade": (
        _decline_trade,
        "Drop one of your pending offers without moving any cards.",
        {
            "type": "object",
            "properties": {"index": {"type": "integer", "description": "Index into your `pending`."}},
            "required": ["index"],
        },
    ),
}

def _tool_list() -> list[dict]:
    return [
        {"name": name, "description": description, "inputSchema": schema}
        for name, (_, description, schema) in _TOOLS.items()
    ]


def _call_tool(name: str, arguments: dict) -> dict:
    entry = _TOOLS.get(name)
    if entry is None:
        raise ToolError(f"unknown tool: {name}")
    handler, _, _ = entry
    try:
        result = handler(**arguments)
    except TypeError as error:
        raise ToolError(f"bad arguments for {name}: {error}") from error
    return {"content": [{"type": "text", "text": json.dumps(result)}], "isError": False}


def _dispatch(message: dict) -> dict | None:
    """One JSON-RPC request -> its response, or `None` for a notification
    (an `id`-less message, which the spec says gets no reply at all — the
    only one a compliant client sends unprompted is `notifications/initialized`
    right after `initialize`, and nothing here needs to react to it)."""
    method = message.get("method")
    request_id = message.get("id")
    if request_id is None:
        return None

    if method == "initialize":
        result = {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": SERVER_INFO,
        }
        return {"jsonrpc": "2.0", "id": request_id, "result": result}

    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": request_id, "result": {"tools": _tool_list()}}

    if method == "tools/call":
        params = message.get("params") or {}
        try:
            result = _call_tool(params.get("name"), params.get("arguments") or {})
        except ToolError as error:
            result = {"content": [{"type": "text", "text": str(error)}], "isError": True}
        return {"jsonrpc": "2.0", "id": request_id, "result": result}

    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": -32601, "message": f"method not found: {method}"},
    }


def main() -> None:
    print(f"hexset MCP server: talking to {BASE_URL}", file=sys.stderr)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError as error:
            print(f"bad JSON-RPC line, skipped: {error}", file=sys.stderr)
            continue
        if not isinstance(message, dict):
            print(f"bad JSON-RPC message, skipped: {message!r}", file=sys.stderr)
            continue
        try:
            response = _dispatch(message)
        except Exception as error:  # noqa: BLE001 — one bad request must not kill the process
            print(f"unhandled error dispatching {message.get('method')}: {error}", file=sys.stderr)
            response = {
                "jsonrpc": "2.0",
                "id": message.get("id"),
                "error": {"code": -32603, "message": str(error)},
            }
        if response is not None:
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
