"""An MCP server so an LLM can take a seat at a HexSet table, over stdio.

This is a thin client of the HTTP API `api.py` defines and `web.py` serves,
not a second game engine binding: every tool below is a `urllib` call to a
running `python -m hexset_ui.web` (see `HEXSET_UI_BASE_URL`). That server can
be anywhere — this is how an LLM joins a table on a machine that actually has
ONNX Runtime while running somewhere that does not.

Identity is the seat token the API mints (see `api.py`), held in this process
for its lifetime. One MCP connection is one seat at one table: `new_table`
deals a fresh one, `join` takes an empty seat at somebody else's by its code,
and either way the token that comes back is what every later tool acts with.

Standard library only, deliberately: the official `mcp` SDK pulls in
`pydantic` (a compiled, Rust-built dependency `onnxruntime` and `numpy` don't
ask for anywhere else in this project), and the stdio wire format it would
save writing here is a handful of JSON-RPC 2.0 methods — `initialize`,
`tools/list`, `tools/call` — small enough to hand-roll directly against the
MCP spec instead, matching the same "standard library only" choice
`web.py`'s own docstring already made for the HTTP side.

Run it with (from `src/`, alongside an already-running web)::

    python -m hexset_ui.mcp

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
SERVER_INFO = {"name": "hexset-ui", "version": "0.1.0"}

# The seat this connection is playing, set by new_table/join and sent on every
# request after. A module global for the same reason the cookie jar it
# replaced was one: the process is the client, and there is exactly one of it.
_token: str | None = None

# advance_one_seat, looped, is what settles a bot cascade after a human
# action; a real game never has more seats left to settle than this, so a loop
# still running past it means something is wedged rather than merely a long
# turn, and continuing to spin at that point would starve the LLM of a
# response with nothing to show for it.
_MAX_CASCADE_STEPS = 64


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
            "— is `python -m hexset_ui.web` running?"
        ) from error


def _request_ok(method: str, path: str, body: dict | None = None) -> dict:
    result = _request(method, path, body)
    if "error" in result:
        raise ToolError(result["error"])
    return result


def _seated() -> None:
    if _token is None:
        raise ToolError("not at a table yet — call new_table() or join(code) first")


def _settle(state: dict) -> dict:
    """Runs `state` forward through the bot seats on move.

    Stops as soon as a person is on move, which is not the same as "our turn"
    now that a table can seat several: another human's turn is nobody's to
    play through, so the LLM is handed the state and polls `state()` until it
    comes back around. The browser drives this same cascade one response at a
    time for UI pacing an LLM has no use for.

    Read off `to_move` and not `current_player`, which are the same seat only
    most of the time: a bot's trade offer and a seven's discards both put
    somebody else on move in the middle of that bot's turn, and asking the
    wrong one there means advancing a seat that is waiting on a person, over
    and over, until this loop gives up.
    """
    for _ in range(_MAX_CASCADE_STEPS):
        if state.get("game_over") or state.get("to_move") in (state.get("human_seats") or []):
            return state
        # advance_blocked (unlike awaiting_confirm, which is only ever true
        # for the one seat actually holding it) is set whenever *any* seat's
        # pending confirm has the cascade gate closed — including one that
        # isn't ours to clear, which nothing here can do anything about.
        if state.get("advance_blocked") and not state.get("awaiting_confirm"):
            return state
        path = "/api/confirm" if state.get("awaiting_confirm") else "/api/advance"
        state = _request_ok("POST", path)
    raise ToolError("bot cascade did not settle — is a bot stuck?")


def _seat(result: dict) -> dict:
    """Records the token a join or a deal handed back, and hides it again.

    The LLM never needs to see it — it is sent on its behalf by `_request` —
    and a token in the transcript is a token in the context window of whatever
    reads that transcript next.
    """
    global _token
    _token = result.pop("token")
    return result


def _models() -> dict:
    return _request_ok("GET", "/api/models")


def _new_table(opponents: list[str] | None = None, open_seats: int = 0, name: str | None = None) -> dict:
    body: dict = {"open_seats": open_seats}
    if opponents:
        body["bots"] = opponents
    if name:
        body["name"] = str(name).strip()[:40]
    return _seat(_request_ok("POST", "/api/tables", body))


def _join(code: str, name: str | None = None) -> dict:
    if not isinstance(code, str) or not code.strip():
        raise ToolError("code must be a table's six-character join code")
    body: dict = {"code": code.strip().upper()}
    if name:
        body["name"] = str(name).strip()[:40]
    return _seat(_request_ok("POST", "/api/join", body))


def _start() -> dict:
    _seated()
    return _settle(_request_ok("POST", "/api/start"))


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
    return _settle(_request_ok("POST", "/api/action", {"action": options[index]}))


def _undo() -> dict:
    _seated()
    return _request_ok("POST", "/api/undo")


# name -> (handler, description, JSON Schema for `arguments`)
_TOOLS: dict[str, tuple] = {
    "models": (
        _models,
        "List the opponent names new_table's `opponents` argument accepts.",
        {"type": "object", "properties": {}},
    ),
    "new_table": (
        _new_table,
        "Deal a new table with you in the first seat, returning its join code. "
        "Nothing is played until start() — leave open seats and share the code "
        "if other people are joining.",
        {
            "type": "object",
            "properties": {
                "opponents": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Names from models(), one per bot seat. Omit for the server's "
                        "own default lineup, which shrinks to fit `open_seats`."
                    ),
                },
                "open_seats": {
                    "type": "integer",
                    "description": (
                        "How many seats to leave empty for other people to join by "
                        "code. A seat nobody takes is not dealt in at all."
                    ),
                },
                "name": {"type": "string", "description": "Your display name, up to 40 characters."},
            },
        },
    ),
    "join": (
        _join,
        "Take an empty seat at an existing table by its six-character code. Fails "
        "if the table has no empty seat or has already started — a game in "
        "progress has only the seats it was dealt with.",
        {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "The table's six-character join code."},
                "name": {"type": "string", "description": "Your display name, up to 40 characters."},
            },
            "required": ["code"],
        },
    ),
    "start": (
        _start,
        "Deal the cards and begin play, dropping any seat still empty. Anyone at "
        "the table may call it. Returns the state once every bot ahead of a person "
        "in the opening turn order has played.",
        {"type": "object", "properties": {}},
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
        "The full current game state: every seat's public info (and your own hand), "
        "the board's dynamic contents (roads, settlements, cities, the robber), and "
        "`legal_actions` — a 0-indexed list of the actions act() currently accepts, "
        "empty when it is not your turn. Poll this while another person is thinking.",
        {"type": "object", "properties": {}},
    ),
    "act": (
        _act,
        "Play legal_actions[index] from the most recent state() (call state() "
        "first if unsure what's legal right now). Returns the state afterward, "
        "every bot's reply already played out, stopping if a person is next.",
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
    print(f"hexset-ui MCP server: talking to {BASE_URL}", file=sys.stderr)
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
