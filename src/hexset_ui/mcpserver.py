"""An MCP server so an LLM can play HexSet as the human seat, over stdio.

This is a thin client of `webserver.py`'s existing HTTP API, not a second
game engine binding: every tool below is a `urllib` call to a running
`python -m hexset_ui.webserver` (see `HEXSET_UI_BASE_URL`), reusing the same
`hexset_id` cookie a browser gets, held here in an `http.cookiejar` for the
life of this process. One MCP connection is one identity, the same "one game
per browser" shape `webserver.py`'s module docstring describes — just with
an LLM holding the cookie instead of a tab.

Standard library only, deliberately: the official `mcp` SDK pulls in
`pydantic` (a compiled, Rust-built dependency `onnxruntime` and `numpy` don't
ask for anywhere else in this project), and the stdio wire format it would
save writing here is a handful of JSON-RPC 2.0 methods — `initialize`,
`tools/list`, `tools/call` — small enough to hand-roll directly against the
MCP spec instead, matching the same "standard library only" choice
`webserver.py`'s own docstring already made for the HTTP side.

Run it with (from `src/`, alongside an already-running webserver)::

    python -m hexset_ui.mcpserver

stdin/stdout carry the protocol; nothing else may write to stdout, so every
log line here goes to stderr instead.
"""

from __future__ import annotations

import http.cookiejar
import json
import os
import sys
import urllib.error
import urllib.request

BASE_URL = os.environ.get("HEXSET_UI_BASE_URL", "http://127.0.0.1:8770").rstrip("/")
PROTOCOL_VERSION = "2024-11-05"
SERVER_INFO = {"name": "hexset-ui", "version": "0.1.0"}

# One cookie jar for this whole process — see the module docstring on why
# that's the right lifetime (one MCP connection is one identity).
_opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))

# advance_one_seat, looped, is what settles a bot cascade after a human
# action (see webserver._handle_advance); a real game never has more seats
# left to settle than this, so a loop still running past it means something
# is wedged rather than merely a long turn, and continuing to spin at that
# point would starve the LLM of a response with nothing to show for it.
_MAX_CASCADE_STEPS = 64


class ToolError(Exception):
    """Raised by a tool implementation to report the failure back to the
    LLM as a normal (not protocol-level) tool result — see `_call_tool`."""


def _request(method: str, path: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    request = urllib.request.Request(f"{BASE_URL}{path}", data=data, method=method)
    if data is not None:
        request.add_header("Content-Type", "application/json")
    try:
        with _opener.open(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        # webserver's own error responses are still a JSON body (see
        # Handler._json's status= callers) — read it rather than raising past
        # it, so a 400 ("it is not your turn to act") reaches the LLM as the
        # same message a browser's fetch() would have gotten.
        return json.loads(error.read().decode("utf-8"))
    except urllib.error.URLError as error:
        raise ToolError(
            f"could not reach the HexSet server at {BASE_URL} ({error.reason}) "
            "— is `python -m hexset_ui.webserver` running?"
        ) from error


def _request_ok(method: str, path: str, body: dict | None = None) -> dict:
    result = _request(method, path, body)
    if "error" in result:
        raise ToolError(result["error"])
    return result


def _settle(state: dict) -> dict:
    """Runs `state` forward through however many seats are on move that
    aren't the human's, the same cascade a browser's own polling settle()
    loop (see webserver._handle_advance) drives one response at a time for
    UI pacing an LLM has no use for — this collapses it into the one
    `act`/`new_game` call the LLM already made.
    """
    for _ in range(_MAX_CASCADE_STEPS):
        if state.get("game_over") or state.get("to_move") == state.get("human_seat"):
            return state
        path = "/api/confirm" if state.get("awaiting_confirm") else "/api/advance"
        state = _request_ok("POST", path)
    raise ToolError("bot cascade did not settle — is a bot stuck?")


def _register(name: str) -> dict:
    if not isinstance(name, str) or not name.strip():
        raise ToolError("name must be a non-empty string")
    return _request_ok("POST", "/api/register", {"name": name.strip()[:40]})


def _models() -> dict:
    return _request_ok("GET", "/api/models")


def _new_game(opponents: list[str] | None = None) -> dict:
    body = {"bot_models": opponents} if opponents else {}
    return _settle(_request_ok("POST", "/api/new", body))


def _board() -> dict:
    return _request_ok("GET", "/api/board")


def _state() -> dict:
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
    return _settle(_request_ok("POST", "/api/action", options[index]))


def _undo() -> dict:
    return _request_ok("POST", "/api/undo")


# name -> (handler, description, JSON Schema for `arguments`)
_TOOLS: dict[str, tuple] = {
    "register": (
        _register,
        "Register a display name for the human seat, before or during a game. "
        "Optional — the game plays the same without it.",
        {
            "type": "object",
            "properties": {"name": {"type": "string", "description": "Up to 40 characters."}},
            "required": ["name"],
        },
    ),
    "models": (
        _models,
        "List the opponent names new_game's `opponents` argument accepts.",
        {"type": "object", "properties": {}},
    ),
    "new_game": (
        _new_game,
        "Deal a fresh game, ending whatever game is already in progress for this "
        "connection. Returns the settled state() once every bot ahead of the human "
        "seat in the opening turn order has played.",
        {
            "type": "object",
            "properties": {
                "opponents": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Exactly 3 names from models(), one per bot seat. Omit for the "
                        "server's own default lineup."
                    ),
                }
            },
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
        "The full current game state: every seat's public info (and the human "
        "seat's own hand), the board's dynamic contents (roads, settlements, "
        "cities, the robber), and `legal_actions` — a 0-indexed list of the "
        "actions act() currently accepts, empty when it is not the human's turn.",
        {"type": "object", "properties": {}},
    ),
    "act": (
        _act,
        "Play legal_actions[index] from the most recent state() (call state() "
        "first if unsure what's legal right now). Returns the settled state() "
        "afterward — every bot's reply already played out, same as new_game().",
        {
            "type": "object",
            "properties": {"index": {"type": "integer", "description": "Index into legal_actions."}},
            "required": ["index"],
        },
    ),
    "undo": (
        _undo,
        "Undo the human seat's own most recent build or bank trade, if state()'s "
        "can_undo is true. Anything else (a bot's move, a played development card) "
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
