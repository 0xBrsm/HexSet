"""The HTTP transport in front of `hexset.server.api`, and the CLI that starts it.

Standard library only: `http.server` for the transport, `json` for the wire
format. The frontend is one static HTML file (`static/index.html`) with inline
SVG and vanilla JS, served as-is.

Nothing about a game lives here. Games, seats, codes, tokens and every rule
about who may do what are `api.py`'s, and this module does three things around
them: read a request, hand it to `Tables.handle`, and write the answer back.
An `ApiError` carries its own status, so even the error mapping is a one-liner.

## MCP is served here too, over HTTP

`POST /mcp` is the MCP Streamable HTTP transport (spec: modelcontextprotocol.io
/specification/2025-06-18/basic/transports) for the tool layer `mcptools.py`
defines -- there is no longer a separate `python -m hexset.server.mcp` stdio
program. `initialize` mints an `Mcp-Session-Id` (`secrets.token_urlsafe`) and
keeps the seat (`mcptools.Session`) it stands up in memory on this server for
as long as that id lives; every later request on that session must carry the
header back: a missing one is a 400, an unknown one a 404 (a client that sees
a 404 just calls `initialize` again — a fresh Mcp-Session-Id, an unseated `Session`,
same as a fresh process used to be). `DELETE /mcp` drops a session early;
`GET /mcp` is 405 -- this server never pushes anything to a client outside of
one `tools/call`'s own response. A `tools/call` for `wait_for_turn` is the one
tool answered as `text/event-stream` rather than one JSON object (see
`_mcp_wait_for_turn_stream` below); everything else is a single JSON-RPC
response, same as `initialize`/`tools/list`.

`Origin` is checked the way the spec's security section asks: present and not
127.0.0.1/localhost/::1/this server's own `--host` is a 403; absent (every
non-browser client) is let through, since DNS rebinding is a browser attack.

## Codes in the URL, tokens in the header

The address is the game, and there is no page in front of it. `GET /` deals
one and moves to its address; `GET /<code>` is that game — the same HTML,
which reads the code out of its own URL and either claims an open seat or, if
there isn't one, renders read-only as an observer. Sharing the URL is the
whole invitation: everyone who opens it lands at the same table, and an open
seat that nobody takes can be given to a bot from the board itself (see
`api.Tables.seat_bot`). Both paths are just the file — the server does not
resolve the code, because a code that does not exist (or a game that is full)
should say so in the page rather than as a raw 404.

Identity is the token `api.py` mints, sent back on `X-HexSet-Token` and kept in
the browser's localStorage. It replaced a cookie, which could not survive the
premise that one browser might hold seats at more than one game.

Run it with (from `src/`)::

    python -m hexset.server.web

then open the printed URL. Opponents come from `api.model_options()`: `heximax`
(handcrafted, no checkpoint needed) plus one entry per `*.onnx` file found in
`HEXSET_UI_MODELS_DIR` (default: `<repo root>/models`) — drop a file in, it
shows up in the picker, no restart, no code change. Pass `--checkpoint <name>`
to seat copies of one opponent instead of the per-seat default lineup.

How a checkpoint plays — a single forward pass or a search over its own
priors, and with what budget — is declared in the file's own metadata and read
by `hexset.clients.onnxbot`. Nothing here knows the difference.
"""

from __future__ import annotations

import argparse
import json
import secrets
import threading
import traceback
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import hexset

from . import journal, mcptools
from .api import (
    CODE_ALPHABET,
    CODE_LENGTH,
    MAX_SEATS,
    ApiError,
    Config,
    Tables,
    model_options,
)
from .constants import TOKEN_HEADER

STATIC_DIR = Path(__file__).resolve().parent / "static"
INDEX_HTML = STATIC_DIR / "index.html"

# Protocol versions this server understands, and what it answers with when a
# client's `initialize` names something else -- the spec's own version
# negotiation (modelcontextprotocol.io/specification/2025-06-18/basic/lifecycle).
MCP_PROTOCOL_VERSIONS = {"2024-11-05", "2025-03-26", "2025-06-18"}
DEFAULT_MCP_PROTOCOL_VERSION = "2025-06-18"


def is_code(path: str) -> bool:
    """Whether a URL path is a table's own code — six characters, every one
    of them in `CODE_ALPHABET` — as opposed to a typo or a missing asset.
    """
    code = path.lstrip("/")
    return len(code) == CODE_LENGTH and all(c in CODE_ALPHABET for c in code.lower())


def looks_like_a_code_attempt(path: str) -> bool:
    """Six characters — the length of a real code — even one using a
    character `CODE_ALPHABET` deliberately excludes as too easily confused
    with another (0/o, 1/i/l). `is_code` above still decides what actually
    opens a table once the page loads and asks the API; this only decides
    that a path this shape belongs on that page rather than getting a bare
    404, the way `/favicon` (the wrong length for a code at all) still does.
    """
    return len(path.lstrip("/")) == CODE_LENGTH


class HexSetServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], tables: Tables) -> None:
        super().__init__(address, Handler)
        self.tables = tables
        # This server's own configured host, for Origin validation -- an
        # Origin naming anything else (besides 127.0.0.1/localhost/::1,
        # always allowed) is a 403 (`web.py`'s module docstring).
        self.host = address[0]
        # Every live MCP session: Mcp-Session-Id -> mcptools.Session. A plain
        # dict behind a lock, the same shape `Tables` uses for its own
        # registry, and for the same reason (a handful of live sessions, not
        # millions) -- see `Handler._mcp_*`.
        self.mcp_sessions: dict[str, mcptools.Session] = {}
        self.mcp_lock = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    server: HexSetServer  # narrows the inherited attribute's type for readability

    def _json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path: Path, content_type: str) -> None:
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        # No caching headers here meant no explicit signal either way, and a
        # phone browser reopening a background tab is exactly the case that
        # falls back to a stale copy rather than refetching — indistinguishable
        # from "the fix didn't work" without this.
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _serve(self, method: str, payload: dict) -> None:
        try:
            self._json(
                self.server.tables.handle(
                    method, self.path, payload, self.headers.get(TOKEN_HEADER)
                )
            )
        except ApiError as error:
            self._json({"error": str(error)}, status=error.status)
        except (BrokenPipeError, ConnectionResetError):
            # A read the page parked on and then navigated away from. There
            # is nobody left to answer, and the traceback socketserver would
            # print for it says nothing about this server.
            pass
        except Exception as error:
            # Anything the API did not expect — a checkpoint that will not
            # load, a bug. Left in the log in full, but answered rather than
            # dropped: http.server's default is to close the connection
            # mid-response, which reaches the browser as a network failure and
            # tells whoever is playing nothing at all.
            traceback.print_exc()
            self._json({"error": f"{type(error).__name__}: {error}"}, status=500)

    def do_GET(self) -> None:  # noqa: N802 (http.server's naming convention)
        if self.path == "/mcp":
            # This server never pushes a message to a client outside of one
            # tools/call's own response (`wait_for_turn`'s SSE) -- there is
            # no standing stream to open here (spec: a GET may 405).
            self.send_error(405)
        elif self.path in ("/", "/index.html") or is_code(self.path):
            self._file(INDEX_HTML, "text/html; charset=utf-8")
        elif self.path.startswith("/api/"):
            self._serve("GET", {})
        elif looks_like_a_code_attempt(self.path):
            self._file(INDEX_HTML, "text/html; charset=utf-8")
        else:
            self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802
        if self.path == "/mcp":
            self._mcp_post()
        else:
            self._with_body("POST")

    def do_DELETE(self) -> None:  # noqa: N802
        if self.path != "/mcp":
            self.send_error(404)
            return
        session_id = self.headers.get("Mcp-Session-Id")
        with self.server.mcp_lock:
            removed = self.server.mcp_sessions.pop(session_id, None) if session_id else None
        if removed is None:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_PUT(self) -> None:  # noqa: N802
        # No route answers PUT -- `PUT /api/games/<code>/valuation` was the
        # only one, and it is gone with the public valuation layer
        # (`agents/reference/trading-final.md`, item 1). The human-trading
        # surface that followed (item 5) added no PUT route back: every new
        # endpoint is `GET`/`POST` (`api.py`'s `_seated`). Left wired anyway
        # -- an unrecognised PUT still deserves `Tables.handle`'s own 404
        # rather than a bare transport-level error.
        self._with_body("PUT")

    def _with_body(self, method: str) -> None:
        if not self.path.startswith("/api/"):
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            payload = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            self._json({"error": "invalid JSON body"}, status=400)
            return
        if not isinstance(payload, dict):
            self._json({"error": "body must be a JSON object"}, status=400)
            return
        self._serve(method, payload)

    # --- MCP: POST /mcp (JSON-RPC), DELETE /mcp above -----------------------

    def _origin_ok(self) -> bool:
        """The spec's Origin check: absent is fine (every non-browser client
        sends none), present is only fine naming this machine -- see the
        module docstring."""
        origin = self.headers.get("Origin")
        if not origin:
            return True
        host = urllib.parse.urlparse(origin).hostname
        return host in {"127.0.0.1", "localhost", "::1", self.server.host}

    def _mcp_error(self, status: int, message: str) -> None:
        self._json({"error": message}, status=status)

    def _mcp_result(self, request_id, result: dict) -> None:
        self._json({"jsonrpc": "2.0", "id": request_id, "result": result})

    def _mcp_post(self) -> None:
        if not self._origin_ok():
            self._mcp_error(403, "Origin not allowed")
            return
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            message = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            self._mcp_error(400, "invalid JSON body")
            return
        if not isinstance(message, dict):
            self._mcp_error(400, "the body must be a single JSON-RPC message")
            return

        method = message.get("method")
        if method == "initialize":
            self._mcp_initialize(message)
            return

        session_id = self.headers.get("Mcp-Session-Id")
        if not session_id:
            self._mcp_error(400, "missing Mcp-Session-Id -- call initialize first")
            return
        with self.server.mcp_lock:
            session = self.server.mcp_sessions.get(session_id)
        if session is None:
            self._mcp_error(404, "unknown Mcp-Session-Id -- call initialize again")
            return

        if "id" not in message:
            # A notification (`notifications/initialized`, or any other
            # id-less message): the spec's 202, no body, no reply.
            self.send_response(202)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return

        request_id = message["id"]
        if method == "ping":
            self._mcp_result(request_id, {})
        elif method == "tools/list":
            self._mcp_result(request_id, {"tools": mcptools.tool_list()})
        elif method == "tools/call":
            params = message.get("params") or {}
            name = params.get("name")
            arguments = params.get("arguments")
            arguments = arguments if isinstance(arguments, dict) else {}
            if name == "wait_for_turn":
                self._mcp_wait_for_turn_stream(request_id, session, arguments)
            else:
                self._mcp_call_tool(request_id, session, name, arguments)
        else:
            self._json({"jsonrpc": "2.0", "id": request_id, "error": {"code": -32601, "message": f"method not found: {method}"}})

    def _mcp_initialize(self, message: dict) -> None:
        params = message.get("params") or {}
        requested = params.get("protocolVersion")
        protocol_version = requested if requested in MCP_PROTOCOL_VERSIONS else DEFAULT_MCP_PROTOCOL_VERSION
        session_id = secrets.token_urlsafe(24)
        with self.server.mcp_lock:
            self.server.mcp_sessions[session_id] = mcptools.Session()
        result = {
            "protocolVersion": protocol_version,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "hexset", "version": hexset.__version__},
        }
        body = json.dumps({"jsonrpc": "2.0", "id": message.get("id"), "result": result}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Mcp-Session-Id", session_id)
        self.end_headers()
        self.wfile.write(body)

    def _mcp_call_tool(self, request_id, session: mcptools.Session, name: str, arguments: dict) -> None:
        try:
            result = mcptools.call_tool(self.server.tables, session, name, arguments)
            payload = {"content": [{"type": "text", "text": json.dumps(result)}], "isError": False}
        except mcptools.ToolError as error:
            payload = {"content": [{"type": "text", "text": str(error)}], "isError": True}
        self._mcp_result(request_id, payload)

    def _mcp_wait_for_turn_stream(self, request_id, session: mcptools.Session, arguments: dict) -> None:
        """The one tool answered as `text/event-stream`: a `: keepalive`
        comment between each 15s wait tick (`mcptools._wait_for_turn_events`),
        then one `event: message` carrying the JSON-RPC response, then the
        connection closes -- this server defaults to HTTP/1.0 (no keep-alive)
        so nothing further is needed to make that happen cleanly."""
        timeout = arguments.get("timeout")
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            for item in mcptools._wait_for_turn_events(self.server.tables, session, timeout=timeout):
                if item is mcptools._KEEPALIVE:
                    self.wfile.write(b": keepalive\n\n")
                else:
                    payload = {"content": [{"type": "text", "text": json.dumps(item)}], "isError": False}
                    response = {"jsonrpc": "2.0", "id": request_id, "result": payload}
                    self.wfile.write(f"event: message\ndata: {json.dumps(response)}\n\n".encode("utf-8"))
                self.wfile.flush()
        except mcptools.ToolError as error:
            response = {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {"content": [{"type": "text", "text": str(error)}], "isError": True},
            }
            try:
                self.wfile.write(f"event: message\ndata: {json.dumps(response)}\n\n".encode("utf-8"))
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
        except (BrokenPipeError, ConnectionResetError):
            # The client gave up waiting and closed its side -- nobody left
            # to write to, and http.server's own traceback for it says
            # nothing about this server (same reasoning as `_serve`).
            pass


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "An opponent (see api.model_options() — 'heximax' or a .onnx name), "
            "seated at every bot seat a new game is dealt with when the "
            "request creating it doesn't name its own lineup. There is no "
            "automatic mixed default any more — omit this and a fresh game "
            "seats only its creator, every other seat open."
        ),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--seed", type=int, default=None, help="Board/RNG seed.")
    parser.add_argument(
        "--no-trade",
        action="store_true",
        help="Switch trading off for every bot seat (max_trades=0).",
    )
    parser.add_argument("--device", default="cpu", help="Inference device (default: cpu).")
    parser.add_argument("--no-browser", action="store_true", help="Do not auto-open a browser tab.")
    parser.add_argument(
        "--games-dir",
        default=None,
        help=(
            "Where to journal every game in full, hidden cards and all "
            f"(default: ${journal.ENV_DIR}, itself defaulting to "
            f"'{journal.DEFAULT_DIR}'). Pass an empty string to journal nothing."
        ),
    )
    args = parser.parse_args(argv)

    if args.checkpoint and args.checkpoint not in model_options():
        parser.error(f"unknown checkpoint: {args.checkpoint}")

    config = Config(
        device=args.device,
        max_trades=0 if args.no_trade else None,
        games_dir=args.games_dir,
        seed=args.seed,
        default_bots=[args.checkpoint] * (MAX_SEATS - 1) if args.checkpoint else None,
    )
    server = HexSetServer((args.host, args.port), Tables(config))

    url = f"http://{args.host}:{args.port}/"
    print(f"HexSet board: {url}  (models={list(model_options())})")
    games_dir = args.games_dir if args.games_dir is not None else journal.configured_dir()
    if games_dir:
        print(f"Every game will be journalled in full under {games_dir}/")
    else:
        print("Games will not be journalled.")
    if not args.no_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
