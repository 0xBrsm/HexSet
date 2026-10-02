"""The HTTP transport in front of `hexset.server.api`, and the CLI that starts it.

Standard library only. Nothing about a game lives here: this module reads a
request, hands it to `Tables.handle`, and writes the answer back, an
`ApiError` carrying its own status. The frontend is one static HTML file.

## MCP, over HTTP

`POST /mcp` is the MCP Streamable HTTP transport (spec:
modelcontextprotocol.io/specification/2025-06-18/basic/transports) for the
tools `mcptools.py` defines. `initialize` mints an `Mcp-Session-Id` and keeps
the seat (`mcptools.Session`) it stands up in memory for as long as that id
lives; every later request must carry the header back, a missing one being a
400 and an unknown one a 404 (on which a client calls `initialize` again for a
fresh, unseated session). `DELETE /mcp` drops a session early; `GET /mcp` is 405, the server never pushing anything outside one `tools/call`'s own
response. Every `tools/call` is answered as `text/event-stream`
(`_mcp_stream_tool`), since acting tools block until the seat's next move and
may sit for minutes; `initialize`/`tools/list`/`ping` are single JSON-RPC
responses.

`Origin` is checked as the spec's security section asks: present and naming
anything but 127.0.0.1/localhost/::1/this server's own `--host` is a 403;
absent (every non-browser client) is allowed.

## Codes in the URL, tokens in the header

The address is the game. `GET /` is the page, which deals a game
(`POST /api/games`) and moves to its address; `GET /<code>` is that game —
the same HTML, which reads the code out of its own URL and either claims an
open seat or renders read-only as an observer. Both paths just return the
file, so a code that does not exist (or a full game) says so in the page
rather than as a raw 404.

Identity is the token `api.py` mints, sent back on `X-HexSet-Token` and kept
in the browser's localStorage — not a cookie, because one browser may hold
seats at more than one game.

Run it with `python -m hexset.server.web`. Opponents come from
`api.model_options()`: every preset the `--runtime` modules register, plus one
entry per `*.onnx` file in `HEXSET_UI_MODELS_DIR` (default:
`<repo root>/models`), picked up without a restart.
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
from hexset.arena import RUNTIME_HELP, load_runtime

from . import _journal as journal, mcptools
from .api import (
    CODE_ALPHABET,
    CODE_LENGTH,
    ApiError,
    Config,
    Tables,
    model_options,
)
from .constants import TOKEN_HEADER

__all__ = [
    "DEFAULT_MCP_PROTOCOL_VERSION",
    "MCP_PROTOCOL_VERSIONS",
    "STATIC_DIR",
    "Handler",
    "HexSetServer",
    "main",
]


STATIC_DIR = Path(__file__).resolve().parent / "static"
INDEX_HTML = STATIC_DIR / "index.html"

# MCP protocol versions understood, and what `initialize` answers with when a
# client names something else.
MCP_PROTOCOL_VERSIONS = {"2024-11-05", "2025-03-26", "2025-06-18"}
DEFAULT_MCP_PROTOCOL_VERSION = "2025-06-18"


def is_code(path: str) -> bool:
    """Whether a URL path is a table's own code: `CODE_LENGTH` characters,
    every one in `CODE_ALPHABET`."""
    code = path.lstrip("/")
    return len(code) == CODE_LENGTH and all(c in CODE_ALPHABET for c in code.lower())


def looks_like_a_code_attempt(path: str) -> bool:
    """A path the length of a code, even using characters `CODE_ALPHABET`
    excludes as confusable. Such a path gets the page, which reports the bad
    code, rather than a bare 404."""
    return len(path.lstrip("/")) == CODE_LENGTH


class HexSetServer(ThreadingHTTPServer):
    """The served table over HTTP: `tables` behind `Handler`, one thread per
    request, with the live MCP sessions."""

    daemon_threads = True

    def __init__(self, address: tuple[str, int], tables: Tables) -> None:
        super().__init__(address, Handler)
        self.tables = tables
        # This server's configured host, for Origin validation.
        self.host = address[0]
        # Every live MCP session: Mcp-Session-Id -> mcptools.Session.
        self.mcp_sessions: dict[str, mcptools.Session] = {}
        self.mcp_lock = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    """One HTTP request: the page, `/api/*` through `Tables.handle`, and
    the MCP transport at `/mcp`."""

    server: HexSetServer  # narrows the inherited attribute's type

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
        # Without this a browser reopening a background tab can serve a stale
        # copy of the page.
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
            # The client navigated away from a parked read; nobody left to
            # answer.
            pass
        except Exception as error:
            # Anything the API did not expect: logged in full, but answered as
            # a 500 rather than dropped mid-response as http.server would.
            traceback.print_exc()
            self._json({"error": f"{type(error).__name__}: {error}"}, status=500)

    def do_GET(self) -> None:  # noqa: N802 (http.server's naming convention)
        if self.path == "/mcp":
            # No standing stream to open; the spec allows a GET to 405.
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
        # No route answers PUT; still wired so an unrecognised one gets
        # `Tables.handle`'s 404 rather than a transport-level error.
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
        """The spec's Origin check: absent is fine, present only if it names
        this machine."""
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
            # An id-less message is a notification: the spec's 202, no reply.
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
            self._mcp_stream_tool(request_id, session, name, arguments)
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

    def _mcp_stream_tool(self, request_id, session: mcptools.Session, name: str, arguments: dict) -> None:
        """Answer a tool call as `text/event-stream`: a `: keepalive` comment
        per wait tick, then one `event: message` carrying the JSON-RPC response
        (a `ToolError` is the same message with `isError`), then close."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            for item in mcptools.call_tool_events(self.server.tables, session, name, arguments):
                if item is mcptools.KEEPALIVE:
                    self.wfile.write(b": keepalive\n\n")
                else:
                    text = item if isinstance(item, str) else json.dumps(item, separators=(",", ":"))
                    payload = {"content": [{"type": "text", "text": text}], "isError": False}
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
            # The client gave up waiting and closed its side.
            pass


def main(argv: list[str] | None = None) -> None:
    """`python -m hexset.server.web`: serve the board and the API over HTTP
    until interrupted."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", action="append", default=[], help=RUNTIME_HELP)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--seed", type=int, default=None, help="Board/RNG seed.")
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
    load_runtime(*args.runtime)

    config = Config(
        device=args.device,
        games_dir=args.games_dir,
        seed=args.seed,
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
