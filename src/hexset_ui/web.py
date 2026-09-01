"""The HTTP transport in front of `hexset_ui.api`, and the CLI that starts it.

Standard library only: `http.server` for the transport, `json` for the wire
format. The frontend is one static HTML file (`static/index.html`) with inline
SVG and vanilla JS, served as-is.

Nothing about a game lives here. Games, seats, codes, tokens and every rule
about who may do what are `api.py`'s, and this module does three things around
them: read a request, hand it to `Tables.handle`, and write the answer back.
An `ApiError` carries its own status, so even the error mapping is a one-liner.

## Codes in the URL, tokens in the header

`GET /` is the front page, where a game is dealt — immediately playable, no
lobby to wait through. `GET /<CODE>` is that game: the same HTML, which reads
the code out of its own URL and either claims an open seat or, if there isn't
one, renders read-only as an observer. Both are just the file — the server
does not resolve the code, because a code that does not exist (or a game that
is full) should say so in the page rather than as a raw 404.

Identity is the token `api.py` mints, sent back on `X-HexSet-Token` and kept in
the browser's localStorage. It replaced a cookie, which could not survive the
premise that one browser might hold seats at more than one game.

Run it with (from `src/`)::

    python -m hexset_ui.web

then open the printed URL. Opponents come from `api.model_options()`: `search2`
(handcrafted, no checkpoint needed) plus one entry per `*.onnx` file found in
`HEXSET_UI_MODELS_DIR` (default: `<repo root>/models`) — drop a file in, it
shows up in the picker, no restart, no code change. Pass `--checkpoint <name>`
to seat copies of one opponent instead of the per-seat default lineup.

How a checkpoint plays — a single forward pass or a search over its own
priors, and with what budget — is declared in the file's own metadata and read
by `hexset_ui.onnxbot`. Nothing here knows the difference.
"""

from __future__ import annotations

import argparse
import json
import os
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import journal
from .api import (
    CODE_ALPHABET,
    CODE_LENGTH,
    MAX_SEATS,
    SEAT_GRACE_SECONDS,
    ApiError,
    Config,
    Tables,
    model_options,
)
from .constants import TOKEN_HEADER

# The per-seat setup-lock grace window's env override — see `api.py`'s
# `SEAT_GRACE_SECONDS` for what it gates and `Config.seat_grace` for the
# default this falls back to absent either one.
ENV_SEAT_GRACE = "HEXSET_UI_SEAT_GRACE"

STATIC_DIR = Path(__file__).resolve().parent / "static"
INDEX_HTML = STATIC_DIR / "index.html"


def is_code(path: str) -> bool:
    """Whether a URL path is a table's own code — six characters, every one
    of them in `CODE_ALPHABET` — as opposed to a typo or a missing asset.
    """
    code = path.lstrip("/")
    return len(code) == CODE_LENGTH and all(c in CODE_ALPHABET for c in code.upper())


def looks_like_a_code_attempt(path: str) -> bool:
    """Six characters — the length of a real code — even one using a
    character `CODE_ALPHABET` deliberately excludes as too easily confused
    with another (0/O, 1/I/L). `is_code` above still decides what actually
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
        except Exception as error:
            # Anything the API did not expect — a checkpoint that will not
            # load, a bug. Left in the log in full, but answered rather than
            # dropped: http.server's default is to close the connection
            # mid-response, which reaches the browser as a network failure and
            # tells whoever is playing nothing at all.
            traceback.print_exc()
            self._json({"error": f"{type(error).__name__}: {error}"}, status=500)

    def do_GET(self) -> None:  # noqa: N802 (http.server's naming convention)
        if self.path in ("/", "/index.html") or is_code(self.path):
            self._file(INDEX_HTML, "text/html; charset=utf-8")
        elif self.path.startswith("/api/"):
            self._serve("GET", {})
        elif looks_like_a_code_attempt(self.path):
            self._file(INDEX_HTML, "text/html; charset=utf-8")
        else:
            self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802
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
        self._serve("POST", payload)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "An opponent (see api.model_options() — 'search2' or a .onnx name), "
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
        "--max-offers",
        type=int,
        default=1,
        help=(
            "Trade-offer budget every bot seat is held to, overriding whatever "
            "each checkpoint trained under (default: 1)."
        ),
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
    parser.add_argument(
        "--seat-grace",
        type=float,
        default=None,
        help=(
            "Seconds an empty seat the setup snake is waiting on stays open "
            f"before it locks out for good (default: ${ENV_SEAT_GRACE}, itself "
            f"defaulting to {SEAT_GRACE_SECONDS:g}). A game can override this "
            "per creation too (POST /api/games's `seat_grace`); 0 deals a "
            "solo game immediately, useful for trying the board out alone."
        ),
    )
    args = parser.parse_args(argv)

    if args.checkpoint and args.checkpoint not in model_options():
        parser.error(f"unknown checkpoint: {args.checkpoint}")

    if args.seat_grace is not None:
        seat_grace = args.seat_grace
    elif os.environ.get(ENV_SEAT_GRACE):
        seat_grace = float(os.environ[ENV_SEAT_GRACE])
    else:
        seat_grace = SEAT_GRACE_SECONDS

    config = Config(
        device=args.device,
        max_offers=args.max_offers,
        games_dir=args.games_dir,
        seed=args.seed,
        default_bots=[args.checkpoint] * (MAX_SEATS - 1) if args.checkpoint else None,
        seat_grace=seat_grace,
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
