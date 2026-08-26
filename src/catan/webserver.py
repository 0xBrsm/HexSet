"""A local, dependency-free HTTP server so a human can play base Catan against
a dropped-in ONNX checkpoint.

Standard library only: `http.server` for the transport, `json` for the wire
format. The frontend is one static HTML file (`web/index.html`) with inline SVG
and vanilla JS, served as-is.

`catan.webplay` (the session, the board layout math, the wire format) is
onnxruntime-free and importable on its own; the network policy is imported
here, inside `main`, so anything that only needs the session can be tested
without onnxruntime installed — the same split `catan.onnxbot` already draws.

## One game per browser, not one game for the whole server

Every request carries (or, on its first visit, gets handed) an opaque
`catan_id` cookie — a random token identifying the browser, nothing more, no
accounts. `CatanServer` keys its games off that: `GET /api/state` from one
cookie never sees, and can never affect, another cookie's game. This was
deliberately a cookie and not the request's source IP, which would collide
two people on the same wifi/NAT into one game and lose the session the moment
either of them changed networks mid-play.

Idle games (nobody's polled `/api/state` in a while) are dropped after
`SESSION_TTL_SECONDS`, since nothing else ever will be — there's no login to
log out of and no "leave game" button, just a tab someone closed or forgot.

Run it with (from `src/`)::

    python -m catan.webserver

then open the printed URL. Opponents come from `model_options()`: `search2`
(handcrafted, no checkpoint needed) plus one entry per `*.onnx` file found in
`CATAN_WEB_MODELS_DIR` (default: `<repo root>/models`) — drop a file in, it
shows up in the picker, no restart, no code change. Pass `--checkpoint <spec>`
to seat 3 copies of one entrant instead of the per-seat default lineup.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import secrets
import sys
import threading
import time
import webbrowser
from dataclasses import dataclass
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable

from .actions import Action
from .board.board import random_base_board
from .game import Game, start, to_move
from .webplay import Bot, GameSession, board_layout

STATIC_DIR = Path(__file__).resolve().parent / "web"
INDEX_HTML = STATIC_DIR / "index.html"
REPO_ROOT = Path(__file__).resolve().parents[2]
NUM_PLAYERS = 4

COOKIE_NAME = "catan_id"
# 30 days: long enough that closing and reopening the tab tomorrow still
# finds the same game. Not tied to how long a game is actually kept alive
# server-side (see SESSION_TTL_SECONDS below) — an idle game is dropped long
# before its cookie would expire on its own either way.
COOKIE_MAX_AGE_SECONDS = 30 * 24 * 60 * 60
# How long a game survives with nobody polling it. An active tab calls
# GET /api/state far more often than this; a closed or forgotten one doesn't
# call it at all. 24 hours comfortably outlasts a real, paused-mid-game
# session without holding onto abandoned ones indefinitely.
SESSION_TTL_SECONDS = 24 * 60 * 60

MODELS_DIR = Path(os.environ.get("CATAN_WEB_MODELS_DIR", REPO_ROOT / "models"))


def model_options() -> dict[str, str]:
    """Display name -> entrant spec, for the per-seat model picker.

    `search2` (handcrafted, no checkpoint) is always first; every `*.onnx`
    file under `MODELS_DIR` follows, filename stem as the display name. The
    client only ever sends a name back (see `_resolve_bot_models`); specs
    never cross the wire, so a request can't point a bot at an arbitrary
    file — this function is the one chokepoint that turns a name into a
    loadable path, same as if it were still a static dict.

    Scanned fresh on every call rather than cached: a directory listing of a
    handful of files is sub-millisecond, and the entire point of this
    function existing (over the static dict it replaced) is that dropping a
    file in shows up without a restart.
    """
    options = {"search2": "search2"}
    for path in sorted(MODELS_DIR.glob("*.onnx")):
        options[path.stem] = f"network:{path}"
    return options


@dataclass
class _Entry:
    """One identity's live game: the session itself, its board layout (cached
    alongside it — see `/api/board` — rather than recomputed per request,
    same as when there was only ever one of these), and when it was last
    touched, for `CatanServer._evict_stale` to act on."""

    session: GameSession
    layout: dict
    last_seen: float


class CatanServer(ThreadingHTTPServer):
    """Holds one game session per browser identity (see the module docstring)
    instead of one for the whole process.

    Two lock granularities, not one: `_registry_lock` guards the `sessions`/
    `_locks` dicts themselves (fast — a handful of dict operations), while
    each identity gets its own `threading.Lock` (see `lock_for`) around the
    actual game mutation, which can be slow (a cascade of bot moves, each an
    ONNX forward pass). One shared lock for everything would mean one
    player's turn stalls every *other* player's request for its duration —
    fine when this served one human total, wrong now that the whole point is
    serving several at once.
    """

    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        new_session: Callable[[list[tuple[str, str]] | None], GameSession],
        device: str = "cpu",
        max_offers: int | None = None,
    ) -> None:
        super().__init__(address, Handler)
        self.sessions: dict[str, _Entry] = {}
        self.new_session = new_session
        # Carried so a live seat-bot swap (`/api/bot`) can build a bot the same
        # way `_build_session` did, without threading them through every call.
        self.device = device
        self.max_offers = max_offers
        self._locks: dict[str, threading.Lock] = {}
        self._registry_lock = threading.Lock()

    def lock_for(self, identity: str) -> threading.Lock:
        """The lock to hold around any mutation of `identity`'s game."""
        with self._registry_lock:
            lock = self._locks.get(identity)
            if lock is None:
                lock = self._locks[identity] = threading.Lock()
            return lock

    def entry(self, identity: str, bots: list[tuple[str, str]] | None = None) -> _Entry:
        """`identity`'s game, dealing a fresh one on first contact. Caller is
        responsible for holding `lock_for(identity)` first if this might run
        concurrently with a mutation of that same identity's game — this only
        protects the registry (the dict of *all* identities' entries), not
        the one game it hands back.
        """
        with self._registry_lock:
            now = time.monotonic()
            self._evict_stale(now)
            found = self.sessions.get(identity)
            if found is not None:
                found.last_seen = now
                return found

        # Building a session can be slow (constructing an mcts: searcher, a
        # cold NetworkBot) — done outside _registry_lock so one identity's
        # first-ever request doesn't stall every other identity's dict
        # lookups for however long that takes.
        session = self.new_session(bots)
        built = _Entry(session, board_layout(session.game.state.board), time.monotonic())
        with self._registry_lock:
            # Another request for this same brand-new identity could have
            # built and registered its own entry while the above ran —
            # first one in wins, so anyone already handed that entry (or
            # already mutating its game under lock_for) isn't left holding
            # a reference this call silently discarded.
            found = self.sessions.get(identity)
            if found is not None:
                found.last_seen = time.monotonic()
                return found
            self.sessions[identity] = built
            return built

    def replace(self, identity: str, bots: list[tuple[str, str]] | None) -> _Entry:
        """Unconditionally deals `identity` a new game, replacing whatever it
        had — what `POST /api/new` does. Caller holds `lock_for(identity)`,
        which rules out the same first-one-wins race `entry` guards against:
        nothing else can be creating this identity's entry at the same time.
        """
        session = self.new_session(bots)  # see entry()'s own note on why this is outside the lock
        entry = _Entry(session, board_layout(session.game.state.board), time.monotonic())
        with self._registry_lock:
            self.sessions[identity] = entry
            return entry

    def _evict_stale(self, now: float) -> None:
        """Must be called with `_registry_lock` held. Drops any identity's
        game (and its lock) untouched for longer than `SESSION_TTL_SECONDS`
        — an abandoned tab, not an active one (see module docstring).

        Not guarded against evicting an identity whose own request is what's
        currently running: `entry`, the only caller, is always invoked while
        the caller already holds `lock_for(identity)` for that very
        identity, so "is this identity's lock held" can never distinguish
        "another concurrent request is using it" from "I am, right now,
        resolving it" — the two cases this sweep actually needs to tell
        apart. Evicting the second one is exactly the intended behavior
        anyway (`entry` immediately rebuilds whatever it just deleted), so
        there is nothing to guard there. What's left unguarded is a single
        request that runs longer than `SESSION_TTL_SECONDS` itself — hours —
        which would need to be evicted by some *other* identity's traffic
        arriving mid-request; accepted as a known gap rather than one worth
        the complexity of closing.
        """
        stale = [i for i, e in self.sessions.items() if now - e.last_seen > SESSION_TTL_SECONDS]
        for identity in stale:
            del self.sessions[identity]
            self._locks.pop(identity, None)


class Handler(BaseHTTPRequestHandler):
    server: CatanServer  # narrows the inherited attribute's type for readability

    def _read_identity(self) -> None:
        """Reads this request's `catan_id` cookie, or mints one — cached on
        `self` (a fresh `Handler` per request) so every handler downstream,
        plus `_json`/`_file`, share the one value: the latter need it to know
        whether to actually set the cookie this response, which only a
        brand-new identity's first response must do.
        """
        identity = None
        header = self.headers.get("Cookie")
        if header:
            cookies = SimpleCookie()
            cookies.load(header)
            morsel = cookies.get(COOKIE_NAME)
            if morsel is not None:
                identity = morsel.value
        self._new_identity = identity is None
        self.identity = identity or secrets.token_urlsafe(18)

    def _json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self._set_identity_cookie_if_new()
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
        self._set_identity_cookie_if_new()
        self.end_headers()
        self.wfile.write(body)

    def _set_identity_cookie_if_new(self) -> None:
        if self._new_identity:
            self.send_header(
                "Set-Cookie",
                f"{COOKIE_NAME}={self.identity}; Path=/; Max-Age={COOKIE_MAX_AGE_SECONDS}; SameSite=Lax",
            )

    def do_GET(self) -> None:  # noqa: N802 (http.server's naming convention)
        self._read_identity()
        if self.path in ("/", "/index.html"):
            self._file(INDEX_HTML, "text/html; charset=utf-8")
        elif self.path == "/api/board":
            # Reading entry.layout/session.state_view() stays inside the
            # lock, same as every mutating handler below — otherwise this
            # GET can race a concurrent POST for the same identity (e.g. the
            # client's own periodic /api/state poll landing mid-action) and
            # read a GameSession mid-mutation.
            with self.server.lock_for(self.identity):
                entry = self.server.entry(self.identity)
                self._json(entry.layout)
        elif self.path == "/api/state":
            with self.server.lock_for(self.identity):
                entry = self.server.entry(self.identity)
                self._json(entry.session.state_view())
        elif self.path == "/api/models":
            # Names only, never paths: the picker builds its 3 dropdowns from
            # this list and sends names back, same as it received them.
            self._json({"models": list(model_options())})
        else:
            self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802
        self._read_identity()
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            payload = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            self._json({"error": "invalid JSON body"}, status=400)
            return

        if self.path == "/api/action":
            self._handle_action(payload)
        elif self.path == "/api/new":
            self._handle_new(payload)
        elif self.path == "/api/bot":
            self._handle_swap_bot(payload)
        elif self.path == "/api/undo":
            self._handle_undo(payload)
        else:
            self.send_error(404)

    def _handle_new(self, payload: dict) -> None:
        bot_models = payload.get("bot_models")
        bots = None
        if bot_models is not None:
            try:
                bots = _resolve_bot_models(bot_models)
            except ValueError as exc:
                self._json({"error": str(exc)}, status=400)
                return
        with self.server.lock_for(self.identity):
            entry = self.server.replace(self.identity, bots)
        self._json(entry.session.state_view())

    def _handle_swap_bot(self, payload: dict) -> None:
        """Re-seat one non-human seat's bot mid-game — models can be swapped
        at any point, not just between games. Rebuilding the network from
        `model_options()` (rather than accepting a spec) keeps this the same
        chokepoint `_resolve_bot_models` already is: a request names a bot, it
        never hands one a path.
        """
        try:
            seat = int(payload["seat"])
            name = str(payload["model"])
        except (KeyError, TypeError, ValueError):
            self._json({"error": "expected {seat: int, model: str}"}, status=400)
            return
        with self.server.lock_for(self.identity):
            session = self.server.entry(self.identity).session
            if seat == session.human_seat or seat not in session.bot.bots_by_seat:
                self._json({"error": f"seat {seat} has no bot to swap"}, status=400)
                return
            try:
                spec = model_options()[name]
            except KeyError:
                self._json({"error": f"unknown model: {name}"}, status=400)
                return
            new_bot = _spawn_bot(
                spec, session.game.state.board, random.Random(), self.server.device, self.server.max_offers
            )
            session.bot.bots_by_seat[seat] = new_bot
            session.bot_names[seat] = name
        self._json(session.state_view())

    def _handle_action(self, payload: dict) -> None:
        with self.server.lock_for(self.identity):
            session = self.server.entry(self.identity).session
            try:
                session.apply_human_action(payload)
                session.advance_bots()
            except ValueError as exc:
                self._json({"error": str(exc), **session.state_view()}, status=400)
                return
            except RuntimeError as exc:
                self._json({"error": str(exc), **session.state_view()}, status=500)
                return
            self._json(session.state_view())

    def _handle_undo(self, payload: dict) -> None:
        # No body expected — `payload` just keeps this the same shape as
        # every other _handle_* the do_POST dispatch calls.
        with self.server.lock_for(self.identity):
            session = self.server.entry(self.identity).session
            try:
                session.undo_last_build()
            except ValueError as exc:
                self._json({"error": str(exc), **session.state_view()}, status=400)
                return
            self._json(session.state_view())

    def log_message(self, fmt: str, *args) -> None:  # noqa: A003
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))


@dataclass
class SeatBot:
    """Routes to a different bot per seat.

    `GameSession` holds exactly one `Bot`; this is that one bot, fanning out
    to whichever underlying `NetworkBot` actually owns the seat on move so
    three independently-picked checkpoints can share a game.
    """

    bots_by_seat: dict[int, Bot]

    def choose(self, game: Game) -> Action:
        return self.bots_by_seat[to_move(game)].choose(game)


def _resolve_bot_models(names: list[str]) -> list[tuple[str, str]]:
    """`names` (client-supplied model-picker labels) -> (name, spec) pairs.

    The only point where a request's model choice turns into something that
    can load a file, and the only names it will accept are ones already in
    `model_options()` — never a spec the client sent directly. The name rides
    along so `_build_session` can label the seat with it, not just the spec.
    """
    try:
        options = model_options()
        return [(name, options[name]) for name in names]
    except KeyError as exc:
        raise ValueError(f"unknown model: {exc.args[0]}") from exc


def _spawn_bot(
    spec: str, board, rng: random.Random, device: str, max_offers: int | None
) -> Bot:
    """One entrant spec (see `model_options()`) as a live bot on `board`.

    Routes `network:`/`mcts:` checkpoints through here rather than through
    `catan.arena.spawn` so `--device` reaches them — `arena.spawn` always
    loads a checkpoint onto its default provider, which is fine for a
    benchmark process pool and wrong for one interactive run wanting a GPU
    execution provider if one's available. Everything else (`search2`,
    `greedy`, ...) is handcrafted and device-free, so it goes through
    `arena.spawn` unchanged.
    """
    from dataclasses import replace

    from .arena import entrant_from_name, spawn

    entrant = entrant_from_name(spec)
    if max_offers is not None:
        entrant = replace(entrant, max_offers=max_offers)

    if entrant.kind == "network":
        from .onnxbot import network_bot  # onnxruntime-free import boundary

        assert isinstance(entrant.weights, str)
        return network_bot(entrant.weights, board, max_offers=entrant.max_offers, device=device)
    if entrant.kind == "mcts":
        from .onnxbot import searcher  # onnxruntime-free import boundary

        assert isinstance(entrant.weights, str)
        return searcher(
            entrant.weights,
            board,
            simulations=entrant.simulations,
            wave=entrant.wave,
            max_offers=entrant.max_offers,
            device=device,
            rng=rng,
        )
    return spawn(entrant, board, rng)


def _build_session(
    bots: list[tuple[str, str]],
    human_seat: int | None,
    seed: int | None,
    device: str,
    max_offers: int | None,
    record_path: str | None = None,
) -> GameSession:
    if len(bots) != NUM_PLAYERS - 1:
        raise ValueError(f"expected {NUM_PLAYERS - 1} bots, got {len(bots)}")

    # Always resolved to a concrete int, even when the caller left it to chance,
    # so a finished game can be written out as a catan.record.Record and later
    # replayed from that same seed — `random.Random()` alone has no int to hand
    # back for that.
    if seed is None:
        seed = random.SystemRandom().randrange(2**31)
    # Left to chance the same way: a fixed --human-seat is for testing one
    # spot on the table, but the default experience shouldn't always deal the
    # human seat 0 (and, with it, the first move of every setup snake).
    if human_seat is None:
        human_seat = random.SystemRandom().randrange(NUM_PLAYERS)
    # Two separate Random instances from the same seed, not one shared stream —
    # matching catan.record's own convention (see record_game / test_record.py):
    # replay() rebuilds the board from stored data (consuming no randomness) and
    # then seeds a *fresh* random.Random(seed) for the game itself. Consuming
    # this seed's stream here to build the board first, then handing the same
    # object on to `start`, would leave `start`'s rng at a different position
    # than replay() reconstructs, and a recorded game would fail its own replay.
    board = random_base_board(random.Random(seed))
    # Bots are assigned to non-human seats in ascending seat order — the
    # picker's 3 dropdowns don't know which seats they'll land on, since
    # human_seat is only resolved above, a moment before this runs. Each bot
    # gets its own rng, seeded off the game seed and its seat rather than
    # sharing one stream, so an `mcts:` bot's search sampling on one seat
    # cannot perturb another's.
    non_human_seats = [s for s in range(NUM_PLAYERS) if s != human_seat]
    bots_by_seat = {}
    names_by_seat = {}
    for seat, (name, spec) in zip(non_human_seats, bots):
        bots_by_seat[seat] = _spawn_bot(spec, board, random.Random(seed * 4 + seat), device, max_offers)
        names_by_seat[seat] = name
    bot = SeatBot(bots_by_seat)
    game = start(board, NUM_PLAYERS, random.Random(seed))
    session = GameSession(
        game=game,
        human_seat=human_seat,
        bot=bot,
        seed=seed,
        record_path=record_path,
        bot_names=names_by_seat,
    )
    session.advance_bots()  # in case the human is not first in the setup snake
    return session


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "An entrant spec (see model_options() — a .onnx path, "
            "'search2', or 'mcts:<path>@<sims>'), used for all 3 bot seats "
            "until the web picker (GET /api/models, POST /api/new) overrides "
            "it. Defaults to search2 plus one of each .onnx file found in "
            "the models directory, rather than 3 copies of one bot."
        ),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--seed", type=int, default=None, help="Board/RNG seed.")
    parser.add_argument(
        "--human-seat",
        type=int,
        default=None,
        choices=range(NUM_PLAYERS),
        help="Which of the 4 seats the human plays (0-3). Random each game if omitted.",
    )
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
        "--record",
        default=None,
        help=(
            "Append each finished game to this path as a catan.record.Record "
            "(JSON lines). Off by default."
        ),
    )
    args = parser.parse_args(argv)

    # model_options() can return any number of entries (zero .onnx files
    # dropped in means just "search2"; a dozen means a dozen) — cycle rather
    # than slice so there's always a valid 3-bot lineup regardless of how
    # many models happen to be in MODELS_DIR right now, down to the empty
    # case (3x search2).
    default_bots = (
        [(args.checkpoint, args.checkpoint)] * (NUM_PLAYERS - 1)
        if args.checkpoint
        else list(itertools.islice(itertools.cycle(model_options().items()), NUM_PLAYERS - 1))
    )

    def new_session(bots: list[tuple[str, str]] | None = None) -> GameSession:
        return _build_session(
            bots or default_bots, args.human_seat, args.seed, args.device, args.max_offers, args.record
        )

    # No session is built here anymore — each browser deals its own on first
    # contact (see CatanServer.entry), so there's no single "the" game to
    # build before the server can even start.
    server = CatanServer((args.host, args.port), new_session, args.device, args.max_offers)

    url = f"http://{args.host}:{args.port}/"
    names = [name for name, _ in default_bots]
    print(f"Catan web board: {url}  (bots={names})")
    if args.record:
        print(f"Finished games will be appended to {args.record}")
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
