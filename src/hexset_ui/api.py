"""Tables, seats and the `/api/*` surface everything plays through.

One place decides what a game is and who may touch it. The browser, a script
driving a seat over HTTP, and an LLM over MCP (see `mcp.py`) are all clients of
this module and get no special treatment from it — the same join, the same
token, the same `state`/`act` pair. `web.py` supplies the HTTP transport and
the static frontend; nothing about a game lives there.

## A table, a code, a seat

A table is created with a fixed row of seats, each of which is a bot, a person,
or empty. It is reachable by a six-character code (`ABCDEF` — see `new_code`),
which is the only thing anyone needs to find it: opening `/<code>` in a browser
or calling `join` with it takes an empty seat and hands back a token.

That token, not the request's source or a cookie, is the identity here. It
names one seat at one table, it is the only way to act on that seat, and it is
what `state` reads to decide whose hand to show. There are no accounts and
nothing to log out of.

## Empty seats never reach the engine

A seat nobody took is not a player who does nothing — it is not dealt in at
all. `Table.start` drops the empty seats and deals a game for exactly the
occupied ones, so a four-seat table that two people joined is a two-player
game, and the engine (which supports 2-6 players natively — see
`hexset_ui.state.new_game`) never has to learn what "empty" means. Nothing in
the turn order, the setup snake or the trade responders needs a special case,
because there is no seat there to skip.

The consequence is that seats are renumbered at start time, and after that a
seat number means an engine seat and nothing else. That is also why nobody may
join a game in progress: the seats it was dealt with are the seats it has.
"""

from __future__ import annotations

import itertools
import os
import random
import secrets
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from . import journal
from .actions import Action
from .board.board import Board, random_base_board
from .game import Game, start, to_move
from .search2 import search2
from .webplay import Bot, GameSession, ResumeError, board_layout

# The one opponent that is not a file. Everything else in the picker is a path
# to a checkpoint, and what it does is the checkpoint's business.
HANDCRAFTED = "search2"

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = Path(os.environ.get("HEXSET_UI_MODELS_DIR", REPO_ROOT / "models"))

# The base board seats four, so a table does too. The engine itself will deal
# 2-6 (see `hexset_ui.state.new_game`); what caps this is the board's 19 hexes
# and the resource bag that was balanced for them, not the code.
MAX_SEATS = 4
MIN_PLAYERS = 2

# Digits and uppercase letters, minus the pairs that get misread aloud or
# retyped wrong: 0/O, 1/I/L. 31 characters, so 31**6 is about 887 million
# codes — far more than enough, since a code only has to be unique among the
# tables alive right now, not every game ever played, and `new_code` re-rolls
# on the collisions that do happen.
CODE_ALPHABET = "23456789ABCDEFGHJKMNPQRSTUVWXYZ"
CODE_LENGTH = 6

# How long a table survives with nobody touching it. An open browser polls far
# more often than this; a closed tab or an abandoned lobby doesn't at all. 24
# hours comfortably outlasts a real game paused over a lunch break without
# holding onto dead ones indefinitely.
TABLE_TTL_SECONDS = 24 * 60 * 60


class ApiError(Exception):
    """A request that cannot be served, with the HTTP status to say so with.

    Carries the status because the alternative is every caller re-deriving it
    from the message: "no empty seats" is a 409 whether it arrived over HTTP,
    over MCP, or from a test, and that fact belongs next to the rule that
    raises it rather than in each transport.
    """

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


def new_code(taken: set[str]) -> str:
    """A fresh table code, avoiding the ones already in use.

    Re-rolls rather than trusting the space: collisions are rare (see
    CODE_ALPHABET) but a duplicate code would hand two tables the same address,
    and the registry knows exactly which codes are live, so there is no reason
    to gamble on it.
    """
    rng = random.SystemRandom()
    while True:
        code = "".join(rng.choice(CODE_ALPHABET) for _ in range(CODE_LENGTH))
        if code not in taken:
            return code


def model_options() -> dict[str, str]:
    """Display name -> entrant spec, for the per-seat model picker.

    `search2` (handcrafted, no checkpoint) is always first; every `*.onnx`
    file under `MODELS_DIR` follows, filename stem as the display name. A
    client only ever sends a name back; specs never cross the wire, so a
    request cannot point a bot at an arbitrary file — this function is the one
    chokepoint that turns a name into a loadable path.

    Scanned fresh on every call rather than cached: a directory listing of a
    handful of files is sub-millisecond, and the entire point of this function
    is that dropping a file in shows up without a restart.
    """
    options = {HANDCRAFTED: HANDCRAFTED}
    for path in sorted(MODELS_DIR.glob("*.onnx")):
        options[path.stem] = str(path)
    return options


@dataclass
class Config:
    """How this server builds the games it deals — the CLI's business (see
    `web.py`), passed in once rather than threaded through every call."""

    device: str = "cpu"
    max_offers: int | None = 1
    games_dir: str | None = None
    seed: int | None = None
    # The lineup to seat when a table is created without naming one, for
    # `--checkpoint` to pin every bot seat to one opponent. `None` means the
    # mixed default (see `default_lineup`).
    default_bots: list[str] | None = None


class SeatKind(str, Enum):
    EMPTY = "empty"
    BOT = "bot"
    PLAYER = "player"


@dataclass
class Seat:
    """One place at a table, before and after the deal.

    A seat holds a bot, a person, or nobody. `token` is the secret that proves
    a request is that person (see the module docstring) and never leaves this
    process except in the one response that mints it.
    """

    kind: SeatKind = SeatKind.EMPTY
    name: str | None = None
    spec: str | None = None
    token: str | None = field(default=None, repr=False)

    def public(self, seat: int) -> dict:
        """What anyone may see about this seat. Never the token."""
        return {
            "seat": seat,
            "kind": self.kind.value,
            "name": self.name,
        }


@dataclass
class Table:
    """A row of seats, a code to reach them by, and — once started — the game
    they are playing.

    `session` is `None` for as long as the table is a lobby, which is also
    exactly when its seats may still change. Starting is one-way.
    """

    code: str
    seats: list[Seat]
    config: Config
    session: GameSession | None = None
    layout: dict | None = None
    last_seen: float = field(default_factory=time.monotonic)
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def started(self) -> bool:
        return self.session is not None

    def seat_of(self, token: str) -> int:
        for index, seat in enumerate(self.seats):
            if seat.token is not None and secrets.compare_digest(seat.token, token):
                return index
        raise ApiError("that token is not for a seat at this table", status=403)

    def join(self, name: str | None) -> tuple[int, str]:
        """Seats a person in the first empty seat, returning it and their token.

        Refused once the game is dealt, and not as a policy that could be
        relaxed: a game in progress has only the seats it was dealt with (see
        the module docstring), so there is no empty seat left to give.
        """
        if self.started:
            raise ApiError("this game has already started", status=409)
        for index, seat in enumerate(self.seats):
            if seat.kind is SeatKind.EMPTY:
                token = secrets.token_urlsafe(18)
                self.seats[index] = Seat(
                    kind=SeatKind.PLAYER, name=name, token=token
                )
                return index, token
        raise ApiError("this table has no empty seats", status=409)

    def occupied(self) -> list[Seat]:
        return [seat for seat in self.seats if seat.kind is not SeatKind.EMPTY]

    def start(self) -> None:
        """Deals the game, dropping every empty seat as it goes.

        After this the table's seats *are* the engine's seats, renumbered and
        one-to-one, which is what lets everything downstream stop distinguishing
        the two.
        """
        if self.started:
            raise ApiError("this game has already started", status=409)
        seats = self.occupied()
        if len(seats) < MIN_PLAYERS:
            raise ApiError(
                f"a game needs at least {MIN_PLAYERS} players; this table has {len(seats)}",
                status=409,
            )
        self.seats = seats
        self.session = build_session(self.code, seats, self.config)
        self.layout = board_layout(self.session.game.state.board)

    def lobby_view(self, viewer: int | None = None) -> dict:
        return {
            "code": self.code,
            "started": self.started,
            "seat": viewer,
            "seats": [seat.public(i) for i, seat in enumerate(self.seats)],
            "can_start": not self.started and len(self.occupied()) >= MIN_PLAYERS,
        }

    def view(self, viewer: int | None = None) -> dict:
        """The lobby's own view before the deal, the game's after it. One
        endpoint either way, so a client polls the same place from the moment
        it joins to the moment the game ends."""
        if self.session is None:
            return self.lobby_view(viewer)
        return {**self.lobby_view(viewer), **self.session.state_view(viewer)}


def spawn_bot(spec: str, board: Board, rng: random.Random, config: Config) -> Bot:
    """One spec (see `model_options()`) as a live bot on `board`.

    Two cases, and no third: `search2` is the handcrafted opponent, and
    anything else is a path to a checkpoint. How a checkpoint wants to be
    played — a single forward pass, or a search over its own priors, and with
    what budget — is read out of the file itself by `onnxbot.spawn`, so this
    function never learns what a simulation is.
    """
    if spec == HANDCRAFTED:
        return search2(board, rng, max_offers=config.max_offers)

    from .onnxbot import spawn  # onnxruntime-free import boundary

    return spawn(spec, board, rng=rng, device=config.device, max_offers=config.max_offers)


@dataclass
class SeatBot:
    """Routes to a different bot per seat.

    `GameSession` holds exactly one `Bot`; this is that one bot, fanning out to
    whichever underlying bot actually owns the seat on move, so several
    independently-picked checkpoints can share a game.
    """

    bots_by_seat: dict[int, Bot]

    def choose(self, game: Game) -> Action:
        return self.bots_by_seat[to_move(game)].choose(game)


def seat_bots(seats: list[Seat], board: Board, seed: int, config: Config) -> SeatBot:
    """One bot per bot seat, each with its own rng seeded off the game seed and
    its seat rather than sharing one stream, so an `mcts:` bot's search sampling
    on one seat cannot perturb another's."""
    return SeatBot(
        {
            index: spawn_bot(seat.spec or seat.name or HANDCRAFTED, board, random.Random(seed * 4 + index), config)
            for index, seat in enumerate(seats)
            if seat.kind is SeatKind.BOT
        }
    )


def build_session(code: str, seats: list[Seat], config: Config) -> GameSession:
    """A fresh game for exactly these seats, in this order."""
    seed = config.seed
    # Always resolved to a concrete int, even when the caller left it to
    # chance, so the journal can name the seed a resumed game rebuilds its
    # board from — `random.Random()` alone has no int to hand back for that.
    if seed is None:
        seed = random.SystemRandom().randrange(2**31)
    # Two separate Random instances from the same seed, not one shared stream.
    # `resume_session` rebuilds a game exactly this way, so the two must agree:
    # consuming this seed's stream to build the board and then handing the same
    # object on to `start` would leave `start`'s rng at a position resuming
    # cannot reconstruct, and a journalled game would fail to resume.
    board = random_base_board(random.Random(seed))
    game = start(board, len(seats), random.Random(seed))
    session = GameSession(
        game=game,
        human_seats=frozenset(i for i, s in enumerate(seats) if s.kind is SeatKind.PLAYER),
        bot=seat_bots(seats, board, seed, config),
        seed=seed,
        journal=journal.open_journal(seed, config.games_dir),
        bot_names={i: s.name for i, s in enumerate(seats) if s.kind is SeatKind.BOT and s.name},
        bot_specs={i: s.spec for i, s in enumerate(seats) if s.kind is SeatKind.BOT and s.spec},
        player_names={i: s.name for i, s in enumerate(seats) if s.kind is SeatKind.PLAYER and s.name},
        code=code,
    )
    session.advance_bots()  # in case no person is first in the setup snake
    return session


def resume_session(code: str, seats: list[Seat], config: Config) -> GameSession | None:
    """The game this code left unfinished, played back to where it stopped —
    or `None` if there isn't one, in which case the caller deals.

    A session lives in memory, so it used to be lost to anything that ended the
    process: a deploy, a crash, or simply going quiet long enough to be
    evicted. The journal is the whole game though (see `hexset_ui.journal`), and
    it replays exactly, so the loss was never necessary.

    Bots are the one thing that does not come back: they are re-seated from the
    same specs, but each gets a fresh rng, so a resumed game's opponents play on
    from here rather than replaying the moves they would have made. Their past
    moves are read off the file, not asked for again, so nothing already played
    can change under anyone.
    """
    where = config.games_dir if config.games_dir is not None else journal.configured_dir()
    path = journal.resumable(where, code)
    if path is None:
        return None

    events = journal.read(path)
    header = events[0]
    seed = header["seed"]
    board = random_base_board(random.Random(seed))
    session = GameSession(
        game=start(board, header["num_players"], random.Random(seed)),
        human_seats=frozenset(header.get("human_seats", [])),
        bot=seat_bots(seats, board, seed, config),
        seed=seed,
        bot_names={i: s.name for i, s in enumerate(seats) if s.kind is SeatKind.BOT and s.name},
        bot_specs={i: s.spec for i, s in enumerate(seats) if s.kind is SeatKind.BOT and s.spec},
        player_names={i: s.name for i, s in enumerate(seats) if s.kind is SeatKind.PLAYER and s.name},
        code=code,
    )
    try:
        session.restore(
            journal.replayable(events),
            journal.Journal(directory=str(path.parent), game_id=path.stem),
        )
    except (ResumeError, ValueError, KeyError) as error:
        # Kept rather than deleted: a journal that will not replay is the one
        # copy of a game that did happen, and is worth more as evidence of
        # whatever broke than the disk space is. Closed, though, so the next
        # request tries to resume it once and then deals instead of failing
        # this way forever.
        print(f"could not resume {path.name}: {error}")
        journal.Journal(directory=str(path.parent), game_id=path.stem).abandoned()
        return None

    session.advance_bots()  # the file may end mid-cascade, on a bot's turn
    return session


class Tables:
    """Every live table, and the operations the API is made of.

    Two lock granularities, not one: `_registry_lock` guards the dict of tables
    itself (fast — a handful of dict operations), while each table carries its
    own lock around the actual game mutation, which can be slow (a cascade of
    bot moves, each an ONNX forward pass). One shared lock would mean one
    table's turn stalls every other table's requests for its duration.
    """

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config()
        self._tables: dict[str, Table] = {}
        self._registry_lock = threading.Lock()

    # --- registry ---------------------------------------------------------

    def create(self, bots: list[str] | None = None, open_seats: int = 0, name: str | None = None) -> tuple[Table, str]:
        """A new table with the caller seated first, returning it and their token.

        `bots` names the checkpoints to seat (see `model_options()`), and
        `open_seats` how many places to leave for other people to join. What is
        left empty at `start` time is simply not dealt in.

        A caller who names its bots gets exactly those and an error if they do
        not fit; a caller who leaves the lineup to the default is asking for a
        full table, so the default gives way to the seats it asked to keep open
        rather than refusing to make the table at all.
        """
        if open_seats < 0:
            raise ApiError("open_seats cannot be negative")
        if bots is None:
            bots = (self.config.default_bots or default_lineup())[: MAX_SEATS - 1 - open_seats]
        options = model_options()
        for entry in bots:
            if entry not in options:
                raise ApiError(f"unknown model: {entry}")
        seats = [Seat(kind=SeatKind.PLAYER, name=name, token=secrets.token_urlsafe(18))]
        seats += [Seat(kind=SeatKind.BOT, name=entry, spec=options[entry]) for entry in bots]
        seats += [Seat() for _ in range(open_seats)]
        if len(seats) > MAX_SEATS:
            raise ApiError(f"a table seats at most {MAX_SEATS}; asked for {len(seats)}")

        with self._registry_lock:
            self._evict_stale(time.monotonic())
            code = new_code(set(self._tables))
            table = Table(code=code, seats=seats, config=self.config)
            self._tables[code] = table
        token = seats[0].token
        assert token is not None
        return table, token

    def get(self, code: str) -> Table:
        with self._registry_lock:
            table = self._tables.get(code.upper())
        if table is None:
            raise ApiError(f"no table with code {code}", status=404)
        table.last_seen = time.monotonic()
        return table

    def by_token(self, token: str | None) -> tuple[Table, int]:
        """The table and seat a token names, or a 401/403.

        A linear scan of live tables. There are tens of these, not millions,
        and a second index would be one more thing to keep in step with
        eviction for no measurable gain.
        """
        if not token:
            raise ApiError("this needs a player token — join a table first", status=401)
        with self._registry_lock:
            tables = list(self._tables.values())
        for table in tables:
            for seat in table.seats:
                if seat.token is not None and secrets.compare_digest(seat.token, token):
                    table.last_seen = time.monotonic()
                    return table, table.seat_of(token)
        raise ApiError("unknown or expired player token", status=403)

    def _evict_stale(self, now: float) -> None:
        """Must be called with `_registry_lock` held. Drops any table untouched
        for longer than `TABLE_TTL_SECONDS` — an abandoned lobby or a closed
        tab, not an active game."""
        stale = [c for c, t in self._tables.items() if now - t.last_seen > TABLE_TTL_SECONDS]
        for code in stale:
            del self._tables[code]

    # --- play -------------------------------------------------------------

    def session_of(self, table: Table) -> GameSession:
        if table.session is None:
            raise ApiError("this game has not started yet", status=409)
        return table.session

    def act(self, table: Table, seat: int, wire: dict) -> dict:
        session = self.session_of(table)
        session.apply_human_action(seat, wire)
        # At most the one seat this action just handed off to — a no-op of its
        # own accord if that is another person (see advance_one_seat). A setup
        # road may have left this seat's own confirm pending instead, in which
        # case that waits for POST /api/confirm rather than running here. Any
        # further bots beyond this one seat wait for the client's own follow-up
        # POST /api/advance, same as after any other response.
        if session.awaiting_confirm is None:
            session.advance_one_seat()
        return table.view(seat)

    def advance(self, table: Table, seat: int) -> dict:
        session = self.session_of(table)
        if session.awaiting_confirm is None:
            session.advance_one_seat()
        return table.view(seat)

    def confirm(self, table: Table, seat: int) -> dict:
        session = self.session_of(table)
        session.confirm_setup_turn(seat)
        return table.view(seat)

    def undo(self, table: Table, seat: int) -> dict:
        self.session_of(table).undo_last_build(seat)
        return table.view(seat)

    def rename(self, table: Table, seat: int, name: str) -> dict:
        table.seats[seat].name = name
        if table.session is not None:
            table.session.player_names[seat] = name
        return table.view(seat)

    def swap_bot(self, table: Table, seat: int, model: str) -> dict:
        """Re-seat one bot mid-game — models can be swapped at any point, not
        just between games. Rebuilding from `model_options()` (rather than
        accepting a spec) keeps this the same chokepoint: a request names a
        bot, it never hands one a path."""
        session = self.session_of(table)
        if not 0 <= seat < len(table.seats) or table.seats[seat].kind is not SeatKind.BOT:
            raise ApiError(f"seat {seat} has no bot to swap")
        try:
            spec = model_options()[model]
        except KeyError:
            raise ApiError(f"unknown model: {model}") from None
        board = session.game.state.board
        session.bot.bots_by_seat[seat] = spawn_bot(spec, board, random.Random(), self.config)
        session.bot_names[seat] = model
        session.bot_specs[seat] = spec
        table.seats[seat].name = model
        table.seats[seat].spec = spec
        if session.journal is not None:
            session.journal.seated(seat=seat, name=model, spec=spec)
        return table.view(seat)

    # --- the /api/* surface -----------------------------------------------

    def handle(self, method: str, path: str, payload: dict, token: str | None) -> dict:
        """One request, dispatched. Raises `ApiError` for anything refused.

        Every transport in the project ends up here: `web.py` calls it with a
        parsed HTTP request, and `mcp.py` reaches it over that same HTTP from
        wherever the LLM is running. Routing lives with the rules rather than in
        the transport so the two cannot drift into serving different games.
        """
        if method == "GET" and path == "/api/models":
            return {"models": list(model_options())}

        # The only read that needs no token: what a browser opening /<code>
        # shows someone who has not joined yet.
        if method == "GET" and path.startswith("/api/table/"):
            return self.get(path[len("/api/table/") :]).lobby_view()

        if method == "POST" and path == "/api/tables":
            table, new_token = self.create(
                bots=payload.get("bots"),
                open_seats=int(payload.get("open_seats", 0)),
                name=payload.get("name"),
            )
            return {"token": new_token, **table.lobby_view(0)}

        if method == "POST" and path == "/api/join":
            table = self.get(str(payload.get("code", "")))
            with table.lock:
                seat, new_token = table.join(payload.get("name"))
                return {"token": new_token, **table.lobby_view(seat)}

        # Everything past here acts on a seat, so it needs the token that names
        # one. Resolved once, here, rather than in each branch.
        table, seat = self.by_token(token)
        with table.lock:
            # The session and the engine refuse in Python's terms — a
            # ValueError for a move that is not legal, not this seat's, or not
            # this phase's. That is a bad request and nothing more, so it is
            # answered as one here rather than reaching the transport as an
            # unhandled error.
            try:
                return self._seated(table, seat, method, path, payload)
            except ValueError as error:
                raise ApiError(str(error)) from None

    def _seated(self, table: Table, seat: int, method: str, path: str, payload: dict) -> dict:
        """The routes that act on one seat. Called with the table's lock held."""
        if method == "GET" and path == "/api/state":
            return table.view(seat)
        if method == "GET" and path == "/api/board":
            if table.layout is None:
                raise ApiError("this game has not started yet", status=409)
            return table.layout
        if method == "POST" and path == "/api/start":
            table.start()
            return table.view(seat)
        if method == "POST" and path == "/api/action":
            return self.act(table, seat, payload.get("action") or {})
        if method == "POST" and path == "/api/advance":
            return self.advance(table, seat)
        if method == "POST" and path == "/api/confirm":
            return self.confirm(table, seat)
        if method == "POST" and path == "/api/undo":
            return self.undo(table, seat)
        if method == "POST" and path == "/api/name":
            return self.rename(table, seat, str(payload.get("name", "")))
        if method == "POST" and path == "/api/bot":
            return self.swap_bot(table, int(payload.get("seat", -1)), str(payload.get("model", "")))
        raise ApiError(f"no such endpoint: {method} {path}", status=404)


def default_lineup() -> list[str]:
    """Three opponents when a caller didn't pick any.

    `model_options()` can return any number of entries (no `.onnx` files
    dropped in means just `search2`; a dozen means a dozen) — cycled rather
    than sliced so there is always a valid lineup regardless of how many models
    happen to be present, down to the empty case (3x search2).
    """
    return [name for name, _ in itertools.islice(itertools.cycle(model_options().items()), MAX_SEATS - 1)]
