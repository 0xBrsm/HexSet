"""Games, seats and the `/api/*` surface everything plays through.

One place decides what a game is and who may touch it. The browser, a script
driving a seat over HTTP, and an LLM over MCP (see `mcp.py`) are all clients of
this module and get no special treatment from it — the same join, the same
token, the same `state`/`act` pair. A bot (embedded or external — see
`botclient.py`) is no different: it is a client like any other, submitting its
own actions through the same token-gated route, never a privileged code path
inside this one. `web.py` supplies the HTTP transport and the static frontend;
nothing about a game lives there.

## A game, an ID, a seat

A game has a unique ID — a six-character code (`abcdef` — see `new_code`),
the only thing anyone needs to find it. There is no lobby: `POST /api/games`
deals a full `MAX_SEATS`-seat game immediately, with the creator seated at one
random seat and every other seat open. Opening `/<id>` in a browser, or
calling `join` with it, claims a random still-open seat and hands back a
token.

A still-open seat can also be given to a bot from the board itself
(`POST /api/bot`, see `Tables.seat_bot`) right up until the setup snake
retires it — the same request that swaps one bot for another, because with no
lobby the player list is the only place a table says who else is playing.

That token, not the request's source or a cookie, is the identity here. It
names one seat at one game, it is the only way to act on that seat, and it is
what `state` reads to decide whose hand to show. There are no accounts and
nothing to log out of — and, deliberately, the token never touches disk (see
`journal.py`): a restart cannot hand a lost token back to anyone, so a table
reopened after one simply treats every non-bot seat as open again (see
`Tables._reopen`).

## A seat locks; it is never dropped

`MAX_SEATS` seats are dealt into the engine from the moment a game exists,
whether or not anyone has claimed them yet — there is no "start" that
renumbers a partial roster down to just the seats somebody's in. Instead, the
setup snake retires an empty seat the moment it reaches one
(`Table._settle_locks`, `hexset.server.seating.lock_seat`): a game that begins
setup with two or three seats occupied stays that size for its whole duration,
one seat's decision at a time rather than one cutoff for the table.
`Table.join` only ever offers a seat that is both empty and unlocked.

There is no waiting window in front of that, and deliberately so. A turn only
ever advances because the seat holding it says so, so somebody expecting a
friend just does not finish their own placement until that friend has sat
down — the wait is a person choosing to wait, which a timer can only cut
short.

## What a seat is told, and what it is not

Every response here is built for one viewer. Two of the filters are not
obvious and both are load-bearing:

Every seat's **valuation vector** is public, and so is the turn's trade log:
those are what a table hears (`hexset.trading`), and `PUT
/api/games/<code>/valuation` is how a seat sets its own. Nothing else about
trading is on the wire, because nothing else exists -- there is no offer to
address, accept or decline.

The `action_mask`/`options` on `GET /api/record` are the engine's own
`legal_actions`, and so is what an embedded bot searches: one list for every
seat, honest by construction now that no action's legality depends on
another seat's hand.

## Liveness is people, not bots

`Table.last_seen` is refreshed by a request from a person or an external
client, never by an embedded bot runner's poll — a runner polls once a second
until its game ends, so counting those would mean no table with a bot at it
could ever go stale. Eviction (`_evict_stale`) runs on every `get` as well as
on `create`, and closing a table stops its runners before the journal.
"""

from __future__ import annotations

import os
import random
import secrets
import threading
import time
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any

import hexset.bots  # noqa: F401 -- registers the "heximax" presets with hexset.arena
from hexset.actions import build_space
from hexset.arena import PRESETS, spawn as spawn_entrant
from hexset.board.board import Board, random_base_board
from hexset.bots import Bot
from hexset.clients.botclient import BotRunner, LocalSearchBrain, LocalTransport
from hexset.game import Phase, is_over, to_move
from hexset.onnx_record import record_from_game

from . import journal
from hexset.actions import legal_actions
from .seating import lock_seat, locked_of, start_at
from .webplay import (
    GameSession,
    ResumeError,
    action_to_wire,
    board_layout,
)

# The opponents that are not files. Everything else in the picker is a path to
# a checkpoint, and what it does is the checkpoint's business.
#
# Both are `hexset.arena` presets, built through `hexset.arena.spawn` so this
# server seats the same bot the training repo duels — `heximax` is the honest
# handcrafted search (it reads the public ledger, never a hidden hand) and is
# the default embedded opponent; `search2` is the older depth-two bot every
# ladder number on record was measured against, kept by name so a game can
# still be played against it. `heximax` is registered as a preset by the
# sibling `heximax` package (imported above for that side effect), not by
# `hexset` itself -- see `heximax`'s "registration" section.
HANDCRAFTED = "heximax"
HANDCRAFTED_ENTRANTS = ("heximax", "search2")

REPO_ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = Path(os.environ.get("HEXSET_UI_MODELS_DIR", REPO_ROOT / "models"))

# The base board seats four, so a game does too. The engine itself will deal
# 2-6 (see `hexset.state.new_game`); what caps this is the board's 19 hexes
# and the resource bag that was balanced for them, not the code. Fixed for
# every game regardless of how many seats end up claimed: a served checkpoint
# hard-rejects a mismatched player count (`onnxbot.py`'s `_check_players`),
# and a partial roster is exactly what the setup lock exists to allow without
# renumbering the engine's own idea of how many seats there are.
MAX_SEATS = 4

# Digits and lowercase letters, minus the pairs that get misread aloud or
# retyped wrong: 0/o, 1/i/l. 31 characters, so 31**6 is about 887 million
# codes — far more than enough, since a code only has to be unique among the
# games alive right now, not every one ever played, and `new_code` re-rolls
# on the collisions that do happen.
#
# Lowercase because the code is only ever seen as a URL, and a URL that is
# read aloud, typed on a phone or pasted into a chat reads better without
# the shift key. Case is not part of the identity either way: every lookup
# normalises (see `Tables.get`), so a code typed in capitals still opens its
# game.
CODE_ALPHABET = "23456789abcdefghjkmnpqrstuvwxyz"
CODE_LENGTH = 6

# The cap every client already enforces (mcp.py, index.html) on a display
# name, applied here too: those are conveniences, not the check, since a raw
# POST to /api/games, /api/join or /api/name bypasses both of them.
MAX_NAME_LENGTH = 40

# How long a game survives with nobody touching it. An open browser or a bot
# runner polls far more often than this; a closed tab or an abandoned game
# doesn't at all. 24 hours comfortably outlasts a real game paused over a
# lunch break without holding onto dead ones indefinitely.
TABLE_TTL_SECONDS = 24 * 60 * 60


class ApiError(Exception):
    """A request that cannot be served, with the HTTP status to say so with.

    Carries the status because the alternative is every caller re-deriving it
    from the message: "no open seats" is a 409 whether it arrived over HTTP,
    over MCP, or from a test, and that fact belongs next to the rule that
    raises it rather than in each transport.
    """

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


def new_code(taken: set[str]) -> str:
    """A fresh game code, avoiding the ones already in use.

    Re-rolls rather than trusting the space: collisions are rare (see
    CODE_ALPHABET) but a duplicate code would hand two games the same address,
    and the registry knows exactly which codes are live, so there is no reason
    to gamble on it.
    """
    rng = random.SystemRandom()
    while True:
        code = "".join(rng.choice(CODE_ALPHABET) for _ in range(CODE_LENGTH))
        if code not in taken:
            return code


def clean_name(name: str | None) -> str | None:
    """A display name, trimmed and capped the same way everywhere one is
    accepted, or `None` if there's nothing left of it once trimmed."""
    if name is None:
        return None
    name = name.strip()[:MAX_NAME_LENGTH]
    return name or None


def model_options() -> dict[str, str]:
    """Display name -> entrant spec, for the per-seat model picker.

    The handcrafted entrants (no checkpoint) come first; every `*.onnx`
    file under `MODELS_DIR` follows, filename stem as the display name. A
    client only ever sends a name back; specs never cross the wire, so a
    request cannot point a bot at an arbitrary file — this function is the one
    chokepoint that turns a name into a loadable path.

    Scanned fresh on every call rather than cached: a directory listing of a
    handful of files is sub-millisecond, and the entire point of this function
    is that dropping a file in shows up without a restart.
    """
    options = {name: name for name in HANDCRAFTED_ENTRANTS}
    for path in sorted(MODELS_DIR.glob("*.onnx")):
        options[path.stem] = str(path)
    return options


@dataclass
class Config:
    """How this server builds the games it deals — the CLI's business (see
    `web.py`), passed in once rather than threaded through every call."""

    device: str = "cpu"
    # The trade off switch every bot seat is held to (`hexset.trading`):
    # `0` seats bots that never trade, `None` (the default) lets them.
    max_trades: int | None = None
    games_dir: str | None = None
    seed: int | None = None
    # The lineup to seat at creation when a request doesn't name its own, for
    # `--checkpoint` to pin every bot seat it fills to one opponent. `None`
    # means no bots at all — every seat but the creator's starts open (see the
    # module docstring); there is no automatic mixed-lineup default any more,
    # since filling the table is now an explicit choice, not the assumption a
    # lobby used to make on a caller's behalf.
    default_bots: list[str] | None = None


class SeatKind(str, Enum):
    EMPTY = "empty"
    BOT = "bot"
    PLAYER = "player"


@dataclass
class Seat:
    """One place at a game, claimed or not.

    A seat holds a bot, a person, or nobody. `token` is the secret that proves
    a request is that seat (see the module docstring) and never leaves this
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
    """A row of seats, a code to reach them by, and the game they're playing
    — dealt the instant this exists, never gated behind a lobby.

    `runners` is every embedded bot-client thread playing a seat at this
    table (see `Tables._spawn_local_bots`) — `close` has to stop those
    before the journal, or one could still be mid-decision when its game is
    marked abandoned.
    """

    code: str
    seats: list[Seat]
    config: Config
    session: GameSession
    layout: dict

    runners: list[tuple[BotRunner, threading.Thread]] = field(default_factory=list, repr=False)
    last_seen: float = field(default_factory=time.monotonic)
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def seat_of(self, token: str) -> int:
        for index, seat in enumerate(self.seats):
            if seat.token is not None and secrets.compare_digest(seat.token, token):
                return index
        raise ApiError("that token is not for a seat at this game", status=403)

    def stop_runners(self) -> None:
        """Stops every embedded bot runner at this table and waits for it.

        Signalled in one pass and joined in a second, so three bots take one
        poll interval between them rather than three.
        """
        for runner, _ in self.runners:
            runner.stop.set()
        for _, thread in self.runners:
            thread.join(timeout=2.0)
        self.runners.clear()

    def close(self) -> None:
        """Stops every embedded bot runner at this table, then closes the
        journal. Order matters: a runner still mid-decision when the journal
        is marked abandoned could otherwise submit one more action into a
        game already filed as over."""
        self.stop_runners()
        if self.session.journal is not None:
            self.session.journal.abandoned()

    def _settle_locks(self) -> None:
        """Must be called with `self.lock` held. Retires every empty seat the
        setup snake has reached, on sight.

        There is no waiting window, because there is nothing a window would
        add. A turn only ever advances because the seat holding it says so,
        so somebody expecting a friend simply does not finish their own
        placement until that friend is in — the wait is a person deciding to
        wait, not a timer counting down behind them. A window on top of that
        could only ever retire a seat somebody was still holding the table
        for.

        No timer thread either, for the same reason: nothing here is
        time-dependent any more. It runs at the top of every request that
        touches the game (`view`, `join`, `Tables.act`, `Tables.seat_bot`),
        which is exactly when the snake can have moved.
        """
        game = self.session.game
        while not is_over(game) and game.phase in (Phase.SETUP_SETTLEMENT, Phase.SETUP_ROAD):
            seat = to_move(game)
            if self.seats[seat].kind is not SeatKind.EMPTY or seat in locked_of(game):
                return
            lock_seat(game, seat)
            if self.session.journal is not None:
                self.session.journal.locked(seat, at_step=self.session._steps)

    def join(self, name: str | None) -> tuple[int, str]:
        """Seats a person (or an external bot process) at a random still-open,
        still-unlocked seat, returning it and their token."""
        self._settle_locks()
        candidates = [
            i
            for i, seat in enumerate(self.seats)
            if seat.kind is SeatKind.EMPTY and i not in locked_of(self.session.game)
        ]
        if not candidates:
            raise ApiError("this game has no open seats", status=409)
        index = random.SystemRandom().choice(candidates)
        token = secrets.token_urlsafe(18)
        clean = clean_name(name)
        self.seats[index] = Seat(kind=SeatKind.PLAYER, name=clean, token=token)
        self.session.claim(index, clean)
        return index, token

    def view(self, viewer: int | None = None, *, omniscient: bool = False) -> dict:
        """The whole game as `viewer` (a seat, or `None` for a spectator) is
        allowed to see it — reachable from the moment a game exists, since
        there's no separate lobby shape any more.

        `omniscient` is for a spectator only, and `state_view` refuses it
        alongside a seat rather than trusting the caller: it holds back
        nothing at all, which is the right answer for somebody watching and
        the wrong one for anybody playing."""
        self._settle_locks()
        state = self.session.state_view(viewer, omniscient=omniscient)
        state["code"] = self.code
        state["seats"] = [seat.public(i) for i, seat in enumerate(self.seats)]
        return state


def spawn_bot(spec: str, board: Board, rng: random.Random, config: Config) -> Bot:
    """One spec (see `model_options()`) as a live bot on `board`.

    Two cases, and no third: `search2` is the handcrafted opponent, and
    anything else is a path to a checkpoint. How a checkpoint wants to be
    played — a single forward pass, or a search over its own priors, and with
    what budget — is read out of the file itself by `onnxbot.spawn`, so this
    function never learns what a simulation is.

    Called by `botclient.LocalSearchBrain`, not by this module any more — a
    bot plays its seat from outside the session, the same as any other
    client (see the module docstring).
    """
    if spec in PRESETS:
        # `hexset.arena` is the training repo's own bot registry, and the one
        # `search2` every duel on record was played against — built here
        # rather than re-declared, so "the handcrafted opponent" means one
        # thing in both repos. The trade switch is this deployment's
        # (`Config.max_trades`), not the preset's.
        return spawn_entrant(replace(PRESETS[spec], max_trades=config.max_trades), board, rng)

    from hexset.clients.onnxbot import spawn  # onnxruntime-free import boundary

    return spawn(spec, board, rng=rng, device=config.device, max_trades=config.max_trades)


def _seat_labels(seats: list[Seat]) -> tuple[dict[int, str], dict[int, str], dict[int, str]]:
    """The three name/spec maps `GameSession` is built with, read off `seats`
    the same way whether the game is being dealt fresh or replayed back from
    a journal — the two must agree on how a seat's kind decides which map it
    lands in, or a resumed game's labels would silently diverge from a fresh
    one's."""
    bot_names = {i: s.name for i, s in enumerate(seats) if s.kind is SeatKind.BOT and s.name}
    bot_specs = {i: s.spec for i, s in enumerate(seats) if s.kind is SeatKind.BOT and s.spec}
    player_names = {i: s.name for i, s in enumerate(seats) if s.kind is SeatKind.PLAYER and s.name}
    return bot_names, bot_specs, player_names


def build_session(code: str, seats: list[Seat], config: Config, *, first: int) -> GameSession:
    """A fresh `MAX_SEATS`-seat game, `first` the creator's own seat (see
    `hexset.server.seating.start_at`). Every seat not already claimed here (an empty
    one, or a named bot's) is simply left for `Table.join`/`lock_seat` to
    resolve as the game itself unfolds."""
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
    game = start_at(board, MAX_SEATS, random.Random(seed), first=first)
    bot_names, bot_specs, player_names = _seat_labels(seats)
    claimed = {i for i, s in enumerate(seats) if s.kind is not SeatKind.EMPTY}
    return GameSession(
        game=game,
        claimed_seats=claimed,
        seed=seed,
        journal=journal.open_journal(seed, config.games_dir),
        bot_names=bot_names,
        bot_specs=bot_specs,
        player_names=player_names,
        code=code,
    )


def resume_session(code: str, seats: list[Seat], config: Config) -> GameSession | None:
    """The game this code left unfinished, played back to where it stopped —
    or `None` if there isn't one, in which case the caller deals.

    A session lives in memory, so it used to be lost to anything that ended the
    process: a deploy, a crash, or simply going quiet long enough to be
    evicted. The journal is the whole game though (see `hexset.server.journal`), and
    it replays exactly, so the loss was never necessary.

    `seats` names only the seats the caller wants pre-claimed on the rebuilt
    game (see `Tables._reopen`) — a bot's, whose identity is just its spec and
    needs no lost token back; every other seat, including one a person held
    before, comes back open. `game.locked` is seeded from the journal's own
    `locked` events before replay runs, which is provably equivalent to
    locking each seat at the step it actually happened (see
    `hexset.server.seating`'s own note on `advance_setup`).
    """
    where = config.games_dir if config.games_dir is not None else journal.configured_dir()
    path = journal.resumable(where, code)
    if path is None:
        return None

    events = journal.read(path)
    header = events[0]
    seed = header["seed"]
    first = header.get("first", 0)
    board = random_base_board(random.Random(seed))
    bot_names, bot_specs, player_names = _seat_labels(seats)
    game = start_at(board, MAX_SEATS, random.Random(seed), first=first)
    game.locked = journal.locked_seats(events)  # noqa: attribute, see seating.py
    claimed = {i for i, s in enumerate(seats) if s.kind is not SeatKind.EMPTY}
    session = GameSession(
        game=game,
        claimed_seats=claimed,
        seed=seed,
        bot_names=bot_names,
        bot_specs=bot_specs,
        player_names=player_names,
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

    return session


class Tables:
    """Every live game, and the operations the API is made of.

    Two lock granularities, not one: `_registry_lock` guards the dict of
    games itself (fast — a handful of dict operations), while each table
    carries its own lock around the actual game mutation. One shared lock
    would mean one game's turn stalls every other game's requests for its
    duration.
    """

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config()
        self._tables: dict[str, Table] = {}
        self._registry_lock = threading.Lock()

    # --- registry ---------------------------------------------------------

    def create(
        self,
        bots: list[str] | None = None,
        name: str | None = None,
    ) -> tuple[Table, str]:
        """A new game, dealt immediately: the creator at a random seat, any
        named bots seated (and tokened) alongside them, everything else
        open. `bots` names the checkpoints to seat (see `model_options()`)
        — left unnamed, this falls back to `Config.default_bots`
        (`--checkpoint`'s pin) or, absent that, no bots at all; a caller
        wanting the table filled says so explicitly, there is no automatic
        mixed lineup any more (see `Config.default_bots`'s own docstring).
        """
        if bots is None:
            bots = list(self.config.default_bots or [])
        options = model_options()
        for entry in bots:
            if entry not in options:
                raise ApiError(f"unknown model: {entry}")
        if 1 + len(bots) > MAX_SEATS:
            raise ApiError(f"a game seats at most {MAX_SEATS}; asked for {1 + len(bots)}")

        with self._registry_lock:
            evicted = self._evict_stale(time.monotonic())
            code = new_code(set(self._tables))
        for table in evicted:
            table.close()

        creator_seat = random.SystemRandom().randrange(MAX_SEATS)
        seats: list[Seat] = [Seat() for _ in range(MAX_SEATS)]
        seats[creator_seat] = Seat(
            kind=SeatKind.PLAYER, name=clean_name(name), token=secrets.token_urlsafe(18)
        )
        remaining = [i for i in range(MAX_SEATS) if i != creator_seat]
        for entry, seat_index in zip(bots, remaining):
            seats[seat_index] = Seat(
                kind=SeatKind.BOT, name=entry, spec=options[entry], token=secrets.token_urlsafe(18)
            )

        session = build_session(code, seats, self.config, first=creator_seat)
        table = Table(
            code=code,
            seats=seats,
            config=self.config,
            session=session,
            # true state: the board is public.
            layout=board_layout(session.game.state(0, hidden=False).board),
        )

        # Registered *before* any bot runner thread starts: a runner's first
        # move is to look itself up by token (`Tables.by_token`, scanning
        # `_tables`), and starting it earlier would race that lookup against
        # this very insert.
        with self._registry_lock:
            self._tables[code] = table
        self._spawn_local_bots(table)

        token = seats[creator_seat].token
        assert token is not None
        return table, token

    def _spawn_local_bots(self, table: Table) -> None:
        """Starts one embedded `LocalSearchBrain` runner thread per bot seat
        `table` already has, tokened and all — the in-process half of
        `botclient.py`'s two brains (see its module docstring). `spawn_bot`
        (unchanged) builds whatever the spec asks for — `search2`, a plain
        checkpoint, or a search over one — and the runner drives it through
        the same token-gated `/api/action` route an external client would
        use, never a direct write to the session."""
        # true state: the board is public.
        board = table.session.game.state(0, hidden=False).board
        transport = LocalTransport(self)
        for index, seat in enumerate(table.seats):
            if seat.kind is not SeatKind.BOT:
                continue
            assert seat.spec is not None and seat.token is not None
            bot = spawn_bot(seat.spec, board, random.Random(), self.config)
            # A bot seat brings its own trading (`hexset.trading`): the
            # engine asks this object for the seat's vector and gate, the
            # same object the runner plays the seat with.
            table.session.set_trader(index, bot)
            brain = LocalSearchBrain(bot=bot, game=table.session.game)
            runner = BotRunner(seat=index, token=seat.token, transport=transport, brain=brain)
            thread = threading.Thread(
                target=runner.run, name=f"bot-{table.code}-{index}", daemon=True
            )
            table.runners.append((runner, thread))
            thread.start()

    def get(self, code: str) -> Table:
        code = code.lower()
        with self._registry_lock:
            # Every lookup, not just `create`: a box that deals one game and
            # is then only ever read would otherwise never reap anything.
            evicted = self._evict_stale(time.monotonic(), keep=code)
            table = self._tables.get(code)
            if table is None:
                table = self._reopen(code)
        for stale in evicted:
            stale.close()
        if table is None:
            raise ApiError(f"no game with code {code}", status=404)
        table.last_seen = time.monotonic()
        return table

    def _reopen(self, code: str) -> Table | None:
        """Must be called with `_registry_lock` held. Puts a game back
        together from its journal for a code the registry has lost — a
        restart — if that game is still in progress. Every seat comes back
        open except a bot's (re-tokened fresh; a checkpoint's identity is
        just its spec, nothing a lost token was protecting) and a locked
        one (still locked) — see `resume_session`'s own docstring for why a
        human's old seat is not, and cannot be, specially recovered.

        Registers the rebuilt table itself, before spawning any bot runner
        (same ordering reason as `create`) — the lock is already held by the
        caller either way, so there is no separate window to race.
        """
        where = self.config.games_dir if self.config.games_dir is not None else journal.configured_dir()
        path = journal.resumable(where, code)
        if path is None:
            return None
        events = journal.read(path)
        seats = [Seat() for _ in range(MAX_SEATS)]
        for seat, (bot_name, spec) in journal.seating(events).items():
            seats[seat] = Seat(kind=SeatKind.BOT, name=bot_name, spec=spec, token=secrets.token_urlsafe(18))
        session = resume_session(code, seats, self.config)
        if session is None:
            return None
        table = Table(
            code=code,
            seats=seats,
            config=self.config,
            session=session,
            # true state: the board is public.
            layout=board_layout(session.game.state(0, hidden=False).board),
        )
        self._tables[code] = table
        self._spawn_local_bots(table)
        return table

    def by_token(self, token: str | None) -> tuple[Table, int]:
        """The game and seat a token names, or a 401/403.

        A linear scan of live games. There are tens of these, not millions,
        and a second index would be one more thing to keep in step with
        eviction for no measurable gain.

        **A bot seat's request does not refresh `last_seen`.** An embedded
        runner polls once a second for the life of the game, so counting its
        polls as activity meant `now - last_seen` could never reach
        `TABLE_TTL_SECONDS` while a runner lived, and a runner lives until the
        game is over: a human who dealt against three bots and closed the tab
        left three threads polling a table that could never be evicted, with
        its journal handle never closed (PR #2 defect 5). Liveness means *a
        person or an external client is still here*, which is exactly the
        seats that are not this server's own bots.
        """
        if not token:
            raise ApiError("this needs a seat token — join a game first", status=401)
        with self._registry_lock:
            tables = list(self._tables.values())
        for table in tables:
            for seat in table.seats:
                if seat.token is not None and secrets.compare_digest(seat.token, token):
                    index = table.seat_of(token)
                    if table.seats[index].kind is not SeatKind.BOT:
                        table.last_seen = time.monotonic()
                    return table, index
        raise ApiError("unknown or expired seat token", status=403)

    def _evict_stale(self, now: float, keep: str | None = None) -> list[Table]:
        """Must be called with `_registry_lock` held. Drops any game untouched
        for longer than `TABLE_TTL_SECONDS` — an abandoned game or a closed
        tab, not an active one. `keep` is the code the caller is about to look
        up, spared so a request can never evict the game it came for.

        A game evicted mid-play leaves its journal open otherwise: nothing
        else ever calls `Journal.abandoned` for it, so a reader that treats an
        unclosed file as "still in flight" would wait on this one forever.

        Returns the popped tables rather than closing them: `close` joins each
        runner thread with a two-second timeout, and doing that while holding
        the registry lock stalls every other game's `by_token` for up to two
        seconds per runner. The caller closes them once it has let go.
        """
        stale = [
            c
            for c, t in self._tables.items()
            if c != keep and now - t.last_seen > TABLE_TTL_SECONDS
        ]
        return [self._tables.pop(code) for code in stale]

    def close(self) -> None:
        """Stop every runner at every table and close every journal.

        Nothing in a long-lived server calls this — eviction does the same job
        one table at a time — but a test, or a shutting-down process, has no
        other way to get its threads back, and a suite that deals a few dozen
        bot games leaks a runner thread per bot seat without it (PR #2 defect
        5: 67 live `bot-*` threads after `test_api.py` + `test_web.py`).
        """
        with self._registry_lock:
            tables = list(self._tables.values())
            self._tables.clear()
        for table in tables:
            table.close()

    # --- play -------------------------------------------------------------

    def act(self, table: Table, seat: int, wire: dict) -> dict:
        table._settle_locks()
        table.session.submit(seat, wire)
        return table.view(seat)

    def undo(self, table: Table, seat: int) -> dict:
        table.session.undo_last_build(seat)
        return table.view(seat)

    def rename(self, table: Table, seat: int, name: str) -> dict:
        name = clean_name(name)
        table.seats[seat].name = name
        table.session.player_names[seat] = name
        return table.view(seat)

    def seat_bot(self, table: Table, viewer: int, seat: int, model: str) -> dict:
        """Put a bot on `seat`: a fresh one where nobody is sitting, or a
        different one in place of the bot already there.

        Both halves are the same request because, with no lobby to pick a
        lineup in, the player list on the board is the only place a table
        decides who else is playing — filling an open seat and changing your
        mind about a bot already in one are the same gesture there. An open
        seat can be filled right up until the setup snake retires it (see
        `Table._settle_locks`), and a bot can be swapped at any point in the
        game, not just between games.

        Rebuilding from `model_options()` (rather than accepting a spec)
        keeps this the same chokepoint: a request names a bot, it never hands
        one a path. A seat that already held a bot has that runner thread
        stopped and a fresh one started on the new spec — the old bot's own
        in-flight decision, if any, still lands (it was already submitted
        through `/api/action` like any other move), but nothing further comes
        from it.

        A person's seat is never taken over, and a retired seat is never
        revived: both refuse.

        `viewer` is the seat that *asked* — whose game this answers with —
        and is not `seat`. Answering as `seat` instead was a hidden-hand
        leak: the response is built for one viewer (see `Table.view`), so a
        view of the bot's seat handed back that bot's whole hand to whoever
        touched its picker, and left their client believing it was sitting
        somewhere it was not.
        """
        if not 0 <= seat < len(table.seats):
            raise ApiError(f"there is no seat {seat} at this game")
        kind = table.seats[seat].kind
        if kind is SeatKind.PLAYER:
            raise ApiError(f"seat {seat} belongs to a player")
        # Before the open/retired test, not after: the window a seat has left
        # only advances when somebody asks (see `_settle_locks`), and this is
        # exactly such a request.
        table._settle_locks()
        if kind is SeatKind.EMPTY and seat in locked_of(table.session.game):
            raise ApiError(f"seat {seat} has been retired from this game")
        try:
            spec = model_options()[model]
        except KeyError:
            raise ApiError(f"unknown model: {model}") from None

        if kind is SeatKind.EMPTY:
            token = secrets.token_urlsafe(18)
            table.seats[seat] = Seat(kind=SeatKind.BOT, name=model, spec=spec, token=token)
            # A seat only counts as playable once the session agrees it is
            # claimed — `GameSession.apply_human_action` refuses a seat that
            # is not, and the runner about to start plays through exactly
            # that route.
            table.session.claimed_seats.add(seat)
        else:
            table.seats[seat].name = model
            table.seats[seat].spec = spec
        table.session.bot_names[seat] = model
        table.session.bot_specs[seat] = spec
        if table.session.journal is not None:
            table.session.journal.seated(seat=seat, name=model, spec=spec)

        for i, (runner, thread) in enumerate(table.runners):
            if runner.seat == seat:
                runner.stop.set()
                thread.join(timeout=2.0)
                del table.runners[i]
                break
        # true state: the board is public.
        board = table.session.game.state(0, hidden=False).board
        bot = spawn_bot(spec, board, random.Random(), self.config)
        table.session.set_trader(seat, bot)
        brain = LocalSearchBrain(bot=bot, game=table.session.game)
        token = table.seats[seat].token
        assert token is not None
        new_runner = BotRunner(seat=seat, token=token, transport=LocalTransport(self), brain=brain)
        new_thread = threading.Thread(
            target=new_runner.run, name=f"bot-{table.code}-{seat}", daemon=True
        )
        table.runners.append((new_runner, new_thread))
        new_thread.start()

        return table.view(viewer)

    def set_valuation(self, table: Table, seat: int, vector) -> dict:
        """`PUT /api/games/<code>/valuation`: this seat publishes its vector.

        The one trading endpoint there is, and the whole of the mechanic's
        API surface (`hexset.trading`): five numbers in [-1, 1], positive
        for "I want more of this". It may be set at any time, takes effect
        at the next trade event, and is public to the whole table the moment
        it lands -- `state_view`'s `valuations` block carries every seat's.
        """
        if vector is None:
            raise ApiError("send a `valuation` of five numbers between -1 and 1")
        table.session.publish(seat, vector)
        return table.view(seat)

    def record(self, table: Table, seat: int) -> dict:
        """`GET /api/record`: the information-set record `hexset.onnx_record`
        builds for a checkpoint, byte-for-byte what an in-process bot would
        compute — the wire a bot client (`clients/botclient.py`) actually
        plays from, rather than reconstructing one from `state_view`'s
        human-shaped fields (see the module docstring's note on why:
        `legal_wire_actions`'s own options are already the one fair
        option list every client gets, `hexset.actions.legal_actions`)."""
        game = table.session.game
        if is_over(game) or to_move(game) != seat:
            raise ApiError("it is not your turn to act", status=409)
        options = legal_actions(game)
        # true state: the board and `num_players` are public.
        state = game.state(seat, hidden=False)
        topology = state.board.topology
        space = build_space(
            topology.num_vertices, topology.num_edges, topology.num_hexes, state.num_players
        )
        record: dict[str, Any] = record_from_game(game, seat, space, options)
        return {
            **{key: value.tolist() for key, value in record.items()},
            "options": [action_to_wire(a) for a in options],
            "space": {
                "num_vertices": topology.num_vertices,
                "num_edges": topology.num_edges,
                "num_hexes": topology.num_hexes,
                "players": state.num_players,
            },
        }

    # --- the /api/* surface -----------------------------------------------

    def handle(self, method: str, path: str, payload: dict, token: str | None) -> dict:
        """One request, dispatched. Raises `ApiError` for anything refused.

        Every transport in the project ends up here: `web.py` calls it with a
        parsed HTTP request, `mcp.py` reaches it over that same HTTP from
        wherever the LLM is running, and `botclient.py` reaches it either the
        same way (a real external process) or in-process, directly, for a
        locally-embedded bot. Routing lives with the rules rather than in the
        transport so none of them can drift into serving different games.
        """
        if method == "GET" and path == "/api/models":
            return {"models": list(model_options())}

        # The reads that need no token, and the whole of what "every game is
        # public" means: `GET /api/table/<code>` is the game as a spectator
        # sees it and `.../board` is the layout that view is drawn on. A link
        # is all either one takes.
        #
        # **A spectator sees everything** — every hand, every dev card, every
        # true victory-point count, and a transcript that names the card
        # bought, the card stolen and the cards discarded. That is the point
        # of watching, and it is also the one place in this module where
        # hidden information leaves it, so it is worth being plain about what
        # follows: this route is not authenticated and cannot be, since
        # holding the link is the whole qualification. Anyone playing at the
        # table holds the link. A seat that opens its own game's public view
        # is therefore reading every opponent's hand, and nothing here can
        # tell that apart from a bystander doing the same.
        #
        # Every route that *acts* still answers a token and still gets its
        # own seat's honest view (`state_view` refuses `omniscient` for a
        # seat outright), so nothing a bot or a training run reads is
        # affected. The exposure is to people, at a table, who choose to look.
        if method == "GET" and path.startswith("/api/table/"):
            code, _, tail = path[len("/api/table/") :].partition("/")
            table = self.get(code)
            if not tail:
                return table.view(None, omniscient=True)
            if tail == "board":
                return table.layout
            raise ApiError(f"no such endpoint: {method} {path}", status=404)

        if method == "POST" and path == "/api/games":
            table, new_token = self.create(
                bots=payload.get("bots"),
                name=payload.get("name"),
            )
            return {"token": new_token, **table.view(table.seat_of(new_token))}

        if method == "POST" and path == "/api/join":
            table = self.get(str(payload.get("code", "")))
            with table.lock:
                seat, new_token = table.join(payload.get("name"))
                return {"token": new_token, **table.view(seat)}

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
                return self._seated(table, seat, method, path, payload, token)
            except ValueError as error:
                raise ApiError(str(error)) from None

    def _seated(
        self, table: Table, seat: int, method: str, path: str, payload: dict, token: str
    ) -> dict:
        """The routes that act on one seat. Called with the table's lock held."""
        if method == "GET" and path == "/api/state":
            return table.view(seat)
        if method == "GET" and path == "/api/board":
            return table.layout
        if method == "GET" and path == "/api/record":
            return self.record(table, seat)
        if method == "POST" and path == "/api/action":
            return self.act(table, seat, payload.get("action") or {})
        if method == "POST" and path == "/api/undo":
            return self.undo(table, seat)
        if method == "POST" and path == "/api/name":
            return self.rename(table, seat, str(payload.get("name", "")))
        if method == "POST" and path == "/api/bot":
            return self.seat_bot(
                table, seat, int(payload.get("seat", -1)), str(payload.get("model", ""))
            )
        if method == "PUT" and path == f"/api/games/{table.code}/valuation":
            return self.set_valuation(table, seat, payload.get("valuation"))
        raise ApiError(f"no such endpoint: {method} {path}", status=404)
