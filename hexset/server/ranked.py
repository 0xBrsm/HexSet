"""Ranked play: a queue the server deals tables from, for clients that do not
choose their opponents.

Only a ranked game reaches the rankings (`rankings.py`). A client of
the HTTP API or a model over MCP joins a game type's queue (`POST
/api/ranked`, MCP `ranked_game`) and is handed a seat once matched; nobody
picks the table, the board or the seat order. Casual tables
(`POST /api/games`) are untouched and never count.

## Matching

Every `MATCH_SECONDS`, for each game type, the waiting tickets are taken
oldest first. Each is grouped with the waiting clients it has met least in
ranked games so far, no client id twice at one table, until the table is
full. One seat still empty goes to the server's fill bot
(`Config.ranked_bot`), and no more: like any player it sits once at a
table, so a table short by two or more waits for players.

Two players, each an id at one `version` of its profile (`entrant`), meet
at no more than `PAIR_CAP` ranked tables: the fill bot and the server's
entrants count as players like any client, so a new version of either side
meets the other afresh. A client may hold any number of tickets, and so
play any number of ranked games at once, within its address's
unfinished-games limit (`api.Tables.admit_ranked`); ranked play counts
against no game allowance.

## Server bots as players

A server bot named in `Config.ranked_entrants` plays ranked games as a
client does, under the id `bot:<entry>` and its profile's version
(`api.bot_profile`): before each pass the queue gives it a ticket at every
game type it is offered at (`api.model_catalogue`) where it has none waiting
and no ranked game in play, so it plays one game of each type at a time. Its
tickets are matched like any other, under the same pair cap; the journal
names its seats (`entrants`), so only the other bot seats are fill.

## The clock

A ranked table that goes `TURN_SECONDS` without a change is aborted
(`api.Tables.check_clocks`): the journal's closing line names the seats it
was waiting on and every seat's points, and what that is worth is the
ranker's call.
"""

from __future__ import annotations

import random
import secrets
import threading
import time
import traceback
from collections import Counter
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

from hexset.rules import GAME_TYPES

from . import _journal as journal

__all__ = [
    "PAIR_CAP",
    "FILL_SEATS",
    "MATCH_SECONDS",
    "TICKET_SECONDS",
    "TURN_SECONDS",
    "RankedQueue",
    "Ticket",
    "entrant",
]

# How often the queue is matched and the ranked tables' clocks read.
MATCH_SECONDS = 10.0

# How long a ranked table may wait on one move before it is aborted.
TURN_SECONDS = 60.0

# How long a waiting ticket lives unread: a client that stopped asking is not
# seated. A matched one is kept this many times longer, for the client to read.
TICKET_SECONDS = 60.0
MATCHED_KEPT = 10

# Ranked tables two players, each an id at one version, may share.
PAIR_CAP = 100

# Seats the fill bot takes at one table: one, as any player sits once.
FILL_SEATS = 1


def entrant(client: dict) -> tuple[str, str | None]:
    """The pair cap's unit: a client id at one version of its profile."""
    return client["id"], client.get("version")


def _bot_entrant(bot: dict) -> tuple[str, str | None]:
    """A bot seat's pair-cap unit, from its journal entry (`name`, `version`)."""
    return f"bot:{bot['name']}", bot.get("version")


def _player(seat_client: dict | None, bot: dict | None) -> str | None:
    """Who a ranked seat is for the meeting counts: a client by its id, a bot
    by its entry name."""
    if bot is not None:
        return f"bot:{bot['name']}"
    if seat_client and seat_client.get("id"):
        return seat_client["id"]
    return None


@dataclass
class Ticket:
    """One client waiting for a ranked game of `game_type`, until `match`
    holds the table it was seated at (`code`, `seat`, `token`). `origin` is
    the address it queued from, which its table counts toward that
    address's unfinished games."""

    id: str
    game_type: str
    client: dict
    origin: object = None
    seen: float = field(default_factory=time.monotonic)
    match: dict | None = None
    # The server bot this ticket seats (`Config.ranked_entrants`), whose
    # `client` is only its id and version; None for a client's.
    bot: str | None = None


class RankedQueue:
    """Every waiting ticket, and the ranked history matching reads: how often
    each pair of players has met, and how many fill-bot tables each entrant
    has sat at. Its worker starts with the first ticket and runs `tick` every
    `MATCH_SECONDS`."""

    def __init__(self, tables) -> None:
        self.tables = tables
        self._tickets: dict[str, Ticket] = {}
        self._changed = threading.Condition()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._meetings: Counter | None = None
        self._pairs: Counter | None = None

    # --- clients ------------------------------------------------------------

    def join(self, game_type: str, client: dict, origin=None) -> Ticket:
        """A new ticket for `client`, already admitted (`api.Tables.admit_ranked`)."""
        ticket = Ticket(secrets.token_urlsafe(18), game_type, dict(client), origin)
        with self._changed:
            self._tickets[ticket.id] = ticket
        self.start()
        return ticket

    def wait(self, ticket_id: str, timeout: float) -> Ticket | None:
        """The ticket once matched, or as it stands after `timeout` seconds;
        `None` for one that is not (or no longer) waiting. Reading it keeps
        it in the queue."""
        deadline = time.monotonic() + timeout
        with self._changed:
            while True:
                ticket = self._tickets.get(ticket_id)
                if ticket is None:
                    return None
                ticket.seen = time.monotonic()
                remaining = deadline - time.monotonic()
                if ticket.match is not None or remaining <= 0:
                    return ticket
                self._changed.wait(remaining)

    def leave(self, ticket_id: str) -> bool:
        """Takes a ticket out of the queue; whether it was still waiting."""
        with self._changed:
            ticket = self._tickets.get(ticket_id)
            if ticket is None or ticket.match is not None:
                return False
            del self._tickets[ticket_id]
        return True

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    # --- the worker -----------------------------------------------------------

    def start(self) -> None:
        """Starts the worker, if it is not running: the first ticket does, and
        a server with entrants (`Config.ranked_entrants`) at start."""
        with self._changed:
            if self._thread is not None:
                return
            self._thread = threading.Thread(target=self._run, name="ranked", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(MATCH_SECONDS):
            try:
                self.tick()
            except Exception:
                traceback.print_exc()

    def tick(self) -> None:
        """One pass: forget tickets nobody reads, queue the server's
        entrants, deal what can be matched, and abort the ranked tables whose
        clock has run out."""
        self._expire()
        self._enter()
        for game_type in GAME_TYPES:
            self._match(game_type)
        self.tables.check_clocks()

    def _enter(self) -> None:
        """A ticket for each server entrant at each game type it is offered
        at, where it has none waiting and no ranked game there in play."""
        entries = self.tables.config.ranked_entrants
        if not entries:
            return
        from .api import bot_profile, model_catalogue  # local: `api` imports this module

        catalogue = model_catalogue()
        with self._changed:
            waiting = {(t.bot, t.game_type) for t in self._tickets.values() if t.bot and t.match is None}
        for entry in entries:
            version = bot_profile(entry).get("version")
            for game_type in GAME_TYPES:
                if entry not in catalogue.get(game_type, {}) or (entry, game_type) in waiting:
                    continue
                if self.tables.in_ranked_play(entry, game_type):
                    continue
                client = {"id": f"bot:{entry}", **({"version": version} if version else {})}
                ticket = Ticket(secrets.token_urlsafe(18), game_type, client, bot=entry)
                with self._changed:
                    self._tickets[ticket.id] = ticket

    def _expire(self) -> None:
        now = time.monotonic()
        with self._changed:
            # An entrant's ticket has no reader: it waits until matched, and
            # goes once it is.
            gone = [t for t in self._tickets.values()
                    if (t.match is not None if t.bot else
                        now - t.seen > TICKET_SECONDS * (MATCHED_KEPT if t.match else 1))]
            for ticket in gone:
                del self._tickets[ticket.id]

    # --- matching -------------------------------------------------------------

    def _match(self, game_type: str) -> None:
        seats = max(GAME_TYPES[game_type].seats)
        bot = self.tables.config.ranked_bot
        meetings, pairs = self._history()
        filler = self._filler(bot)
        with self._changed:
            waiting = [t for t in self._tickets.values() if t.match is None and t.game_type == game_type]
        taken: set[str] = set()
        for first in waiting:
            if first.id in taken:
                continue
            group = [first]
            pool = [t for t in waiting if t.id not in taken and t is not first]
            while len(group) < seats:
                ids = {t.client["id"] for t in group}
                pool = [t for t in pool if t.client["id"] not in ids
                        and all(pairs[frozenset((entrant(t.client), entrant(g.client)))] < PAIR_CAP
                                for g in group)]
                if not pool:
                    break
                pick = min(pool, key=lambda t: (
                    sum(meetings[frozenset((t.client["id"], g.client["id"]))] for g in group), random.random()))
                group.append(pick)
                pool.remove(pick)
            if len(group) < seats:
                if filler is None:
                    continue
                group = [t for t in group if pairs[frozenset((entrant(t.client), filler))] < PAIR_CAP]
                if first not in group or seats - len(group) > FILL_SEATS:
                    continue
            fill = seats - len(group)
            try:
                dealt = self.tables.deal_ranked(game_type, [t.bot or t.client for t in group], bot, fill,
                                                [t.origin for t in group])
            except Exception:
                traceback.print_exc()
                continue
            self._count([t.client for t in group], [filler] * fill)
            with self._changed:
                for ticket, (seat, token) in zip(group, dealt["seats"]):
                    ticket.match = {"code": dealt["code"], "seat": seat, "token": token}
                    ticket.seen = time.monotonic()
                self._changed.notify_all()
            taken.update(t.id for t in group)

    def _filler(self, bot: str | None) -> tuple[str, str | None] | None:
        """The fill bot's pair-cap unit at the version it plays now, or None
        for no fill bot."""
        if bot is None:
            return None
        from .api import bot_profile  # local: `api` imports this module

        return _bot_entrant({"name": bot, "version": bot_profile(bot).get("version")})

    def _history(self) -> tuple[Counter, Counter]:
        """The meeting counts by id and by versioned pair (`entrant`), read
        from the ranked journals the first time they are asked for and kept
        up to date by `_count`."""
        if self._meetings is None:
            meetings, pairs = Counter(), Counter()
            for header in _ranked_headers(self.tables.journal_directory()):
                clients = {int(s): c for s, c in (header.get("clients") or {}).items() if c}
                bots = {int(s): b for s, b in (header.get("bots") or {}).items()}
                locked = set(header.get("locked") or ())
                seats = [s for s in range(int(header.get("num_players", 0))) if s not in locked]
                players = [_player(clients.get(s), bots.get(s)) for s in seats]
                versioned = [_bot_entrant(bots[s]) if s in bots
                             else entrant(clients[s]) if clients.get(s, {}).get("id") else None for s in seats]
                _add(meetings, pairs, players, versioned)
            self._meetings, self._pairs = meetings, pairs
        return self._meetings, self._pairs

    def _count(self, clients: list[dict], fill: list[tuple[str, str | None]]) -> None:
        players = [c["id"] for c in clients] + [unit[0] for unit in fill]
        _add(self._meetings, self._pairs, players, [entrant(c) for c in clients] + fill)


def _add(meetings: Counter, pairs: Counter, players: list[str | None],
         versioned: list[tuple[str, str | None] | None]) -> None:
    for a, b in combinations([p for p in players if p], 2):
        meetings[frozenset((a, b))] += 1
    for a, b in combinations([v for v in versioned if v and v[0]], 2):
        pairs[frozenset((a, b))] += 1


def _ranked_headers(directory: str | None) -> list[dict]:
    """The header of every ranked game journalled in `directory`."""
    if not directory:
        return []
    try:
        paths = sorted(Path(directory).glob("*.jsonl"))
    except OSError:
        return []
    return [h for h in (journal.header_of(p) for p in paths) if h and h.get("ranked")]
