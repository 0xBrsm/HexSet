"""Rankings the server shows but does not compute.

The server ranks nothing itself. `--ranker MODULE:FUNCTION` names a function
it calls with every ranked game its journals hold to an end; what that
function returns is the rankings, served as `GET /api/rankings`, the MCP
`rankings` tool and the page at `/rankings`. With no ranker there are no
rankings. Only a table the ranked queue dealt counts (`ranked.py`); a
casual one never reaches the ranker.

## The ranker

`rank(games: list[dict]) -> dict`, called on a thread of its own: once when
the server starts, over every ranked game in the journal directory, and
again after each one that ends. `games` is every ranked game that ended,
played out or timed out, oldest first by when it ended, each as
`finished_game` reads it; what a timeout is worth is the ranker's to say. The ranker gets
its own copy and keeps no state between calls, so a change to it takes
effect from a restart, over the whole history. It returns
`{"boards": [...]}`, each board a `key`, a `title`, an optional `note`, its
`columns` (`{"key", "label"}`) and its `rows` in the order shown, each row a
`name`, an optional `id` and `games`, and a value for any of the columns.
`games` is how many games it played; a row under the server's minimum games
is not shown.

## Profiles

A client id is the identity; the rest of what a client says about itself,
its `name`, `version`, `owner`, `url` and `description`
(`constants.PROFILE_FIELDS`), is whatever the latest request carrying that id
sent (`seen`), and its `name` whatever a seat it holds was renamed to since
(`renamed`), read back from the journals at start. A row whose `id` has a
profile is served with it in place of anything the ranker wrote there, its
`name` too when the profile has one; a profile changing re-serves the board
without ranking again. A server bot has no client: a row whose `id` is
`bot:<entry>` (`BOT_ID`) carries the profile its preset or its model file
declares (`api.bot_profile`), the version the server runs now.

A ranker that raises, or returns something `check` refuses, is logged and
the last good rankings stay up. A game is handed over as its journal
holds it, client addresses included; what is served is only each row's
`name`, `id`, `games`, profile and declared columns, so anything else a
row carries is dropped rather than published.
"""

from __future__ import annotations

import copy
import importlib
import math
import queue
import threading
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from . import _journal as journal
from .constants import PROFILE_FIELDS, profile_value

__all__ = [
    "BOT_ID",
    "MIN_GAMES",
    "Rankings",
    "Ranker",
    "check",
    "finished_game",
    "finished_games",
    "load_ranker",
    "profiles",
]

Ranker = Callable[[list[dict]], dict]

# A row whose `id` is this and a bot's entry name stands for that bot, and is
# served with the profile its preset or model file declares (`api.bot_profile`).
BOT_ID = "bot:"

# Rows that played fewer games than this are not shown, by default.
MIN_GAMES = 10

# Fields a row may carry beyond its columns: its identity, then what its
# client says of itself.
_ROW_FIELDS = ("name", "id", "games", "pinned", *PROFILE_FIELDS[1:])

# What a row's value may be: what a table cell can show.
_SCALARS = (str, int, float, bool, type(None))

# The queue item that stops the worker.
_STOP = object()


def load_ranker(spec: str) -> Ranker:
    """The function `spec` names as `MODULE:FUNCTION`, imported now so a
    misnamed ranker stops the server at start rather than failing quietly
    later."""
    module_name, colon, name = spec.partition(":")
    if not colon or not module_name or not name:
        raise ValueError(f"a ranker is named MODULE:FUNCTION, not {spec!r}")
    ranker = getattr(importlib.import_module(module_name), name, None)
    if not callable(ranker):
        raise ValueError(f"{module_name} has no function {name}")
    return ranker


def finished_game(events: list[dict]) -> dict | None:
    """One journal's game as a ranker reads it, or `None` for a game that
    neither played out to a result nor timed out.

    `id`, `code`, `game_type`, `board_mode`, `ranked`, `dealt_at`,
    `ended_at`, `turns` and `winner` (a seat, `None` for a timeout) come from
    the header and the closing line. `ended` is `"finished"`, or `"timeout"`
    for a ranked game aborted on its clock, `timed_out` then the seats it was
    waiting on (`_journal.Journal.timed_out`). `imported` is a game this server
    did not deal but filed from a record (`_journal.journal_of`). `seats` holds
    every seat that took part, a seat
    closed before the first move being none: its `seat`, `name`, `points`
    (true victory points at the end), `won`, `left` (retired part-way through,
    its pieces still counting), and who held it at the end, either `bot`
    (`{"name", "spec"}`, the entry it was seated as, and the `version` its
    profile declared then) with `swapped` saying
    another bot held it after play began, or `client`, the person's or
    model's client record as journalled (`id`, `kind`, and where it came
    from), `None` when it sent none."""
    if not events or events[0].get("kind") != "game":
        return None
    result = next((e for e in reversed(events) if e.get("kind") == "result"
                   or (e.get("kind") == "abandoned" and e.get("reason") == "timeout")), None)
    if result is None:
        return None
    timed_out = result.get("kind") == "abandoned"
    header = events[0]
    bots = _bots(events)
    people = journal.players(events)
    clients = journal.clients(events)
    retired = journal.retirements(events)
    closed = set(header.get("locked") or ()) | set(retired.get(0, ()))
    left = {seat for step, seats in retired.items() if step > 0 for seat in seats}
    swapped = _swapped(events)
    points = result.get("points") or []
    seats = []
    for seat in range(int(header.get("num_players", len(points)))):
        if seat in closed:
            continue
        bot = bots.get(seat)
        seats.append({
            "seat": seat,
            "name": bot["name"] if bot else people.get(seat, ""),
            "points": points[seat] if seat < len(points) else 0,
            "won": seat == result.get("winner"),
            "left": seat in left,
            "bot": bot,
            "swapped": seat in swapped,
            "client": None if bot else clients.get(seat),
        })
    return {
        "id": header.get("id"),
        "code": header.get("code"),
        "game_type": header.get("game_type", "standard"),
        "board_mode": header.get("board_mode"),
        "ranked": bool(header.get("ranked")),
        "imported": header.get("seed") is None,
        "dealt_at": header.get("at"),
        "ended_at": result.get("at"),
        "ended": "timeout" if timed_out else "finished",
        "timed_out": list(result.get("seats") or []) if timed_out else [],
        "turns": result.get("turns"),
        "winner": None if timed_out else result.get("winner"),
        "seats": seats,
    }


def _bots(events: list[dict]) -> dict[int, dict]:
    """Seat -> the record of the bot on it at the end (`name`, `spec`, and a
    `version` where one was journalled): `_journal.seating`'s fold, keeping
    the whole record."""
    header = events[0] if events else {}
    seats = {int(seat): dict(bot) for seat, bot in (header.get("bots") or {}).items()}
    for event in events:
        if event.get("kind") == "seated":
            if event.get("spec"):
                seats[event["seat"]] = {k: event[k] for k in ("name", "spec", "version") if k in event}
            else:
                seats.pop(event["seat"], None)
        elif event.get("kind") == "unseated":
            seats.pop(event["seat"], None)
    return seats


def _swapped(events: list[dict]) -> set[int]:
    """The seats a bot was seated on after the first move, replacing the
    one there; steps counted as `_journal.retirements` counts them, so a
    swap undone back to the start is a swap before play."""
    seats: set[int] = set()
    steps = 0
    for event in events:
        kind = event.get("kind")
        if kind in ("action", "trade"):
            steps += 1
        elif kind == "undo":
            steps = min(steps, int(event["back_to"]))
        elif kind == "seated" and event.get("spec") and steps > 0:
            seats.add(int(event["seat"]))
    return seats


def finished_games(directory: str | Path | None) -> list[dict]:
    """Every game journalled in `directory` that ended, played out or timed
    out, oldest first by when it ended (`finished_game`). A file that never
    mentions either is passed over unparsed."""
    if not directory:
        return []
    try:
        paths = sorted(Path(directory).glob("*.jsonl"))
    except OSError:
        return []
    games = []
    for path in paths:
        try:
            raw = path.read_bytes()
            if b'"result"' not in raw and b'"timeout"' not in raw:
                continue
        except OSError:
            continue
        game = finished_game(journal.read(path))
        if game is not None:
            games.append(game)
    return sorted(games, key=_finish_order)


def profiles(directory: str | Path | None) -> dict[str, dict]:
    """Client id -> the profile the latest claim journalled in `directory`
    carried (`constants.PROFILE_FIELDS`, only those it sent), its name as any
    later rename of a seat it held left it, with `at`, when that was. Every
    journal counts, finished or not."""
    found: dict[str, dict] = {}
    if not directory:
        return found
    try:
        paths = sorted(Path(directory).glob("*.jsonl"))
    except OSError:
        return found
    for path in paths:
        events = journal.read(path)
        if not events or events[0].get("kind") != "game":
            continue
        at = str(events[0].get("at") or "")
        held = {int(seat): client for seat, client in (events[0].get("clients") or {}).items() if client}
        for client in held.values():
            _note(found, client, at)
        for event in events:
            kind, when = event.get("kind"), str(event.get("at") or "")
            if kind == "seated" and event.get("client"):
                held[int(event["seat"])] = event["client"]
                _note(found, event["client"], when)
            elif kind == "unseated":
                held.pop(int(event["seat"]), None)
            elif kind == "renamed" and (held.get(int(event["seat"])) or {}).get("id"):
                _rename(found, held[int(event["seat"])]["id"], event.get("name") or None, when)
    return found


def _note(found: dict[str, dict], client: dict | None, at: str) -> bool:
    """Make `client`'s profile its id's in `found`, unless one later than
    `at` is there; whether that changed anything."""
    if not client or not client.get("id"):
        return False
    known = found.get(client["id"])
    if known is not None and known["at"] > at:
        return False
    profile = {field: client[field] for field in PROFILE_FIELDS if client.get(field)}
    if known is not None and {k: v for k, v in known.items() if k != "at"} == profile:
        known["at"] = at
        return False
    found[client["id"]] = {**profile, "at": at}
    return True


def _rename(found: dict[str, dict], client_id: str, name: str | None, at: str) -> bool:
    """Give `client_id`'s profile in `found` the name `name` (none clears
    it), the rest kept, unless a later one is there; whether that changed
    anything."""
    known = found.get(client_id)
    if known is not None and known["at"] > at:
        return False
    profile = {k: v for k, v in (known or {}).items() if k not in ("at", "name")}
    if name:
        profile["name"] = name
    changed = known is None or {k: v for k, v in known.items() if k != "at"} != profile
    found[client_id] = {**profile, "at": at}
    return changed


def _finish_order(game: dict) -> tuple[str, str]:
    return str(game.get("ended_at") or ""), str(game.get("id") or "")


def check(value: object, min_games: int = 0) -> list[dict]:
    """A ranker's return value as served: its boards with every row cut to
    `name`, `id`, `games`, its profile and the board's own columns, less the rows
    whose `games` is under `min_games` (a row that says nothing of its games
    stays). Raises `ValueError` naming the first thing that is not a
    set of rankings."""
    if not isinstance(value, dict) or not isinstance(value.get("boards"), list):
        raise ValueError('a ranker returns {"boards": [...]}')
    boards = []
    keys: set[str] = set()
    for b, board in enumerate(value["boards"]):
        where = f"boards[{b}]"
        if not isinstance(board, dict):
            raise ValueError(f"{where} is not an object")
        key = board.get("key")
        if not isinstance(key, str) or not key:
            raise ValueError(f"{where}.key is not a name")
        if key in keys:
            raise ValueError(f"two boards are keyed {key!r}")
        keys.add(key)
        if not isinstance(board.get("title"), str):
            raise ValueError(f"{where}.title is not text")
        note = board.get("note")
        if note is not None and not isinstance(note, str):
            raise ValueError(f"{where}.note is not text")
        columns = board.get("columns")
        if not isinstance(columns, list):
            raise ValueError(f"{where}.columns is not a list")
        for c, column in enumerate(columns):
            if not (isinstance(column, dict) and isinstance(column.get("key"), str)
                    and isinstance(column.get("label"), str)):
                raise ValueError(f'{where}.columns[{c}] is not {{"key", "label"}}')
        named = [column["key"] for column in columns]
        if len(set(named)) != len(named) or {"name", "id"} & set(named):
            raise ValueError(f"{where}.columns repeat a key, or name one name or id")
        # `games` may also be shown as a column; its check below holds.
        rows = board.get("rows")
        if not isinstance(rows, list):
            raise ValueError(f"{where}.rows is not a list")
        served = []
        for r, row in enumerate(rows):
            if not isinstance(row, dict) or not isinstance(row.get("name"), str):
                raise ValueError(f"{where}.rows[{r}] has no name")
            if row.get("id") is not None and not isinstance(row["id"], str):
                raise ValueError(f"{where}.rows[{r}].id is not text")
            games = row.get("games")
            if games is not None and (type(games) is not int or games < 0):
                raise ValueError(f"{where}.rows[{r}].games is not a count")
            if games is not None and games < min_games:
                continue
            cut = {"name": row["name"], "id": row.get("id"), "games": games}
            # A rating the ranker holds fixed, a scale's reference: served only when set.
            if row.get("pinned") is not None and not isinstance(row["pinned"], bool):
                raise ValueError(f"{where}.rows[{r}].pinned is not true or false")
            if row.get("pinned"):
                cut["pinned"] = True
            for field in PROFILE_FIELDS[1:]:
                text = row.get(field)
                try:
                    cut[field] = None if text is None else profile_value(field, text)
                except ValueError as error:
                    raise ValueError(f"{where}.rows[{r}].{error}") from None
            for name in named:
                if name in _ROW_FIELDS:
                    continue
                cell = row.get(name)
                if not isinstance(cell, _SCALARS) or (isinstance(cell, float) and not math.isfinite(cell)):
                    raise ValueError(f"{where}.rows[{r}].{name} is not a number, text, true/false or null")
                cut[name] = cell
            served.append(cut)
        boards.append({
            "key": key,
            "title": board["title"],
            "note": note,
            "columns": [{"key": c["key"], "label": c["label"]} for c in columns],
            "rows": served,
        })
    return boards


class Rankings:
    """The server's rankings: the ranked games read so far, and the
    last good board `ranker` made of them. Every read and every call to the
    ranker is on one worker thread, so a game finishing never waits on
    either; `view` is the board as it stands."""

    def __init__(self, ranker: Ranker, directory: str | None, min_games: int = MIN_GAMES) -> None:
        self._ranker = ranker
        self._directory = directory
        self._min_games = min_games
        self._games: dict[str, dict] = {}
        self._profiles: dict[str, dict] = {}
        # The last good ranking, before profiles: (updated, games, boards).
        self._ranked: tuple[str, int, list[dict]] | None = None
        self._board: dict = {"updated": None, "games": 0, "min_games": min_games, "boards": []}
        self._lock = threading.Lock()
        self._queue: queue.Queue = queue.Queue()
        # `None` reads the whole directory: the worker's first job.
        self._queue.put(None)
        self._thread = threading.Thread(target=self._work, name="rankings", daemon=True)
        self._thread.start()

    def ended(self, path: Path) -> None:
        """A game just wrote its ending to `path` (`_journal.Journal.on_end`):
        read it and, a ranked one, rank again."""
        self._queue.put(Path(path))

    def seen(self, client: dict) -> None:
        """A client just claimed a seat (`api.Tables`): what it sent is now
        its id's profile."""
        self._queue.put(("seen", dict(client), _now()))

    def renamed(self, client_id: str, name: str | None) -> None:
        """A seat `client_id` holds was renamed (`api.Tables.rename`): its
        profile goes by `name` now, none clearing it."""
        self._queue.put(("renamed", client_id, name, _now()))

    def view(self) -> dict:
        """`{"updated", "games", "min_games", "boards"}`: when the board was
        ranked (null before the first ranking), over how many finished games,
        the fewest games a row shown has played, and the boards `check`
        passed. Not to be modified."""
        with self._lock:
            return self._board

    def wait(self) -> None:
        """Block until everything handed in so far is read and ranked."""
        self._queue.join()

    def close(self) -> None:
        """Stop the worker once what it already holds is done."""
        self._queue.put(_STOP)
        self._thread.join(timeout=5)

    def _work(self) -> None:
        dirty = restyled = False
        while True:
            item = self._queue.get()
            try:
                if item is _STOP:
                    return
                if isinstance(item, tuple) and item[0] == "renamed":
                    restyled = _rename(self._profiles, *item[1:]) or restyled
                elif isinstance(item, tuple):
                    restyled = _note(self._profiles, item[1], item[2]) or restyled
                else:
                    dirty = self._take(item) or dirty
                # Several games finishing together are ranked once, and a
                # profile changing re-serves the last ranking as it stands.
                if self._queue.empty():
                    if dirty:
                        self._publish()
                    if dirty or restyled:
                        self._serve()
                    dirty = restyled = False
            except Exception:
                traceback.print_exc()
            finally:
                self._queue.task_done()

    def _take(self, item: Path | None) -> bool:
        if item is None:
            for game in finished_games(self._directory):
                if game["ranked"]:
                    self._games[str(game["id"])] = game
            for client_id, profile in profiles(self._directory).items():
                _note(self._profiles, {"id": client_id, **profile}, profile["at"])
            return True
        game = finished_game(journal.read(item))
        if game is None or not game["ranked"] or str(game["id"]) in self._games:
            return False
        self._games[str(game["id"])] = game
        return True

    def _publish(self) -> None:
        games = sorted(self._games.values(), key=_finish_order)
        try:
            boards = check(self._ranker(copy.deepcopy(games)), self._min_games)
        except Exception:
            print("rankings: the ranker failed; the last good board stays up")
            traceback.print_exc()
            return
        self._ranked = (_now(), len(games), boards)

    def _serve(self) -> None:
        """The last ranking, each row with an id carrying that id's profile."""
        if self._ranked is None:
            return
        updated, games, ranked = self._ranked
        boards = [{**board, "rows": [self._profiled(row) for row in board["rows"]]} for board in ranked]
        board = {"updated": updated, "games": games, "min_games": self._min_games, "boards": boards}
        with self._lock:
            self._board = board

    def _profiled(self, row: dict) -> dict:
        if row["id"] and row["id"].startswith(BOT_ID):
            from .api import bot_profile  # local: `api` imports this module

            profile = bot_profile(row["id"][len(BOT_ID):])
            return {**row, **{field: profile.get(field) for field in PROFILE_FIELDS[1:]}} if profile else row
        profile = self._profiles.get(row["id"]) if row["id"] else None
        if profile is None:
            return row
        return {
            **row,
            "name": profile.get("name") or row["name"],
            **{field: profile.get(field) for field in PROFILE_FIELDS[1:]},
        }


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
