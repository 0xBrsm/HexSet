"""Games, seats and the `/api/*` surface everything plays through.

Browser, HTTP script, LLM over MCP (`mcptools.py`), an outside bot
(`hexset.clients.botclient` is an example) and the server's own bot seats
(`_runner.py`) are all clients of this module and get no special treatment:
the same token-gated routes, the same `state`/`act` pair, and the same
`your_move` saying what each seat owes. `web.py` supplies the HTTP
transport; nothing about a game lives there.

## A game, an ID, a seat

A game's ID is a six-character code. There is no lobby: `POST /api/games`
deals a full `MAX_SEATS`-seat game immediately, the creator at one random seat
and every other seat open. The body may name the board mode and the game
type: a type played at fewer seats than that (the duel variant, at two) is
dealt with the seats past its largest count retired, and its seats are fixed
from the deal. `GET /<id>` or `POST /api/join` claims a random
still-open seat and returns a token; `POST /api/bot` gives one to a bot
instead, until the seat is filled or closed.

That token, not the request's source or a cookie, is the identity: it names
one seat at one game, it is the only way to act on that seat, and it is what
`state` reads to decide whose hand to show. It never touches disk
(`_journal.py`), so a restart cannot hand a lost token back; what a restart
does put back is who was sitting where, so a reopened table comes back with
every seat claimed and none tokened, and `POST /api/reclaim` trades a
returning client's own secret for a fresh token on its seat.

## A seat resolves before anyone moves

`MAX_SEATS` seats exist from the moment a game does, claimed or not. During
setup nobody moves while any seat is still `SeatKind.EMPTY` and unlocked
(`Table.waiting_for`, on every view as `waiting_for`): a seat resolves by a
person taking it, a bot being seated on it, or `POST /api/close` closing it,
and only once none is left does play start, from seat 0. A seat locks but is
never dropped, so a game's size is fixed once setup begins. There is no timer
anywhere: a turn advances only because the seat holding it says so.

## What a seat is told, and what it is not

Every response is built for one viewer; the turn's trade log is public to all
of them. Trading between seats is the trade round (`hexset.trading`;
`docs/onnx.md` §3): `POST /api/games/<code>/trade/round` broadcasts one offer
to every other seat, `.../trade/round/answer` is a seat's accept, counter or
pass on a broadcast its `pending` shows, and `.../trade/round/choose` is the
actor's pick among the answers, or a decline of them all. Every manual seat
is a `PendingGate`, so nothing is ever agreed on its behalf; while a bot's
broadcast waits on one, the table holds that bot's turn (`trade_wait`,
`to_move` None).
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import random
import re
import secrets
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs

from hexset.actions import build_space, legal_actions
from hexset.arena import BOT_PROFILE_FIELDS, PRESETS, preset_profile, registered_presets, spawn as spawn_entrant
from hexset.board.board import BOARD_MODES, Board, base_board
from hexset.bots import Bot
from hexset.chance import for_rules
from hexset.clients.onnxmeta import PROFILE_KEYS, modes_of, read_metadata
from hexset.game import Game, Phase, is_over, lock_seat, may_act, pending_free_roads, start
from hexset.onnx_record import record_from_game
from hexset.record import from_events, open_record
from hexset.rules import GAME_TYPES, STANDARD_GAME, GameType, rules_from
from hexset.trading import holds

from . import _journal as journal
from ._runner import BotRunner, LocalSearchBrain, LocalTransport
from ._seating import SETUP_PHASES, unlock_seat
from ._webplay import GameSession, ResumeError
from .constants import PROFILE_FIELDS, profile_value
from .rankings import MIN_GAMES as RANKINGS_MIN_GAMES, Rankings
from .ranked import TURN_SECONDS, RankedQueue
from .wire import action_to_wire, board_layout, round_offer_from_wire, signed_bundle_from_wire

__all__ = [
    "API_VERSION",
    "API_DOCS",
    "MAX_SEATS",
    "CODE_ALPHABET",
    "CODE_LENGTH",
    "MAX_NAME_LENGTH",
    "PROFILE_FIELDS",
    "REQUIRED_PROFILE",
    "TABLE_TTL_SECONDS",
    "MAX_WAIT_SECONDS",
    "CLIENT_KINDS",
    "MODELS_DIR",
    "ApiError",
    "Config",
    "client_kind",
    "LIVE_IDLE_SECONDS",
    "Origin",
    "Seat",
    "SeatKind",
    "Table",
    "Tables",
    "default_models_dir",
    "listed_models",
    "model_catalogue",
    "model_options",
    "spawn_bot",
    "your_move",
]


# The version of *this API*, served by `GET /api/version`. Bumped when a client
# written against the previous contract can break (a route removed or renamed,
# a request field no longer accepted, a response shape changed); additive work
# leaves it alone. Not `hexset.__version__`, a checkpoint's `contract`, or the
# per-table `version` an acting route carries.
API_VERSION = 4

# Where a caller who lands on the bare `GET /api` finds the routes: this API's
# reference in the published repository.
API_DOCS = "https://github.com/0xBrsm/HexSet/blob/main/docs/api.md"


def your_move(view: dict) -> tuple[str, list[int]]:
    """`(your_move, waiting_on)` for one seat's view: what the table wants
    from that seat now, the first of these that holds, and while that is
    `"wait"`, the seats it waits on (empty when it waits on nobody in
    particular).

    `game_over`; `discard`, a seven's discard this seat owes, whoever is
    `to_move`; `answer_trade`, an offer in its `pending`; `choose_trade`, its
    own round with every answer in; `act`, an action from `legal_actions`,
    only while the seat is `to_move`; else `wait`. `legal_actions` can list
    moves `POST /api/action` refuses: while open seats hold up setup, while a
    bot's offer waits on a person, and while another seat holds its setup
    turn open. A seat whose own round still awaits answers may act, and the
    round closes with its turn."""
    if view.get("game_over"):
        return "game_over", []
    seat = view.get("seat")
    legal = view.get("legal_actions") or []
    if legal and view.get("phase") == Phase.DISCARD.name:
        return "discard", []
    if view.get("pending"):
        return "answer_trade", []
    trade_round = view.get("trade_round")
    if trade_round is not None and not trade_round.get("awaiting"):
        return "choose_trade", []
    if legal and seat is not None and view.get("to_move") == seat:
        return "act", []
    if trade_round is not None:
        return "wait", list(trade_round.get("awaiting") or [])
    for held in (view.get("trade_wait"), view.get("waiting_for")):
        if held:
            return "wait", list(held)
    owing = [s for s, n in enumerate(view.get("discard_quota") or []) if n]
    if owing:
        return "wait", owing
    to_move = view.get("to_move")
    return "wait", [] if to_move is None else [to_move]


def default_models_dir() -> Path:
    """Where the model picker looks when `HEXSET_UI_MODELS_DIR` names
    nothing: `models/` at the root of the source tree this package was loaded
    from (the directory holding `pyproject.toml` beside `hexset/`), or, for an
    installed package, `models/` under the working directory."""
    root = Path(__file__).resolve().parents[2]
    if (root / "pyproject.toml").is_file():
        return root / "models"
    return Path.cwd() / "models"


MODELS_DIR = Path(os.environ.get("HEXSET_UI_MODELS_DIR") or default_models_dir())

# The base board seats four, so a game does too, whatever its claimed seats: a
# served checkpoint hard-rejects a mismatched player count.
MAX_SEATS = 4

# Digits and lowercase letters, minus 0/o and 1/i/l. Case is not part of a
# code's identity: every lookup normalises, so capitals still work.
CODE_ALPHABET = "23456789abcdefghjkmnpqrstuvwxyz"
CODE_LENGTH = 6

# The display-name cap, enforced here because a raw POST bypasses clients.
MAX_NAME_LENGTH = 40

# What a client of the HTTP API must say of itself to play (`require_profile`):
# all of its profile but the `url`.
REQUIRED_PROFILE = ("name", "version", "owner", "description")

# How long a game survives with nobody touching it.
TABLE_TTL_SECONDS = 24 * 60 * 60

# How long a code with no game behind it is remembered as one, so a burst of
# reads for it scans the journal directory once.
MISS_SECONDS = 5.0

# The cap on a read's `wait`: the longest it may park for a change.
MAX_WAIT_SECONDS = 25.0

# Sent with a new game when its caller already has another in play.
PARALLEL_NOTICE = "tables share one CPU core; parallel games each run slower"

# How long an unfinished table counts against its creator's live-table limit
# after the last time a person read it. An abandoned table stops counting this
# long after it was left, not when `TABLE_TTL_SECONDS` evicts it.
LIVE_IDLE_SECONDS = 15 * 60

# The longest `User-Agent` a journal keeps.
MAX_USER_AGENT_LENGTH = 200


class ApiError(Exception):
    """A request that cannot be served, carrying the HTTP status to answer
    with — the same status over HTTP, over MCP and in-process."""

    def __init__(
        self, message: str, status: int = 400, *,
        fields: dict | None = None, retry_after: int | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        # Machine-readable detail beside `error` in the body, and a
        # `Retry-After` in seconds, where the refusal has them.
        self.fields = fields or {}
        self.retry_after = retry_after


def require_live(game: Game) -> None:
    """Refuses anything that would change a game already over: 409, a conflict
    with the state of the thing rather than a malformed request. Deliberately
    not a blanket gate in `Tables.handle` — dealing a new game and reading a
    finished one both stay allowed."""
    if is_over(game):
        raise ApiError("the game is already over", status=409)


def new_code(taken: set[str]) -> str:
    """A fresh game code, re-rolling until it is not one of `taken`."""
    rng = random.SystemRandom()
    while True:
        code = "".join(rng.choice(CODE_ALPHABET) for _ in range(CODE_LENGTH))
        if code not in taken:
            return code


def clean_name(name: object) -> str | None:
    """A display name, trimmed and capped the same way everywhere one is
    accepted, or `None` if there's nothing left of it once trimmed. Anything
    but a string or `None` is a 400."""
    if name is None:
        return None
    if not isinstance(name, str):
        raise ApiError("a name is a string")
    name = name.strip()[:MAX_NAME_LENGTH]
    return name or None


def seat_field(payload: dict, key: str = "seat") -> int:
    """A seat number out of a request body: an integer, or a 400. Whether
    the table has that seat is the route's own check."""
    value = payload.get(key)
    if type(value) is not int:
        raise ApiError(f"`{key}` is a seat number")
    return value


# What a client is on the wire: a hash of a secret it alone holds, used to
# correlate games in the journal and to reclaim a seat after a token is gone.
# Not an account: seat theft via a leaked secret is accepted. Its kind is the
# server's call, never the client's (`client_kind`): the page's own requests
# are "web", the MCP endpoint's "mcp", anything else "api".
CLIENT_KINDS = ("web", "api", "mcp")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")

# The name a claimed seat gets when nobody sent one, so `player_names` always
# holds something.
_DEFAULT_SEAT_NAME = {"web": "human", "api": "api", "mcp": "mcp"}


def default_seat_name(kind: str) -> str:
    """The fallback display name for a seat claimed by a `kind` of client."""
    return _DEFAULT_SEAT_NAME.get(kind, "api")


@dataclass(frozen=True)
class Origin:
    """Where a request came from, as the transport saw it: `ip` the client's
    address (behind a proxy, the one its forwarding header carried), `via` the
    route it actually arrived on (`"http"` for `/api/*`, `"mcp"` for a tool
    call), its `User-Agent`, the API key it sent, if any, and `page`, whether
    the browser marked it as sent by this server's own page (`web.py`). In-process
    callers (the server's own bot seats, tests) pass none, and are neither
    stamped nor limited."""

    ip: str | None = None
    via: str = "http"
    user_agent: str | None = None
    key: str | None = None
    page: bool = False


def client_kind(origin: Origin | None) -> str:
    """What claimed a seat, decided from where the request came from: the
    MCP endpoint is `"mcp"`, a request the browser marked as this server's
    own page is `"web"`, and anything else, an in-process caller included,
    is `"api"`. The page's marks can be forged by a client set on it, as any
    header can; nothing a client sends in its body is asked."""
    if origin is None:
        return "api"
    if origin.via == "mcp":
        return "mcp"
    return "web" if origin.page else "api"


def stamp_client(client: dict, origin: Origin | None, holder: str | None = None) -> dict:
    """`client` (`parse_client`'s) as the journal keeps it: its `kind`
    (`client_kind`), plus `via`, `ip` and `ua` from `origin`, and `key`, the
    name of the `holder` of the API key it dealt with (never the key)."""
    kind = client_kind(origin)
    if origin is None:
        return {**client, "kind": kind}
    ua = (origin.user_agent or "")[:MAX_USER_AGENT_LENGTH] or None
    stamped = {**client, "kind": kind, "via": origin.via, "ip": origin.ip, "ua": ua}
    return {**stamped, "key": holder} if holder else stamped


@dataclass(frozen=True)
class ApiKey:
    """An API key's holder, as the journal names them, and the new games the
    key may deal per `Config.period_days`, in place of an address's
    `Config.games_per_period` (0 is no limit)."""

    name: str
    games_per_period: int


def load_api_keys(path: str) -> dict[str, ApiKey]:
    """The keys file: `{"<key>": {"name": "<holder>", "games_per_period":
    <n>}}`, every name distinct."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    keys = {key: ApiKey(name=str(entry["name"]), games_per_period=int(entry["games_per_period"]))
            for key, entry in raw.items()}
    names = [k.name for k in keys.values()]
    if len(set(names)) != len(names):
        raise ValueError(f"{path}: two keys share a holder's name")
    return keys


def _allowance(ip: str, holder: str | None) -> str:
    """Whose period allowance a deal counts against: its API key's holder,
    else its address."""
    return f"key:{holder}" if holder else ip


def parse_client(payload: dict) -> dict:
    """The optional `client` on `POST /api/games`/`/api/join`: `{"id": <64-hex
    sha256>}` plus, optionally, its profile: `name`, `version`, `owner`,
    `url` and `description` (`PROFILE_FIELDS`), each as
    `constants.profile_value` takes it and left out when empty. Absent ->
    `{"id": None}`. Anything else it carries is ignored, a `kind` included
    (`client_kind` decides that). An `id` that is not a 64-hex sha256 digest,
    or a profile field `profile_value` refuses, is a 400."""
    raw = payload.get("client")
    if raw is None:
        return {"id": None}
    if not isinstance(raw, dict):
        raise ApiError("client must be an object")
    client_id = raw.get("id")
    if client_id is not None and not (isinstance(client_id, str) and _HEX64.match(client_id)):
        raise ApiError("client.id must be a 64-character hex sha256 digest")
    client: dict = {"id": client_id}
    for field in PROFILE_FIELDS:
        value = raw.get(field)
        if value is None:
            continue
        try:
            value = profile_value(field, value)
        except ValueError as error:
            raise ApiError(f"client.{error}") from None
        if value:
            client[field] = value
    return client


def require_profile(client: dict, origin: Origin | None) -> None:
    """The price of the HTTP API: a client that is neither this server's page
    nor the MCP endpoint plays only with an `id` and a profile naming itself
    (`REQUIRED_PROFILE`), so every one of its games counts under that id. A
    400 naming what is missing otherwise. In-process callers are not asked."""
    if origin is None or client["kind"] != "api":
        return
    missing = [f"client.{field}" for field in ("id", *REQUIRED_PROFILE) if not client.get(field)]
    if missing:
        raise ApiError(
            f"a client of the HTTP API says what it is; missing {', '.join(missing)} "
            f"({API_DOCS}#client-identity-and-seat-recovery)"
        )


def catanatron_seatable() -> bool:
    """Whether the `catanatron` opponent can be seated in this install. The
    extra is optional; without it the picker simply has one fewer opponent.
    Importing the adapter is what registers the preset, so this is called
    before the picker is built and before a spec is spawned."""
    try:
        import hexset.catanatron.bot  # noqa: F401 -- registers the preset
    except ImportError:
        return False
    return True


@dataclass(frozen=True)
class Offered:
    """One entry the picker offers: the entrant spec it seats, for a model
    file the `name`, `version` and `commit` the file declares, and the
    `profile` a preset was registered with (`hexset.arena.preset_profile`)
    or a model file declares (`owner`, `url`, `description`,
    `hexset.clients.onnxmeta.PROFILE_KEYS`)."""

    spec: str
    name: str | None = None
    version: str | None = None
    commit: str | None = None
    profile: tuple[tuple[str, str], ...] = ()

    def public(self) -> dict:
        fields = (("name", self.name), ("version", self.version), ("commit", self.commit), *self.profile)
        return {key: value for key, value in fields if value}


def entry_name(name: str, version: str | None) -> str:
    """What the picker offers a model file as: `<name>-<version>`, or the
    name alone for a file that declares no version."""
    return f"{name}-{version}" if version else name


@lru_cache(maxsize=256)
def _declared(path: str, mtime_ns: int, size: int) -> dict[str, str] | None:
    """The metadata the file at `path` declares, or `None` for one that is
    not a model. Keyed on the file's mtime and size, so a replaced file is
    read again."""
    try:
        return read_metadata(Path(path).read_bytes())
    except (OSError, ValueError, UnicodeDecodeError):
        return None


_warned: set[str] = set()


def _warn_once(message: str) -> None:
    if message not in _warned:
        _warned.add(message)
        print(message, file=sys.stderr)


def model_catalogue() -> dict[str, dict[str, Offered]]:
    """Game type -> entry -> what that entry seats there, for every
    `hexset.rules.GAME_TYPES` entry: every preset a runtime registered
    (`hexset.arena.registered_presets`, so the server's `--runtime`, and
    catanatron where its extra is installed) at every type, then each
    `*.onnx` under `MODELS_DIR` as `<name>-<version>` (`entry_name`) from the
    `name` and `version` it declares, at the game types its `modes` declare
    (`hexset.clients.onnxmeta`; the standard game when it declares none).
    The filename plays no part: files may share a name, each version its
    own entry.

    A file that is not a model, declares no `name`, or whose entry is a
    preset's name is not offered; two files declaring one name and version
    are neither offered. Each is logged once. Specs never cross the wire, so
    a request cannot point a bot at an arbitrary file. Scanned on every
    call, so a new file needs no restart."""
    catanatron_seatable()  # registers its preset where the extra is installed
    presets = {name: Offered(name, profile=tuple(preset_profile(name).items())) for name in registered_presets()}
    catalogue = {kind: dict(presets) for kind in GAME_TYPES}
    files: dict[str, list[tuple[Path, dict[str, str]]]] = {}
    for path in sorted(MODELS_DIR.glob("*.onnx")):
        try:
            stat = path.stat()
        except OSError:
            continue
        meta = _declared(str(path), stat.st_mtime_ns, stat.st_size)
        if meta is None:
            _warn_once(f"models: {path} is not an ONNX model; not offered")
            continue
        name = meta.get("name", "").strip()
        if not name:
            _warn_once(f"models: {path} declares no name; not offered")
            continue
        entry = entry_name(name, meta.get("version", "").strip() or None)
        if entry in presets:
            _warn_once(f"models: {path} is offered as {entry!r}, a registered bot's name; not offered")
            continue
        files.setdefault(entry, []).append((path, meta))
    for entry, declaring in files.items():
        if len(declaring) > 1:
            paths = ", ".join(str(path) for path, _ in declaring)
            _warn_once(f"models: {paths} all declare {entry!r}; none is offered")
            continue
        path, meta = declaring[0]
        offered = Offered(
            str(path), meta["name"].strip(), meta.get("version", "").strip() or None,
            meta.get("commit", "").strip() or None, _declared_profile(path, meta),
        )
        for kind in modes_of(meta):
            if kind not in catalogue:
                _warn_once(f"models: {path} declares mode {kind!r}, not a game type this server deals")
                continue
            catalogue[kind][entry] = offered
    return catalogue


def _declared_profile(path: Path, meta: dict[str, str]) -> tuple[tuple[str, str], ...]:
    """The profile fields a model file declares, each held to a client's
    lengths (`constants.profile_value`); one it would refuse is logged once
    and left out."""
    profile = []
    for key in PROFILE_KEYS:
        if key not in meta:
            continue
        try:
            value = profile_value(key, meta[key])
        except ValueError as error:
            _warn_once(f"models: {path} declares a {key} it may not: {error}; left out")
            continue
        if value:
            profile.append((key, value))
    return tuple(profile)


def bot_profile(entry: str) -> dict[str, str]:
    """The profile the server shows for its bot `entry` (`BOT_PROFILE_FIELDS`):
    the one its preset was registered with, else the version and profile the
    model file offered as `entry` declares; empty for neither."""
    profile = preset_profile(entry)
    if profile:
        return profile
    for offered in model_catalogue().values():
        if entry in offered and offered[entry].name is not None:
            found = offered[entry]
            fields = {"version": found.version, **dict(found.profile)}
            return {key: value for key, value in fields.items() if key in BOT_PROFILE_FIELDS and value}
    return {}


def model_options(game_type: str = STANDARD_GAME.name) -> dict[str, str]:
    """Entry -> entrant spec, for the picker at a `game_type` table
    (`model_catalogue`)."""
    return {name: offered.spec for name, offered in model_catalogue().get(game_type, {}).items()}


def listed_models() -> list[str]:
    """Every entry available to browser and API clients at any game type,
    each once."""
    return list(dict.fromkeys(name for offered in model_catalogue().values() for name in offered))


def _models_view() -> dict:
    """`GET /api/models`: every entry, and by game type what each entry is
    there."""
    catalogue = model_catalogue()
    return {
        "models": list(dict.fromkeys(name for offered in catalogue.values() for name in offered)),
        "game_types": {
            kind: {name: entry.public() for name, entry in offered.items()}
            for kind, offered in catalogue.items()
        },
    }


def _seated_spec(model: str, game_type: str) -> str:
    """The spec `model` seats at a `game_type` table, or a 400 saying why
    there is none."""
    catalogue = model_catalogue()
    offered = catalogue.get(game_type, {})
    if model in offered:
        return offered[model].spec
    if any(model in names for names in catalogue.values()):
        raise ApiError(f"{model} does not play {game_type} games")
    raise ApiError(f"unknown model: {model}")


def wait_query(query: str) -> tuple[int | None, float]:
    """`after`/`wait` out of a read's query string. No `after` means answer
    now; `wait` is capped at `MAX_WAIT_SECONDS`."""
    params = parse_qs(query)
    raw = params.get("after", [None])[0]
    if raw is None:
        return None, 0.0
    try:
        after = int(raw)
        wait = float(params.get("wait", ["0"])[0])
    except ValueError:
        raise ApiError("`after` is a version number and `wait` is seconds") from None
    return after, max(0.0, min(wait, MAX_WAIT_SECONDS))


def round_query(query: str) -> int:
    """The `round` a replay read is stopping at. The caller clamps it against
    the game's last round rather than refusing an out-of-range one."""
    params = parse_qs(query)
    raw = params.get("round", [None])[0]
    if raw is None:
        raise ApiError("a replay read names the `round` to stop at")
    try:
        return int(raw)
    except ValueError:
        raise ApiError("`round` is a round number") from None


def _check_version(table: "Table", version: int | None) -> None:
    """The optional `version` an acting request can send: 409 if the table has
    moved since the caller read `state()`, rather than applying
    `index`/`seat`/`bundle` meant for an older position. `None` skips the
    check; anything but an integer is a 400."""
    if version is not None and type(version) is not int:
        raise ApiError("`version` is the table's version number")
    if version is not None and version != table.version:
        raise ApiError(f"the table has moved (version {table.version}); read state again", status=409)


@dataclass
class Config:
    """How this server builds the games it deals (`web.py`'s CLI sets it)."""

    device: str = "cpu"
    games_dir: str | None = None
    seed: int | None = None
    # How every board is dealt: a `hexset.board.board.BOARD_MODES` name. The
    # lettered spiral here, where the engine's own default is "random".
    board_mode: str = "spiral"
    # Per client address: games it may deal in any rolling `period_days`,
    # counted across restarts from the journals, and unfinished tables it may
    # have in play at once (`LIVE_IDLE_SECONDS`). 0 is no limit. Addresses in
    # `limit_exempt` (CIDRs) and in-process callers are never limited. A
    # deal made with one of `api_keys` counts against that key's own
    # allowance instead of its address's.
    games_per_period: int = 100
    period_days: float = 30.0
    live_tables_per_ip: int = 5
    limit_exempt: tuple[str, ...] = ("127.0.0.0/8", "::1/128")
    api_keys: dict[str, ApiKey] = field(default_factory=dict)
    # Where to ask for a larger allowance; sent with a game-limit refusal and
    # with an allowance running low. None sends nothing.
    contact: str | None = None
    # Makes the rankings from every ranked game that ended (`rankings`); None
    # keeps no rankings. Rows that played fewer than `rankings_min_games`
    # games are not shown.
    ranker: Callable[[list[dict]], dict] | None = None
    rankings_min_games: int = RANKINGS_MIN_GAMES
    # The bot the ranked queue seats wherever no client fills a table
    # (`ranked.py`): an entry `model_options` offers at the game type. None
    # fills nothing, so a ranked table waits for clients to fill it.
    ranked_bot: str | None = None
    # Server bots the ranked queue seats as players of their own: each keeps a
    # ticket waiting at every game type `model_catalogue` offers it at, one
    # ranked game at a time, and is ranked as `bot:<entry>` (`ranked.py`).
    ranked_entrants: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.board_mode not in BOARD_MODES:
            raise ValueError(f"unknown board mode {self.board_mode!r}: {sorted(BOARD_MODES)}")
        self.exempt_networks = tuple(ipaddress.ip_network(n, strict=False) for n in self.limit_exempt)

    def is_exempt(self, ip: str | None) -> bool:
        """Whether `ip` is under no per-address limit. A request with no
        address cannot be told apart from any other, so it is limited."""
        if ip is None:
            return False
        try:
            address = ipaddress.ip_address(ip)
        except ValueError:
            return False
        return any(address in network for network in self.exempt_networks)


class SeatKind(str, Enum):
    """Who holds a `Seat`: nobody, a bot the server runs, or a player over
    the API."""

    EMPTY = "empty"
    BOT = "bot"
    PLAYER = "player"


@dataclass
class Seat:
    """One place at a game: a bot, a person, or nobody. `token` is the secret
    that proves a request is that seat, and never leaves this process except
    in the one response that mints it."""

    kind: SeatKind = SeatKind.EMPTY
    name: str | None = None
    spec: str | None = None
    token: str | None = field(default=None, repr=False)
    # `{"id": <64-hex sha256 or None>, "kind": "web"|"api"|"mcp"}`, or `None`
    # for a bot or unclaimed seat. Never in `public`: it exists for the journal
    # and `POST /api/reclaim`.
    client: dict | None = None

    def public(self, seat: int) -> dict:
        """What anyone may see about this seat. Never the token."""
        return {
            "seat": seat,
            "kind": self.kind.value,
            "name": self.name,
        }


@dataclass
class Table:
    """A row of seats, a code to reach them by, and the game they're playing.
    `runners` is every embedded bot-client thread playing a seat here, and
    `stopping` every one told to stop when its seat changed bot, which
    finishes on its own once the table's lock is free."""

    code: str
    seats: list[Seat]
    config: Config
    session: GameSession
    layout: dict

    runners: list[tuple[BotRunner, threading.Thread]] = field(default_factory=list, repr=False)
    stopping: list[tuple[BotRunner, threading.Thread]] = field(default_factory=list, repr=False)
    last_seen: float = field(default_factory=time.monotonic)
    # The address that dealt this table, for the per-address limits; `None`
    # for one dealt in-process or reopened from a journal. A ranked table has
    # every seated client's address in `addresses` instead.
    creator_ip: str | None = None
    addresses: frozenset = frozenset()
    # When a view last changed (`bump`): a ranked table's clock.
    changed_at: float = field(default_factory=time.monotonic)
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # Every change a view can show bumps `version` and wakes `changed`, so a
    # reader waits on the table rather than a timer. Its own lock, never
    # `self.lock`: a waiter holding that could not be changed.
    version: int = 0
    changed: threading.Condition = field(default_factory=threading.Condition, repr=False)

    def bump(self) -> None:
        """Marks this table changed and wakes everyone parked on it."""
        with self.changed:
            self.version += 1
            self.changed_at = time.monotonic()
            self.changed.notify_all()

    def wait_for_change(self, after: int, timeout: float) -> None:
        """Blocks until `version` passes `after`, or `timeout` seconds go by.
        Must be called with no other lock held."""
        with self.changed:
            self.changed.wait_for(lambda: self.version > after, timeout=timeout)

    def seat_of(self, token: str) -> int:
        """The seat `token` holds at this table, or a 403."""
        for index, seat in enumerate(self.seats):
            if seat.token is not None and secrets.compare_digest(seat.token, token):
                return index
        raise ApiError("that token is not for a seat at this game", status=403)

    def stop_runners(self) -> None:
        """Stop every embedded bot runner at this table and wait for it.
        Signalled, woken and joined in separate passes, so three bots take one
        wake between them."""
        for runner, _ in self.runners:
            runner.stop.set()
        # A runner parked on a long poll cannot see `stop` until woken, and one
        # mid-decision finishes first, which the join below allows for.
        self.bump()
        for _, thread in self.runners + self.stopping:
            thread.join(timeout=15.0)
        self.runners.clear()
        self.stopping.clear()

    def close(self, timed_out: list[int] | None = None) -> None:
        """Stop every embedded bot runner, then close the journal: abandoned,
        or, with `timed_out`, aborted on a ranked clock that ran out on those
        seats (`Journal.timed_out`). Order matters: a runner still
        mid-decision could otherwise submit one more action into a game
        already filed as over."""
        self.stop_runners()
        if self.session.journal is not None:
            if timed_out is None:
                self.session.journal.abandoned()
            else:
                self.session.journal.timed_out(self.session.game, timed_out)

    def owing(self) -> list[int]:
        """The seats the game is waiting on: each one `your_move` asks
        something of. Called with the table's lock held."""
        game = self.session.game
        return [s for s in range(game.num_players)
                if s not in game.locked and your_move(self.view(s))[0] not in ("wait", "game_over")]

    def waiting_for(self) -> list[int]:
        """The still-empty, still-unlocked seats holding up setup; always empty
        once setup has ended, since a seat can only resolve, never re-empty.
        Nobody moves while it is non-empty: `Tables.act` 409s every
        `POST /api/action` and every view's `to_move` reads `None`."""
        game = self.session.game
        if game.phase not in SETUP_PHASES:
            return []
        return [i for i, seat in enumerate(self.seats) if seat.kind is SeatKind.EMPTY and i not in game.locked]

    def join(self, name: str | None, client: dict | None = None) -> tuple[int, str]:
        """Seat a person or an LLM at a random still-open, still-unlocked seat,
        returning that seat and its token. 409 when none is open, which is also
        how a finished game refuses, since a seat never re-empties. Every
        manual seat gets a `PendingGate` as it is claimed, and an unnamed one
        falls back to `default_seat_name(client.kind)`."""
        client = client or {"id": None, "kind": "api"}
        clean = clean_name(name) or client.get("name") or default_seat_name(client["kind"])
        candidates = [
            i
            for i, seat in enumerate(self.seats)
            if seat.kind is SeatKind.EMPTY and i not in self.session.game.locked
        ]
        if not candidates:
            raise ApiError("this game has no open seats", status=409)
        index = random.SystemRandom().choice(candidates)
        token = secrets.token_urlsafe(18)
        self.seats[index] = Seat(kind=SeatKind.PLAYER, name=clean, token=token, client=client)
        self.session.claim(index, clean, client)
        self.session.confirm_mode(index)
        self.bump()
        return index, token

    def view(
        self, viewer: int | None = None, *, omniscient: bool = False, finished: bool = False
    ) -> dict:
        """The whole game as `viewer` (a seat, or `None` for a spectator) is
        allowed to see it (`GameSession.state_view`). `omniscient` holds
        nothing back and is for a spectator only; `state_view` refuses it
        alongside a seat. `finished` says the game was played out even though
        this position is not its end, which only a replay stand-in can be.
        Called with the table's lock held, or on a table nobody else can
        reach."""
        state = self.session.state_view(viewer, omniscient=omniscient, finished=finished)
        waiting = self.waiting_for()
        state["waiting_for"] = waiting
        # The manual seats a bot's trade offer waits on. Nobody moves while
        # either list is non-empty: `to_move` reads `None`.
        state["trade_wait"] = self.session.trade_wait()
        if waiting or state["trade_wait"]:
            state["to_move"] = None
        state["code"] = self.code
        state["seats"] = [seat.public(i) for i, seat in enumerate(self.seats)]
        state["version"] = self.version
        state["your_move"], state["waiting_on"] = your_move(state)
        return state


def spawn_bot(
    spec: str, board: Board, rng: random.Random, config: Config, players: int | None = None,
    game_type: str | None = None,
) -> Bot:
    """One spec (`model_options()`) as a live bot on `board`: a preset name is
    a registered opponent, anything else a path to a checkpoint. A preset is
    seated with the table's `game_type` as its `Entrant.game_type`. How a
    checkpoint wants to be played is read out of the file by `onnxbot.spawn`,
    which refuses one trained for other than the table's `players`, or one
    whose `modes` do not hold its `game_type`.
    """
    if spec not in PRESETS:
        catanatron_seatable()  # a resumed game names a spec no picker built
    if spec in PRESETS:
        # A registered opponent, with its preset's own defaults.
        return spawn_entrant(replace(PRESETS[spec], game_type=game_type), board, rng)

    from hexset.clients.onnxbot import spawn  # onnxruntime-free import boundary

    return spawn(spec, board, rng=rng, device=config.device, players=players, game_type=game_type)


def seat_spawn(
    name: str | None, spec: str, board: Board, config: Config, players: int, game_type: str | None = None,
) -> Bot:
    """`spawn_bot` for a seat about to be filled, run before anything about
    the table changes. Whatever stops the spec from becoming a bot -- a
    checkpoint naming a trader nobody registered, a file that will not load,
    a missing runtime -- is a 400 naming the opponent and the reason, so the
    caller has nothing to undo."""
    try:
        return spawn_bot(spec, board, random.Random(), config, players, game_type)
    except Exception as error:
        raise ApiError(f"cannot seat {name or spec}: {type(error).__name__}: {error}") from None


def deal(config: Config, board_mode: str | None = None) -> tuple[int, Board]:
    """The seed a new game is dealt from and the board it deals, so the bots
    can be spawned on that board before the game exists. Always a concrete
    seed, so the journal can name the one a resumed game rebuilds from.
    `board_mode` overrides `config.board_mode` for this one game."""
    seed = config.seed
    if seed is None:
        seed = random.SystemRandom().randrange(2**31)
    return seed, base_board(board_mode or config.board_mode, random.Random(seed))


def _seat_labels(
    seats: list[Seat],
) -> tuple[dict[int, str], dict[int, str], dict[int, str], dict[int, dict]]:
    """The name/spec/client maps `GameSession` is built with, read off `seats`
    the same way whether the game is dealt fresh or replayed from a journal —
    the two must agree, or a resumed game's labels would diverge."""
    bot_names = {i: s.name for i, s in enumerate(seats) if s.kind is SeatKind.BOT and s.name}
    bot_specs = {i: s.spec for i, s in enumerate(seats) if s.kind is SeatKind.BOT and s.spec}
    player_names = {i: s.name for i, s in enumerate(seats) if s.kind is SeatKind.PLAYER and s.name}
    clients = {i: s.client for i, s in enumerate(seats) if s.client is not None}
    return bot_names, bot_specs, player_names, clients


def build_session(
    code: str,
    seats: list[Seat],
    config: Config,
    *,
    first: int,
    dealt: tuple[int, Board] | None = None,
    board_mode: str | None = None,
    game_type: GameType = STANDARD_GAME,
    locked: tuple[int, ...] = (),
    ranked: bool = False,
    entrants: tuple[int, ...] = (),
) -> GameSession:
    """A fresh `MAX_SEATS`-seat game of `game_type`, `first` the seat the
    setup snake opens on and `locked` the seats retired at the deal,
    journalled from its header on. `dealt` is the `(seed, board)` `deal`
    returned for `board_mode` (`config.board_mode` when `None`), when the
    caller needed the board first; `None` deals one here. Seats not claimed
    here are left for `Table.join`/`lock_seat`."""
    board_mode = board_mode or config.board_mode
    seed, board = dealt if dealt is not None else deal(config, board_mode)
    # Two Random instances from the same seed, not one shared stream, since
    # `reopen_session` rebuilds a game this way and could not otherwise
    # reconstruct `start`'s rng position.
    game = start(board, MAX_SEATS, random.Random(seed), first=first, game_type=game_type, locked=locked)
    # This table drives its own trading (`GameSession.begin_round`, spread over
    # as many requests as its seats need).
    game.trade_mode = "external"
    bot_names, bot_specs, player_names, clients = _seat_labels(seats)
    claimed = {i for i, s in enumerate(seats) if s.kind is not SeatKind.EMPTY}
    return GameSession(
        game=game,
        claimed_seats=claimed,
        seed=seed,
        board_mode=board_mode,
        game_type=game_type,
        journal=journal.open_journal(seed, config.games_dir),
        bot_names=bot_names,
        bot_specs=bot_specs,
        player_names=player_names,
        clients=clients,
        code=code,
        ranked=ranked,
        entrants=entrants,
    )


def _stop_runner(table: Table, seat: int) -> None:
    """Stops the bot runner playing `seat`, if any."""
    for i, (runner, thread) in enumerate(table.runners):
        if runner.seat == seat:
            runner.stop.set()
            # A runner parked on a long poll sees `stop` only once woken,
            # and then waits for this table's lock to read the view it
            # woke for: joined at `close`, never here under that lock.
            table.bump()
            table.stopping = [(r, t) for r, t in table.stopping if t.is_alive()]
            table.stopping.append(table.runners.pop(i))
            return


def _unranked(table: Table) -> None:
    """A 409 at a ranked table, whose seats are the queue's to fill."""
    if table.session.ranked:
        raise ApiError("a ranked table's seats are fixed by the queue", status=409)


def _fixed_seats(table: Table) -> str:
    """Why a seat retired at a fixed-seat deal cannot be opened."""
    game_type = table.session.game_type
    return (f"a {game_type.name} game is played at "
            f"{' or '.join(str(n) for n in game_type.seats)} seats; its seats are fixed")


def journal_dir(config: Config) -> str | None:
    """Where this server's journals live: whatever `config` named, or the
    environment's directory when it named nothing. An empty string means
    journalling is off, which is not the same as `None`."""
    return config.games_dir if config.games_dir is not None else journal.configured_dir()


def reopened_seats(events: list[dict]) -> list[Seat]:
    """The row of seats a journalled game comes back with. A bot's seat is
    re-tokened fresh; a person's comes back claimed but **untokened**, since a
    token never touches disk, so `POST /api/reclaim` is the way back in.
    Claimed-but-untokened rather than `EMPTY` is what stops `Table.join`
    handing a seat mid-game to whoever opens the link next."""
    seats = [Seat() for _ in range(MAX_SEATS)]
    for seat, (bot_name, spec) in journal.seating(events).items():
        seats[seat] = Seat(kind=SeatKind.BOT, name=bot_name, spec=spec, token=secrets.token_urlsafe(18))
    for seat, name in journal.players(events).items():
        seats[seat] = Seat(kind=SeatKind.PLAYER, name=name or None, token=None)
    # A bot's seat carries no client identity; every other seat gets its back,
    # which `POST /api/reclaim` matches a returning client against.
    for seat, client in journal.clients(events).items():
        if seats[seat].kind is not SeatKind.BOT:
            seats[seat].client = client
    return seats


def _opening_session(code: str, seats: list[Seat], events: list[dict]) -> GameSession | None:
    """The game `events` opens with — dealt from the header's seed, seated as
    `seats` says, nothing played into it and no seat retired yet (`restore`
    retires each where the file says) — or `None` when the file does not open
    like a game.

    A header with no seed is a game this server did not deal
    (`journal.journal_of`): its board is the header's own, and its chance
    stream the deck and the effects its action lines recorded."""
    if not events or events[0].get("kind") != "game":
        return None
    header = events[0]
    seed = header["seed"]
    first = header.get("first", 0)
    bot_names, bot_specs, player_names, clients = _seat_labels(seats)
    # Journals from before there was a choice of deal name none, and were
    # dealt at random; those from before there was a choice of game type, or
    # of seats retired at the deal, were standard games with none retired.
    board_mode = header.get("board_mode", "random")
    game_type = journal.game_type_of(
        header.get("game_type", STANDARD_GAME.name), rules_from(header.get("rules", {})),
        header.get("num_players", MAX_SEATS),
    )
    if seed is None:
        game = open_record(from_events(events, partial=True))
    else:
        board = base_board(board_mode, random.Random(seed))
        # A game journalled before 1.10.0 drew its dice off the deck's stream;
        # dealt split, it would replay its actions against other dice.
        rng = random.Random(seed)
        chance = for_rules(game_type.rules, rng, split=header.get("split_streams", False))
        game = start(board, MAX_SEATS, rng, first=first, chance=chance,
                     game_type=game_type, locked=header.get("locked", ()))
    # The trade round is this table's protocol, driven by the session.
    game.trade_mode = "external"
    claimed = {i for i, s in enumerate(seats) if s.kind is not SeatKind.EMPTY}
    return GameSession(
        game=game,
        claimed_seats=claimed,
        seed=seed,
        board_mode=board_mode,
        game_type=game_type,
        bot_names=bot_names,
        bot_specs=bot_specs,
        player_names=player_names,
        clients=clients,
        code=code,
        ranked=bool(header.get("ranked")),
        entrants=tuple(header.get("entrants") or ()),
    )


def replay_session(code: str, seats: list[Seat], events: list[dict], upto: int) -> GameSession | None:
    """The same game `reopen_session` rebuilds, stopped after `upto` steps. No
    `Journal` is attached and the live table's session is never touched, so a
    reader stepping back cannot move the game everyone else is playing."""
    session = _opening_session(code, seats, events)
    if session is None:
        return None
    steps, _ = journal.replayable_rounds(events)
    notes = {at: note for at, note in journal.notes_of(events).items() if at < upto}
    retired = {at: seats for at, seats in journal.retirements(events).items() if at <= upto}
    session.restore(steps[:upto], notes=notes, retired=retired)
    return session


def reopen_session(
    code: str, seats: list[Seat], path: Path, events: list[dict], *, attach: bool = True
) -> GameSession | None:
    """The game `path` records, replayed back to exactly where it stopped, or
    `None` when there is nothing left to hand back. Which of three cases a file
    is, is decided from the *replayed game*, never from its closing line, which
    both a won and an abandoned game write: `is_over` comes back read-only with
    no `Journal`, still in play comes back live and journalling onward into the
    same file, and closed but not over is gone (`None`, a 404). `attach=False`
    leaves a live game unjournalled for the caller to `resume_journal` once
    nothing else can fail."""
    session = _opening_session(code, seats, events)
    if session is None:
        return None
    closed = journal.is_closed(events)
    try:
        # No journal until the replay says whether this game is still live.
        session.restore(
            journal.replayable(events),
            notes=journal.notes_of(events),
            retired=journal.retirements(events),
        )
    except (ResumeError, ValueError, KeyError) as error:
        # The file is kept, but closed, so the next request deals a game
        # instead of failing this way forever.
        print(f"could not reopen {path.name}: {error}")
        if not closed:
            journal.Journal(directory=str(path.parent), game_id=path.stem).abandoned()
        return None

    if is_over(session.game):
        return session
    if closed:
        return None
    if attach:
        resume_journal(session, path)
    return session


def resume_journal(session: GameSession, path: Path) -> None:
    """Journal a reopened live game onward into the file it was replayed
    from, marking the seam."""
    session.journal = journal.Journal(directory=str(path.parent), game_id=path.stem)
    session.journal.reopened(at_step=session.steps)


class Tables:
    """Every live game, and the operations the API is made of. Two lock
    granularities: `_registry_lock` guards the dict of games, while each table
    carries its own lock around the game mutation, so one game's turn does not
    stall every other game's requests. Nothing slow runs under
    `_registry_lock`: a reopen reads, replays and spawns with it released."""

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config()
        self._tables: dict[str, Table] = {}
        self._registry_lock = threading.Lock()
        # One lock per code being reopened, so two reads of it reopen it once.
        self._reopening: dict[str, threading.Lock] = {}
        # Code -> when it was last found to have no game behind it.
        self._misses: dict[str, float] = {}
        # Address -> when (epoch seconds) each game it dealt in the current
        # period was dealt, oldest first, seeded from the journals.
        self._dealt: dict[str, deque[float]] = self._dealt_from_journals()
        self._close_interrupted()
        self.queue = RankedQueue(self)
        self.rankings = (
            Rankings(self.config.ranker, journal_dir(self.config), self.config.rankings_min_games)
            if self.config.ranker is not None else None
        )

    def _dealt_from_journals(self) -> dict[str, deque[float]]:
        """Every deal in the last `period_days` the journals record an address
        for (the creator's client, in each header), so a restart does not
        reset the count. Ranked games count against no allowance."""
        directory = journal_dir(self.config)
        if not directory or not os.path.isdir(directory):
            return {}
        since = time.time() - self.config.period_days * 86400
        dealt: dict[str, list[float]] = {}
        for path in Path(directory).glob("*.jsonl"):
            header = journal.header_of(path)
            if header is None or header.get("ranked"):
                continue
            try:
                at = datetime.strptime(header["at"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()
            except (KeyError, TypeError, ValueError):
                continue
            if at < since:
                continue
            for client in (header.get("clients") or {}).values():
                ip = (client or {}).get("ip")
                if ip and not self.config.is_exempt(ip):
                    dealt.setdefault(_allowance(ip, (client or {}).get("key")), []).append(at)
        return {ip: deque(sorted(times)) for ip, times in dealt.items()}

    def holder(self, origin: Origin | None) -> str | None:
        """The name of whoever `origin`'s API key was issued to, `None` when
        it sent none; a key not in `Config.api_keys` is a 401."""
        if origin is None or origin.key is None:
            return None
        api_key = self.config.api_keys.get(origin.key)
        if api_key is None:
            raise ApiError("unknown API key", status=401)
        return api_key.name

    def unadmit(self, origin: Origin | None, holder: str | None) -> None:
        """Takes back the deal `admit` just recorded for `origin`."""
        if origin is None or self.config.is_exempt(origin.ip):
            return
        with self._registry_lock:
            dealt = self._dealt.get(_allowance(origin.ip or "?", holder))
            if dealt:
                dealt.pop()

    def admit(self, origin: Origin | None, holder: str | None) -> None:
        """Refuses `POST /api/games` from an address over either limit in
        `Config` with a 429; records the deal otherwise. A deal with an API
        key, its `holder`'s, counts against the key's allowance in place of
        the address's. In-process callers and exempt addresses always pass."""
        if origin is None or self.config.is_exempt(origin.ip):
            return
        cap, whose = self._cap(origin, holder)
        with self._registry_lock:
            dealt = self._window(origin, holder)
            if cap and len(dealt) >= cap:
                slot = dealt[0] + self.config.period_days * 86400
                at = time.strftime("%Y-%m-%dT%H:%MZ", time.gmtime(slot))
                fields = {"next_slot": at}
                more = ""
                if self.config.contact:
                    fields["contact"] = self.config.contact
                    more = f"; contact: {self.config.contact}"
                raise ApiError(
                    f"game limit: {cap} per {self.config.period_days:g} days for {whose}; "
                    f"next slot {at}{more}",
                    status=429, fields=fields, retry_after=max(1, int(slot - time.time())),
                )
            self._refuse_live(origin)
            dealt.append(time.time())

    def admit_ranked(self, origin: Origin | None) -> None:
        """Refuses a ranked ticket (`POST /api/ranked`) from an address at its
        unfinished-games limit with a 429. Ranked play counts against no
        period allowance, an API key's or an address's: the queue's pair cap
        (`ranked.PAIR_CAP`) bounds it."""
        if origin is None or self.config.is_exempt(origin.ip):
            return
        with self._registry_lock:
            self._refuse_live(origin)

    def _refuse_live(self, origin: Origin) -> None:
        """A 429 for an address at `Config.live_tables_per_ip`. Call under
        `_registry_lock`."""
        live_cap = self.config.live_tables_per_ip
        if live_cap and self._live(origin) >= live_cap:
            raise ApiError(
                f"live table limit: {live_cap} unfinished games at once for this address",
                status=429,
            )

    def _cap(self, origin: Origin, holder: str | None) -> tuple[int, str]:
        """The period allowance a deal counts against, and whose it is."""
        if holder is None:
            return self.config.games_per_period, "this address"
        return self.config.api_keys[origin.key].games_per_period, "this API key"

    def _window(self, origin: Origin, holder: str | None) -> deque:
        """The deals inside the rolling period, oldest first, pruned in
        place. Call under `_registry_lock`."""
        dealt = self._dealt.setdefault(_allowance(origin.ip or "?", holder), deque())
        while dealt and time.time() - dealt[0] > self.config.period_days * 86400:
            dealt.popleft()
        return dealt

    def _live(self, origin: Origin) -> int:
        """Unfinished, recently seen tables dealt from `origin`'s address.
        Call under `_registry_lock`."""
        now = time.monotonic()
        return sum(
            1
            for t in self._tables.values()
            if (t.creator_ip == (origin.ip or "?") or origin.ip in t.addresses)
            and not is_over(t.session.game)
            and now - t.last_seen < LIVE_IDLE_SECONDS
        )

    def allowance(self, origin: Origin | None, holder: str | None) -> dict | None:
        """`GET /api/limits`, and `allowance` on `POST /api/games`: what is left
        of this caller's limits. None for a caller under no limit. `contact`
        rides along once ten or fewer games are left."""
        if origin is None or self.config.is_exempt(origin.ip):
            return None
        cap, _whose = self._cap(origin, holder)
        with self._registry_lock:
            used = len(self._window(origin, holder))
            live = self._live(origin)
        out = {
            "games_left": max(0, cap - used) if cap else None,
            "games_per_period": cap or None,
            "period_days": self.config.period_days,
            "live_tables": live,
            "live_tables_limit": self.config.live_tables_per_ip or None,
        }
        if cap and cap - used <= 10 and self.config.contact:
            out["contact"] = self.config.contact
        return out

    # --- registry ---------------------------------------------------------

    def create(
        self,
        bots: list[str] | None = None,
        name: str | None = None,
        client: dict | None = None,
        board_mode: str | None = None,
        game_type: str | None = None,
    ) -> tuple[Table, str]:
        """A new game, dealt immediately: the creator at a random seat, any
        named bots seated alongside them, everything else open. `bots` names
        opponents offered at its game type (`model_catalogue`); none named is none seated. Turn
        order is seat order. Every bot
        is spawned before the game is dealt or journalled, so one that will
        not spawn is a 400 that leaves nothing behind.

        `board_mode` names a `BOARD_MODES` entry (`config.board_mode` when
        `None`) and `game_type` a `GAME_TYPES` one (standard when `None`).
        The table always has `MAX_SEATS` seats; those past the type's largest
        seat count are retired at the deal, so a duel-variant table plays
        seats 0 and 1 and its creator sits at one of them."""
        if board_mode is not None and (not isinstance(board_mode, str) or board_mode not in BOARD_MODES):
            raise ApiError(f"unknown board mode: {board_mode!r}; one of {sorted(BOARD_MODES)}")
        if game_type is not None and (not isinstance(game_type, str) or game_type not in GAME_TYPES):
            raise ApiError(f"unknown game type: {game_type!r}; one of {sorted(GAME_TYPES)}")
        contract = GAME_TYPES[game_type] if game_type is not None else STANDARD_GAME
        playing = min(max(contract.seats), MAX_SEATS)
        if bots is None:
            bots = []
        if not isinstance(bots, list) or not all(isinstance(entry, str) for entry in bots):
            raise ApiError("bots is a list of model names")
        specs = {entry: _seated_spec(entry, contract.name) for entry in bots}
        if 1 + len(bots) > playing:
            raise ApiError(f"a {contract.name} game seats at most {playing}; asked for {1 + len(bots)}")

        creator_seat = random.SystemRandom().randrange(playing)
        client = client or {"id": None, "kind": "api"}
        clean = clean_name(name) or client.get("name") or default_seat_name(client["kind"])
        seats: list[Seat] = [Seat() for _ in range(MAX_SEATS)]
        seats[creator_seat] = Seat(
            kind=SeatKind.PLAYER, name=clean, token=secrets.token_urlsafe(18), client=client
        )
        remaining = [i for i in range(playing) if i != creator_seat]
        for entry, seat_index in zip(bots, remaining):
            seats[seat_index] = Seat(
                kind=SeatKind.BOT, name=entry, spec=specs[entry], token=secrets.token_urlsafe(18)
            )
        table = self._open(seats, contract, board_mode or self.config.board_mode)
        token = seats[creator_seat].token
        assert token is not None
        return table, token

    def _open(self, seats: list[Seat], contract: GameType, board_mode: str, *,
              ranked: bool = False, entrants: tuple[int, ...] = (),
              addresses: frozenset = frozenset()) -> Table:
        """Deals `seats` a `contract` game on a `board_mode` board and puts the
        table in play: every bot spawned first (a 400 for one that will not,
        leaving nothing behind), every person's seat a `PendingGate`, the seats
        past the contract's largest count retired, registered, then the bots'
        runners started."""
        dealt = deal(self.config, board_mode)
        spawned = self._spawn_seats(seats, dealt[1], contract.name)

        with self._registry_lock:
            evicted = self._evict_stale(time.monotonic())
            code = new_code(set(self._tables))
        for table in evicted:
            table.close()

        playing = min(max(contract.seats), MAX_SEATS)
        session = build_session(
            code, seats, self.config, first=0, dealt=dealt, board_mode=board_mode,
            game_type=contract, locked=tuple(range(playing, MAX_SEATS)), ranked=ranked,
            entrants=entrants,
        )
        self._watch(session)
        for index, seat in enumerate(seats):
            if seat.kind is SeatKind.PLAYER:
                session.confirm_mode(index)
        for index, bot in spawned.items():
            # A bot seat brings its own trading gate (`hexset.trading`).
            session.set_trader(index, bot)
        table = Table(
            code=code,
            seats=seats,
            config=self.config,
            session=session,
            # true state: the board is public.
            layout=board_layout(session.game.state(0, hidden=False).board),
            addresses=addresses,
        )

        # Registered *before* any bot runner starts: a runner's first move is
        # to look itself up by token, which scans `_tables`.
        with self._registry_lock:
            self._tables[code] = table
        self._start_runners(table, spawned)
        return table

    # --- ranked play (`ranked.py`) --------------------------------------------

    def deal_ranked(self, game_type: str, players: list[dict | str], bot: str | None, fill: int,
                    origins: list) -> dict:
        """A ranked table of `game_type` for `players`, each a client or the
        entry of a server bot entered as a player (`Config.ranked_entrants`),
        with `fill` more seats for `bot`, every seat placed at random:
        `{"code", "seats"}`, `seats` each player's `(seat, token)` in
        `players`' order. Its journal is marked ranked, with the entrants'
        seats, and its seats are fixed."""
        contract = GAME_TYPES[game_type]
        playing = min(max(contract.seats), MAX_SEATS)
        if len(players) + fill != playing:
            raise ValueError(f"a ranked {game_type} table seats {playing}")
        spec = _seated_spec(bot, contract.name) if fill else None
        order = random.SystemRandom().sample(range(playing), playing)
        seats: list[Seat] = [Seat() for _ in range(MAX_SEATS)]
        placed = []
        entrants = []
        for player, index in zip(players, order):
            token = secrets.token_urlsafe(18)
            if isinstance(player, str):
                seats[index] = Seat(kind=SeatKind.BOT, name=player, spec=_seated_spec(player, contract.name),
                                    token=token)
                entrants.append(index)
            else:
                name = player.get("name") or default_seat_name(player["kind"])
                seats[index] = Seat(kind=SeatKind.PLAYER, name=name, token=token, client=player)
            placed.append((index, token))
        for index in order[len(players):]:
            seats[index] = Seat(kind=SeatKind.BOT, name=bot, spec=spec, token=secrets.token_urlsafe(18))
        addresses = frozenset(o.ip for o in origins if o is not None and o.ip)
        table = self._open(seats, contract, self.config.board_mode, ranked=True,
                           entrants=tuple(sorted(entrants)), addresses=addresses)
        return {"code": table.code, "seats": placed}

    def in_ranked_play(self, entry: str, game_type: str) -> bool:
        """Whether `entry` holds an entrant's seat at a ranked `game_type`
        table still in play here."""
        with self._registry_lock:
            tables = list(self._tables.values())
        for table in tables:
            session = table.session
            if (session.ranked and session.game_type.name == game_type and not is_over(session.game)
                    and any(table.seats[i].name == entry for i in session.entrants)):
                return True
        return False

    def check_clocks(self) -> list[str]:
        """Aborts every ranked table that has waited on one move longer than
        `ranked.TURN_SECONDS` (`Table.close`, naming the seats it waited on);
        the codes aborted."""
        now = time.monotonic()
        with self._registry_lock:
            late = [t for t in self._tables.values()
                    if t.session.ranked and not is_over(t.session.game) and now - t.changed_at > TURN_SECONDS]
            for table in late:
                self._tables.pop(table.code, None)
        for table in late:
            with table.lock:
                owing = table.owing()
            table.close(timed_out=owing)
        return [t.code for t in late]

    def queue_join(self, payload: dict, origin: Origin | None) -> dict:
        """`POST /api/ranked`: a ticket for a ranked game of `game_type`
        (standard when unnamed), refused only at the caller's unfinished-games
        limit (`admit_ranked`). For a client of the HTTP API (with its profile) or a model
        over MCP, never the page, and always with a `client.id`."""
        game_type = payload.get("game_type", STANDARD_GAME.name)
        if not isinstance(game_type, str) or game_type not in GAME_TYPES:
            raise ApiError(f"unknown game type: {game_type!r}; one of {sorted(GAME_TYPES)}")
        holder = self.holder(origin)
        client = stamp_client(parse_client(payload), origin, holder)
        if client["kind"] == "web":
            raise ApiError("ranked play is for clients of the API and MCP", status=403)
        require_profile(client, origin)
        if not client.get("id"):
            raise ApiError("ranked play needs a client.id: its games count under it")
        self.admit_ranked(origin)
        ticket = self.queue.join(game_type, client, origin)
        self._seen(client)
        return {"ticket": ticket.id, "game_type": game_type, "status": "waiting"}

    def queue_wait(self, ticket_id: str, query: str) -> dict:
        """`GET /api/ranked/<ticket>?wait=N`: `{"status": "waiting"}`, or once
        the queue has dealt its table, `"matched"` with the seat's `token` and
        its view, as `POST /api/join` answers. Read at least every
        `ranked.TICKET_SECONDS`, or the ticket is dropped (404)."""
        try:
            wait = max(0.0, min(float(parse_qs(query).get("wait", ["0"])[0]), MAX_WAIT_SECONDS))
        except ValueError:
            raise ApiError("`wait` is seconds") from None
        ticket = self.queue.wait(ticket_id, wait)
        if ticket is None:
            raise ApiError("no such ticket: it was left, or not read for too long", status=404)
        if ticket.match is None:
            return {"ticket": ticket.id, "game_type": ticket.game_type, "status": "waiting"}
        table = self.get(ticket.match["code"])
        with table.lock:
            return {"ticket": ticket.id, "status": "matched", "token": ticket.match["token"],
                    **table.view(ticket.match["seat"])}

    def journal_directory(self) -> str | None:
        """Where this server's journals live (`journal_dir`)."""
        return journal_dir(self.config)

    def _close_interrupted(self) -> None:
        """Files every ranked game a restart cut short as abandoned, so none
        comes back without its clock; its seats are told so by a 404 and
        queue again."""
        directory = journal_dir(self.config)
        if not directory or not os.path.isdir(directory):
            return
        for path in Path(directory).glob("*.jsonl"):
            header = journal.header_of(path)
            if header and header.get("ranked") and not journal.is_closed(journal.read(path)):
                journal.Journal(directory=str(path.parent), game_id=path.stem).abandoned(reason="restart")

    def _seen(self, client: dict) -> None:
        """A client just claimed a seat: its profile is now whatever this
        request said (`rankings.Rankings.seen`)."""
        if self.rankings is not None and client.get("id"):
            self.rankings.seen(client)

    def _watch(self, session: GameSession) -> None:
        """Has `session`'s journal tell the rankings when the game ends."""
        if self.rankings is not None and session.journal is not None:
            session.journal.on_end = self.rankings.ended

    def _spawn_seats(self, seats: list[Seat], board: Board, game_type: str) -> dict[int, Bot]:
        """Seat -> a live bot for every bot seat in `seats`, spawned on
        `board` for a `game_type` table (`seat_spawn`, a 400 for any that will
        not spawn)."""
        spawned: dict[int, Bot] = {}
        for index, seat in enumerate(seats):
            if seat.kind is SeatKind.BOT:
                assert seat.spec is not None
                spawned[index] = seat_spawn(seat.name, seat.spec, board, self.config, len(seats), game_type)
        return spawned

    def _start_runners(self, table: Table, bots: dict[int, Bot]) -> None:
        """Start one embedded `LocalSearchBrain` runner thread per bot in
        `bots`, already the trader of its seat. The runner drives its bot
        through the same token-gated `/api/action` route an external client
        would use, never a direct write to the session."""
        transport = LocalTransport(self)
        for index, bot in bots.items():
            token = table.seats[index].token
            assert token is not None
            brain = LocalSearchBrain(bot=bot, game=table.session.game, lock=table.lock)
            runner = BotRunner(seat=index, token=token, transport=transport, brain=brain)
            thread = threading.Thread(
                target=runner.run, name=f"bot-{table.code}-{index}", daemon=True
            )
            table.runners.append((runner, thread))
            thread.start()

    def get(self, code: str) -> Table:
        """The live table for `code`, reopened from its journal if the registry
        has lost it, or a 404. Codes are matched case-insensitively."""
        code = code.lower()
        with self._registry_lock:
            # Every lookup, not just `create`, or a server that only ever reads
            # would never reap anything.
            evicted = self._evict_stale(time.monotonic(), keep=code)
            table = self._tables.get(code)
        for stale in evicted:
            stale.close()
        if table is None:
            table = self._reopen_once(code)
        if table is None:
            raise ApiError(f"no game with code {code}", status=404)
        table.last_seen = time.monotonic()
        return table

    def _reopen_once(self, code: str) -> Table | None:
        """`_reopen` for one code at a time, with `_registry_lock` held only
        to look up and to insert. A second read of the same code waits for the
        first and gets its table; a code found to have nothing behind it is
        remembered for `MISS_SECONDS`."""
        with self._registry_lock:
            gate = self._reopening.setdefault(code, threading.Lock())
        try:
            with gate:
                with self._registry_lock:
                    table = self._tables.get(code)
                    missed = self._misses.get(code)
                if table is not None:
                    return table
                if missed is not None and time.monotonic() - missed < MISS_SECONDS:
                    return None
                reopened = self._reopen(code)
                now = time.monotonic()
                with self._registry_lock:
                    if reopened is None:
                        for old in [c for c, at in self._misses.items() if now - at >= MISS_SECONDS]:
                            del self._misses[old]
                        self._misses[code] = now
                        return None
                    table, spawned = reopened
                    # Only a reader that arrived as the gate was being dropped
                    # can have got here first; its table is the one in play.
                    first = self._tables.setdefault(code, table)
                    self._misses.pop(code, None)
                if first is not table:
                    return first
                self._start_runners(table, spawned)
                return table
        finally:
            with self._registry_lock:
                if self._reopening.get(code) is gate:
                    del self._reopening[code]

    def _reopen(self, code: str) -> tuple[Table, dict[int, Bot]] | None:
        """Puts a game back together from its journal for a code the registry
        has lost, with the bots to start once it is registered; `None` for a
        code with no game behind it, or one walked away from unfinished. A
        game already over gets no bot. Every bot is spawned before the file is
        written to, so one that will not spawn is a 400 that leaves the
        journal as it was."""
        path = journal.most_recent(journal_dir(self.config), code)
        if path is None:
            return None
        # The one read of this file; both callers below work off it.
        events = journal.read(path)
        seats = reopened_seats(events)
        session = reopen_session(code, seats, path, events, attach=False)
        if session is None:
            return None
        live = not is_over(session.game)
        # true state: the board is public.
        board = session.game.state(0, hidden=False).board
        spawned = self._spawn_seats(seats, board, session.game_type.name) if live else {}
        # A manual seat is a `PendingGate` from the moment it is claimed.
        for index, seat in enumerate(seats):
            if seat.kind is SeatKind.PLAYER:
                session.confirm_mode(index)
        for index, bot in spawned.items():
            session.set_trader(index, bot)
        if live:
            resume_journal(session, path)
            self._watch(session)
        table = Table(
            code=code,
            seats=seats,
            config=self.config,
            session=session,
            layout=board_layout(board),
        )
        return table, spawned

    def by_token(self, token: str | None) -> tuple[Table, int]:
        """The game and seat a token names, or a 401 (no token) / 403 (unknown
        token). **A bot seat's request does not refresh `last_seen`**: an
        embedded runner reads the table for the life of the game, so counting
        its reads would keep a table alive forever after everyone left."""
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

    def reclaim(self, code: str, secret: str) -> tuple[Table, int, str]:
        """`POST /api/reclaim`: a fresh token for the seat whose `client.id`
        equals `sha256(secret).hexdigest()` — the one way back into a seat once
        its token is gone. A seat still `PLAYER` just gets a new token and its
        old one then fails; an `EMPTY` seat is revived as `Table.join` would
        seat it; a bot seat and a locked one refuse, and no match is a 403.
        Works on a finished game on purpose, `_seated` still refusing every
        mutation."""
        table = self.get(code)
        digest = hashlib.sha256(secret.encode("utf-8")).hexdigest()
        with table.lock:
            locked = table.session.game.locked
            for index, seat in enumerate(table.seats):
                if seat.kind is SeatKind.BOT or index in locked:
                    continue
                client = seat.client
                if not client or not client.get("id"):
                    continue
                if not secrets.compare_digest(client["id"], digest):
                    continue
                token = secrets.token_urlsafe(18)
                seat.token = token
                if seat.kind is SeatKind.EMPTY:
                    seat.kind = SeatKind.PLAYER
                    seat.name = seat.name or default_seat_name(client["kind"])
                    table.session.claim(index, seat.name, client)
                    table.session.confirm_mode(index)
                table.bump()
                return table, index, token
        raise ApiError("no seat matches that secret", status=403)

    def _evict_stale(self, now: float, keep: str | None = None) -> list[Table]:
        """Must be called with `_registry_lock` held. Drops any game untouched
        for longer than `TABLE_TTL_SECONDS`, sparing `keep`, the code the
        caller is about to look up. Returns the popped tables rather than
        closing them: `close` joins runner threads, which would stall every
        other game's `by_token` under this lock."""
        stale = [
            c
            for c, t in self._tables.items()
            if c != keep and now - t.last_seen > TABLE_TTL_SECONDS
        ]
        return [self._tables.pop(code) for code in stale]

    def close(self) -> None:
        """Stop every runner at every table, close every journal and stop the
        ranked queue's and the rankings' workers. Eviction
        does the same job one table at a time in a long-lived server; this is
        for a shutting-down process or a test, which would otherwise leak a
        runner thread per bot seat."""
        with self._registry_lock:
            tables = list(self._tables.values())
            self._tables.clear()
        for table in tables:
            table.close()
        self.queue.close()
        if self.rankings is not None:
            self.rankings.close()

    # --- play -------------------------------------------------------------

    def act(self, table: Table, seat: int, wire: dict, version: int | None = None) -> dict:
        """`POST /api/action`: apply `wire` as `seat`'s move and return its
        view. 409 while any seat is still to resolve or a trade round is
        awaiting answers, and for a stale `version`."""
        _check_version(table, version)
        waiting = table.waiting_for()
        if waiting:
            names = ", ".join(str(s) for s in waiting)
            raise ApiError(f"waiting for seats: {names}", status=409)
        if table.session.trade_wait():
            raise ApiError("waiting for trade answers", status=409)
        table.session.submit(seat, wire)
        table.bump()
        return table.view(seat)

    def undo(self, table: Table, seat: int) -> dict:
        """`POST /api/undo`: take back `seat`'s most recent build, bank trade,
        or Road Building or Knight play, while it is still the last thing that
        happened (`GameSession.undo_last_build`)."""
        table.session.undo_last_build(seat)
        table.bump()
        return table.view(seat)

    def rename(self, table: Table, seat: int, name: object) -> dict:
        """`POST /api/name`: set `seat`'s display name. An empty one falls
        back to the seat's default (`default_seat_name`), as at claim time.
        Journalled, so a reopened table keeps it, and the name its client id
        goes by from here on (`rankings.Rankings.renamed`)."""
        client = table.seats[seat].client or {}
        clean = clean_name(name) or default_seat_name(client.get("kind", "api"))
        table.seats[seat].name = clean
        table.session.rename(seat, clean)
        table.bump()
        if self.rankings is not None and client.get("id"):
            self.rankings.renamed(client["id"], clean)
        return table.view(seat)

    def seat_bot(self, table: Table, viewer: int, seat: int, model: str) -> dict:
        """`POST /api/bot`: put a bot on `seat` — a fresh one where nobody is
        sitting, or a different one in place of the bot already there. An open
        seat can be filled until somebody closes it, and a bot swapped at any
        point during the game but not after `is_over`. A request names a bot,
        never a path. Swapping stops the old runner and starts a fresh one; the
        old bot's in-flight decision, if any, still lands. A person's seat is
        never taken over, a retired seat never revived, and a seat closed at a
        seat retired at the deal of a table whose seats are fixed
        (`GameSession.fixed_seats`) never opened.

        The bot is spawned before anything changes, so one that will not
        spawn is a 400 that leaves the seat, the session and the journal as
        they were. The view comes back as `viewer`, the seat that *asked*, not
        as `seat`, which would hand that seat's hand to whoever touched its
        picker.
        """
        game = table.session.game
        require_live(game)
        _unranked(table)
        if not 0 <= seat < len(table.seats):
            raise ApiError(f"there is no seat {seat} at this game")
        kind = table.seats[seat].kind
        if kind is SeatKind.PLAYER:
            raise ApiError(f"seat {seat} belongs to a player")
        # Before the first move a closed seat is still the table's to change:
        # seating a bot there reopens it. After, it is retired.
        reopening = kind is SeatKind.EMPTY and seat in game.locked
        if reopening and table.session.steps > 0:
            raise ApiError(f"seat {seat} has been retired from this game")
        if reopening and seat in table.session.fixed_seats:
            raise ApiError(_fixed_seats(table))
        spec = _seated_spec(model, table.session.game_type.name)
        # true state: the board is public.
        bot = seat_spawn(
            model, spec, game.state(0, hidden=False).board, self.config, game.num_players,
            table.session.game_type.name,
        )

        if reopening:
            unlock_seat(game, seat)
            if table.session.journal is not None:
                table.session.journal.unlocked(seat, at_step=table.session.steps)
        if kind is SeatKind.EMPTY:
            token = secrets.token_urlsafe(18)
            table.seats[seat] = Seat(kind=SeatKind.BOT, name=model, spec=spec, token=token)
            # A seat only counts as playable once the session agrees it is
            # claimed, which is the route the new runner plays through.
            table.session.claimed_seats.add(seat)
        else:
            table.seats[seat].name = model
            table.seats[seat].spec = spec
        table.session.bot_names[seat] = model
        table.session.bot_specs[seat] = spec
        if table.session.journal is not None:
            table.session.journal.seated(seat=seat, name=model, spec=spec)

        _stop_runner(table, seat)
        table.session.set_trader(seat, bot)
        self._start_runners(table, {seat: bot})

        table.bump()
        return table.view(viewer)

    def _unseat_bot(self, table: Table, seat: int) -> None:
        """Takes the bot off `seat`, leaving it empty: only `close_seat` and
        `open_seat` call it, after their own before-the-first-move checks,
        so a bot picked by mistake can be taken back."""
        _stop_runner(table, seat)
        table.seats[seat] = Seat()
        table.session.claimed_seats.discard(seat)
        table.session.bot_names.pop(seat, None)
        table.session.bot_specs.pop(seat, None)
        table.session.set_trader(seat, None)
        if table.session.journal is not None:
            table.session.journal.unseated(seat=seat)

    def close_seat(self, table: Table, viewer: int, seat: int) -> dict:
        """`POST /api/close`: close `seat` outright — no bot, no person, for the
        rest of this game. Any seated person may. Refuses a person's seat,
        takes a bot off its seat first, is idempotent for one already closed,
        and works only before the first move (409 after) at a table whose
        seat was not retired at a fixed-seat deal (`GameSession.fixed_seats`,
        409). `open_seat` is
        the reverse, under the same rule."""
        _unranked(table)
        if not 0 <= seat < len(table.seats):
            raise ApiError(f"there is no seat {seat} at this game")
        if table.seats[seat].kind is SeatKind.PLAYER:
            raise ApiError(f"seat {seat} belongs to a player")
        if table.session.steps > 0:
            raise ApiError("seats are fixed once play has started", status=409)
        if seat in table.session.fixed_seats:
            raise ApiError(_fixed_seats(table), status=409)
        if table.seats[seat].kind is SeatKind.BOT:
            self._unseat_bot(table, seat)
            table.bump()
        if seat not in table.session.game.locked:
            lock_seat(table.session.game, seat)
            if table.session.journal is not None:
                table.session.journal.locked(seat, at_step=table.session.steps)
            table.bump()
        return table.view(viewer)

    def open_seat(self, table: Table, viewer: int, seat: int) -> dict:
        """`POST /api/open`: reopen a closed `seat`, or take the bot off one.
        Empty again, it holds the table (`Table.waiting_for`) until it is
        filled or closed once more. Refused once play has started, like
        `close_seat`, and for a person's seat; idempotent for a seat already
        open."""
        _unranked(table)
        if not 0 <= seat < len(table.seats):
            raise ApiError(f"there is no seat {seat} at this game")
        if table.seats[seat].kind is SeatKind.PLAYER:
            raise ApiError(f"seat {seat} belongs to a player")
        if table.session.steps > 0:
            raise ApiError("seats are fixed once play has started", status=409)
        if seat in table.session.fixed_seats:
            raise ApiError(_fixed_seats(table), status=409)
        if table.seats[seat].kind is SeatKind.BOT:
            self._unseat_bot(table, seat)
            table.bump()
        if seat in table.session.game.locked:
            unlock_seat(table.session.game, seat)
            if table.session.journal is not None:
                table.session.journal.unlocked(seat, at_step=table.session.steps)
            table.bump()
        return table.view(viewer)

    def leave_seat(self, table: Table, viewer: int) -> dict:
        """`POST /api/leave`: retire your own seat for the rest of this game.
        Nothing happens to its hand or pieces, only to whose turn comes next,
        and permanently. Refuses while a trade round is open naming you as its
        actor or as a seat it is owed an answer from: resolve that first. The
        last seat still in the game cannot leave it (409), there being nobody
        to hand the turn to. A setup turn the seat was holding open, and its
        own take-back, go with it (`GameSession.release`). When the seats
        left are all bots, the table is closed: its runners stop and its
        journal is filed as abandoned (`Table.close`)."""
        game = table.session.game
        require_live(game)
        if viewer in game.locked:
            return table.view(viewer)
        round_ = table.session.open_round
        if round_ is not None and (round_.offer.actor == viewer or viewer in round_.awaiting):
            raise ApiError("resolve the open trade round first (answer_trade/choose_trade), then leave")
        if all(s == viewer or s in game.locked for s in range(game.num_players)):
            raise ApiError("you are the last seat in this game; there is nobody to hand it to", status=409)
        lock_seat(game, viewer)
        table.session.release(viewer)
        if table.session.journal is not None:
            table.session.journal.locked(viewer, at_step=table.session.steps)
        table.bump()
        if not any(
            s not in game.locked and seat.kind is not SeatKind.BOT
            for s, seat in enumerate(table.seats)
        ):
            # Only bots are left, with no seat a player could still take: the
            # game is abandoned rather than played out to nobody. Closed off
            # this request's thread, since `close` waits on runners that may
            # need the table this request holds.
            threading.Thread(target=table.close, daemon=True).start()
        return table.view(viewer)

    # --- the trade round (`hexset.trading`, "The trade round") --------------

    def open_round(self, table: Table, seat: int, payload: dict) -> dict:
        """`POST /api/games/<CODE>/trade/round`: broadcast one offer to every
        other seat. `{"give": [5 ints], "want": [5 ints]}`, unsigned counts in
        `RESOURCE_NAMES` order, on disjoint resources and covered by the hand;
        `"give_any"`/`"want_any"` add any cards to one side, which makes the
        offer an invitation to counter (`hexset.trading.Offer.any`).
        How many cards is the seat's own business. Only on `seat`'s own turn
        in MAIN with no Road Building roads still to place (409), and only
        while `seat` holds `give` (400). Bots answer at once; a manual seat
        answers later. The returned view's `trade_round` carries the answers
        so far and who is still to answer."""
        game = table.session.game
        if game.phase is not Phase.MAIN:
            raise ApiError(f"trading is only open in MAIN, not {game.phase.name}", status=409)
        if game.current_player != seat:
            raise ApiError("it is not your turn to open a round", status=409)
        if pending_free_roads(game):
            raise ApiError("place the roads Road Building owes first", status=409)
        received, any_cards = round_offer_from_wire(
            payload.get("give") or [], payload.get("want") or [],
            payload.get("give_any", 0), payload.get("want_any", 0),
        )
        # true state: the engine is the referee for coverage.
        state = game.state(0, hidden=False)
        if not holds(state, seat, [max(0, -n) for n in received]):
            raise ApiError("you cannot cover your side of that offer", status=400)
        table.session.open_round_for(seat, received, any_cards)
        table.bump()
        return table.view(seat)

    def answer_round(self, table: Table, seat: int, payload: dict) -> dict:
        """`POST /api/games/<CODE>/trade/round/answer`: `{"actor": <seat>,
        "received": [5 ints], "kind": "accept"|"counter"|"pass", "bundle":
        [5 ints]?}`. `received` is the exact offer this seat's `pending`
        showed, signed towards `actor`; `bundle` is the counter, signed the
        same way. 409 for an offer no longer open, for a seat the round is not
        waiting on, and for an accept or counter the seat cannot cover."""
        _check_version(table, payload.get("version"))
        actor = payload.get("actor")
        if not isinstance(actor, int):
            raise ApiError("send the offer's `actor` seat")
        kind = payload.get("kind")
        if kind not in ("accept", "counter", "pass"):
            raise ApiError('kind must be "accept", "counter" or "pass"')
        received = signed_bundle_from_wire(payload.get("received") or [])
        bundle = (
            signed_bundle_from_wire(payload.get("bundle"))
            if kind != "pass" and payload.get("bundle") is not None
            else (received if kind == "accept" else None)
        )
        try:
            table.session.answer_round(seat, actor, received, kind, bundle)
        except ValueError as error:
            raise ApiError(str(error), status=409) from None
        table.bump()
        return table.view(seat)

    def choose_round(self, table: Table, seat: int, payload: dict) -> dict:
        """`POST /api/games/<CODE>/trade/round/choose`: `{"seat": <seat that
        answered>, "bundle": [5 ints]}` executes that exact recorded answer;
        `{"decline": true}` closes the round with nothing moved."""
        _check_version(table, payload.get("version"))
        if payload.get("decline"):
            try:
                table.session.decline_round(seat)
            except ValueError as error:
                raise ApiError(str(error), status=409) from None
            table.bump()
            return table.view(seat)
        responder = payload.get("seat")
        if not isinstance(responder, int):
            raise ApiError("send the responding `seat`")
        bundle = signed_bundle_from_wire(payload.get("bundle") or [])
        try:
            table.session.execute_round_choice(seat, responder, bundle)
        except ValueError as error:
            raise ApiError(str(error), status=409) from None
        table.bump()
        return table.view(seat)

    def record(self, table: Table, seat: int) -> dict:
        """`GET /api/record`: the information-set record `hexset.onnx_record`
        builds for a checkpoint, byte-for-byte what an in-process bot would
        compute, plus the `options` it is masked against and the action
        `space` dimensions. 409 unless this seat may act."""
        game = table.session.game
        # `may_act`, not `to_move`: every seat owing cards to a seven may act
        # at once, so each is owed a record.
        if is_over(game) or not may_act(game, seat):
            raise ApiError("it is not your turn to act", status=409)
        options = legal_actions(game, seat)
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

    def await_change(self, table: Table, query: str) -> None:
        """Park a read until `table` has changed past the `after` it named, or
        its `wait` runs out. Must be called with no lock held: a waiter
        holding one could not be woken by the mutation it waits for."""
        after, wait = wait_query(query)
        if after is not None:
            table.wait_for_change(after, wait)

    def _replay_viewer(self, code: str, token: str | None) -> int | None:
        """The seat `token` holds at `code`, or `None` for everyone else.
        Lenient where `by_token` is strict: a replay read is open to anyone
        holding the link, so an unknown token reads as a spectator rather than
        an error."""
        if not token:
            return None
        try:
            table, seat = self.by_token(token)
        except ApiError:
            return None
        return seat if table.code == code else None

    def replay_view(self, code: str, query: str, token: str | None = None) -> dict:
        """`GET /api/table/<code>/replay?round=N`: the board at the *end* of
        round N — once every one of its steps has been played, so round 0 is
        the finished opening. The round asked for is clamped, not refused, and
        the position is read from the journal every time, with no cache, since
        a game in progress can `undo`, which rewrites rather than appends.

        **Read as whoever is asking.** A seat gets its own seat's view of that
        round, hiding every other hand exactly as its live view does; only a
        reader with no seat gets the omniscient one. Answering omniscient to a
        token holder would turn stepping back into a way of reading hidden
        cards and then playing on with them.
        """
        code = code.lower()
        asked = round_query(query)
        path = journal.most_recent(journal_dir(self.config), code)
        if path is None:
            raise ApiError(f"no game with code {code}", status=404)
        events = journal.read(path)
        steps, rounds = journal.replayable_rounds(events)
        last = max(rounds, default=0)
        wanted = max(0, min(asked, last))
        # The first step *past* the wanted round.
        upto = next((i for i, played in enumerate(rounds) if played > wanted), len(steps))
        seats = reopened_seats(events)
        session = replay_session(code, seats, events, upto)
        if session is None:
            raise ApiError(f"no game with code {code}", status=404)
        # Read through a `Table` of its own, so everything `Table.view` adds
        # is present and cannot drift from the live shape.
        standin = Table(
            code=code,
            seats=seats,
            config=self.config,
            session=session,
            layout=board_layout(session.game.state(0, hidden=False).board),
        )
        viewer = self._replay_viewer(code, token)
        # Whether the *game* was played out, which no stand-in stopped at an
        # earlier round can tell from its own position.
        finished = journal.is_finished(events)
        view = standin.view(viewer, omniscient=viewer is None, finished=finished)
        # A replayed round is inert: the stand-in is a real session, so a seat
        # would otherwise get back the moves it had *then* and send them to the
        # live table at a later position entirely.
        view["legal_actions"] = []
        view["can_undo"] = False
        # Nothing is owed at a replayed round, an offer it shows pending
        # included.
        view["your_move"] = "game_over" if view.get("game_over") else "wait"
        view["waiting_on"] = []
        # The round being *read*: the cut for round N falls after its last
        # END_TURN, which has already ticked `session.round` to N+1.
        view["round"] = wanted
        # `version` here is the stand-in's own and means nothing outside it:
        # replay steps by round, not by long poll. `finished` rides along so a
        # client can match the live finished view's observer footing.
        view["replay"] = {"round": wanted, "last_round": last, "finished": finished}
        return view

    def handle(
        self, method: str, path: str, payload: dict, token: str | None, origin: Origin | None = None
    ) -> dict:
        """One request, dispatched. Raises `ApiError` for anything refused.
        Every transport ends up here, so routing cannot drift between them,
        and the query string is split off here, so `/api/state?after=7&wait=20`
        means the same thing over HTTP and in-process. `origin` is where the
        transport says the request came from: it is journalled with every seat
        a request claims, and limits who may deal (`admit`)."""
        path, _, query = path.partition("?")
        if method == "GET" and path in ("/api", "/api/"):
            return {"api": API_VERSION, "docs": API_DOCS}
        if method == "GET" and path == "/api/version":
            return {"api": API_VERSION}
        if method == "GET" and path == "/api/models":
            return _models_view()
        if method == "GET" and path == "/api/rankings":
            if self.rankings is None:
                raise ApiError("this server keeps no rankings", status=404)
            return self.rankings.view()

        # The reads that need no token: `GET /api/table/<code>` is the game as
        # a spectator sees it, `.../board` the layout, `.../replay` a past
        # round. **A spectator sees everything** — every hand, every dev card,
        # every true victory-point count, an unredacted transcript — and the
        # route cannot be authenticated, holding the link being the whole
        # qualification. Every route that *acts* still answers a token and gets
        # its own seat's honest view.
        if method == "GET" and path.startswith("/api/table/"):
            code, _, tail = path[len("/api/table/") :].partition("/")
            table = self.get(code)
            if not tail:
                self.await_change(table, query)
                table.last_seen = time.monotonic()
                with table.lock:
                    return table.view(None, omniscient=True)
            if tail == "board":
                return table.layout
            if tail == "replay":
                return self.replay_view(code, query, token)
            raise ApiError(f"no such endpoint: {method} {path}", status=404)

        if method == "POST" and path == "/api/games":
            holder = self.holder(origin)
            client = stamp_client(parse_client(payload), origin, holder)
            require_profile(client, origin)
            self.admit(origin, holder)
            try:
                table, new_token = self.create(
                    bots=payload.get("bots"),
                    name=payload.get("name"),
                    client=client,
                    board_mode=payload.get("board_mode"),
                    game_type=payload.get("game_type"),
                )
            except ApiError:
                # A deal refused for its own reasons costs no allowance.
                self.unadmit(origin, holder)
                raise
            if origin is not None:
                table.creator_ip = origin.ip or "?"
            self._seen(client)
            limits = self.allowance(origin, holder)
            with table.lock:
                out = {"token": new_token, **table.view(table.seat_of(new_token))}
            if limits is not None:
                out["allowance"] = limits
                if limits["live_tables"] > 1:
                    out["notice"] = PARALLEL_NOTICE
            return out

        if method == "GET" and path == "/api/limits":
            return self.allowance(origin, self.holder(origin)) or {}

        if method == "POST" and path == "/api/join":
            table = self.get(str(payload.get("code", "")))
            client = stamp_client(parse_client(payload), origin)
            require_profile(client, origin)
            with table.lock:
                seat, new_token = table.join(payload.get("name"), client)
                joined = {"token": new_token, **table.view(seat)}
            self._seen(client)
            return joined

        if method == "POST" and path == "/api/ranked":
            return self.queue_join(payload, origin)
        if path.startswith("/api/ranked/"):
            ticket_id, _, tail = path[len("/api/ranked/"):].partition("/")
            if method == "GET" and not tail:
                return self.queue_wait(ticket_id, query)
            if method == "POST" and tail == "leave":
                return {"left": self.queue.leave(ticket_id)}
            raise ApiError(f"no such endpoint: {method} {path}", status=404)

        if method == "POST" and path == "/api/reclaim":
            table, seat, new_token = self.reclaim(
                str(payload.get("code", "")), str(payload.get("secret", ""))
            )
            with table.lock:
                return {"token": new_token, **table.view(seat)}

        # Everything past here acts on a seat, so it needs a token.
        table, seat = self.by_token(token)
        if method == "GET" and path == "/api/state":
            # Outside the lock below, which the mutation this waits for needs.
            self.await_change(table, query)
            if table.seats[seat].kind is not SeatKind.BOT:
                table.last_seen = time.monotonic()
        with table.lock:
            # The session raises `ValueError` for an illegal or out-of-turn
            # move, which is a 400.
            try:
                return self._seated(table, seat, method, path, payload, token)
            except ValueError as error:
                raise ApiError(str(error)) from None

    def _seated(
        self, table: Table, seat: int, method: str, path: str, payload: dict, token: str
    ) -> dict:
        """The routes that act on one seat. Called with the table's lock held.
        Every route but the three reads is a mutation, and one `require_live`
        gate here refuses every POST once `is_over`, so a finished game is
        read-only for its seats as well as for spectators."""
        if method == "POST":
            require_live(table.session.game)
        if method == "GET" and path == "/api/state":
            return table.view(seat)
        if method == "GET" and path == "/api/board":
            return table.layout
        if method == "GET" and path == "/api/record":
            return self.record(table, seat)
        if method == "POST" and path == "/api/action":
            return self.act(table, seat, payload.get("action") or {}, payload.get("version"))
        if method == "POST" and path == "/api/undo":
            return self.undo(table, seat)
        if method == "POST" and path == "/api/name":
            return self.rename(table, seat, payload.get("name"))
        if method == "POST" and path == "/api/bot":
            return self.seat_bot(table, seat, seat_field(payload), str(payload.get("model", "")))
        if method == "POST" and path == "/api/open":
            return self.open_seat(table, seat, seat_field(payload))
        if method == "POST" and path == "/api/close":
            return self.close_seat(table, seat, seat_field(payload))
        if method == "POST" and path == "/api/leave":
            return self.leave_seat(table, seat)
        if method == "POST" and path == f"/api/games/{table.code}/trade/round":
            return self.open_round(table, seat, payload)
        if method == "POST" and path == f"/api/games/{table.code}/trade/round/answer":
            return self.answer_round(table, seat, payload)
        if method == "POST" and path == f"/api/games/{table.code}/trade/round/choose":
            return self.choose_round(table, seat, payload)
        raise ApiError(f"no such endpoint: {method} {path}", status=404)
