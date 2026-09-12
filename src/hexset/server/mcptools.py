"""The MCP tool layer: what an LLM seat calls, translated to/from the wire
`api.py` speaks -- served over HTTP by `web.py`'s `POST /mcp` now, not a
separate stdio process (see `web.py`'s module docstring on why that program
is gone).

Every tool below reaches the game through `tables.handle(method, path,
payload, token)` -- the same in-process seam `web.py` and
`clients.botclient.LocalTransport` use, never `Tables`/`Table` internals
directly. `ApiError` becomes `ToolError`, the shape `_call_tool`/`web.py`
report back to the LLM as a normal (not protocol-level) tool result.

Identity used to be a handful of module globals (`_token`/`_code`/`_identity`)
because one stdio process was one seat. An HTTP server serves many MCP
sessions at once, so that state now lives in a `Session` object -- one per
`Mcp-Session-Id` -- threaded through every call instead.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Iterator

from .api import ApiError, Tables

# Resource order every wire bundle (5 signed or unsigned ints) uses -- see
# `catanatron/names.py`'s RESOURCE_NAMES, Title-cased as the wire's hand/bank
# dicts spell them.
RESOURCES = ("Wood", "Brick", "Sheep", "Wheat", "Ore")


class ToolError(Exception):
    """Raised by a tool implementation to report the failure back to the
    LLM as a normal (not protocol-level) tool result."""


@dataclass
class Session:
    """One MCP session's seat: the token `tables.handle` acts with, the game
    code the trade routes are addressed by, and the `identity` string the seat's
    client id was minted from (needed if `resume_game` is ever called with it again).
    Scoped to one `Mcp-Session-Id`, held in memory by `web.py` for as long as
    that session lives -- nothing here ever touches disk."""

    token: str | None = None
    code: str | None = None
    identity: str | None = None
    # How many transcript lines this session has been sent so far, or `None`
    # for none yet -- the automatic cursor `_trim_for` trims the next reply's
    # `log` against, so a caller that never sends `log_after` still gets only
    # what is new. Reset whenever the session takes a seat (`_seat`,
    # `_resume_game`): a new seat owes a whole transcript.
    log_sent: int | None = None
    # The annotated board (`_layout`) for the game this seat is at, fetched
    # once per seat -- it never changes after the deal -- and read by every
    # reply's `summary`. `None` until the seat is taken.
    board: dict | None = None


def _call_status(tables: Tables, session: Session, method: str, path: str, body: dict | None = None) -> tuple[int, dict]:
    try:
        return 200, tables.handle(method, path, body or {}, session.token)
    except ApiError as error:
        return error.status, {"error": str(error)}


def _call_ok(tables: Tables, session: Session, method: str, path: str, body: dict | None = None) -> dict:
    status, result = _call_status(tables, session, method, path, body)
    if "error" in result:
        raise ToolError(result["error"])
    return result


def _seated(session: Session) -> None:
    if session.token is None:
        raise ToolError("not at a game yet — call new_game() or join(code) first")


def _seat(session: Session, result: dict) -> dict:
    """Records the token a join or a deal handed back, and hides it again --
    the LLM never needs to see it, it is sent on its behalf by `_call_ok`."""
    session.token = result.pop("token")
    session.code = result.get("code")
    session.log_sent = None
    session.board = None
    return result


def _client_of(identity: str) -> tuple[dict, str]:
    """The `client` wire field and the secret behind it, from an identity
    string: `secret = identity.strip().lower()`, `id = sha256(secret)`, kind
    `"mcp"` -- no env-var override, the caller supplies its own string."""
    if not isinstance(identity, str) or not identity.strip():
        raise ToolError("identity must be a non-empty string naming you, e.g. claude-opus-5")
    secret = identity.strip().lower()
    client = {"id": hashlib.sha256(secret.encode("utf-8")).hexdigest(), "kind": "mcp"}
    return client, secret


def _bots(tables: Tables, session: Session) -> dict:
    return _call_ok(tables, session, "GET", "/api/models")


def _display_name(name: str | None) -> str | None:
    """The 40-character cap every client applies, or `None` for no name at
    all -- the server defaults an unnamed mcp seat to "mcp" itself
    (`api.default_seat_name`), so there is nothing left for this to fall
    back to."""
    return str(name).strip()[:40] if name else None


def _new_game(
    tables: Tables,
    session: Session,
    identity: str,
    opponents: list[str] | None = None,
    name: str | None = None,
    timeout: float | None = None,
    log_after: int | None = None,
    full_log: bool = False,
) -> Iterator:
    client, _ = _client_of(identity)
    session.identity = identity
    body: dict = {"name": _display_name(name), "client": client}
    if opponents:
        body["bots"] = opponents
    # Translated like every other state reply: the deal is the first view an
    # LLM reads, and it used to be the one that came back raw.
    dealt = _seat(session, _call_ok(tables, session, "POST", "/api/games", body))
    _layout(tables, session)  # the board is fixed from here on; fetch it once now
    return _settle(tables, session, dealt, timeout, log_after, full_log)


def _join(
    tables: Tables,
    session: Session,
    code: str,
    identity: str,
    name: str | None = None,
    timeout: float | None = None,
    log_after: int | None = None,
    full_log: bool = False,
) -> Iterator:
    if not isinstance(code, str) or not code.strip():
        raise ToolError("code must be a game's six-character code")
    client, _ = _client_of(identity)
    session.identity = identity
    body: dict = {"code": code.strip().lower(), "name": _display_name(name), "client": client}
    joined = _seat(session, _call_ok(tables, session, "POST", "/api/join", body))
    _layout(tables, session)
    return _settle(tables, session, joined, timeout, log_after, full_log)


def _resume_game(
    tables: Tables,
    session: Session,
    code: str,
    identity: str,
    timeout: float | None = None,
    log_after: int | None = None,
    full_log: bool = False,
) -> Iterator:
    """Reclaims a seat by `POST /api/reclaim`, the same way a browser's own
    reclaim works -- no local cache file any more (there is nothing left to
    cache: a session dies with its `Mcp-Session-Id`, and a fresh MCP
    connection just calls this again with the `code`/`identity` it already
    knows). `secret = identity.strip().lower()`, the same secret `new_game`/
    `join` mint from `identity`."""
    if not isinstance(code, str) or not code.strip():
        raise ToolError("code must be a game's six-character code")
    _, secret = _client_of(identity)
    reclaimed = _call_ok(tables, session, "POST", "/api/reclaim", {"code": code.strip().lower(), "secret": secret})
    session.token = reclaimed.pop("token")
    session.code = reclaimed.get("code")
    session.identity = identity
    session.log_sent = None  # a reclaimed seat is owed the whole transcript
    session.board = None
    _layout(tables, session)
    return _settle(tables, session, reclaimed, timeout, log_after, full_log)


#  --- Board summary -----------------------------------------------------------
#
# `board()`'s `hexes` name a terrain (`FOREST`, not `Wood` -- `hexset.board.
# terrain.Terrain`, a different enum from the resource it yields) and a dice
# token, and nothing joins the two into what a placement decision actually
# weighs: which resources a vertex touches and how likely each is to pay out.
# Working that out by hand from raw hex/vertex adjacency for every placement
# is exactly the kind of arithmetic an LLM does unreliably at the board's
# full size -- so it's done once here instead, from data `/api/board`
# already sends, no server round-trip or engine import added.

# Ways to roll a token on 2d6 -- the standard settlement-value weight; 7 is
# the robber's own roll and never labels a hex, so it never appears here.
_PIPS = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}

# Terrain -> the resource it yields. DESERT/SEA/GOLD are left out (and so
# read back as `None`): a desert and the sea pay nothing, and gold pays the
# collecting seat's own choice, not a fixed one the tile could name.
_TERRAIN_RESOURCE = {
    "FOREST": "Wood",
    "HILLS": "Brick",
    "PASTURE": "Sheep",
    "FIELDS": "Wheat",
    "MOUNTAINS": "Ore",
}
_BOARD_DEAD = ("size", "resources", "dev_cards", "year_of_plenty_pairs")


def _board(tables: Tables, session: Session) -> dict:
    _seated(session)
    return _layout(tables, session)


def _layout(tables: Tables, session: Session) -> dict:
    """The annotated board for this session's game, fetched once and kept on
    the `Session` (`_seat` clears it: a new seat may be a new board). Both
    the `board` tool and every reply's `summary` read it from here."""
    if session.board is not None:
        return session.board
    raw = _call_ok(tables, session, "GET", "/api/board")
    # Render geometry and constant tables (`hexset.board`'s own lists, the
    # Year of Plenty pairs `_translate_action` already names) are the
    # browser's; a seat reads ids, terrain, tokens and adjacency.
    for key in _BOARD_DEAD:
        raw.pop(key, None)
    by_vertex: dict[int, list[tuple[str, int]]] = {}
    for hex_ in raw.get("hexes") or []:
        hex_.pop("x", None)
        hex_.pop("y", None)
        resource = _TERRAIN_RESOURCE.get(hex_["terrain"])
        pips = _PIPS.get(hex_["token"], 0)
        hex_["resource"] = resource
        hex_["pips"] = pips
        if resource is None:
            continue
        for v in hex_["vertex_ids"]:
            by_vertex.setdefault(v, []).append((resource, pips))
    for vertex in raw.get("vertices") or []:
        vertex.pop("x", None)
        vertex.pop("y", None)
        touching = by_vertex.get(vertex["id"], [])
        vertex["pips"] = sum(pips for _, pips in touching)
        vertex["resources"] = sorted({resource for resource, _ in touching})
    session.board = raw
    return raw


def _state(tables: Tables, session: Session, log_after: int | None = None, full_log: bool = False) -> dict:
    _seated(session)
    return _reply(session, _call_ok(tables, session, "GET", "/api/state"), log_after, full_log)


# --- Guarding an index against a table that moved ----------------------------
#
# `act` used to take the table's `version` and refuse if it had changed. That
# counter (`api.Table.version`) bumps on *every* change anybody makes -- an
# opponent answering your trade offer, a bot's move at another seat, a rename,
# even a read that fires a pending trade event -- so in a 4-seat game it was
# stale by the time the reply had been read, and a seat answering a trade
# round could never pass it at all: the other seats' answers to the same
# offer bumped it under them. What an index actually has to be stable
# against is narrower: that `legal_actions[index]` still names the action the
# caller chose. `expect` checks exactly that, and nothing else.

def _expect_check(index: int, chosen: dict, expect: dict | None, num_players: int) -> None:
    """`expect` is the `legal_actions` entry the caller chose, plus its group
    key as `type`. Any key it carries that the entry has -- the named operand
    (`edge`/`vertex`/`hex`/`victim`), a named resource, or the raw `a`/`b` --
    is compared; anything else (`index`, notes) is ignored."""
    if expect is None:
        return
    if not isinstance(expect, dict) or "type" not in expect:
        raise ToolError("expect must be the legal_actions entry you chose, with its group key as `type`")
    translated = _translate_action(chosen)
    actual = {"type": chosen.get("type"), **_group_actions([translated], num_players)[chosen.get("type")][0]}
    del actual["index"]
    comparable = {**actual, "a": chosen.get("a"), "b": chosen.get("b")}
    mismatch = {k: v for k, v in expect.items() if k in comparable and comparable[k] != v}
    if mismatch:
        raise ToolError(
            f"legal_actions moved under you: index {index} is now {actual}, not the "
            f"{mismatch} you chose — call state() again and pick from the fresh list"
        )


def _act(
    tables: Tables,
    session: Session,
    index: int,
    expect: dict | None = None,
    timeout: float | None = None,
    log_after: int | None = None,
    full_log: bool = False,
) -> Iterator:
    _seated(session)
    # The raw list, not the translated one: only `index` is resolved here, and
    # `POST /api/action` reads `type`/`a`/`b` alone (`wire_to_action`).
    raw = _call_ok(tables, session, "GET", "/api/state")
    options = raw.get("legal_actions") or []
    if not isinstance(index, int) or not (0 <= index < len(options)):
        raise ToolError(
            f"index {index!r} out of range — state()'s legal_actions has "
            f"{len(options)} option(s) right now (0..{len(options) - 1})"
            if options
            else "index out of range — state()'s legal_actions is empty; it is not your turn"
        )
    _expect_check(index, options[index], expect, len(raw.get("players") or []))
    played = _call_ok(tables, session, "POST", "/api/action", {"action": options[index]})
    return _settle(tables, session, played, timeout, log_after, full_log)


def _undo(tables: Tables, session: Session, timeout: float | None = None) -> Iterator:
    _seated(session)
    return _settle(tables, session, _call_ok(tables, session, "POST", "/api/undo"), timeout)


def _leave_game(tables: Tables, session: Session) -> dict:
    _seated(session)
    return _reply(session, _call_ok(tables, session, "POST", "/api/leave"))


# --- Trading (docs/bot-api.md §3; the human/LLM surface, `agents/reference/
# trading-final.md` item 5) ---------------------------------------------------
#
# The wire speaks in 5-count arrays, some unsigned and some signed towards
# whichever seat proposed the trade (`hexset.trading`'s docstring on
# `_validate_exchange`) -- easy to get backwards, and not worth an LLM
# reasoning about at all. Everything below this line trades in named dicts
# from each tool caller's own point of view (`you_give`/`you_receive`)
# instead, and resolves an `index` against a fresh, *raw* state fetch the
# same way `_act(index)` resolves one against `legal_actions` -- so nobody
# has to hold a stale bundle across a round-trip either.


def _named(counts: list[int]) -> dict[str, int]:
    return {name: n for name, n in zip(RESOURCES, counts) if n}


def _positional(counts: dict | None) -> list[int]:
    counts = counts or {}
    unknown = set(counts) - set(RESOURCES)
    if unknown:
        raise ToolError(f"not a resource name: {', '.join(sorted(unknown))}")
    return [int(counts.get(name, 0)) for name in RESOURCES]


def _actor_view(bundle: list[int]) -> tuple[dict, dict]:
    """(gives, gets) for the seat a signed bundle is signed towards."""
    return _named([max(0, -n) for n in bundle]), _named([max(0, n) for n in bundle])


def _counterparty_view(bundle: list[int]) -> tuple[dict, dict]:
    """(gives, gets) for the seat on the *other* side of a signed bundle --
    one side's gives are the other's gets, so this is `_actor_view` swapped."""
    actor_gives, actor_gets = _actor_view(bundle)
    return actor_gets, actor_gives


def _bundle_towards_actor(give: dict | None, receive: dict | None) -> list[int]:
    """The inverse of `_counterparty_view`: builds a signed bundle from what
    the *counterparty* (the tool caller) would give and receive."""
    overlap = set(give or {}) & set(receive or {})
    if overlap:
        raise ToolError(f"can't both give and receive the same resource: {', '.join(sorted(overlap))}")
    gives, receives = _positional(give), _positional(receive)
    return [g - r for g, r in zip(gives, receives)]


def _translate_trades(raw: dict) -> dict:
    raw["trades"] = [
        {"a": t["a"], "b": t["b"], "a_gave": _named(t["gave"]), "a_got": _named(t["got"])}
        for t in raw.get("trades") or []
    ]
    raw["pending"] = [
        {"actor": t["actor"], **dict(zip(("you_give", "you_receive"), _counterparty_view(t["bundle"])))}
        for t in raw.get("pending") or []
    ]
    trade_round = raw.get("trade_round")
    if trade_round is not None:
        you_give, you_receive = _actor_view(trade_round["offer"]["bundle"])
        raw["trade_round"] = {
            "you_give": you_give,
            "you_receive": you_receive,
            # A pass carries no bundle: the seat is named so it is told
            # apart from one still in `awaiting`, and nothing can be chosen.
            "responses": [
                {
                    "seat": r["seat"],
                    "kind": r["kind"],
                    **(
                        dict(zip(("you_would_give", "you_would_receive"), _actor_view(r["bundle"])))
                        if r["bundle"] is not None
                        else {}
                    ),
                }
                for r in trade_round["responses"]
            ],
            "awaiting": trade_round["awaiting"],
        }
    return raw


# --- Legal actions still carrying a raw resource index ------------------------
#
# `action_to_wire` (`webplay.py`) is the one wire format both a browser and
# an LLM read `legal_actions` through, so it stays positional there for
# either client to replay verbatim (`wire_to_action` reads only `type`/`a`/
# `b`, ignoring anything else) -- but three action types still spend a
# resource as a bare index into RESOURCES the same way trades used to:
# BANK_TRADE (`a`=give, `b`=want), PLAY_MONOPOLY and DISCARD (`a`=the
# resource). PLAY_YEAR_OF_PLENTY's `a` indexes a *pair* of resources instead
# (`hexset.actions.YEAR_OF_PLENTY_PAIRS`, mirrored below rather than
# imported -- see this module's docstring on staying engine-free).

_YEAR_OF_PLENTY_PAIRS = [(a, b) for a in range(len(RESOURCES)) for b in range(a, len(RESOURCES))]


def _translate_action(action: dict) -> dict:
    kind = action.get("type")
    if kind in ("PLAY_MONOPOLY", "DISCARD"):
        return {**action, "resource": RESOURCES[action["a"]]}
    if kind == "BANK_TRADE":
        return {**action, "give": RESOURCES[action["a"]], "want": RESOURCES[action["b"]]}
    if kind == "PLAY_YEAR_OF_PLENTY":
        pair = _YEAR_OF_PLENTY_PAIRS[action["a"]]
        return {**action, "resources": [RESOURCES[r] for r in pair]}
    return action


def _trim_log(view: dict, log_after: int | None) -> dict:
    """Answer only the transcript lines the caller does not already hold.

    `log_after` is how many it has. Every reply carries `log_total`, the
    full length, so the next call can pass it straight back, and `log_from`,
    the index `log[0]` sits at, so a caller splicing onto its own copy knows
    where to cut. Omitting the cursor returns the whole transcript, which is
    what a fresh reader and a reclaimed seat both want.

    One line of overlap is deliberate, and is what makes the cursor safe to
    splice. `render_log` collapses a burst of engine steps into a single line
    that it *rewrites in place* as the burst grows -- `emit()` pops the line
    it already wrote and appends the longer one -- so the last line a caller
    holds is the one line that can still change under it. Everything before
    it has settled. Re-sending exactly that line covers the rewrite, and
    covers a log that shrank under an undo too: the cursor is clamped into
    range rather than trusted, so a caller holding lines that no longer
    exist is answered with the tail that does.
    """
    lines = view.get("log")
    if not isinstance(lines, list):
        return view
    view["log_total"] = len(lines)
    # The final read is the one re-render a cursor cannot splice. `state_view`
    # asks for the transcript with `omniscient or over` once the game is over
    # (webplay.py), and that lifts redaction across the *whole* history at
    # once: every earlier steal stops being "a card" and names what it was.
    # Those are rewrites of lines the caller already holds, arbitrarily far
    # back, so the cursor is ignored here and the full transcript sent for the
    # client to replace its copy with -- `log_from: 0` is how it knows to.
    if log_after is not None and view.get("game_over"):
        log_after = None
    start = 0 if (log_after is None or not lines) else max(0, min(int(log_after) - 1, len(lines) - 1))
    view["log"] = lines[start:]
    view["log_from"] = start
    return view


# --- your_move: what the table wants from this seat right now -----------------
#
# The answer to "is it me, and which tool?" used to be spread over five
# fields -- `legal_actions`, `pending`, `trade_round.awaiting`, `trade_wait`,
# `to_move` -- and the one LLM game played through these tools spent several
# calls learning how they relate. `_turn_ready` already knew; this says it.

# `your_move` -> the tool that plays it. "wait" and "game_over" name none.
_MOVES = ("discard", "answer_trade", "choose_trade", "act", "wait", "game_over")


def _your_move(view: dict) -> tuple[str, list[int]]:
    """`(your_move, waiting_on)` for a translated view.

    `your_move` is one of `_MOVES`, checked in that order: a seven's discards
    come before anything else the rules allow, an offer standing against you
    before your own play, and your own fully-answered round before the rest
    of your turn. `waiting_on` is the seats that have to act before there is
    anything for you to do -- empty unless `your_move` is `"wait"`."""
    if view.get("game_over"):
        return "game_over", []
    legal = view.get("legal_actions") or []
    if legal and view.get("phase") == "DISCARD":
        return "discard", []
    if view.get("pending"):
        return "answer_trade", []
    trade_round = view.get("trade_round")
    if trade_round is not None and not trade_round.get("awaiting"):
        return "choose_trade", []
    if legal:
        return "act", []
    if trade_round is not None:
        return "wait", list(trade_round.get("awaiting") or [])
    if view.get("trade_wait"):
        return "wait", list(view["trade_wait"])
    if view.get("waiting_for"):
        return "wait", list(view["waiting_for"])
    owing = [seat for seat, n in enumerate(view.get("discard_quota") or []) if n]
    if owing:
        return "wait", owing
    to_move = view.get("to_move")
    return "wait", [] if to_move is None else [to_move]


# --- summary: the derived facts a turn actually turns on ----------------------
#
# `board()` already precomputes per-vertex pips and resources because that
# join is arithmetic an LLM does unreliably at the board's size. The same was
# true of everything below, and the first LLM game through these tools redid
# all of it by hand every turn: what a build costs against the hand, which of
# the legal spots is any good, which hex the robber hurts most, and how far
# the awards are. Each is a pure function of the view and the fixed board, so
# it is computed here once per reply -- engine-free, like `_layout`.

# Build costs, mirrored from `hexset.economy.COSTS` (this module imports no
# engine; see the module docstring). Keys are the `afford` entries.
_COSTS = {
    "road": {"Wood": 1, "Brick": 1},
    "settlement": {"Wood": 1, "Brick": 1, "Sheep": 1, "Wheat": 1},
    "city": {"Wheat": 2, "Ore": 3},
    "dev_card": {"Sheep": 1, "Wheat": 1, "Ore": 1},
}
_BUILD_ACTION = {
    "road": "BUILD_ROAD",
    "settlement": "BUILD_SETTLEMENT",
    "city": "BUILD_CITY",
    "dev_card": "BUY_DEV_CARD",
}
# `hexset.roads.MIN_LONGEST_ROAD` and the rulebook's three knights, mirrored.
_MIN_LONGEST_ROAD = 5
_MIN_LARGEST_ARMY = 3
# Setup offers every open vertex -- fifty-odd -- and the tail of that list by
# pips is never the answer. The count left off is reported as `spots_omitted`.
_SPOTS_CAP = 15
_SPOT_ACTIONS = ("SETUP_SETTLEMENT", "BUILD_SETTLEMENT", "BUILD_CITY")
_CITY = 2  # `hexset.state.Building.CITY`


def _afford(hand: dict, legal: list[dict]) -> dict:
    """Per build: `ok` (the hand covers it), `missing` (what it is short, when
    not), `legal` (whether `legal_actions` offers it right now -- a build can
    be affordable with nowhere to put it, or the phase may not allow it)."""
    offered = {a.get("type") for a in legal}
    out = {}
    for build, cost in _COSTS.items():
        missing = {r: n - hand.get(r, 0) for r, n in cost.items() if hand.get(r, 0) < n}
        entry: dict = {"ok": not missing, "legal": _BUILD_ACTION[build] in offered}
        if missing:
            entry["missing"] = missing
        out[build] = entry
    return out


def _port_label(port: dict) -> str:
    return f"{port['ratio']}:1" if port.get("resource") is None else f"{port['resource']} {port['ratio']}:1"


def _spots(legal: list[dict], board: dict) -> tuple[list[dict], int]:
    """Every settlement/city placement in `legal`, joined to the vertex's
    pips, resources and port, best first. `index` is the `act()` index."""
    vertices = {v["id"]: v for v in board.get("vertices") or []}
    ports = {v: p for p in board.get("ports") or [] for v in p.get("vertices") or []}
    spots = []
    for index, action in enumerate(legal):
        if action.get("type") not in _SPOT_ACTIONS:
            continue
        vertex_id = action.get("a")
        vertex = vertices.get(vertex_id, {})
        spot = {
            "index": index,
            "type": action["type"],
            "vertex": vertex_id,
            "pips": vertex.get("pips", 0),
            "resources": vertex.get("resources", []),
        }
        port = ports.get(vertex_id)
        if port is not None:
            spot["port"] = _port_label(port)
        spots.append(spot)
    spots.sort(key=lambda s: (-s["pips"], s["index"]))
    return spots[:_SPOTS_CAP], max(0, len(spots) - _SPOTS_CAP)


def _robber(legal: list[dict], view: dict, board: dict) -> list[dict]:
    """Every hex `MOVE_ROBBER` may go to, joined to what sits on it: the
    hex's resource and pips, `hits` (each seat's buildings on it, yours
    included) and `options` (one `act()` index per victim the engine offers,
    `victim: null` for nobody to steal from). Most productive hex first."""
    owner = view.get("vertex_owner") or []
    building = view.get("vertex_building") or []
    num_players = len(view.get("players") or [])
    hexes = {h["id"]: h for h in board.get("hexes") or []}
    by_hex: dict[int, dict] = {}
    for index, action in enumerate(legal):
        if action.get("type") != "MOVE_ROBBER":
            continue
        hex_id = action.get("a")
        entry = by_hex.get(hex_id)
        if entry is None:
            hex_ = hexes.get(hex_id, {})
            hits: dict[int, dict] = {}
            for v in hex_.get("vertex_ids") or []:
                if v < len(owner) and owner[v] >= 0:
                    hit = hits.setdefault(owner[v], {"seat": owner[v], "settlements": 0, "cities": 0})
                    hit["cities" if v < len(building) and building[v] == _CITY else "settlements"] += 1
            entry = by_hex[hex_id] = {
                "hex": hex_id,
                "resource": hex_.get("resource"),
                "pips": hex_.get("pips", 0),
                "hits": [hits[s] for s in sorted(hits)],
                "options": [],
            }
        slot = action.get("b")
        victim = None if slot is None or slot >= num_players else slot
        entry["options"].append({"index": index, "victim": victim})
    return sorted(by_hex.values(), key=lambda e: (-e["pips"], e["hex"]))


def _race(view: dict, me: dict) -> dict:
    """Where this seat stands: points and the distance to the win (the
    rule itself is the view's top-level `winning_points`), the leading
    opponent by *public* points (hidden victory-point cards are not
    counted for anyone else), and each award -- yours, who holds it, and
    `need`, the length or knight count that would take it (strictly more
    than the holder, or the minimum if nobody holds it yet)."""
    players = view.get("players") or []
    seat = me.get("seat")
    winning = view.get("winning_points") or 10
    points = me.get("victory_points", 0)
    others = [p for p in players if p.get("seat") != seat]
    leader = max(others, key=lambda p: (p.get("victory_points", 0), -p.get("seat", 0)), default=None)

    def award(flag: str, count: str, minimum: int) -> dict:
        holder = next((p for p in players if p.get(flag)), None)
        entry: dict = {"yours": me.get(count, 0), "held": holder is not None and holder.get("seat") == seat}
        if holder is not None:
            entry["holder"] = holder.get("seat")
            entry["holder_has"] = holder.get(count, 0)
        if not entry["held"]:
            entry["need"] = max(minimum, holder.get(count, 0) + 1) if holder is not None else minimum
        return entry

    return {
        "points": points,
        "to_win": max(0, winning - points),
        "leader": None if leader is None else {"seat": leader.get("seat"), "points": leader.get("victory_points", 0)},
        "longest_road": award("longest_road", "road_length", _MIN_LONGEST_ROAD),
        "largest_army": award("largest_army", "knights_played", _MIN_LARGEST_ARMY),
    }


def _summarize(view: dict, board: dict | None) -> dict:
    """Adds `summary` to a translated view for a seated reader whose hand
    the view reveals. `spots` and `robber` need the board and appear only
    when there is a placement or a robber move to make."""
    seat = view.get("seat")
    players = view.get("players") or []
    me = next((p for p in players if p.get("seat") == seat), None) if seat is not None else None
    if me is None or "hand" not in me:
        return view
    legal = view.get("legal_actions") or []
    summary: dict = {"afford": _afford(me["hand"], legal), "race": _race(view, me)}
    if board is not None:
        spots, omitted = _spots(legal, board)
        if spots:
            summary["spots"] = spots
            if omitted:
                summary["spots_omitted"] = omitted
        robber = _robber(legal, view, board)
        if robber:
            summary["robber"] = robber
    view["summary"] = summary
    return view


# --- The compact reply: grouped legal_actions, sparse occupancy ----------------
#
# The wire's `legal_actions` is a flat list of `{type, a, b}` -- the right
# shape for a browser to replay verbatim, and the shape `act(index)` still
# resolves against (`_act` reads the raw list). For a reader it repeats the
# type string on every one of fifty entries and carries a dead `b: 0` on most.
# The reply groups them by type, names the operand (`edge`, `vertex`, `hex`,
# `victim`, or the resource names `_translate_action` adds) and keeps the flat
# `index` on each entry, which is the one thing `act()` needs back.
#
# Occupancy came as three dense arrays indexed by id (`vertex_owner`,
# `vertex_building`, `edge_owner`), mostly -1. The reply lists what is
# actually there instead: `buildings` and `roads` by seat. Roughly neutral in
# tokens late in a game, cheaper early, and readable throughout -- "seat 2
# has a city at vertex 12" is a fact, position 12 of an array is a lookup.

_OPERANDS = {
    "BUILD_ROAD": ("edge",),
    "SETUP_ROAD": ("edge",),
    "BUILD_SETTLEMENT": ("vertex",),
    "SETUP_SETTLEMENT": ("vertex",),
    "BUILD_CITY": ("vertex",),
    "MOVE_ROBBER": ("hex", "victim"),
}
# The keys `_translate_action` adds for the types that spend a resource.
_NAMED = ("resource", "give", "want", "resources")


def _group_actions(legal: list[dict], num_players: int) -> dict[str, list[dict]]:
    """Translated legal actions -> `{type: [{index, <named operands>}]}`. A
    `MOVE_ROBBER` victim slot at or past the seat count means nobody
    (`hexset.actions.victim_of`) and reads back as `null`."""
    grouped: dict[str, list[dict]] = {}
    for index, action in enumerate(legal):
        kind = action.get("type")
        entry: dict = {"index": index}
        names = _OPERANDS.get(kind)
        if names is not None:
            entry[names[0]] = action.get("a")
            if len(names) > 1:
                slot = action.get("b")
                entry[names[1]] = None if slot is None or slot >= num_players else slot
        elif any(k in action for k in _NAMED):
            entry.update({k: action[k] for k in _NAMED if k in action})
        elif action.get("a") or action.get("b"):
            # A type this table does not know: keep the raw operands rather
            # than lose them.
            entry["a"], entry["b"] = action.get("a"), action.get("b")
        grouped.setdefault(kind, []).append(entry)
    return grouped


def _compact_board(view: dict) -> dict:
    """The three dense occupancy arrays -> `buildings` (vertex, seat, kind)
    and `roads` (edge ids, one list per seat)."""
    owner = view.pop("vertex_owner", None)
    building = view.pop("vertex_building", None)
    edges = view.pop("edge_owner", None)
    if owner is None or building is None or edges is None:
        return view
    view["buildings"] = [
        {"vertex": v, "seat": seat, "kind": "city" if building[v] == _CITY else "settlement"}
        for v, seat in enumerate(owner)
        if seat >= 0
    ]
    seats = max(len(view.get("players") or []), max(edges, default=-1) + 1)
    roads: list[list[int]] = [[] for _ in range(seats)]
    for edge, seat in enumerate(edges):
        if seat >= 0:
            roads[seat].append(edge)
    view["roads"] = roads
    return view


# Wire fields a reader never acts on, dropped from every reply (`_prune`).
# `version`: the whole-table counter no MCP tool takes (see `_expect_check`).
# `claimed_seats`: `players[].kind != "empty"`. `waiting_for`/`trade_wait`:
# `_your_move` has already folded them into `waiting_on`.
_DEAD = ("version", "claimed_seats", "waiting_for", "trade_wait")
# Per-seat counts sent sparse: a missing name is a zero, the convention the
# trade dicts already use (`_named`).
_SPARSE_COUNTS = ("hand", "known", "dev_cards")


def _sparse(counts: dict | None) -> dict:
    return {name: n for name, n in (counts or {}).items() if n}


def _prune(view: dict) -> dict:
    """The wire's redundancies, removed once per reply. `seats` (seat, kind,
    name) repeats `players` but for `kind`, which moves onto each player
    entry instead; a seat's `last_roll` is the table's own `last_roll`
    for the roller and stale for everyone else."""
    for key in _DEAD:
        view.pop(key, None)
    kinds = {s.get("seat"): s.get("kind") for s in view.pop("seats", None) or []}
    players = []
    for player in view.get("players") or []:
        player.pop("last_roll", None)
        for key in _SPARSE_COUNTS:
            if key in player:
                player[key] = _sparse(player[key])
        seat = player.get("seat")
        players.append({"seat": seat, "kind": kinds[seat], **player} if seat in kinds else player)
    if players:
        view["players"] = players
    if "bank" in view:
        view["bank"] = _sparse(view["bank"])
    return view


def _compact(view: dict) -> dict:
    """The last step before a reply goes out: everything above that reads
    the flat list, the dense arrays or the fields `_prune` drops
    (`_your_move`, `_summarize`) has run."""
    legal = view.get("legal_actions") or []
    view["legal_count"] = len(legal)
    view["legal_actions"] = _group_actions(legal, len(view.get("players") or []))
    return _prune(_compact_board(view))


def _translate(raw: dict) -> dict:
    """Every translation a view gets on its way to the LLM, except the
    transcript trim -- which needs to know what the session already holds
    (`_trim_for`), or an explicit cursor (`_translate_view`) -- and the
    `summary` and `_compact` steps `_reply` adds on top."""
    raw = _translate_trades(raw)
    raw["legal_actions"] = [_translate_action(a) for a in raw.get("legal_actions") or []]
    raw["your_move"], raw["waiting_on"] = _your_move(raw)
    return raw


def _translate_view(raw: dict, log_after: int | None = None) -> dict:
    """`_translate` plus a trim against an explicit cursor -- no session, so
    nothing here remembers what was sent. `_reply` is the one every tool
    answers through; this is for the polls `wait_for_turn` reads and throws
    away, and for tests of the translation alone."""
    return _trim_log(_translate(raw), log_after)


def _trim_for(session: Session, view: dict, log_after: int | None = None, full_log: bool = False) -> dict:
    """The transcript trim every tool reply gets, against the cursor the
    caller meant: `log_after` if it sent one, otherwise how many lines this
    session has already been sent (`Session.log_sent`), or nothing at all for
    `full_log`. The caller does not have to remember a thing -- which is the
    point: a caller that forgets an optional argument on one of seven tools
    for forty calls is the caller this API actually has.

    `full_log` is the reset for a reply that went missing (a dropped stream):
    `log_from` coming back larger than the lines a client holds is how it
    notices. Records what was sent afterwards, so the next reply continues
    from here -- a reply that errors before this point records nothing."""
    cursor = None if full_log else (log_after if log_after is not None else session.log_sent)
    view = _trim_log(view, cursor)
    if "log_total" in view:
        session.log_sent = view["log_total"]
    return view


def _finish(session: Session, view: dict, log_after: int | None = None, full_log: bool = False) -> dict:
    """The one seam every view crosses on its way out, whichever tool is
    answering: `summary` (`_summarize`, against the board cached on the
    session), then the compact shape (`_compact`), then the transcript trim
    against this session's cursor (`_trim_for`). `view` is already
    `_translate`d -- `_reply` does that for a fresh wire dict, and
    `wait_for_turn` for the poll it decided to hand over.

    Two call sites used to apply these steps separately and drifted twice:
    the streamed view first lacked `summary`, then came back with the flat
    `legal_actions` while `state()` grouped them. The shape of a reply must
    not depend on which tool returned it, so the steps live here and nowhere
    else."""
    return _trim_for(session, _compact(_summarize(view, session.board)), log_after, full_log)


def _reply(session: Session, raw: dict, log_after: int | None = None, full_log: bool = False) -> dict:
    """A raw wire view as the tool reply the LLM reads: translated, then
    `_finish`ed."""
    return _finish(session, _translate(raw), log_after, full_log)


def _offer_trade(
    tables: Tables,
    session: Session,
    give: dict,
    want: dict,
    timeout: float | None = None,
    log_after: int | None = None,
    full_log: bool = False,
) -> Iterator:
    _seated(session)
    body = {"give": _positional(give), "want": _positional(want)}
    offered = _call_ok(tables, session, "POST", f"/api/games/{session.code}/trade/round", body)
    return _settle(tables, session, offered, timeout, log_after, full_log)


def _answer_trade(
    tables: Tables,
    session: Session,
    index: int,
    kind: str,
    give: dict | None = None,
    receive: dict | None = None,
    timeout: float | None = None,
    log_after: int | None = None,
    full_log: bool = False,
) -> Iterator:
    _seated(session)
    # No `version` here on purpose: the wire refuses anything but the exact
    # open offer by `actor` + `received` (`GameSession.answer_round`), which is
    # the only staleness that can hurt this call. See `_expect_check`.
    raw = _call_ok(tables, session, "GET", "/api/state")
    pending = raw.get("pending") or []
    if not isinstance(index, int) or not (0 <= index < len(pending)):
        raise ToolError(
            f"index {index!r} out of range — state()'s pending has "
            f"{len(pending)} offer(s) right now (0..{len(pending) - 1})"
            if pending
            else "index out of range — state()'s pending is empty; nobody has an open offer against you"
        )
    offer = pending[index]
    body = {"actor": offer["actor"], "received": offer["bundle"], "kind": kind}
    if kind == "counter":
        body["bundle"] = _bundle_towards_actor(give, receive)
    answered = _call_ok(tables, session, "POST", f"/api/games/{session.code}/trade/round/answer", body)
    return _settle(tables, session, answered, timeout, log_after, full_log)


def _choose_trade(
    tables: Tables,
    session: Session,
    index: int | None = None,
    decline: bool = False,
    timeout: float | None = None,
    log_after: int | None = None,
    full_log: bool = False,
) -> Iterator:
    _seated(session)
    # No `version` (see `_answer_trade`): the wire matches the chosen answer
    # by `seat` + `bundle` exactly (`GameSession.execute_round_choice`).
    raw = _call_ok(tables, session, "GET", "/api/state")
    if decline:
        declined = _call_ok(tables, session, "POST", f"/api/games/{session.code}/trade/round/choose", {"decline": True})
        return _settle(tables, session, declined, timeout, log_after, full_log)
    responses = ((raw.get("trade_round") or {}).get("responses")) or []
    if not isinstance(index, int) or not (0 <= index < len(responses)):
        raise ToolError(
            f"index {index!r} out of range — state()'s trade_round.responses has "
            f"{len(responses)} answer(s) right now (0..{len(responses) - 1})"
            if responses
            else "index out of range — state()'s trade_round.responses is empty; "
            "nobody has answered your offer yet"
        )
    response = responses[index]
    if response.get("kind") == "pass" or response.get("bundle") is None:
        raise ToolError(
            f"index {index} is a pass -- seat {response['seat']} turned your offer down; "
            "choose an accept or counter, or `decline: true`"
        )
    body = {"seat": response["seat"], "bundle": response["bundle"]}
    chosen = _call_ok(tables, session, "POST", f"/api/games/{session.code}/trade/round/choose", body)
    return _settle(tables, session, chosen, timeout, log_after, full_log)


# --- Settling: every acting tool returns at this seat's next decision ---------
#
# A reply that comes back the instant after an action leaves the caller to
# decide when to look again -- and the one LLM game played through the
# earlier tools showed what that costs: a seat that reads `wait`, does
# something else, and stalls its own turn. So no acting tool answers until
# there is something for this seat to do. `act(END_TURN)` returns when the
# other seats have played round to you (or an offer lands against you, or
# the game ends); `act(BUILD_ROAD)` returns at once, since it is still your
# turn. The loop a seat runs is then act -> act -> act, and `your_move` is
# never `wait` unless the wait was cut short by `timeout` or `_MAX_WAIT`.
#
# `web.py` answers every `tools/call` as `text/event-stream`, writing a
# keepalive for each `_KEEPALIVE` yielded here so the connection (and
# whatever proxy sits in front of it) does not decide the server has gone
# quiet during a long wait. `call_tool_events` is what it drives;
# `call_tool` drains the same generator for any caller that wants one
# blocking call.

_KEEPALIVE = object()
_WAIT_TICK = 15.0
# The most any one call blocks, whatever `timeout` says: a human seat that
# walked away must not hold a connection open for ever. On expiry the reply
# is whatever the table looks like then, `your_move: wait` included.
_MAX_WAIT = 600.0


def _turn_ready(view: dict) -> bool:
    """Whether a translated view gives this seat something to do -- the same
    question `your_move` answers, so it is the same code."""
    return _your_move(view)[0] != "wait"


def _poll_raw(tables: Tables, session: Session, after: int | None = None, wait: float = 0.0) -> dict:
    query = "" if after is None else f"?after={after}&wait={wait}"
    return _call_ok(tables, session, "GET", f"/api/state{query}")


def _covers(hand: dict, cost: dict) -> bool:
    return all(hand.get(name, 0) >= n for name, n in cost.items())


# The most forced moves one `_forced` pass plays before handing the view
# over regardless -- a bound, not a budget; a turn has at most one roll
# and one open offer per other seat.
_FORCED_CAP = 8


def _forced(tables: Tables, session: Session, raw: dict) -> dict:
    """Plays what no seat would decide differently, so no reply asks: a
    lone `ROLL` (with a Knight also legal the seat chooses, so that is
    left alone), and a `pass` on any offer this hand cannot cover -- the
    first game through these tools spent one round-trip per bot turn
    passing on those. Returns the latest raw view."""
    for _ in range(_FORCED_CAP):
        legal = raw.get("legal_actions") or []
        if len(legal) == 1 and legal[0].get("type") == "ROLL":
            raw = _call_ok(tables, session, "POST", "/api/action", {"action": legal[0]})
            continue
        me = next((p for p in raw.get("players") or [] if p.get("seat") == raw.get("seat")), None)
        hand = None if me is None else me.get("hand")
        unaffordable = next(
            (
                offer
                for offer in raw.get("pending") or []
                if hand is not None and not _covers(hand, _counterparty_view(offer["bundle"])[0])
            ),
            None,
        )
        if unaffordable is not None and session.code:
            body = {"actor": unaffordable["actor"], "received": unaffordable["bundle"], "kind": "pass"}
            raw = _call_ok(tables, session, "POST", f"/api/games/{session.code}/trade/round/answer", body)
            continue
        return raw
    return raw


def _settle(
    tables: Tables,
    session: Session,
    raw: dict,
    timeout: float | None = None,
    log_after: int | None = None,
    full_log: bool = False,
) -> Iterator:
    """From `raw` -- the wire view an action (or a seat, or a plain read)
    just handed back -- plays the forced moves (`_forced`), then yields
    `_KEEPALIVE` per wait tick until this seat has something to decide, then
    the one `_finish`ed reply. Immediately, if `raw` already says so. Only
    that final view moves the session's cursor.

    Every acting tool ends here, so every reply is the same shape as
    `state()`'s: summarised, grouped, trimmed. The polls in between are read
    for `version` and `_turn_ready` alone."""
    # `_translate`, not `_finish`, until the last line: the views in between
    # are never seen by the caller, so they must not move the session's
    # cursor (`Session.log_sent`).
    view = _translate(_forced(tables, session, raw))
    limit = _MAX_WAIT if timeout is None else max(0.0, min(float(timeout), _MAX_WAIT))
    deadline = time.monotonic() + limit
    while not _turn_ready(view):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        yield _KEEPALIVE
        raw = _poll_raw(tables, session, after=view.get("version"), wait=min(_WAIT_TICK, remaining))
        view = _translate(_forced(tables, session, raw))
    yield _finish(session, view, log_after, full_log)


def _wait_for_turn(
    tables: Tables,
    session: Session,
    timeout: float | None = None,
    log_after: int | None = None,
    full_log: bool = False,
) -> Iterator:
    _seated(session)
    return _settle(tables, session, _call_ok(tables, session, "GET", "/api/state"), timeout, log_after, full_log)


# The transcript-cursor arguments every state-returning tool takes, spelled
# once. The cursor itself is automatic (`_trim_for`): these are the overrides.
# `_WAIT_ARGS` adds the one every settling tool takes (`_settle`).
_CURSOR_ARGS = {
    "log_after": {
        "type": "integer",
        "description": "Override the automatic `log` cursor: the line count you hold. Normally omit.",
    },
    "full_log": {
        "type": "boolean",
        "description": "Send the whole `log` (after a lost reply).",
    },
}
_WAIT_ARGS = {
    "timeout": {
        "type": "number",
        "description": "Max seconds to wait for your next move (default/cap 600; 0 = reply now).",
    },
    **_CURSOR_ARGS,
}

_IDENTITY_ARG = {
    "type": "string",
    "description": "A string naming you, e.g. claude-opus-5. It is your key for resume_game().",
}
_CODE_ARG = {"type": "string", "description": "The game's six-character code."}
_NAME_ARG = {"type": "string", "description": "Display name, up to 40 characters."}


def _resource_dict(description: str) -> dict:
    return {"type": "object", "additionalProperties": {"type": "integer"}, "description": description}


# Every tool but `bots`/`board` answers with the same state reply, documented
# once, on `state`. The other descriptions say only what differs.
#
# name -> (handler, description, JSON Schema for `arguments`)
_TOOLS: dict[str, tuple] = {
    "bots": (
        _bots,
        "Bot names new_game's `opponents` accepts.",
        {"type": "object", "properties": {}},
    ),
    "new_game": (
        _new_game,
        "Deal a new game and take a random seat; play starts at once. `opponents` "
        "fill bot seats; any other seat stays open for others to join by the "
        "returned `code`. Nothing is traded on your behalf. Replies at your first "
        "move, as state().",
        {
            "type": "object",
            "properties": {
                "identity": _IDENTITY_ARG,
                "opponents": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Names from bots(), one per bot seat. Omit for no bots.",
                },
                "name": _NAME_ARG,
                **_WAIT_ARGS,
            },
            "required": ["identity"],
        },
    ),
    "join": (
        _join,
        "Take a random open seat at an existing game. Fails when none is open "
        "(a seat left or locked out stays closed for that game). Replies at your "
        "first move, as state().",
        {
            "type": "object",
            "properties": {"code": _CODE_ARG, "identity": _IDENTITY_ARG, "name": _NAME_ARG, **_WAIT_ARGS},
            "required": ["code", "identity"],
        },
    ),
    "board": (
        _board,
        "The fixed board; read once per game. `hexes`: terrain, number `token`, "
        "`resource` it pays (null for desert/sea/gold) and `pips` (2d6 ways to "
        "roll the token: 5 for 6/8 down to 1 for 2/12). `vertices`: `pips` summed "
        "and `resources` de-duplicated over touching hexes, i.e. settlement value. "
        "`edges` (v0/v1), `ports` (vertices, resource or null for 3:1, ratio).",
        {"type": "object", "properties": {}},
    ),
    "state": (
        _state,
        "Game state; every playing tool replies with it at your next decision. "
        "Forced moves are played for you: a lone ROLL, and passing offers your "
        "hand can't cover.\n"
        "`your_move`: `act`, `discard`, `answer_trade` or `choose_trade` = the tool "
        "to call; `game_over`; `wait` only after a `timeout` or while seats are open "
        "(`waiting_on`; call wait_for_turn()).\n"
        "`summary.afford` per build: `ok`, `missing`, `legal` now. `summary.race`: "
        "`points`, `to_win`, `leader`, per award yours/holder's/`need`. When "
        "offered: `spots` (legal settlement/city vertices with pips, resources, "
        "port; best first) and `robber` (hexes with pips, `hits`, an `index` per "
        "victim).\n"
        "`legal_actions`: grouped by type, each entry an `index` for act() plus "
        "`edge` (roads), `vertex` (settlement/city), `hex`+`victim` (MOVE_ROBBER, "
        "null = nobody), `give`/`want` (BANK_TRADE), `resource` (MONOPOLY/DISCARD), "
        "`resources` (YEAR_OF_PLENTY); ROLL, END_TURN, BUY_DEV_CARD, PLAY_KNIGHT, "
        "PLAY_ROAD_BUILDING index only.\n"
        "`players`: `kind`, your `hand`/`dev_cards`, public ledger `known`/`unknown`. "
        "`buildings`, `roads` (edge ids per seat), `bank`, `trade_ratios`, `robber` "
        "hex. Resource dicts omit zeros.\n"
        "`pending`: offers to you (`actor`, `you_give`, `you_receive`) -> "
        "answer_trade(index). `trade_round`: your open offer's `responses` (`seat`, "
        "`kind`, `you_would_give`/`you_would_receive`) -> choose_trade(index). "
        "`trades`: done this turn.\n"
        "`log`: new transcript lines plus the last one you hold (it may be "
        "rewritten); splice at `log_from`. Full on a new/resumed seat and at "
        "game over.",
        {"type": "object", "properties": {**_CURSOR_ARGS}},
    ),
    "wait_for_turn": (
        _wait_for_turn,
        "Wait until `your_move` is not `wait`, then reply as state(). Only needed "
        "after a reply whose `timeout` ran out.",
        {"type": "object", "properties": {**_WAIT_ARGS}},
    ),
    "act": (
        _act,
        "Play `legal_actions` entry `index` from the latest state() (same indexes "
        "in `summary.spots`/`robber`). END_TURN is an action like any other. "
        "Replies at your next move, as state().",
        {
            "type": "object",
            "properties": {
                "index": {"type": "integer", "description": "Index into legal_actions."},
                "expect": {
                    "type": "object",
                    "description": "The entry you chose with its group key as `type`, e.g. "
                    '{"type": "BUILD_ROAD", "edge": 17}. Refuses if that index now names '
                    "something else.",
                },
                **_WAIT_ARGS,
            },
            "required": ["index"],
        },
    ),
    "undo": (
        _undo,
        "Undo your own last build, bank trade, Road Building or Knight while "
        "`can_undo` is true (a Knight until its robber move is made). Nothing else "
        "can be undone.",
        {"type": "object", "properties": {"timeout": _WAIT_ARGS["timeout"]}},
    ),
    "leave_game": (
        _leave_game,
        "Give up your seat for good; pieces and hand stay, your turns are skipped. "
        "Refuses while a trade round involving you is open.",
        {"type": "object", "properties": {}},
    ),
    "get_table": (
        _state,
        "Same as state().",
        {"type": "object", "properties": {**_CURSOR_ARGS}},
    ),
    "offer_trade": (
        _offer_trade,
        "On your own turn in MAIN, offer a trade to every other seat: 1-3 cards a "
        "side, no resource on both sides. Replies once every seat has answered, "
        "with `trade_round.responses` for choose_trade().",
        {
            "type": "object",
            "properties": {
                "give": _resource_dict('Resource -> count you give, e.g. {"Wood": 1}.'),
                "want": _resource_dict("Resource -> count you want."),
                **_WAIT_ARGS,
            },
            "required": ["give", "want"],
        },
    ),
    "answer_trade": (
        _answer_trade,
        "Answer `pending[index]`: `accept` as offered, `counter` with your own "
        "`give`/`receive`, or `pass`. The actor then picks one answer. Replies at "
        "your next move.",
        {
            "type": "object",
            "properties": {
                "index": {"type": "integer", "description": "Index into pending."},
                "kind": {"type": "string", "enum": ["accept", "counter", "pass"]},
                "give": _resource_dict("counter only: resource -> count you give."),
                "receive": _resource_dict("counter only: resource -> count you receive."),
                **_WAIT_ARGS,
            },
            "required": ["index", "kind"],
        },
    ),
    "choose_trade": (
        _choose_trade,
        "Execute `trade_round.responses[index]`, or `decline: true` to close your "
        "round with no trade. Replies at your next move.",
        {
            "type": "object",
            "properties": {
                "index": {"type": "integer", "description": "Index into trade_round.responses."},
                "decline": {"type": "boolean"},
                **_WAIT_ARGS,
            },
        },
    ),
    "resume_game": (
        _resume_game,
        "Reclaim your seat with the `code` and the exact `identity` you gave "
        "new_game()/join(). Use after a new MCP session, which starts seatless. "
        "Replies at your next move, as state().",
        {
            "type": "object",
            "properties": {
                "code": _CODE_ARG,
                "identity": {"type": "string", "description": "The identity new_game()/join() was called with."},
                **_WAIT_ARGS,
            },
            "required": ["code", "identity"],
        },
    ),
}


def tool_list() -> list[dict]:
    return [
        {"name": name, "description": description, "inputSchema": schema}
        for name, (_, description, schema) in _TOOLS.items()
    ]


def call_tool_events(tables: Tables, session: Session, name: str, arguments: dict) -> Iterator:
    """One tool call as `web.py` streams it: `_KEEPALIVE` for every wait
    tick, then the one raw result dict -- or a raised `ToolError`, before or
    between items. The `{"content": [...], "isError": ...}` MCP result
    envelope is `web.py`'s job, not this module's."""
    entry = _TOOLS.get(name)
    if entry is None:
        raise ToolError(f"unknown tool: {name}")
    handler, _, _ = entry
    try:
        result = handler(tables, session, **arguments)
    except TypeError as error:
        raise ToolError(f"bad arguments for {name}: {error}") from error
    if isinstance(result, dict):
        yield result
    else:
        yield from result


def call_tool(tables: Tables, session: Session, name: str, arguments: dict) -> dict:
    """`call_tool_events` drained: one blocking call -> its result dict."""
    result: dict = {}
    for item in call_tool_events(tables, session, name, arguments):
        if item is not _KEEPALIVE:
            result = item
    return result
