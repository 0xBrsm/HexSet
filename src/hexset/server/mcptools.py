"""The MCP tool layer: what an LLM seat calls, translated to/from the wire
`api.py` speaks -- served over HTTP by `web.py`'s `POST /mcp` now, not a
separate stdio process (see `web.py`'s module docstring on why that program
is gone).

Every tool below reaches the game through `tables.handle(method, path,
payload, token)` -- the same in-process seam `web.py` and
`clients.botclient.LocalTransport` use, never `Tables`/`Table` internals
directly. `ApiError` becomes `ToolError`, the shape `_call_tool`/`web.py`
report back to the LLM as a normal (not protocol-level) tool result.

Identity used to be a handful of module globals (`_token`/`_code`/`_model`)
because one stdio process was one seat. An HTTP server serves many MCP
sessions at once, so that state now lives in a `Session` object -- one per
`Mcp-Session-Id` -- threaded through every call instead.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

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
    code the trade routes are addressed by, and the `model` string identity
    was minted from (needed if `resume_game` is ever called with it again).
    Scoped to one `Mcp-Session-Id`, held in memory by `web.py` for as long as
    that session lives -- nothing here ever touches disk."""

    token: str | None = None
    code: str | None = None
    model: str | None = None
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


def _client_of(model: str) -> tuple[dict, str]:
    """The `client` wire field and the secret behind it, from a model
    string: `secret = model.strip().lower()`, `id = sha256(secret)`, kind
    `"mcp"` -- no env-var override, the model supplies its own string."""
    if not isinstance(model, str) or not model.strip():
        raise ToolError("model must be a non-empty string identifying you, e.g. claude-opus-5")
    secret = model.strip().lower()
    client = {"id": hashlib.sha256(secret.encode("utf-8")).hexdigest(), "kind": "mcp"}
    return client, secret


def _models(tables: Tables, session: Session) -> dict:
    return _call_ok(tables, session, "GET", "/api/models")


def _display_name(name: str | None) -> str | None:
    """The 40-character cap every client applies, or `None` for no name at
    all -- the server defaults an unnamed mcp seat to "mcp" itself
    (`api.default_seat_name`), so there is nothing left for this to fall
    back to."""
    return str(name).strip()[:40] if name else None


def _new_game(tables: Tables, session: Session, model: str, opponents: list[str] | None = None, name: str | None = None) -> dict:
    client, _ = _client_of(model)
    session.model = model
    body: dict = {"name": _display_name(name), "client": client}
    if opponents:
        body["bots"] = opponents
    # Translated like every other state reply: the deal is the first view an
    # LLM reads, and it used to be the one that came back raw.
    dealt = _seat(session, _call_ok(tables, session, "POST", "/api/games", body))
    _layout(tables, session)  # the board is fixed from here on; fetch it once now
    return _reply(session, dealt)


def _join(tables: Tables, session: Session, code: str, model: str, name: str | None = None) -> dict:
    if not isinstance(code, str) or not code.strip():
        raise ToolError("code must be a game's six-character code")
    client, _ = _client_of(model)
    session.model = model
    body: dict = {"code": code.strip().lower(), "name": _display_name(name), "client": client}
    joined = _seat(session, _call_ok(tables, session, "POST", "/api/join", body))
    _layout(tables, session)
    return _reply(session, joined)


def _resume_game(tables: Tables, session: Session, code: str, model: str) -> dict:
    """Reclaims a seat by `POST /api/reclaim`, the same way a browser's own
    reclaim works -- no local cache file any more (there is nothing left to
    cache: a session dies with its `Mcp-Session-Id`, and a fresh MCP
    connection just calls this again with the `code`/`model` it already
    knows). `secret = model.strip().lower()`, the same secret `new_game`/
    `join` mint from `model`."""
    if not isinstance(code, str) or not code.strip():
        raise ToolError("code must be a game's six-character code")
    _, secret = _client_of(model)
    reclaimed = _call_ok(tables, session, "POST", "/api/reclaim", {"code": code.strip().lower(), "secret": secret})
    session.token = reclaimed.pop("token")
    session.code = reclaimed.get("code")
    session.model = model
    session.log_sent = None  # a reclaimed seat is owed the whole transcript
    session.board = None
    _layout(tables, session)
    return _reply(session, reclaimed)


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
    by_vertex: dict[int, list[tuple[str, int]]] = {}
    for hex_ in raw.get("hexes") or []:
        resource = _TERRAIN_RESOURCE.get(hex_["terrain"])
        pips = _PIPS.get(hex_["token"], 0)
        hex_["resource"] = resource
        hex_["pips"] = pips
        if resource is None:
            continue
        for v in hex_["vertex_ids"]:
            by_vertex.setdefault(v, []).append((resource, pips))
    for vertex in raw.get("vertices") or []:
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

_ACTION_KEYS = ("type", "a", "b")


def _expect_check(chosen: dict, expect: dict | None) -> None:
    if expect is None:
        return
    if not isinstance(expect, dict) or "type" not in expect:
        raise ToolError("expect must be the legal_actions entry you chose, at least its `type`")
    mismatch = [k for k in _ACTION_KEYS if k in expect and expect[k] != chosen.get(k)]
    if mismatch:
        raise ToolError(
            "legal_actions moved under you: that index is now "
            f"{ {k: chosen.get(k) for k in _ACTION_KEYS} }, not the "
            f"{ {k: expect[k] for k in _ACTION_KEYS if k in expect} } you chose — "
            "call state() again and pick from the fresh list"
        )


def _act(
    tables: Tables,
    session: Session,
    index: int,
    expect: dict | None = None,
    log_after: int | None = None,
    full_log: bool = False,
) -> dict:
    _seated(session)
    # The raw list, not the translated one: only `index` is resolved here, and
    # `POST /api/action` reads `type`/`a`/`b` alone (`wire_to_action`).
    options = _call_ok(tables, session, "GET", "/api/state").get("legal_actions") or []
    if not isinstance(index, int) or not (0 <= index < len(options)):
        raise ToolError(
            f"index {index!r} out of range — state()'s legal_actions has "
            f"{len(options)} option(s) right now (0..{len(options) - 1})"
            if options
            else "index out of range — state()'s legal_actions is empty; it is not your turn"
        )
    _expect_check(options[index], expect)
    return _reply(
        session, _call_ok(tables, session, "POST", "/api/action", {"action": options[index]}), log_after, full_log
    )


def _undo(tables: Tables, session: Session) -> dict:
    _seated(session)
    return _reply(session, _call_ok(tables, session, "POST", "/api/undo"))


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
    """Where this seat stands: points and the distance to the win, the
    leading opponent by *public* points (hidden victory-point cards are not
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
        "winning_points": winning,
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


def _translate(raw: dict) -> dict:
    """Every translation a view gets on its way to the LLM, except the
    transcript trim -- which needs to know what the session already holds
    (`_trim_for`), or an explicit cursor (`_translate_view`) -- and the
    `summary`, which needs the session's board (`_reply`)."""
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


def _reply(session: Session, raw: dict, log_after: int | None = None, full_log: bool = False) -> dict:
    """A raw wire view as the tool reply the LLM reads: translated, given
    its `summary` (`_summarize`), then trimmed against this session's cursor
    (`_trim_for`)."""
    return _trim_for(session, _summarize(_translate(raw), session.board), log_after, full_log)


def _get_table(tables: Tables, session: Session, log_after: int | None = None, full_log: bool = False) -> dict:
    return _state(tables, session, log_after, full_log)


def _offer_trade(
    tables: Tables,
    session: Session,
    give: dict,
    want: dict,
    log_after: int | None = None,
    full_log: bool = False,
) -> dict:
    _seated(session)
    body = {"give": _positional(give), "want": _positional(want)}
    return _reply(
        session, _call_ok(tables, session, "POST", f"/api/games/{session.code}/trade/round", body), log_after, full_log
    )


def _answer_trade(
    tables: Tables,
    session: Session,
    index: int,
    kind: str,
    give: dict | None = None,
    receive: dict | None = None,
    log_after: int | None = None,
    full_log: bool = False,
) -> dict:
    _seated(session)
    # No `version` here on purpose: the wire refuses anything but the exact
    # open offer by `actor` + `received` (`GameSession.answer_round`), which is
    # the only staleness that can hurt this call. See `_expect_check`.
    raw = _call_ok(tables, session, "GET", "/api/state")
    pending = raw.get("pending") or []
    if not isinstance(index, int) or not (0 <= index < len(pending)):
        raise ToolError(
            f"index {index!r} out of range — get_table()'s pending has "
            f"{len(pending)} offer(s) right now (0..{len(pending) - 1})"
            if pending
            else "index out of range — get_table()'s pending is empty; nobody has an open offer against you"
        )
    offer = pending[index]
    body = {"actor": offer["actor"], "received": offer["bundle"], "kind": kind}
    if kind == "counter":
        body["bundle"] = _bundle_towards_actor(give, receive)
    return _reply(
        session,
        _call_ok(tables, session, "POST", f"/api/games/{session.code}/trade/round/answer", body),
        log_after,
        full_log,
    )


def _choose_trade(
    tables: Tables,
    session: Session,
    index: int | None = None,
    decline: bool = False,
    log_after: int | None = None,
    full_log: bool = False,
) -> dict:
    _seated(session)
    # No `version` (see `_answer_trade`): the wire matches the chosen answer
    # by `seat` + `bundle` exactly (`GameSession.execute_round_choice`).
    raw = _call_ok(tables, session, "GET", "/api/state")
    if decline:
        return _reply(
            session,
            _call_ok(tables, session, "POST", f"/api/games/{session.code}/trade/round/choose", {"decline": True}),
            log_after,
            full_log,
        )
    responses = ((raw.get("trade_round") or {}).get("responses")) or []
    if not isinstance(index, int) or not (0 <= index < len(responses)):
        raise ToolError(
            f"index {index!r} out of range — get_table()'s trade_round.responses has "
            f"{len(responses)} answer(s) right now (0..{len(responses) - 1})"
            if responses
            else "index out of range — get_table()'s trade_round.responses is empty; "
            "nobody has answered your offer yet"
        )
    response = responses[index]
    if response.get("kind") == "pass" or response.get("bundle") is None:
        raise ToolError(
            f"index {index} is a pass -- seat {response['seat']} turned your offer down; "
            "choose an accept or counter, or `decline: true`"
        )
    body = {"seat": response["seat"], "bundle": response["bundle"]}
    return _reply(
        session,
        _call_ok(tables, session, "POST", f"/api/games/{session.code}/trade/round/choose", body),
        log_after,
        full_log,
    )


# --- wait_for_turn: a long poll, exposed as an SSE stream (web.py) -----------
#
# `web.py`'s `POST /mcp` answers a `tools/call` for this one tool with
# `text/event-stream` instead of one JSON object, writing a keepalive
# between each 15s wait so the connection (and whatever proxy sits in front
# of it) doesn't decide the server has gone quiet. The generator below is
# the shared loop: `_wait_for_turn_events` yields `_KEEPALIVE` for every tick
# that doesn't resolve the wait and the final translated view once it does
# (or once `timeout` runs out); `web.py` drives it directly for the SSE
# framing, and `_wait_for_turn` (registered as the tool, for tools/list and
# any caller that wants one blocking call) just drains it.

_KEEPALIVE = object()
_WAIT_TICK = 15.0


def _turn_ready(view: dict) -> bool:
    """Whether a translated view gives this seat something to do -- the same
    question `your_move` answers, so it is the same code."""
    return _your_move(view)[0] != "wait"


def _poll_state(tables: Tables, session: Session, after: int | None = None, wait: float = 0.0) -> dict:
    query = "" if after is None else f"?after={after}&wait={wait}"
    # `_translate`, not `_reply`: these polls are never seen by the caller,
    # so they must not move the session's cursor (`Session.log_sent`).
    return _translate(_call_ok(tables, session, "GET", f"/api/state{query}"))


def _wait_for_turn_events(
    tables: Tables,
    session: Session,
    timeout: float | None = None,
    log_after: int | None = None,
    full_log: bool = False,
):
    """Yields `_KEEPALIVE` for each wait tick that doesn't resolve, then the
    final translated view -- immediately, if it's already true.

    Only the view that is actually yielded is trimmed (`_trim_for`), and only
    it moves the session's cursor. The polls in between are read for
    `version` and `_turn_ready` alone and are never seen by the caller."""
    _seated(session)
    view = _poll_state(tables, session)
    if _turn_ready(view):
        yield _trim_for(session, view, log_after, full_log)
        return
    elapsed = 0.0
    while timeout is None or elapsed < timeout:
        yield _KEEPALIVE
        remaining = _WAIT_TICK if timeout is None else max(0.0, min(_WAIT_TICK, timeout - elapsed))
        view = _poll_state(tables, session, after=view.get("version"), wait=remaining)
        elapsed += remaining
        if _turn_ready(view) or (timeout is not None and elapsed >= timeout):
            yield _trim_for(session, view, log_after, full_log)
            return
    yield _trim_for(session, view, log_after, full_log)


def _wait_for_turn(
    tables: Tables,
    session: Session,
    timeout: float | None = None,
    log_after: int | None = None,
    full_log: bool = False,
) -> dict:
    result: dict = {}
    for item in _wait_for_turn_events(tables, session, timeout=timeout, log_after=log_after, full_log=full_log):
        if item is not _KEEPALIVE:
            result = item
    return result


# The transcript-cursor arguments every state-returning tool takes, spelled
# once. The cursor itself is automatic (`_trim_for`): these are the overrides.
_CURSOR_ARGS = {
    "log_after": {
        "type": "integer",
        "description": "Optional override of the automatic transcript cursor: how many "
        "`log` lines you hold (a `log_total` from an earlier reply). Normally omit it.",
    },
    "full_log": {
        "type": "boolean",
        "description": "Optional: send the whole transcript, ignoring the cursor -- for "
        "when a reply went missing (`log_from` came back past the lines you hold).",
    },
}

# name -> (handler, description, JSON Schema for `arguments`)
_TOOLS: dict[str, tuple] = {
    "models": (
        _models,
        "List the opponent names new_game's `opponents` argument accepts.",
        {"type": "object", "properties": {}},
    ),
    "new_game": (
        _new_game,
        "Deal a new game, playable immediately: you at one random seat, any "
        "named opponents at others, everything else open for other people (or "
        "other bots) to join by the code this returns. There is no separate "
        "start — the board is live from the first response. Nothing is ever "
        "traded on your behalf: offers to you wait in get_table()'s `pending` "
        "for answer_trade(). `model` identifies you for resume_game() -- keep "
        "it exactly as given if you ever mean to reclaim this seat.",
        {
            "type": "object",
            "properties": {
                "model": {
                    "type": "string",
                    "description": "Your exact model identifier, e.g. claude-opus-5.",
                },
                "opponents": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Names from models(), one per bot seat to fill at the deal. "
                        "Omit for no bots at all — every other seat stays open."
                    ),
                },
                "name": {"type": "string", "description": "Your display name, up to 40 characters."},
            },
            "required": ["model"],
        },
    ),
    "join": (
        _join,
        "Take a random open seat at an existing game by its six-character code. "
        "Fails if every seat is taken or has locked out (see state()'s `locked`) "
        "— a seat somebody closed outright is retired for the rest of that "
        "game, so join before that happens. Your seat is gated the same way "
        "new_game()'s is — see its description. `model` identifies you for "
        "resume_game() the same way it does there.",
        {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "The game's six-character code."},
                "model": {
                    "type": "string",
                    "description": "Your exact model identifier, e.g. claude-opus-5.",
                },
                "name": {"type": "string", "description": "Your display name, up to 40 characters."},
            },
            "required": ["code", "model"],
        },
    ),
    "board": (
        _board,
        "The board's fixed layout: hex positions/terrain/numbers and vertex/edge "
        "adjacency. Unlike state(), this never changes once a game is dealt, so it "
        "only needs reading once per game. Each hex also carries `resource` (its "
        "terrain's payout, e.g. `FOREST` -> `Wood`; `null` for desert/sea/gold, "
        "which pay nothing fixed) and `pips` (its 2d6 odds: 5 for a 6 or 8 down to "
        "1 for a 2 or 12, 0 for none). Each vertex carries the same two, summed "
        "and de-duplicated over every hex it touches -- `pips`/`resources` there "
        "are the settlement-value numbers a placement decision actually turns on.",
        {"type": "object", "properties": {}},
    ),
    "state": (
        _state,
        "The full current game state. Read `your_move` first, then `summary`: "
        "`afford` (per build: `ok`, `missing` resources, `legal` right now), "
        "`race` (your points and `to_win`, the public leader, and for each "
        "award your count, the holder's, and the `need` that takes it), and "
        "when there is a placement or robber move to make, `spots` (each legal "
        "settlement/city vertex with its pips, resources and port, best first, "
        "with the `index` to act() on) and `robber` (each hex the robber may go "
        "to with its pips, whose buildings it `hits`, and an `index` per "
        "victim). `your_move` is `act`, "
        "`discard`, `answer_trade` or `choose_trade` names the tool the table "
        "wants from you now; `wait` means nothing does, and `waiting_on` lists "
        "the seats it is waiting for; `game_over` is the end. Then every seat's "
        "public info (hand size, and "
        "your own hand; the public resource-count ledger for everyone else — "
        "counting isn't hidden information here, only a steal's identity and "
        "dev-card types are), the board's dynamic contents, and `legal_actions` "
        "— a 0-indexed list of the actions act() currently accepts, empty when "
        "it is not your turn. Pass the entry you pick as act()'s `expect` to "
        "have it refuse instead of guessing if the list moved under you. Poll "
        "this while another seat is thinking, or call wait_for_turn() instead "
        "to block until it's worth polling again. A legal_actions entry that "
        "spends a resource names it too, alongside the raw `a`/`b` act() "
        "replays: BANK_TRADE has `give`/`want`, PLAY_MONOPOLY/DISCARD have "
        "`resource`, PLAY_YEAR_OF_PLENTY has `resources` (a 2-list). The "
        "transcript `log` is sent incrementally without you doing anything: "
        "each reply carries only the lines added since this session's previous "
        "reply, plus the one trailing line that may have been rewritten since "
        "(a burst of builds collapses into a single line that grows), with "
        "`log_from` naming the index the slice starts at and `log_total` the "
        "whole length. Keep your own copy and splice at `log_from`. A fresh "
        "seat, a resumed seat and the final read of a finished game get the "
        "whole transcript; pass `full_log: true` to force that at any time.",
        {
            "type": "object",
            "properties": {
                **_CURSOR_ARGS,
            },
        },
    ),
    "wait_for_turn": (
        _wait_for_turn,
        "Block until there is something for you to do: legal_actions is "
        "non-empty, an offer is waiting for answer_trade(), your own open "
        "trade_round has an answer from everyone, or the game is over. "
        "Returns immediately if any of that is already true. Streamed as "
        "Server-Sent Events with a keepalive roughly every 15 seconds while "
        "it waits, so a long turn from another seat doesn't look like a dead "
        "connection.",
        {
            "type": "object",
            "properties": {
                "timeout": {
                    "type": "number",
                    "description": "Give up and return the current state after this many "
                    "seconds. Omit to wait indefinitely.",
                },
                **_CURSOR_ARGS,
            },
        },
    ),
    "act": (
        _act,
        "Play legal_actions[index] from the most recent state() (call state() "
        "first if unsure what's legal right now). Returns the state right after "
        "that one action — ending your own turn is END_TURN, an action like any "
        "other, not something act() infers.",
        {
            "type": "object",
            "properties": {
                "index": {"type": "integer", "description": "Index into legal_actions."},
                "expect": {
                    "type": "object",
                    "description": "Optional: the legal_actions entry you chose (its `type`, "
                    "and `a`/`b` if you have them). If legal_actions[index] no longer matches "
                    "it, act() refuses instead of playing whatever now sits at that index.",
                },
                **_CURSOR_ARGS,
            },
            "required": ["index"],
        },
    ),
    "undo": (
        _undo,
        "Undo your own most recent build, bank trade, Road Building, or Knight, "
        "if state()'s can_undo is true. A Knight stays undoable up through the "
        "forced robber move it opens, until that move is actually made. "
        "Anything else (another seat's move, a rolled seven's own robber move) "
        "cannot be undone.",
        {"type": "object", "properties": {}},
    ),
    "leave_game": (
        _leave_game,
        "Give up your seat for the rest of this game — permanent, and the only "
        "way to do it (there is no re-join). Your pieces and hand stay on the "
        "board exactly as they are; only your turn is skipped from now on, and "
        "the game carries on without you. Refuses while a trade round is open "
        "naming you as its actor or as a seat still owed an answer — resolve it "
        "with answer_trade()/choose_trade() first.",
        {"type": "object", "properties": {}},
    ),
    "get_table": (
        _get_table,
        "Everything the table has said about trading, alongside state(): this "
        "turn's `trades` (each as `a`/`b` seats plus `a_gave`/`a_got`, named "
        "resource -> count); `pending` -- offers broadcast to you, unanswered, "
        "each as `actor` plus `you_give`/`you_receive` (what answering `accept` "
        "would cost/pay you); `trade_round` -- your own open offer, only when "
        "you're the one who broadcast it, as `you_give`/`you_receive` plus "
        "`responses` (each an accept or counter with `you_would_give`/"
        "`you_would_receive` if chosen, or a `pass` with neither) and "
        "`awaiting`, the seats still to answer. Resource dicts omit zero counts. "
        "Use a `pending`/`responses` list's index with answer_trade()/"
        "choose_trade() -- never hand-build a trade from these dicts.",
        {
            "type": "object",
            "properties": {
                **_CURSOR_ARGS,
            },
        },
    ),
    "offer_trade": (
        _offer_trade,
        "On your own turn in MAIN, broadcast one offer to every other seat: "
        "`give` and `want` are named resource -> count, 1-3 cards a side on "
        "different resources. Bots answer at once (accept, counter or pass); "
        "read the answers in get_table()'s `trade_round`, then choose_trade().",
        {
            "type": "object",
            "properties": {
                "give": {
                    "type": "object",
                    "additionalProperties": {"type": "integer"},
                    "description": "Resource name -> count you give, e.g. {\"Wood\": 1}.",
                },
                "want": {
                    "type": "object",
                    "additionalProperties": {"type": "integer"},
                    "description": "Resource name -> count you want.",
                },
                **_CURSOR_ARGS,
            },
            "required": ["give", "want"],
        },
    ),
    "answer_trade": (
        _answer_trade,
        "Answer one of get_table()'s `pending` offers by its index there. "
        "`kind` is `accept` (its `you_give`/`you_receive` as offered), "
        "`counter` (then pass your own `give`/`receive`, named resource -> "
        "count, as the counter-offer) or `pass`. The actor picks among every "
        "seat's answer once all have answered, via choose_trade() on their side.",
        {
            "type": "object",
            "properties": {
                "index": {"type": "integer", "description": "Index into get_table()'s pending."},
                "kind": {"type": "string", "enum": ["accept", "counter", "pass"]},
                "give": {
                    "type": "object",
                    "additionalProperties": {"type": "integer"},
                    "description": "Only for kind=counter: resource -> count you'd give.",
                },
                "receive": {
                    "type": "object",
                    "additionalProperties": {"type": "integer"},
                    "description": "Only for kind=counter: resource -> count you'd receive.",
                },
                **_CURSOR_ARGS,
            },
            "required": ["index", "kind"],
        },
    ),
    "choose_trade": (
        _choose_trade,
        "Execute one answer to your own open offer, by its index into "
        "get_table()'s `trade_round.responses` -- or `decline: true` to close "
        "the round with nothing traded.",
        {
            "type": "object",
            "properties": {
                "index": {"type": "integer", "description": "Index into trade_round.responses."},
                "decline": {"type": "boolean"},
                **_CURSOR_ARGS,
            },
        },
    ),
    "resume_game": (
        _resume_game,
        "Reclaim a seat by its game `code` and the `model` new_game()/join() "
        "were called with — the same way a browser reclaims one after losing "
        "its token, via POST /api/reclaim. Use this after a fresh MCP "
        "connection (a new Mcp-Session-Id starts with no seat at all) or "
        "after a 404 on a request that used to work. Fails if no seat's "
        "client matches that `model`'s secret, or the seat has locked out.",
        {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "The game's six-character code."},
                "model": {
                    "type": "string",
                    "description": "The exact model identifier new_game()/join() were called with.",
                },
            },
            "required": ["code", "model"],
        },
    ),
}


def tool_list() -> list[dict]:
    return [
        {"name": name, "description": description, "inputSchema": schema}
        for name, (_, description, schema) in _TOOLS.items()
    ]


def call_tool(tables: Tables, session: Session, name: str, arguments: dict) -> dict:
    """One tool call -> its raw result dict, or a raised `ToolError`. The
    `{"content": [...], "isError": ...}` MCP result envelope is `web.py`'s
    job, not this module's — it's the one thing that differs between a plain
    JSON response and `wait_for_turn`'s streamed one."""
    entry = _TOOLS.get(name)
    if entry is None:
        raise ToolError(f"unknown tool: {name}")
    handler, _, _ = entry
    try:
        return handler(tables, session, **arguments)
    except TypeError as error:
        raise ToolError(f"bad arguments for {name}: {error}") from error
