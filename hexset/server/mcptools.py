"""The MCP tool layer: what an LLM seat calls, translated to and from the wire
`api.py` speaks. Served over HTTP by `web.py`'s `POST /mcp`.

Every tool reaches the game through `tables.handle(method, path, payload,
token)`, never `Tables`/`Table` internals, and an `ApiError` becomes a
`ToolError` reported back as a normal (not protocol-level) tool result.
Per-seat identity lives on a `Session`, one per `Mcp-Session-Id`. This module
imports no engine.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from dataclasses import dataclass
from typing import Iterator

from .api import ApiError, Tables

__all__ = [
    "RESOURCES",
    "ToolError",
    "Session",
    "KEEPALIVE",
    "tool_list",
    "call_tool_events",
    "call_tool",
]


# Resource order every wire bundle (5 signed or unsigned ints) uses,
# Title-cased as the wire's hand/bank dicts spell them.
RESOURCES = ("Wood", "Brick", "Sheep", "Wheat", "Ore")


class ToolError(Exception):
    """Raised by a tool implementation to report the failure back to the
    LLM as a normal (not protocol-level) tool result."""


@dataclass
class Session:
    """One MCP session's seat: the token `tables.handle` acts with, the game
    code the trade routes are addressed by, and the `identity` the client id
    was minted from (needed by `resume_game`). Held in memory, never on disk."""

    token: str | None = None
    code: str | None = None
    identity: str | None = None
    # How many transcript lines this session has been sent: the automatic
    # cursor `_trim_for` trims against. Reset whenever the session takes a
    # seat, which owes a whole transcript.
    log_sent: int | None = None
    # The annotated board (`_layout`), fetched once per seat and read by every
    # reply's `summary`.
    board: dict | None = None
    # The ranked seats the open `offer_trade` may be settled with, or `None`
    # for anyone -- read by `_settled_round` while that round is open.
    offer_to: list[int] | None = None
    # Replies and their bytes as the client received them, reported as `usage`
    # on the `game_over` reply.
    calls: int = 0
    bytes: int = 0


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
    """Record the token a join or a deal handed back, and remove it from the
    reply: the LLM never sees it, `_call_ok` sends it on its behalf."""
    session.token = result.pop("token")
    session.code = result.get("code")
    session.log_sent = None
    session.board = None
    return result


def _client_of(identity: str) -> tuple[dict, str]:
    """The `client` wire field and the secret behind it, from an identity
    string: `secret = identity.strip().lower()`, `id = sha256(secret)`, kind
    `"mcp"`."""
    if not isinstance(identity, str) or not identity.strip():
        raise ToolError("identity must be a non-empty string naming you, e.g. claude-opus-5")
    secret = identity.strip().lower()
    client = {"id": hashlib.sha256(secret.encode("utf-8")).hexdigest(), "kind": "mcp"}
    return client, secret


def _bots(tables: Tables, session: Session) -> dict:
    return _call_ok(tables, session, "GET", "/api/models")


def _display_name(name: str | None) -> str | None:
    """A name trimmed to 40 characters, or `None`: the server defaults an
    unnamed mcp seat to "mcp" itself."""
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
    """Reclaim a seat through `POST /api/reclaim`, with `secret =
    identity.strip().lower()` — the same secret `new_game`/`join` mint. A
    session dies with its `Mcp-Session-Id`, so a fresh connection calls this."""
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
# The raw board names a terrain and a dice token but joins neither into what a
# placement weighs. Computed here, from data `/api/board` already sends.

# Ways to roll a token on 2d6; 7 never labels a hex.
_PIPS = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}

# Terrain -> the resource it yields. DESERT and SEA read back as `None`.
_TERRAIN_RESOURCE = {
    "FOREST": "Wood",
    "HILLS": "Brick",
    "PASTURE": "Sheep",
    "FIELDS": "Wheat",
    "MOUNTAINS": "Ore",
}
_BOARD_DEAD = ("size", "resources", "dev_cards", "year_of_plenty_pairs")


def _board(tables: Tables, session: Session) -> str:
    _seated(session)
    return _board_text(_layout(tables, session))


def _layout(tables: Tables, session: Session) -> dict:
    """The annotated board for this session's game, fetched once and kept on
    the `Session`."""
    if session.board is not None:
        return session.board
    # A copy: `GET /api/board` hands back the table's own `layout` dict.
    raw = copy.deepcopy(_call_ok(tables, session, "GET", "/api/board"))
    # Render geometry and constant tables are the browser's.
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


def _port_label(port: dict) -> str:
    return f"{port['ratio']}:1" if port.get("resource") is None else f"{port['resource']} {port['ratio']}:1"


def _board_text(board: dict) -> str:
    """The annotated board as an incidence encoding: each hex with the vertices
    touching it, each vertex with its yield, port and reachable vertices, one
    header per table. No `edges` block; `summary.roads` names each legal
    edge's destination vertex in the reply that has it."""
    port_of: dict[int, str] = {}
    for port in board.get("ports") or []:
        for v in port.get("vertices") or []:
            port_of[v] = _port_label(port).replace(" ", "")
    neighbors: dict[int, set[int]] = {}
    for edge in board.get("edges") or []:
        neighbors.setdefault(edge["v0"], set()).add(edge["v1"])
        neighbors.setdefault(edge["v1"], set()).add(edge["v0"])
    lines = ["hexes: id resource pips vertices"]
    for hex_ in board.get("hexes") or []:
        label = hex_.get("resource") or hex_.get("terrain")
        lines.append(f"{hex_['id']} {label} {hex_.get('pips', 0)} {','.join(map(str, hex_.get('vertex_ids') or []))}")
    lines.append("")
    lines.append("vertices: id pips resources port neighbors")
    for vertex in board.get("vertices") or []:
        vid = vertex["id"]
        resources = ",".join(vertex.get("resources") or []) or "-"
        nbrs = ",".join(map(str, sorted(neighbors.get(vid, ()))))
        lines.append(f"{vid} {vertex.get('pips', 0)} {resources} {port_of.get(vid, '-')} {nbrs}")
    supply = board.get("piece_supply") or {}
    if supply:
        lines.append("")
        lines.append("piece_supply: " + " ".join(f"{k}={v}" for k, v in supply.items()))
    return "\n".join(lines)


def _state(tables: Tables, session: Session, log_after: int | None = None, full_log: bool = False) -> dict:
    _seated(session)
    return _reply(session, _call_ok(tables, session, "GET", "/api/state"), log_after, full_log)


# --- Guarding an index against a table that moved ----------------------------
#
# `api.Table.version` bumps on *every* change anybody makes, so it is too broad
# a guard for an index. What an index has to be stable against is that
# `legal_actions[index]` still names the action the caller chose, which is what
# `expect` checks.

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
    # The raw list, not the translated one: `POST /api/action` reads
    # `type`/`a`/`b` alone.
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


def _discard(
    tables: Tables,
    session: Session,
    cards: dict,
    timeout: float | None = None,
    log_after: int | None = None,
    full_log: bool = False,
) -> Iterator:
    """Discard `cards` (resource -> count) in one call, though the engine only
    ever offers one card at a time. `cards` must total this seat's
    `discard_quota` exactly, so a miscount is refused before anything is
    spent; each card is then posted against the freshest `legal_actions`."""
    _seated(session)
    raw = _call_ok(tables, session, "GET", "/api/state")
    if raw.get("phase") != "DISCARD":
        raise ToolError(f"discard() is only for a DISCARD phase; phase is {raw.get('phase')} right now")
    seat = raw.get("seat")
    quota_list = raw.get("discard_quota") or []
    quota = quota_list[seat] if seat is not None and seat < len(quota_list) else 0
    counts = _positional(cards)
    total = sum(counts)
    if total != quota:
        raise ToolError(f"cards must total your discard_quota of {quota} exactly, not {total}")
    me = next((p for p in raw.get("players") or [] if p.get("seat") == seat), None)
    hand = (me or {}).get("hand") or {}
    short = {
        RESOURCES[i]: counts[i] - hand.get(RESOURCES[i], 0)
        for i in range(len(RESOURCES))
        if counts[i] > hand.get(RESOURCES[i], 0)
    }
    if short:
        raise ToolError(f"you don't hold that many to discard: short {short}")
    remaining = dict(zip(RESOURCES, counts))
    while any(remaining.values()):
        resource = next(name for name, left in remaining.items() if left)
        wanted = RESOURCES.index(resource)
        entry = next(
            (a for a in raw.get("legal_actions") or [] if a.get("type") == "DISCARD" and a.get("a") == wanted),
            None,
        )
        if entry is None:
            raise ToolError(f"discarding {resource} is not on offer right now; call state() again")
        raw = _call_ok(tables, session, "POST", "/api/action", {"action": entry})
        remaining[resource] -= 1
    return _settle(tables, session, raw, timeout, log_after, full_log)


def _leave_game(tables: Tables, session: Session) -> dict:
    _seated(session)
    return _reply(session, _call_ok(tables, session, "POST", "/api/leave"))


# --- Trading (docs/onnx.md §3; human and LLM seats are direct gates) ----
#
# The wire speaks in 5-count arrays signed towards whichever seat proposed the
# trade; everything below uses named dicts from the caller's own point of view
# (`you_give`/`you_receive`) and resolves an `index` against a fresh raw state
# fetch, so no caller holds a stale bundle across a round-trip.


def _named(counts: list[int]) -> dict[str, int]:
    """A positional count list as a dict, zeros omitted."""
    return {name: n for name, n in zip(RESOURCES, counts) if n}


def _covers(hand: dict, cost: dict) -> bool:
    return all(hand.get(name, 0) >= n for name, n in cost.items())


def _positional(counts: dict | None) -> list[int]:
    """A named count dict as a positional list in `RESOURCES` order. Every
    count is a non-negative integer."""
    counts = counts or {}
    if not isinstance(counts, dict):
        raise ToolError("resource counts are an object of resource name -> count")
    unknown = set(counts) - set(RESOURCES)
    if unknown:
        raise ToolError(f"not a resource name: {', '.join(sorted(unknown))}")
    bad = {name: n for name, n in counts.items() if type(n) is not int or n < 0}
    if bad:
        raise ToolError(f"a count is a non-negative integer, not {bad}")
    return [counts.get(name, 0) for name in RESOURCES]


def _actor_view(bundle: list[int]) -> tuple[dict, dict]:
    """(gives, gets) for the seat a signed bundle is signed towards."""
    return _named([max(0, -n) for n in bundle]), _named([max(0, n) for n in bundle])


def _counterparty_view(bundle: list[int]) -> tuple[dict, dict]:
    """(gives, gets) for the seat on the *other* side of a signed bundle:
    `_actor_view` swapped."""
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
    me = next((p for p in raw.get("players") or [] if p.get("seat") == raw.get("seat")), None)
    hand = None if me is None else me.get("hand")
    pending = []
    for t in raw.get("pending") or []:
        you_give, you_receive = _counterparty_view(t["bundle"])
        entry = {"actor": t["actor"], "you_give": you_give, "you_receive": you_receive}
        any_cards = int(t.get("any") or 0)
        if any_cards:
            # Signed towards the actor: its any cards to take are yours to
            # choose and give, its any cards to give are yours to name.
            entry["you_give_any" if any_cards > 0 else "you_receive_any"] = abs(any_cards)
        if hand is not None:
            # Whether `accept` is even possible; a counter always is. An
            # offer with any cards is only ever countered, naming them.
            entry["can_accept"] = not any_cards and _covers(hand, you_give)
        pending.append(entry)
    raw["pending"] = pending
    trade_round = raw.get("trade_round")
    if trade_round is not None:
        you_give, you_receive = _actor_view(trade_round["offer"]["bundle"])
        any_cards = int(trade_round["offer"].get("any") or 0)
        raw["trade_round"] = {
            "you_give": you_give,
            "you_receive": you_receive,
            **({"you_receive_any" if any_cards > 0 else "you_give_any": abs(any_cards)} if any_cards else {}),
            # A pass carries no bundle: the seat is named so it is told apart
            # from one still in `awaiting`, and nothing can be chosen.
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


# `legal_actions` stays positional on the wire so any client can replay an
# entry verbatim, but BANK_TRADE (`a`=give, `b`=want), PLAY_MONOPOLY and
# DISCARD (`a`) spend a resource as a bare index into RESOURCES, and
# PLAY_YEAR_OF_PLENTY's `a` indexes a *pair* (mirrored below, not imported).

_YEAR_OF_PLENTY_PAIRS = [(a, b) for a in range(len(RESOURCES)) for b in range(a, len(RESOURCES))]


def _translate_action(action: dict) -> dict:
    """One legal action with its resource indexes named."""
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
    `log_after` is how many it has; every reply carries `log_total`, to pass
    straight back next call, and `log_from`, the index `log[0]` sits at.
    Omitting the cursor returns the whole transcript.

    **One line of overlap is deliberate**: `render_log` rewrites a collapsed
    line in place as its run grows, so the last line a caller holds is the one
    that can still change. The cursor is clamped, so a caller holding lines an
    undo removed gets the tail that survives."""
    lines = view.get("log")
    if not isinstance(lines, list):
        return view
    view["log_total"] = len(lines)
    # The game's end re-renders the whole transcript with redaction lifted, so
    # every earlier steal names what it was. The final reply is still trimmed
    # like any other; `full_log` asks for the lifted history.
    start = 0 if (log_after is None or not lines) else max(0, min(int(log_after) - 1, len(lines) - 1))
    view["log"] = lines[start:]
    view["log_from"] = start
    return view


# --- summary: the derived facts a turn actually turns on ----------------------
#
# What a build costs against the hand, which legal spots are any good, which
# hex the robber hurts most, how far the awards are.

# Build costs, mirrored from `hexset.economy.COSTS`. Keys are `afford`'s.
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
_SPOT_ACTIONS = ("SETUP_SETTLEMENT", "BUILD_SETTLEMENT", "BUILD_CITY")
_ROAD_ACTIONS = ("BUILD_ROAD", "SETUP_ROAD")
_CITY = 2  # `hexset.state.Building.CITY`


# What `_afford` counts a seat's pieces as, against `piece_supply`.
_PIECE_OF = {"road": "road", "settlement": "settlement", "city": "city"}


def _afford(hand: dict, legal: list[dict], view: dict | None = None, board: dict | None = None) -> dict:
    """Per build: `ok` (the hand covers it), `missing` (what it is short),
    `legal` (whether `legal_actions` offers it now), and when the hand covers
    it but it is not offered, `why`: `phase`, `pieces`, `deck` or `spot`.
    `legal` is omitted when `legal_actions` is empty."""
    offered = {a.get("type") for a in legal}
    out = {}
    for build, cost in _COSTS.items():
        missing = {r: n - hand.get(r, 0) for r, n in cost.items() if hand.get(r, 0) < n}
        entry: dict = {"ok": not missing}
        if legal:
            entry["legal"] = _BUILD_ACTION[build] in offered
            if not missing and not entry["legal"] and view is not None:
                entry["why"] = _why_not(build, view, board)
        if missing:
            entry["missing"] = missing
        out[build] = entry
    return out


def _why_not(build: str, view: dict, board: dict | None) -> str:
    if view.get("phase") != "MAIN":
        return "phase"
    if build == "dev_card":
        return "deck" if not view.get("dev_cards_remaining") else "spot"
    supply = ((board or {}).get("piece_supply") or {}).get(_PIECE_OF[build])
    if supply is not None:
        seat = view.get("seat")
        owner, kind = view.get("vertex_owner") or [], view.get("vertex_building") or []
        if build == "road":
            placed = sum(1 for s in view.get("edge_owner") or [] if s == seat)
        elif build == "city":
            placed = sum(1 for v, s in enumerate(owner) if s == seat and v < len(kind) and kind[v] == _CITY)
        else:
            placed = sum(1 for v, s in enumerate(owner) if s == seat and not (v < len(kind) and kind[v] == _CITY))
        if placed >= supply:
            return "pieces"
    return "spot"


def _spots(legal: list[dict], board: dict) -> list[dict]:
    """Every settlement/city placement in `legal`, joined to the vertex's pips,
    resources and port (`port_matches`: a 3:1, or a 2:1 in a resource the
    vertex yields), best first; `index` is the `act()` index. Uncapped, since
    `legal_actions` drops its own settlement/city groups once this exists."""
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
            # A 2:1 port is worth what the vertex produces of that resource;
            # a 3:1 takes anything.
            spot["port_matches"] = port.get("resource") is None or port["resource"] in spot["resources"]
        spots.append(spot)
    spots.sort(key=lambda s: (-s["pips"], s["index"]))
    return spots


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


def _far_endpoint(v0: int, v1: int, own: set[int], neighbors: dict[int, set[int]]) -> int:
    """Which of an edge's two vertices `_roads` calls its `to`: the one this
    seat's network does not already touch, preferring the newer frontier when
    neither end is touched."""
    v0_own, v1_own = v0 in own, v1 in own
    if v0_own != v1_own:
        return v1 if v0_own else v0
    if not v0_own:  # both new
        v0_adj = any(n in own for n in neighbors.get(v0, ()))
        v1_adj = any(n in own for n in neighbors.get(v1, ()))
        if v0_adj != v1_adj:
            return v1 if v0_adj else v0
    return v1  # both already touched, or tied either way: either end


def _roads(legal: list[dict], view: dict, board: dict) -> list[dict]:
    """Every legal road placement, joined to the vertex it reaches: that
    vertex's pips, resources, port and `settle`, plus `then`/`then_pips`, the
    best settleable vertex one road on. A road with both ends already this
    seat's is flagged `link`. Best first."""
    edges = {e["id"]: e for e in board.get("edges") or []}
    vertices = {v["id"]: v for v in board.get("vertices") or []}
    ports = {v: p for p in board.get("ports") or [] for v in p.get("vertices") or []}
    neighbors: dict[int, set[int]] = {}
    for edge in board.get("edges") or []:
        neighbors.setdefault(edge["v0"], set()).add(edge["v1"])
        neighbors.setdefault(edge["v1"], set()).add(edge["v0"])
    seat = view.get("seat")
    owner = view.get("vertex_owner") or []
    own: set[int] = {v for v, s in enumerate(owner) if s == seat}
    for edge_id, s in enumerate(view.get("edge_owner") or []):
        if s == seat:
            edge = edges.get(edge_id)
            if edge is not None:
                own.add(edge["v0"])
                own.add(edge["v1"])

    def built_on(vid: int) -> bool:
        return vid < len(owner) and owner[vid] >= 0

    def settleable(vid: int) -> bool:
        return not built_on(vid) and not any(built_on(n) for n in neighbors.get(vid, ()))

    roads = []
    for index, action in enumerate(legal):
        if action.get("type") not in _ROAD_ACTIONS:
            continue
        edge_id = action.get("a")
        edge = edges.get(edge_id, {})
        v0, v1 = edge.get("v0"), edge.get("v1")
        to = _far_endpoint(v0, v1, own, neighbors) if v0 is not None and v1 is not None else v1
        vertex = vertices.get(to, {})
        row: dict = {
            "index": index,
            "edge": edge_id,
            "to": to,
            "pips": vertex.get("pips", 0),
            "resources": vertex.get("resources", []),
            "settle": settleable(to),
        }
        if v0 in own and v1 in own:
            # Both ends already ours: this road reaches nothing new, so
            # `to`/`pips` describe a vertex the seat already holds.
            row["link"] = True
        port = ports.get(to)
        if port is not None:
            row["port"] = _port_label(port)
        near = v0 if to == v1 else v1
        beyond = [n for n in neighbors.get(to, ()) if n != near and n not in own and settleable(n)]
        if beyond:
            best = max(beyond, key=lambda n: (vertices.get(n, {}).get("pips", 0), -n))
            row["then"] = best
            row["then_pips"] = vertices.get(best, {}).get("pips", 0)
        roads.append(row)
    roads.sort(key=lambda r: (not r["settle"], -r["pips"], -r.get("then_pips", 0), r["index"]))
    return roads


def _race(view: dict, me: dict) -> dict:
    """Where this seat stands: `points` and `to_win` (against the view's
    `winning_points`), `top_opponent` by *public* points (hidden victory-point
    cards are not counted for anyone else), and per award: yours, who holds it,
    and `need`, the count that would take it."""
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
        "top_opponent": None if leader is None else {"seat": leader.get("seat"), "points": leader.get("victory_points", 0)},
        "longest_road": award("longest_road", "road_length", _MIN_LONGEST_ROAD),
        "largest_army": award("largest_army", "knights_played", _MIN_LARGEST_ARMY),
    }


def _summarize(view: dict, board: dict | None) -> dict:
    """Adds `summary` to a translated view for a seated reader whose hand
    the view reveals. `spots`, `robber` and `roads` need the board and
    appear only when there is a placement, a robber move or a road to
    make."""
    seat = view.get("seat")
    players = view.get("players") or []
    me = next((p for p in players if p.get("seat") == seat), None) if seat is not None else None
    if me is None or "hand" not in me:
        return view
    legal = view.get("legal_actions") or []
    summary: dict = {"afford": _afford(me["hand"], legal, view, board), "race": _race(view, me)}
    if board is not None:
        spots = _spots(legal, board)
        if spots:
            summary["spots"] = spots
        robber = _robber(legal, view, board)
        if robber:
            summary["robber"] = robber
        roads = _roads(legal, view, board)
        if roads:
            summary["roads"] = roads
    view["summary"] = summary
    return view


# --- The compact reply: grouped legal_actions, sparse occupancy ----------------
#
# `act()` resolves an index against the wire's flat `legal_actions` list, which
# `_act` reads raw. The reply groups those by type, names the operand (`edge`,
# `vertex`, `hex`, `victim`, or a resource name) and keeps the flat `index`.
# Occupancy becomes `buildings` and `roads` by seat.

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
    `MOVE_ROBBER` victim slot at or past the seat count means nobody and reads
    back as `null`."""
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
            # A type this table does not know: keep the raw operands.
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


# Wire fields a reader never acts on, dropped from every reply (`_prune`):
# `version` (no MCP tool takes it), `claimed_seats` (`players[].kind`),
# `waiting_for`/`trade_wait`/`to_move` (folded into `your_move`/`waiting_on`),
# `awaiting_confirm` (an MCP seat never sees it) and `can_undo` (no undo tool).
_DEAD = ("version", "claimed_seats", "waiting_for", "trade_wait", "to_move", "awaiting_confirm", "can_undo")
# Dropped only while empty.
_DEAD_WHEN_EMPTY = ("locked", "trades")
_DEAD_WHEN_NONE = ("winner",)  # seat 0 wins too: only null is nothing to say
# Per-player fields dropped the same way: `last_roll` is stale off the roller,
# the award flags repeat `summary.race`'s `held`.
_PLAYER_DEAD = ("last_roll", "longest_road", "largest_army")
# Per-seat counts sent sparse: a missing name is a zero.
_SPARSE_COUNTS = ("hand", "known", "dev_cards")


def _sparse(counts: dict | None) -> dict:
    """A count dict with its zero entries dropped."""
    return {name: n for name, n in (counts or {}).items() if n}


def _prune(view: dict) -> dict:
    """The wire's redundancies, removed once per reply: `seats` collapses into
    a `kind` on each player entry, per-seat `last_roll` and award flags go, and
    `discard_quota` is dropped when every entry is zero."""
    for key in _DEAD:
        view.pop(key, None)
    for key in _DEAD_WHEN_EMPTY:
        if key in view and not view[key]:
            del view[key]
    for key in _DEAD_WHEN_NONE:
        if key in view and view[key] is None:
            del view[key]
    if view.get("started"):
        del view["started"]  # only an unstarted table is worth saying
    if not any(view.get("discard_quota") or []):
        view.pop("discard_quota", None)
    kinds = {s.get("seat"): s.get("kind") for s in view.pop("seats", None) or []}
    players = []
    for player in view.get("players") or []:
        for key in _PLAYER_DEAD:
            player.pop(key, None)
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


# --- Tables: one header, then rows -------------------------------------------
#
# Lists of small dicts become one string, `(k1,k2):v,v|v,v`, the keys named
# once. Inside a cell, lists join with `;`, dicts with `:` between key and
# value and `+` between fields; `-` is null.


def _cell(value) -> str:
    """One table cell: `-` for null, `1`/`0` for a bool, `;`/`+` for nesting."""
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, list):
        return ";".join(_cell(v) for v in value)
    if isinstance(value, dict):
        return "+".join(f"{k}:{_cell(v)}" for k, v in value.items())
    return str(value)


def _tabulate(rows: list[dict]) -> str:
    """Rows of dicts as `(keys):row|row`, the keys named once."""
    keys: list[str] = []
    for row in rows:
        keys.extend(k for k in row if k not in keys)
    body = "|".join(",".join(_cell(row.get(k)) for k in keys) for row in rows)
    return f"({','.join(keys)}):{body}"


def _tabulate_summary(summary: dict) -> None:
    """Tabulate `summary`'s `spots`/`robber`/`roads` in place."""
    spots = summary.get("spots")
    if spots:
        # A single kind is named once, up front; a mixed list keeps `type`.
        kinds = {r.get("type") for r in spots}
        if len(kinds) == 1:
            summary["spots"] = f"{kinds.pop()}:" + _tabulate([{k: v for k, v in r.items() if k != "type"} for r in spots])
        else:
            summary["spots"] = _tabulate(spots)
    robber = summary.get("robber")
    if robber:
        rows = [
            {
                **{k: v for k, v in r.items() if k not in ("hits", "options")},
                # seat:settlements+cities, e.g. `0:1s+1c`; `-` for nobody.
                "hits": [f"{h['seat']}:{h['settlements']}s+{h['cities']}c" for h in r["hits"]] or None,
                # act() index:victim, e.g. `12:0`; `12:-` for nobody to rob.
                "options": [f"{o['index']}:{_cell(o['victim'])}" for o in r["options"]],
            }
            for r in robber
        ]
        summary["robber"] = _tabulate(rows)
    roads = summary.get("roads")
    if roads:
        # No `type` to strip: BUILD_ROAD and SETUP_ROAD never overlap, being
        # different phases.
        summary["roads"] = _tabulate(roads)


# `summary.spots`/`summary.robber` already name every settlement/city or
# MOVE_ROBBER entry, so the bare group in `legal_actions` is the same thing
# twice. `act()` still resolves against the raw list, never this grouped one.
def _drop_superseded(grouped: dict[str, list[dict]], summary: dict) -> None:
    if summary.get("spots"):
        for kind in _SPOT_ACTIONS:
            grouped.pop(kind, None)
    if summary.get("robber"):
        grouped.pop("MOVE_ROBBER", None)


def _compact(view: dict, session: Session) -> dict:
    """The last step before a reply goes out. Everything that reads the flat
    list, the dense arrays or the fields `_prune` drops must already have
    run."""
    legal = view.get("legal_actions") or []
    grouped = _group_actions(legal, len(view.get("players") or []))
    _drop_superseded(grouped, view.get("summary") or {})
    view["legal_actions"] = {kind: _tabulate(rows) for kind, rows in grouped.items()}
    if view.get("summary"):
        _tabulate_summary(view["summary"])
    return _prune(_compact_board(view))


def _translate(raw: dict) -> dict:
    """Every translation a view gets on its way to the LLM, except the
    transcript trim and the `summary`/`_compact` steps `_finish` adds."""
    raw = _translate_trades(raw)
    raw["legal_actions"] = [_translate_action(a) for a in raw.get("legal_actions") or []]
    return raw


def _trim_for(session: Session, view: dict, log_after: int | None = None, full_log: bool = False) -> dict:
    """The transcript trim every tool reply gets: `log_after` if the caller
    sent one, otherwise `Session.log_sent`, or nothing for `full_log`, the
    reset for a reply that went missing. What was sent is recorded afterwards,
    so a reply that errors first records nothing."""
    cursor = None if full_log else (log_after if log_after is not None else session.log_sent)
    view = _trim_log(view, cursor)
    if "log_total" in view:
        session.log_sent = view["log_total"]
    return view


def _can_offer(view: dict) -> bool:
    """Whether `offer_trade()` would be accepted right now: `api.open_round`'s
    MAIN-and-on-move rule, with no Road Building roads still to place -- the
    one time in MAIN that `END_TURN` is not on offer -- and no round of this
    seat's own already open."""
    return (
        view.get("phase") == "MAIN"
        and view.get("to_move") == view.get("seat")
        and any(a.get("type") == "END_TURN" for a in view.get("legal_actions") or [])
        and view.get("trade_round") is None
    )


def _finish(session: Session, view: dict, log_after: int | None = None, full_log: bool = False) -> dict:
    """The one seam every view crosses on its way out, so a reply's shape never
    depends on which tool returned it: `can_offer`, `summary`, `_compact`, then
    the transcript trim. `view` must already be `_translate`d, and `can_offer`
    runs before `_compact` reshapes what it reads."""
    view["can_offer"] = _can_offer(view)
    view = _trim_for(session, _compact(_summarize(view, session.board), session), log_after, full_log)
    if view.get("game_over"):
        this = len(json.dumps(view, separators=(",", ":")))
        view["usage"] = {"calls": session.calls + 1, "bytes": session.bytes + this}
    return view


def _reply(session: Session, raw: dict, log_after: int | None = None, full_log: bool = False) -> dict:
    """A raw wire view as the tool reply the LLM reads: translated, then
    `_finish`ed."""
    return _finish(session, _translate(raw), log_after, full_log)


def _offer_trade(
    tables: Tables,
    session: Session,
    give: dict,
    want: dict,
    to: list[int] | None = None,
    timeout: float | None = None,
    log_after: int | None = None,
    full_log: bool = False,
    give_any: int = 0,
    want_any: int = 0,
) -> Iterator:
    _seated(session)
    if to is not None:
        if not isinstance(to, list) or not all(isinstance(seat, int) for seat in to):
            raise ToolError("to must be a list of seat numbers, best first")
        to = list(dict.fromkeys(to))  # ranked, no repeats
    body = {"give": _positional(give or {}), "want": _positional(want or {})}
    if give_any:
        body["give_any"] = give_any
    if want_any:
        body["want_any"] = want_any
    offered = _call_ok(tables, session, "POST", f"/api/games/{session.code}/trade/round", body)
    session.offer_to = to
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
    # No `version`: the wire refuses anything but the exact open offer by
    # `actor` + `received`, which is the only staleness that can hurt this.
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
    # No `version`: the wire matches the chosen answer by `seat` + `bundle`
    # exactly.
    raw = _call_ok(tables, session, "GET", "/api/state")
    session.offer_to = None
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
# No acting tool answers until there is something for this seat to do, so
# `your_move` is never `wait` unless the wait was cut short by `timeout` or
# `_MAX_WAIT`. `web.py` answers every `tools/call` as `text/event-stream`,
# writing a keepalive for each `KEEPALIVE` yielded here; `call_tool` drains
# the same generator for one blocking call.

KEEPALIVE = object()
_WAIT_TICK = 15.0
# The most any one call blocks, whatever `timeout` says. On expiry the reply
# is whatever the table looks like then, `your_move: wait` included.
_MAX_WAIT = 600.0


def _turn_ready(view: dict) -> bool:
    """Whether a view gives this seat something to do: the server's own
    `your_move` (`api.your_move`), which every view carries."""
    return view.get("your_move") != "wait"


def _poll_raw(tables: Tables, session: Session, after: int | None = None, wait: float = 0.0) -> dict:
    """One `GET /api/state`, optionally long-polling past version `after`."""
    query = "" if after is None else f"?after={after}&wait={wait}"
    return _call_ok(tables, session, "GET", f"/api/state{query}")


# The most forced moves one `_forced` pass plays before handing the view over
# regardless -- a bound, not a budget.
_FORCED_CAP = 8


def _settled_round(raw: dict, to: list[int] | None = None) -> dict | None:
    """The `.../trade/round/choose` body for an own round nobody is still to
    answer and nothing is left to decide, or `None`. A counter is always the
    seat's to weigh. With `to` (the seats `offer_trade` ranked) the best-ranked
    listed accept executes and no listed accept closes the round; without `to`,
    one accept executes, none closes, two or more are the seat's call."""
    trade_round = raw.get("trade_round")
    if not trade_round or trade_round.get("awaiting"):
        return None
    responses = trade_round.get("responses") or []
    if any(r.get("kind") == "counter" for r in responses):
        return None
    accepts = [r for r in responses if r.get("kind") == "accept"]
    if to is not None:
        accepts = sorted((a for a in accepts if a.get("seat") in to), key=lambda a: to.index(a["seat"]))
    elif len(accepts) > 1:
        return None
    if not accepts:
        return {"decline": True}
    return {"seat": accepts[0]["seat"], "bundle": accepts[0]["bundle"]}


def _forced(tables: Tables, session: Session, raw: dict) -> dict:
    """Play what no seat would decide differently, so no reply asks: a lone
    `ROLL`, a `pass` on an offer while the hand is empty, and the seat's own
    fully-answered offer when nothing is left to choose. An offer the hand
    cannot *cover* still reaches the seat, flagged `can_accept: false`."""
    for _ in range(_FORCED_CAP):
        legal = raw.get("legal_actions") or []
        if len(legal) == 1 and legal[0].get("type") == "ROLL":
            raw = _call_ok(tables, session, "POST", "/api/action", {"action": legal[0]})
            continue
        settled = _settled_round(raw, session.offer_to)
        if settled is not None and session.code:
            raw = _call_ok(tables, session, "POST", f"/api/games/{session.code}/trade/round/choose", settled)
            session.offer_to = None
            continue
        me = next((p for p in raw.get("players") or [] if p.get("seat") == raw.get("seat")), None)
        hand = None if me is None else me.get("hand")
        pending = raw.get("pending") or []
        unanswerable = pending[0] if pending and hand is not None and not any(hand.values()) else None
        if unanswerable is not None and session.code:
            body = {"actor": unanswerable["actor"], "received": unanswerable["bundle"], "kind": "pass"}
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
    """From `raw` — the wire view an action or a read just handed back — play
    the forced moves, then yield `KEEPALIVE` per wait tick until this seat has
    something to decide, then the one `_finish`ed reply. Every acting tool ends
    here, so every reply has `state()`'s shape, and only that final view moves
    the session's cursor."""
    # `_translate`, not `_finish`, until the last line: the views in between
    # are never seen by the caller, so they must not move the cursor.
    view = _translate(_forced(tables, session, raw))
    limit = _MAX_WAIT if timeout is None else max(0.0, min(float(timeout), _MAX_WAIT))
    deadline = time.monotonic() + limit
    while not _turn_ready(view):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        yield KEEPALIVE
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
# once. The cursor itself is automatic (`_trim_for`); these are the overrides.
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
        "The fixed board as text. Read it once, right after taking a seat, to "
        "plan expansion beyond what `summary` shows (one or two roads out). "
        "`hexes`: id, resource "
        "(DESERT/SEA pay nothing), pips (2d6 ways to roll its token: 5 "
        "for 6/8 down to 1 for 2/12), vertex ids. `vertices`: id, pips summed and "
        "resources de-duplicated over touching hexes (settlement value), port "
        "(e.g. Wheat2:1, 3:1, -), neighbor vertex ids a road reaches. No edge list: "
        "state()'s `summary.roads` names each legal edge's destination vertex.",
        {"type": "object", "properties": {}},
    ),
    "state": (
        _state,
        "Game state; every acting tool replies with it at your next decision. "
        "Played for you: a lone ROLL; passing offers with an empty hand; your own "
        "offer when all pass or one clean accept fits `to`.\n"
        "`your_move`: `act`, `discard`->discard(cards), `answer_trade` or "
        "`choose_trade`: the tool; `game_over`; `wait` only after `timeout` or "
        "with seats open (`waiting_on`; wait_for_turn()).\n"
        "`summary.afford` per build: `ok`, `missing`, `legal`, `why` (phase/spot/pieces/deck). `summary.race`: "
        "`points`, `to_win`, `top_opponent`, awards yours/holder's/`need`. When legal: "
        "`spots` (settlement/city vertices, pips/resources/port/`port_matches`, best "
        "first) and `robber` (hexes, pips, `hits` seat:Ns+Nc = settlements+cities, "
        "`options` actIndex:victimSeat), each replacing its `legal_actions` group; `roads` (`to` "
        "vertex, pips/resources/port, `settle`, `then` = best vertex one road on, "
        "`link` = joins your network).\n"
        "Tables are `(keys):row|row`, cells comma-separated, `-` null, 1/0 bool, "
        "`;` in a list. `legal_actions`: per type, `index` for act() plus `edge` (roads), "
        "`vertex` (settlement/city), `hex`+`victim` (MOVE_ROBBER), `give`/`want` "
        "(BANK_TRADE), `resource` (MONOPOLY/DISCARD), `resources` (YEAR_OF_PLENTY); "
        "the rest index only.\n"
        "`players`: `kind`, your `hand`/`dev_cards`, public ledger `known`/`unknown`. "
        "`buildings`, `roads` (edge ids per seat), "
        "`bank`, `robber` hex, `trade_ratios` (your bank rate; 3 or 2 = a port). "
        "Resource dicts omit zeros.\n"
        "`can_offer`: true if offer_trade() would be accepted now. `pending`: "
        "offers to you (`actor`, `you_give`, `you_receive`, `can_accept`; a counter "
        "is always allowed) -> answer_trade(index). "
        "`trade_round`: your open offer's `responses` (`seat`, `kind`, "
        "`you_would_give`/`you_would_receive`) -> choose_trade(index). `trades`: "
        "done this turn.\n"
        "`log`: new lines plus the last you hold (it may be rewritten); splice at "
        "`log_from`. Full on a new/resumed seat; `full_log` for the whole.",
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
    "discard": (
        _discard,
        "On a seven, discard `cards` (resource -> count) in one call instead of "
        "one act() per card. Must total your discard_quota exactly. Replies at "
        "your next move, as state().",
        {
            "type": "object",
            "properties": {
                "cards": _resource_dict('Resource -> count to discard, e.g. {"Wood": 2, "Ore": 1}.'),
                **_WAIT_ARGS,
            },
            "required": ["cards"],
        },
    ),
    "leave_game": (
        _leave_game,
        "Give up your seat for good; pieces and hand stay, your turns are skipped. "
        "Refuses while a trade round involving you is open.",
        {"type": "object", "properties": {}},
    ),
    "offer_trade": (
        _offer_trade,
        "On your own turn in MAIN, offer a trade to every other seat: any cards you "
        "hold for any you want, no resource on both sides; only while `can_offer`. `give_any`/"
        "`want_any` add any cards to one side, named by whoever answers; such an "
        "offer is only ever countered, never accepted. A lone clean "
        "accept is executed for you; with `to`, the best-ranked listed accepter "
        "is and others are refused. Counters, or several accepts without `to`, "
        "come back for choose_trade().",
        {
            "type": "object",
            "properties": {
                "give": _resource_dict('Resource -> count you give, e.g. {"Wood": 1}.'),
                "want": _resource_dict("Resource -> count you want."),
                "give_any": {"type": "integer", "minimum": 0,
                             "description": "Cards you give, of the answerer's naming. Omit for none."},
                "want_any": {"type": "integer", "minimum": 0,
                             "description": "Cards you take, of the answerer's choosing. Omit for none."},
                "to": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Seats you would trade with, best first. Omit for anyone.",
                },
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
    """Every tool as MCP `tools/list` describes it."""
    return [
        {"name": name, "description": description, "inputSchema": schema}
        for name, (_, description, schema) in _TOOLS.items()
    ]


def call_tool_events(tables: Tables, session: Session, name: str, arguments: dict) -> Iterator:
    """One tool call as `web.py` streams it: `KEEPALIVE` for every wait tick,
    then the one raw result dict — or a raised `ToolError`, before or between
    items. The MCP result envelope is `web.py`'s job."""
    entry = _TOOLS.get(name)
    if entry is None:
        raise ToolError(f"unknown tool: {name}")
    handler, _, _ = entry
    _check_arguments(arguments)
    try:
        result = handler(tables, session, **arguments)
    except TypeError as error:
        raise ToolError(f"bad arguments for {name}: {error}") from error
    if isinstance(result, (dict, str)):
        _count(session, result)
        yield result
    else:
        for item in result:
            if item is not KEEPALIVE:
                _count(session, item)
            yield item


def _check_arguments(arguments: dict) -> None:
    """The arguments every tool shares, checked before any tool runs, so a
    malformed one refuses the call rather than failing it half-done."""
    timeout = arguments.get("timeout")
    if timeout is not None and (type(timeout) not in (int, float) or not math.isfinite(timeout)):
        raise ToolError("timeout is a number of seconds")
    log_after = arguments.get("log_after")
    if log_after is not None and (type(log_after) is not int or log_after < 0):
        raise ToolError("log_after is the number of log lines you hold")


def _count(session: Session, result) -> None:
    """Charge one reply and its byte size to the session's `usage`."""
    session.calls += 1
    session.bytes += len(result) if isinstance(result, str) else len(json.dumps(result, separators=(",", ":")))


def call_tool(tables: Tables, session: Session, name: str, arguments: dict) -> dict | str:
    """`call_tool_events` drained: one blocking call -> its result (a dict, or
    `board`'s text)."""
    result: dict | str = {}
    for item in call_tool_events(tables, session, name, arguments):
        if item is not KEEPALIVE:
            result = item
    return result
