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
    return _seat(session, _call_ok(tables, session, "POST", "/api/games", body))


def _join(tables: Tables, session: Session, code: str, model: str, name: str | None = None) -> dict:
    if not isinstance(code, str) or not code.strip():
        raise ToolError("code must be a game's six-character code")
    client, _ = _client_of(model)
    session.model = model
    body: dict = {"code": code.strip().lower(), "name": _display_name(name), "client": client}
    return _seat(session, _call_ok(tables, session, "POST", "/api/join", body))


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
    return _translate_view(reclaimed)


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
    return raw


def _state(tables: Tables, session: Session) -> dict:
    _seated(session)
    return _translate_view(_call_ok(tables, session, "GET", "/api/state"))


def _version_check(fresh: dict, version: int | None) -> None:
    if version is not None and fresh.get("version") != version:
        raise ToolError(
            f"the table has moved (version {fresh.get('version')}); call state() or "
            "get_table() again and re-index from there"
        )


def _act(tables: Tables, session: Session, index: int, version: int | None = None) -> dict:
    state = _state(tables, session)
    _version_check(state, version)
    options = state.get("legal_actions") or []
    if not isinstance(index, int) or not (0 <= index < len(options)):
        raise ToolError(
            f"index {index!r} out of range — state()'s legal_actions has "
            f"{len(options)} option(s) right now (0..{len(options) - 1})"
            if options
            else "index out of range — state()'s legal_actions is empty; it is not your turn"
        )
    return _translate_view(_call_ok(tables, session, "POST", "/api/action", {"action": options[index]}))


def _undo(tables: Tables, session: Session) -> dict:
    _seated(session)
    return _translate_view(_call_ok(tables, session, "POST", "/api/undo"))


def _leave_game(tables: Tables, session: Session) -> dict:
    _seated(session)
    return _translate_view(_call_ok(tables, session, "POST", "/api/leave"))


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
            "responses": [
                {
                    "seat": r["seat"],
                    "kind": r["kind"],
                    **dict(zip(("you_would_give", "you_would_receive"), _actor_view(r["bundle"]))),
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


def _translate_view(raw: dict) -> dict:
    raw = _translate_trades(raw)
    raw["legal_actions"] = [_translate_action(a) for a in raw.get("legal_actions") or []]
    return raw


def _get_table(tables: Tables, session: Session) -> dict:
    return _state(tables, session)


def _offer_trade(tables: Tables, session: Session, give: dict, want: dict) -> dict:
    _seated(session)
    body = {"give": _positional(give), "want": _positional(want)}
    return _translate_view(_call_ok(tables, session, "POST", f"/api/games/{session.code}/trade/round", body))


def _answer_trade(
    tables: Tables,
    session: Session,
    index: int,
    kind: str,
    give: dict | None = None,
    receive: dict | None = None,
    version: int | None = None,
) -> dict:
    _seated(session)
    raw = _call_ok(tables, session, "GET", "/api/state")
    _version_check(raw, version)
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
    return _translate_view(_call_ok(tables, session, "POST", f"/api/games/{session.code}/trade/round/answer", body))


def _choose_trade(
    tables: Tables,
    session: Session,
    index: int | None = None,
    decline: bool = False,
    version: int | None = None,
) -> dict:
    _seated(session)
    raw = _call_ok(tables, session, "GET", "/api/state")
    _version_check(raw, version)
    if decline:
        return _translate_view(
            _call_ok(tables, session, "POST", f"/api/games/{session.code}/trade/round/choose", {"decline": True})
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
    body = {"seat": response["seat"], "bundle": response["bundle"]}
    return _translate_view(_call_ok(tables, session, "POST", f"/api/games/{session.code}/trade/round/choose", body))


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
    if view.get("game_over"):
        return True
    if view.get("legal_actions"):
        return True
    if view.get("pending"):
        return True
    trade_round = view.get("trade_round")
    if trade_round is not None and not trade_round.get("awaiting"):
        return True
    return False


def _poll_state(tables: Tables, session: Session, after: int | None = None, wait: float = 0.0) -> dict:
    query = "" if after is None else f"?after={after}&wait={wait}"
    return _translate_view(_call_ok(tables, session, "GET", f"/api/state{query}"))


def _wait_for_turn_events(tables: Tables, session: Session, timeout: float | None = None):
    """Yields `_KEEPALIVE` for each wait tick that doesn't resolve, then the
    final translated view -- immediately, if it's already true."""
    _seated(session)
    view = _poll_state(tables, session)
    if _turn_ready(view):
        yield view
        return
    elapsed = 0.0
    while timeout is None or elapsed < timeout:
        yield _KEEPALIVE
        remaining = _WAIT_TICK if timeout is None else max(0.0, min(_WAIT_TICK, timeout - elapsed))
        view = _poll_state(tables, session, after=view.get("version"), wait=remaining)
        elapsed += remaining
        if _turn_ready(view) or (timeout is not None and elapsed >= timeout):
            yield view
            return
    yield view


def _wait_for_turn(tables: Tables, session: Session, timeout: float | None = None) -> dict:
    result: dict = {}
    for item in _wait_for_turn_events(tables, session, timeout=timeout):
        if item is not _KEEPALIVE:
            result = item
    return result


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
        "The full current game state: every seat's public info (hand size, and "
        "your own hand; the public resource-count ledger for everyone else — "
        "counting isn't hidden information here, only a steal's identity and "
        "dev-card types are), the board's dynamic contents, and `legal_actions` "
        "— a 0-indexed list of the actions act() currently accepts, empty when "
        "it is not your turn. Also carries `version`, which bumps on every "
        "change — pass it to act()/answer_trade()/choose_trade() to have them "
        "refuse instead of guessing if the table moved under you. Poll this "
        "while another seat is thinking, or call wait_for_turn() instead to "
        "block until it's worth polling again. A legal_actions entry that "
        "spends a resource names it too, alongside the raw `a`/`b` act() "
        "replays: BANK_TRADE has `give`/`want`, PLAY_MONOPOLY/DISCARD have "
        "`resource`, PLAY_YEAR_OF_PLENTY has `resources` (a 2-list).",
        {"type": "object", "properties": {}},
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
                "version": {
                    "type": "integer",
                    "description": "Optional: pass the `version` from the state() you chose "
                    "the index from. If the table has moved since, act() refuses instead of "
                    "guessing what index still means what you intended.",
                },
            },
            "required": ["index"],
        },
    ),
    "undo": (
        _undo,
        "Undo your own most recent build or bank trade, if state()'s can_undo is "
        "true. Anything else (another seat's move, a played development card) "
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
        "`responses` (each `you_would_give`/`you_would_receive` if chosen) and "
        "`awaiting`, the seats still to answer. Resource dicts omit zero counts. "
        "Use a `pending`/`responses` list's index with answer_trade()/"
        "choose_trade() -- never hand-build a trade from these dicts.",
        {"type": "object", "properties": {}},
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
                "version": {
                    "type": "integer",
                    "description": "Optional: pass the `version` from the state() you chose "
                    "the index from. If the table has moved since, this refuses instead of "
                    "guessing what index still means what you intended.",
                },
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
                "version": {
                    "type": "integer",
                    "description": "Optional: pass the `version` from the state() you chose "
                    "the index from. If the table has moved since, this refuses instead of "
                    "guessing what index still means what you intended.",
                },
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
