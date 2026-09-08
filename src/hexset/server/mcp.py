"""An MCP server so an LLM can take a seat in a HexSet game, over stdio.

This is a thin client of the HTTP API `api.py` defines and `web.py` serves,
not a second game engine binding: every tool below is a `urllib` call to a
running `python -m hexset.server.web` (see `HEXSET_UI_BASE_URL`). That server can
be anywhere — this is how an LLM joins a game on a machine that actually has
ONNX Runtime while running somewhere that does not. A bot checkpoint reaches
the same server the same way — see `botclient.py` — this module and that one
are peers, not one built on the other.

Identity is the seat token the API mints (see `api.py`), held in this process
for its lifetime. One MCP connection is one seat at one game: `new_game`
deals a fresh one, dealt and playable immediately (there is no lobby to
start), `join` takes an open seat at somebody else's by its code, and either
way the token that comes back is what every later tool acts with.

Standard library only, deliberately: the official `mcp` SDK pulls in
`pydantic` (a compiled, Rust-built dependency `onnxruntime` and `numpy` don't
ask for anywhere else in this project), and the stdio wire format it would
save writing here is a handful of JSON-RPC 2.0 methods — `initialize`,
`tools/list`, `tools/call` — small enough to hand-roll directly against the
MCP spec instead, matching the same "standard library only" choice
`web.py`'s own docstring already made for the HTTP side.

Run it with (from `src/`, alongside an already-running web)::

    python -m hexset.server.mcp

stdin/stdout carry the protocol; nothing else may write to stdout, so every
log line here goes to stderr instead.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import urllib.error
import urllib.request

from .constants import TOKEN_HEADER

BASE_URL = os.environ.get("HEXSET_UI_BASE_URL", "http://127.0.0.1:8770").rstrip("/")
PROTOCOL_VERSION = "2024-11-05"
SERVER_INFO = {"name": "hexset", "version": "0.1.0"}

# Resource order every wire bundle (5 signed or unsigned ints) uses -- see
# `catanatron/names.py`'s RESOURCE_NAMES, Title-cased as the wire's hand/bank
# dicts spell them.
RESOURCES = ("Wood", "Brick", "Sheep", "Wheat", "Ore")

# The seat this connection is playing, set by new_game/join and sent on every
# request after. A module global for the same reason the cookie jar it
# replaced was one: the process is the client, and there is exactly one of it.
_token: str | None = None
# The game this connection is seated at, needed to address the trade routes
# (`/api/games/<code>/...`) -- `/api/state`/`/api/action` don't need it, since
# the token alone already names one game, but the trade endpoints are
# addressed by code (`api.py`), so it is remembered here the same way.
_code: str | None = None

# Where a seat's token/code are cached between processes, so an LLM can
# resume a game the same way a human's browser does -- by holding onto the
# token client-side, not because the server remembers anything (`api.py`'s
# module docstring: a token "never touches disk", a restart "cannot hand a
# lost token back to anyone"). One file, since one process is one seat.
_SESSION_FILE = pathlib.Path(
    os.environ.get("HEXSET_MCP_SESSION_FILE", os.path.expanduser("~/.cache/hexset-mcp/session.json"))
)


class ToolError(Exception):
    """Raised by a tool implementation to report the failure back to the
    LLM as a normal (not protocol-level) tool result — see `_call_tool`."""


def _request(method: str, path: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    request = urllib.request.Request(f"{BASE_URL}{path}", data=data, method=method)
    # A real User-Agent, not urllib's default: a `BASE_URL` fronted by
    # Cloudflare (or similar) treats the default one as bot traffic and
    # returns a plain-text 403 instead of the JSON `web.py` would send,
    # which breaks the `json.loads` below for reasons that have nothing
    # to do with the game.
    request.add_header("User-Agent", f"hexset-mcp/{SERVER_INFO['version']}")
    if data is not None:
        request.add_header("Content-Type", "application/json")
    if _token is not None:
        request.add_header(TOKEN_HEADER, _token)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        # The API's own refusals are still a JSON body (see web.Handler._serve)
        # — read it rather than raising past it, so a 400 ("it is not your turn
        # to act") reaches the LLM as the same message a browser would get.
        return json.loads(error.read().decode("utf-8"))
    except urllib.error.URLError as error:
        raise ToolError(
            f"could not reach the HexSet server at {BASE_URL} ({error.reason}) "
            "— is `python -m hexset.server.web` running?"
        ) from error


def _request_ok(method: str, path: str, body: dict | None = None) -> dict:
    result = _request(method, path, body)
    if "error" in result:
        raise ToolError(result["error"])
    return result


def _seated() -> None:
    if _token is None:
        raise ToolError("not at a game yet — call new_game() or join(code) first")


def _seat(result: dict) -> dict:
    """Records the token a join or a deal handed back, and hides it again.

    The LLM never needs to see it — it is sent on its behalf by `_request` —
    and a token in the transcript is a token in the context window of whatever
    reads that transcript next. Also remembers the game's code, off the same
    response (`table.view` always carries `code`), for the trade routes below.
    """
    global _token, _code
    _token = result.pop("token")
    _code = result.get("code")
    _save_session()
    return result


def _save_session() -> None:
    """Persists the current seat so a later process can resume it (see
    `resume_game`) -- the same trick a human's browser plays by holding onto
    the token client-side, since the server itself remembers nothing."""
    try:
        _SESSION_FILE.parent.mkdir(parents=True, exist_ok=True)
        _SESSION_FILE.write_text(json.dumps({"base_url": BASE_URL, "code": _code, "token": _token}))
        _SESSION_FILE.chmod(0o600)
    except OSError as error:
        print(f"could not save session to {_SESSION_FILE}: {error}", file=sys.stderr)


def _load_session() -> dict:
    """Resumes the seat a previous process saved, if any -- called explicitly
    from `resume_game()` rather than at import time, so tests (which reset
    `_token`/`_code` per test via monkeypatch) aren't at the mercy of
    whatever session file happens to sit on disk."""
    try:
        saved = json.loads(_SESSION_FILE.read_text())
    except (OSError, json.JSONDecodeError):
        raise ToolError(f"no saved session at {_SESSION_FILE}")
    if saved.get("base_url") != BASE_URL:
        raise ToolError(
            f"saved session is for {saved.get('base_url')!r}, not this server ({BASE_URL!r})"
        )
    if not saved.get("token"):
        raise ToolError(f"no saved session at {_SESSION_FILE}")
    global _token, _code
    _token = saved["token"]
    _code = saved.get("code")
    return _state()


def _resume_game() -> dict:
    return _load_session()


def _models() -> dict:
    return _request_ok("GET", "/api/models")


def _display_name(name: str | None) -> str:
    """A claimed seat with no name of its own falls back to `webplay.py`'s
    generic "human" label -- indistinguishable in the log/UI from an actual
    person, which is exactly backwards for a seat only an LLM can hold.
    `new_game`/`join` always send a name because of this: `name` if the LLM
    gave one, "mcp" otherwise."""
    return str(name).strip()[:40] if name else "mcp"


def _new_game(opponents: list[str] | None = None, name: str | None = None) -> dict:
    body: dict = {"name": _display_name(name)}
    if opponents:
        body["bots"] = opponents
    return _seat(_request_ok("POST", "/api/games", body))


def _join(code: str, name: str | None = None) -> dict:
    if not isinstance(code, str) or not code.strip():
        raise ToolError("code must be a game's six-character code")
    body: dict = {"code": code.strip().lower(), "name": _display_name(name)}
    return _seat(_request_ok("POST", "/api/join", body))


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


def _board() -> dict:
    _seated()
    raw = _request_ok("GET", "/api/board")
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


def _state() -> dict:
    _seated()
    return _translate_view(_request_ok("GET", "/api/state"))


def _act(index: int) -> dict:
    state = _state()
    options = state.get("legal_actions") or []
    if not isinstance(index, int) or not (0 <= index < len(options)):
        raise ToolError(
            f"index {index!r} out of range — state()'s legal_actions has "
            f"{len(options)} option(s) right now (0..{len(options) - 1})"
            if options
            else "index out of range — state()'s legal_actions is empty; it is not your turn"
        )
    return _translate_view(_request_ok("POST", "/api/action", {"action": options[index]}))


def _undo() -> dict:
    _seated()
    return _translate_view(_request_ok("POST", "/api/undo"))


def _leave_game() -> dict:
    _seated()
    return _translate_view(_request_ok("POST", "/api/leave"))


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


def _get_table() -> dict:
    return _state()


def _offer_trade(give: dict, want: dict) -> dict:
    _seated()
    body = {"give": _positional(give), "want": _positional(want)}
    return _translate_view(_request_ok("POST", f"/api/games/{_code}/trade/round", body))


def _answer_trade(index: int, kind: str, give: dict | None = None, receive: dict | None = None) -> dict:
    _seated()
    raw = _request_ok("GET", "/api/state")
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
    return _translate_view(_request_ok("POST", f"/api/games/{_code}/trade/round/answer", body))


def _choose_trade(index: int | None = None, decline: bool = False) -> dict:
    _seated()
    if decline:
        return _translate_view(
            _request_ok("POST", f"/api/games/{_code}/trade/round/choose", {"decline": True})
        )
    raw = _request_ok("GET", "/api/state")
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
    return _translate_view(_request_ok("POST", f"/api/games/{_code}/trade/round/choose", body))


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
        "for answer_trade().",
        {
            "type": "object",
            "properties": {
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
        },
    ),
    "join": (
        _join,
        "Take a random open seat at an existing game by its six-character code. "
        "Fails if every seat is taken or has locked out (see state()'s `locked`) "
        "— a seat somebody closed outright is retired for the rest of that "
        "game, so join before that happens. Your seat is gated the same way "
        "new_game()'s is — see its description.",
        {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "The game's six-character code."},
                "name": {"type": "string", "description": "Your display name, up to 40 characters."},
            },
            "required": ["code"],
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
        "it is not your turn. Poll this while another seat is thinking: nothing "
        "plays a turn on your behalf, bot seats included. A legal_actions entry "
        "that spends a resource names it too, alongside the raw `a`/`b` act() "
        "replays: BANK_TRADE has `give`/`want`, PLAY_MONOPOLY/DISCARD have "
        "`resource`, PLAY_YEAR_OF_PLENTY has `resources` (a 2-list).",
        {"type": "object", "properties": {}},
    ),
    "act": (
        _act,
        "Play legal_actions[index] from the most recent state() (call state() "
        "first if unsure what's legal right now). Returns the state right after "
        "that one action — ending your own turn is END_TURN, an action like any "
        "other, not something act() infers.",
        {
            "type": "object",
            "properties": {"index": {"type": "integer", "description": "Index into legal_actions."}},
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
            },
        },
    ),
    "resume_game": (
        _resume_game,
        "Reclaim the seat this process (or an earlier run pointed at the same "
        "server) last held, restored from a local cache file rather than the "
        "server -- the same trick a human's browser plays by holding onto its "
        "own session token, since the server itself forgets a seat's token the "
        "moment nothing is holding it. Fails if nothing was saved here, or it "
        "was saved for a different HEXSET_UI_BASE_URL than this process has now.",
        {"type": "object", "properties": {}},
    ),
}

def _tool_list() -> list[dict]:
    return [
        {"name": name, "description": description, "inputSchema": schema}
        for name, (_, description, schema) in _TOOLS.items()
    ]


def _call_tool(name: str, arguments: dict) -> dict:
    entry = _TOOLS.get(name)
    if entry is None:
        raise ToolError(f"unknown tool: {name}")
    handler, _, _ = entry
    try:
        result = handler(**arguments)
    except TypeError as error:
        raise ToolError(f"bad arguments for {name}: {error}") from error
    return {"content": [{"type": "text", "text": json.dumps(result)}], "isError": False}


def _dispatch(message: dict) -> dict | None:
    """One JSON-RPC request -> its response, or `None` for a notification
    (an `id`-less message, which the spec says gets no reply at all — the
    only one a compliant client sends unprompted is `notifications/initialized`
    right after `initialize`, and nothing here needs to react to it)."""
    method = message.get("method")
    request_id = message.get("id")
    if request_id is None:
        return None

    if method == "initialize":
        result = {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": SERVER_INFO,
        }
        return {"jsonrpc": "2.0", "id": request_id, "result": result}

    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": request_id, "result": {"tools": _tool_list()}}

    if method == "tools/call":
        params = message.get("params") or {}
        try:
            result = _call_tool(params.get("name"), params.get("arguments") or {})
        except ToolError as error:
            result = {"content": [{"type": "text", "text": str(error)}], "isError": True}
        return {"jsonrpc": "2.0", "id": request_id, "result": result}

    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": -32601, "message": f"method not found: {method}"},
    }


def main() -> None:
    print(f"hexset MCP server: talking to {BASE_URL}", file=sys.stderr)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError as error:
            print(f"bad JSON-RPC line, skipped: {error}", file=sys.stderr)
            continue
        if not isinstance(message, dict):
            print(f"bad JSON-RPC message, skipped: {message!r}", file=sys.stderr)
            continue
        try:
            response = _dispatch(message)
        except Exception as error:  # noqa: BLE001 — one bad request must not kill the process
            print(f"unhandled error dispatching {message.get('method')}: {error}", file=sys.stderr)
            response = {
                "jsonrpc": "2.0",
                "id": message.get("id"),
                "error": {"code": -32603, "message": str(error)},
            }
        if response is not None:
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
