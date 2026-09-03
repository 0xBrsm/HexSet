"""Game session and wire protocol for the local human-vs-policy web board.

Deliberately torch-free: `hexset.server.web` imports the network bot lazily, so
this module — the board layout math, the wire-format mapping and the session
that drives a game — can be imported and tested without PyTorch, the same way
`hexset.actions` and `hexset.game` can. Anything with a
`choose(game) -> Action` method is all a session needs of its opponent:
`NetworkBot` from `hexset.clients.onnxbot`, `Heximax` or `SearchBot` from
`heximax`/`hexset.bots`, or the `RandomBot` the tests use.

## The wire format

An `Action` is a `NamedTuple` — a plain tuple under the hood — so it survives a
JSON round trip if its fields are given JSON-friendly types. `action_to_wire`
does that (the enum becomes its name, the tuples become lists); `wire_to_action`
undoes it. Round-tripping is exact: `wire_to_action(action_to_wire(a)) == a` for
every `a` `legal_actions` can produce, and `test_webplay.py` pins that across a
played-out game rather than a handful of hand-picked shapes.

## Never build an action the engine did not offer

`GameSession.submit` decodes the wire action and checks it against
a *fresh* call to `rules.is_legal`, not merely against what was on offer at
some earlier poll. A UI bug, a stale page, or a tampered request all fail the same
way: the action is rejected before it reaches `hexset.actions.apply`. That is
also why every clickable thing in the frontend is one of the literal wire
objects `state_view()` already sent, echoed back unchanged — the client never
constructs an `Action` from parts, it only ever repeats one the server offered.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

from hexset.actions import (
    YEAR_OF_PLENTY_PAIRS,
    Action,
    ActionType,
    apply,
    legal_actions,
)
from hexset.board.board import Board
from hexset.board.coords import Hex
from hexset.board.terrain import NUM_RESOURCES, TERRAIN_RESOURCE, Resource
from hexset.board.topology import Topology
from hexset.cards import NUM_DEV_CARDS, DevCard
from hexset.devcards import holdings
from hexset.economy import trade_ratios
from hexset.game import Game, Phase, is_over, run_pending_event, to_move
from hexset.ledger import PublicLedger
from hexset.roads import road_lengths
from hexset.state import MAX_CITIES, MAX_ROADS, MAX_SETTLEMENTS, GameState, copy_state
from hexset.trading import NO_VALUATION, Bundle, Trade, apply_trades
from hexset.victory import public_victory_points, victory_points

from .journal import Journal
from .rules import is_legal
from .seating import locked_of, settle, snapshot

RESOURCE_NAMES: tuple[str, ...] = tuple(r.name.title() for r in Resource)
DEV_CARD_NAMES: tuple[str, ...] = tuple(c.name.title().replace("_", " ") for c in DevCard)


class ResumeError(Exception):
    """A journalled game would not replay — its actions no longer describe a
    legal game under this engine. Recoverable, and by design: the caller deals
    a fresh game rather than failing the request (see `api.resume_session`), so
    an engine change that invalidates old journals costs the games in flight
    at the time and nothing else."""


# --- Hex-to-pixel layout -----------------------------------------------------
#
# Pointy-top orientation (textbook redblobgames algebra): a hex's six corners
# sit at angles 60*i - 30 degrees around its center, and `Topology.hex_vertices`
# already lists a hex's vertices in that same i = 0..5 order (corner i is shared
# with the neighbours in directions i and i+1, and the direction vectors in
# `hexset.board.coords` place direction i at angle 60*(i-1) under this same
# center formula — corner i sits at the midpoint of that, 60*i - 30). One
# consequence worth relying on in tests: a regular hexagon's edge length equals
# its circumradius, so every board edge should measure exactly `size` between
# its two vertex pixels, whatever the board's shape.

SQRT3 = math.sqrt(3.0)


def hex_center(h: Hex, size: float) -> tuple[float, float]:
    x = size * (SQRT3 * h.q + SQRT3 / 2 * h.r)
    y = size * (1.5 * h.r)
    return (x, y)


def hex_corner(center: tuple[float, float], index: int, size: float) -> tuple[float, float]:
    angle = math.radians(60 * index - 30)
    return (center[0] + size * math.cos(angle), center[1] + size * math.sin(angle))


def vertex_pixels(topology: Topology, size: float) -> list[tuple[float, float]]:
    """One pixel position per vertex, agreeing across every hex that touches it."""
    positions: list[tuple[float, float] | None] = [None] * topology.num_vertices
    for h in range(topology.num_hexes):
        center = hex_center(topology.hexes[h], size)
        for corner_index, v in enumerate(topology.hex_vertices[h]):
            if positions[v] is None:
                positions[v] = hex_corner(center, corner_index, size)
    missing = [v for v, p in enumerate(positions) if p is None]
    if missing:
        raise AssertionError(f"vertices with no touching hex: {missing}")
    return positions  # type: ignore[return-value]


def board_layout(board: Board, size: float = 60.0) -> dict:
    """Static board geometry and contents for `/api/board`.

    Sent once per game: nothing here changes as the game is played. Occupancy
    (who owns which vertex/edge, where the robber sits) lives in `state_view`
    instead, which is polled after every move.
    """
    topology = board.topology
    vpix = vertex_pixels(topology, size)
    hexes = []
    for h in range(topology.num_hexes):
        cx, cy = hex_center(topology.hexes[h], size)
        hexes.append(
            {
                "id": h,
                "terrain": board.terrain[h].name,
                "token": board.tokens[h] or None,
                "x": round(cx, 3),
                "y": round(cy, 3),
                # The six corner vertex ids, in order, so the frontend draws the
                # hex outline from the same vertex pixels the buildings sit on
                # rather than recomputing corners of its own.
                "vertex_ids": list(topology.hex_vertices[h]),
            }
        )
    vertices = [
        {"id": v, "x": round(x, 3), "y": round(y, 3)} for v, (x, y) in enumerate(vpix)
    ]
    edges = [
        {"id": e, "v0": a, "v1": b} for e, (a, b) in enumerate(topology.edges)
    ]
    ports = [
        {
            "edge": p.edge,
            "vertices": list(p.vertices),
            "resource": None if p.resource is None else RESOURCE_NAMES[p.resource],
            "ratio": p.ratio,
        }
        for p in board.ports
    ]
    return {
        "size": size,
        "hexes": hexes,
        "vertices": vertices,
        "edges": edges,
        "ports": ports,
        "resources": list(RESOURCE_NAMES),
        "dev_cards": list(DEV_CARD_NAMES),
        "year_of_plenty_pairs": [
            [RESOURCE_NAMES[a], RESOURCE_NAMES[b]] for a, b in YEAR_OF_PLENTY_PAIRS
        ],
        # The engine's own supply caps (`hexset.state`), so the frontend's
        # remaining-piece HUD can't drift from what `can_place_*` actually
        # enforces.
        "piece_supply": {
            "road": MAX_ROADS,
            "settlement": MAX_SETTLEMENTS,
            "city": MAX_CITIES,
        },
    }


# --- Trading -------------------------------------------------------------------


@dataclass(frozen=True)
class PostedValuation:
    """A seat's published valuation vector, as the trade mechanic's seam.

    What a person (or an LLM) at a HexSet table brings to a trade: five
    numbers in [-1, 1] set through `PUT /api/games/<code>/valuation`, and
    nothing else. `accepts` is unconditionally True, and that is the whole
    of the interface on purpose -- the engine only ever asks the gate about
    a bundle whose *public* surplus is already strictly positive for this
    seat (`hexset.trading.trade_event`), so the vector a seat posts is
    exactly the statement it is making. A richer human interface -- a
    per-exchange confirm, a per-opponent rule -- is deferred by the trading
    design, not approximated here.
    """

    vector: tuple[float, ...]

    def valuation(self, view) -> tuple[float, ...]:
        del view
        return self.vector

    def accepts(self, view, received: Bundle, counterparty: int) -> bool:
        del view, received, counterparty
        return True


@dataclass(frozen=True)
class PendingGate:
    """A seat's gate under confirm mode: never clears on its own, and
    records every candidate the table's automatic trade event found for it
    instead (`docs/negotiation-interface.md` §1).

    Parallel to `PostedValuation` -- `valuation` is unchanged, so a
    confirm-mode seat still advertises through the same five numbers -- but
    `accepts` always returns `False`, and its only side effect is appending
    the candidate to `game.pending` for the player (or LLM) to review
    through `POST .../trade/confirm` or `.../decline`.

    Because `_best_clearing` asks the acting seat's own gate before the
    counterparty's and short-circuits on `False`, a candidate is only ever
    recorded for a seat sitting as the *counterparty* once the other side's
    own gate has already accepted that exact bundle -- nothing pending here
    is speculative in the common case (a confirm-mode seat answering another
    seat's trade event). Holds `game` rather than `game.pending` itself so
    that a later event's `game.pending = []` (see `trade_event`) is seen
    through the same reference, not one already left behind.
    """

    game: "Game"
    seat: int
    vector: tuple[float, ...] = NO_VALUATION

    def valuation(self, view) -> tuple[float, ...]:
        del view
        return self.vector

    def accepts(self, view, received: Bundle, counterparty: int) -> bool:
        del view
        self.game.pending.append(Trade(self.seat, counterparty, received))
        return False


# --- Wire format for actions --------------------------------------------------


def action_to_wire(action: Action) -> dict:
    return {"type": action.type.name, "a": action.a, "b": action.b}


def wire_to_action(data: dict) -> Action:
    try:
        kind = ActionType[str(data["type"])]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"unknown action type {data.get('type')!r}") from exc
    try:
        return Action(type=kind, a=int(data.get("a", 0)), b=int(data.get("b", 0)))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"malformed action payload: {data!r}") from exc


# --- Wire format for a manually composed trade ---------------------------------


def bundle_from_wire(give: dict, receive: dict) -> Bundle:
    """A signed `Bundle`, positive towards the proposer, from the named
    amounts `POST .../trade`'s body carries (`{"Wood": 2}`, matching
    `RESOURCE_NAMES`'s own resource-name convention). Raises `ValueError` --
    the same as a malformed action -- for an unknown name or a non-integer
    count."""
    index = {name: r for r, name in enumerate(RESOURCE_NAMES)}
    counts = [0] * NUM_RESOURCES
    for side, sign in ((give, -1), (receive, 1)):
        for name, n in (side or {}).items():
            r = index.get(name)
            if r is None:
                raise ValueError(f"unknown resource: {name!r}")
            try:
                counts[r] += sign * int(n)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"malformed amount for {name!r}: {n!r}") from exc
    return tuple(counts)


# --- Human-readable log -------------------------------------------------------


def _bundle_text(bundle: tuple[int, ...]) -> str:
    parts = [f"{n} {RESOURCE_NAMES[r]}" for r, n in enumerate(bundle) if n]
    return ", ".join(parts) if parts else "nothing"


def _hex_label(board: Board, hex_id: int) -> str:
    """A hex the way a player reads the board, not by its internal id: the
    dice number and resource it produces, e.g. "8 Wood". The desert has
    neither.
    """
    resource = TERRAIN_RESOURCE[board.terrain[hex_id]]
    if resource is None:
        return "the desert"
    return f"{board.tokens[hex_id]} {RESOURCE_NAMES[resource]}"


def _resource_counts(counts: list[int]) -> str:
    """`[2, 1, 0, 0, 0]` -> `"2 Wood, 1 Brick"`; `[]`-equivalent -> `""`.

    Deliberately not `_list_with_counts`: that one pluralises, which is right
    for the countable things it lists ("2 roads") and wrong for every
    resource name in the game — "2 Sheep" and "2 Wood", never "2 Sheeps".
    """
    return ", ".join(f"{n} {RESOURCE_NAMES[r]}" for r, n in enumerate(counts) if n)


def _hand_gains(before: list[int], after: list[int]) -> str | None:
    return _resource_counts([after[r] - before[r] for r in range(NUM_RESOURCES)]) or None


# (verb, noun) for every action that folds into one "placed/built ..." run
# per actor instead of a line each — see render_log.
_BUILD_KIND = {
    ActionType.SETUP_SETTLEMENT: ("placed", "settlement"),
    ActionType.SETUP_ROAD: ("placed", "road"),
    ActionType.BUILD_ROAD: ("built", "road"),
    ActionType.BUILD_SETTLEMENT: ("built", "settlement"),
    ActionType.BUILD_CITY: ("built", "city"),
}


def _list_with_counts(items: list[str]) -> str:
    """`['settlement', 'road']` -> `'a settlement and a road'`;
    `['road', 'road']` -> `'2 roads'` — one count per distinct item, in the
    order first seen, not one entry per occurrence."""
    order = list(dict.fromkeys(items))
    parts = [f"a {item}" if items.count(item) == 1 else f"{items.count(item)} {item}s" for item in order]
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + f" and {parts[-1]}"


@dataclass
class _Snapshot:
    hands: list[list[int]]
    held: list[list[int]]


# The placement and bank/port-trade actions any seat can take back — its own
# only (see `_apply`'s `_UndoPoint.actor` and `undo_last_build`): an
# opponent's misclick, or a bot's, isn't a different seat's to undo. Road
# Building's free roads need no special case: they still arrive as ordinary
# BUILD_ROAD
# actions (see game.build_road), so restoring game.free_roads alongside the
# board covers them too. PLAY_ROAD_BUILDING itself is in here too, for the
# instant right after the card is played but before either free road has
# landed — the only way to give the card back once it's already spent
# server-side (unlike a Knight, which the client never sends until a victim
# is chosen, so it never needs a matching undo point).
_UNDOABLE_BUILDS: frozenset[ActionType] = frozenset(
    {
        ActionType.SETUP_SETTLEMENT,
        ActionType.SETUP_ROAD,
        ActionType.BUILD_ROAD,
        ActionType.BUILD_SETTLEMENT,
        ActionType.BUILD_CITY,
        ActionType.BANK_TRADE,
        ActionType.PLAY_ROAD_BUILDING,
    }
)


@dataclass
class _UndoPoint:
    """Everything one placement action could have touched, from just before
    it ran — restoring these fields is restoring the whole session to that
    instant, without having to compute any action's reverse by hand (refund
    which resources, recompute longest road, work out whose turn a setup
    placement handed off to, ...).

    state/free_roads cover a paid Main-phase build. phase/current_player/
    setup_step/last_settlement only ever move during setup (see
    game.place_initial_settlement/place_initial_road) and are otherwise
    exactly what they were, so restoring them unconditionally is correct
    either way rather than needing two separate cases."""

    state: GameState
    # The public-knowledge ledger, snapshotted with the state and restored
    # with it. Not optional and not derivable: `PublicLedger` is built
    # forward from the moves that were public, so an undone BANK_TRADE that
    # left the ledger alone would leave it certifying a floor the hand no
    # longer supports -- `known[wheat] = 1` against a hand holding none, a
    # falsehood every seat then reads out of `/api/state` and `/api/record`
    # (PR #2 defect 2). `spend`'s own clamp hides it rather than fixing it.
    ledger: PublicLedger
    free_roads: int
    phase: Phase
    current_player: int
    setup_step: int
    last_settlement: int
    events: int
    steps: int
    # Whose action this would take back. Only its own actor may undo it: with
    # more than one human at a table, "the last undoable action" is not
    # automatically the asking player's to reach for.
    actor: int


def _snapshot(game: Game) -> _Snapshot:
    # true state: the server's own omniscient observer snapshot.
    state = game.state(0, hidden=False)
    return _Snapshot(
        hands=[hand[:] for hand in state.hands],
        held=[holdings(state, p)[:] for p in range(state.num_players)],
    )


@dataclass
class _Event:
    """One applied action, with everything the log could ever need to describe
    it — held in full, redacted for nobody.

    The log used to be rendered to sentences the instant an action landed,
    from the one human's point of view, and stored that way. With more than
    one human at a table that is not expressible: a steal names the resource
    to the thief and the victim and hides it from everyone else, and a
    discard spells out the cards only for the seat that lost them (see
    `_describe` and `render_log`). One shared list of sentences cannot say two
    different things at once, so what is stored here is the truth and the
    hiding happens per reader instead — the same split `hexset.server.journal`
    already makes, one layer up.

    `after` is the state the action produced, which is what most lines are
    actually about (what a roll paid out, what a Monopoly swept). `before` is
    kept alongside it because every one of those is a *difference*, and the
    engine only keeps the current hand.
    """

    round_num: int
    actor: int
    action: Action
    before: _Snapshot
    after: _Snapshot
    last_roll: int | None
    # The exchanges the engine cleared inside this action (`hexset.trading`).
    # The trade event runs on the way into the main phase and again after
    # every MAIN action the current player takes (owner review against the
    # rulebook, 2026-09-03: trade and build interleave), so any of those --
    # not only the roll or robber move that enters MAIN -- can carry several.
    # Empty for setup, discards, and every other seat's actions.
    trades: tuple[Trade, ...] = ()


def _who(seat: int, labels: dict[int, str]) -> str:
    """"Player N (label)" — the label is whoever holds the seat, the player's
    own registered name or the model name, the same one `state_view` already
    puts on that seat (see `GameSession.seat_labels`). One form for every
    seat, actor or bystander, rather than a special case for the reader that
    the rest of the log has to match — and with several humans at a table,
    "human" would no longer identify anyone anyway. Player numbers are
    1-indexed for the log even though `seat` itself stays 0-indexed
    everywhere else — "Player 0" reads as a bug to anyone who isn't a
    programmer.
    """
    return f"Player {seat + 1} ({labels.get(seat, 'bot')})"


def _describe(
    event: _Event,
    board: Board,
    labels: dict[int, str],
    viewer: int | None,
    *,
    omniscient: bool = False,
) -> str:
    """One event as a sentence, told to `viewer`.

    `viewer` is the seat reading the log, and the only thing it changes is
    what stays hidden: a card bought, a card stolen. Everything else reads
    identically to everyone, including whoever wasn't at the table at all
    (`viewer=None`), which is what a replay gets.

    `omniscient` names every one of those instead — the card bought, the card
    stolen, the cards discarded — for a reader outside the game entirely. It
    is not a seat's view with more in it: a seat may never be told these
    things about another seat, and nothing that acts on this game is ever
    handed a log built this way (see `state_view`).
    """
    actor, action = event.actor, event.action
    before, after = event.before, event.after
    num_players = len(after.hands)
    kind = action.type
    who = _who(actor, labels)

    if kind is ActionType.ROLL:
        line = f"{who} rolled {event.last_roll}."
        gains = [
            f"{_who(p, labels)} collects {g}."
            for p in range(num_players)
            if (g := _hand_gains(before.hands[p], after.hands[p])) is not None
        ]
        return " ".join([line, *gains])

    if kind is ActionType.SETUP_SETTLEMENT:
        line = f"{who} placed a settlement."
        gains = _hand_gains(before.hands[actor], after.hands[actor])
        if gains:
            line += f" {who} collects {gains}."
        return line

    if kind is ActionType.SETUP_ROAD:
        return f"{who} placed a road."

    if kind is ActionType.BUILD_ROAD:
        return f"{who} built a road."

    if kind is ActionType.BUILD_SETTLEMENT:
        return f"{who} built a settlement."

    if kind is ActionType.BUILD_CITY:
        return f"{who} built a city."

    if kind is ActionType.BUY_DEV_CARD:
        gained = [c for c in range(NUM_DEV_CARDS) if before.held[actor][c] < after.held[actor][c]]
        if (omniscient or actor == viewer) and gained:
            return f"{who} bought a {DEV_CARD_NAMES[gained[0]]}."
        return f"{who} bought a development card."

    if kind is ActionType.PLAY_ROAD_BUILDING:
        return f"{who} played Road Building."

    if kind in (ActionType.MOVE_ROBBER, ActionType.PLAY_KNIGHT):
        prefix = f"{who} played a Knight and " if kind is ActionType.PLAY_KNIGHT else f"{who} "
        victim = action.b if action.b < num_players else None
        line = f"{prefix}moved the robber to {_hex_label(board, action.a)}"
        if victim is None:
            return line + "."
        stolen = _hand_gains(before.hands[victim], after.hands[victim])
        # Named only to the two seats who already know it — the thief saw what
        # they took, the victim saw what left. To everyone else a steal is a
        # card, and which one is exactly the hidden-hand information the rest
        # of this module works to keep hidden.
        if (omniscient or viewer in (actor, victim)) and stolen:
            resource = next(
                RESOURCE_NAMES[r]
                for r in range(NUM_RESOURCES)
                if after.hands[victim][r] < before.hands[victim][r]
            )
            return line + f" and stole {resource} from {_who(victim, labels)}."
        return line + f" and stole a card from {_who(victim, labels)}."

    if kind is ActionType.PLAY_MONOPOLY:
        resource = RESOURCE_NAMES[action.a]
        swept = after.hands[actor][action.a] - before.hands[actor][action.a]
        return f"{who} played Monopoly on {resource} and collected {swept} card(s)."

    if kind is ActionType.PLAY_YEAR_OF_PLENTY:
        r0, r1 = YEAR_OF_PLENTY_PAIRS[action.a]
        return f"{who} played Year of Plenty for {RESOURCE_NAMES[r0]} and {RESOURCE_NAMES[r1]}."

    return f"{who} played {kind.name}."


def _trade_lines(event: _Event, labels: dict[int, str]) -> list[str]:
    """One sentence per exchange the engine cleared inside this action.

    Fully public: both hands, both bundles, both seats. A trade is an
    announced exchange at a real table, and the ledger already certifies
    every card that moved, so there is nothing here to redact per reader.
    """
    out = []
    for trade in event.trades:
        got = tuple(max(0, n) for n in trade.received)
        gave = tuple(max(0, -n) for n in trade.received)
        out.append(
            f"{_who(trade.a, labels)} traded {_bundle_text(gave)} "
            f"to {_who(trade.b, labels)} for {_bundle_text(got)}."
        )
    return out


def render_log(
    events: list[_Event],
    board: Board,
    labels: dict[int, str],
    viewer: int | None,
    *,
    omniscient: bool = False,
) -> list[str]:
    """Every event as the sidebar transcript `viewer` should see.

    A pure fold, run fresh per reader, because two humans at one table are
    owed two different transcripts (see `_Event`). That it is recomputed
    rather than appended to is also what makes undo trivial: dropping the
    events drops their lines, with no separate log to wind back.

    Three kinds of action arrive as a burst of engine steps that a reader
    would only ever want as one sentence, and each collapses into a run
    rewritten in place as it grows:

      builds     one "placed/built ..." per actor per round
      discards   one line per actor per discard, however many cards
      bank       consecutive trades of the same pair, summed

    Only ever one run is open at a time — anything that doesn't continue the
    current one clears it, so a run can never reach back across an intervening
    line to join something older (a second seven in the same round starts a
    fresh discard line rather than swelling the first).

    Trades are not actions and so are not events of their own: the engine
    clears them inside the roll or the robber move that opened the main
    phase, and `_trade_lines` writes one public sentence per exchange
    straight after that action's own line. END_TURN writes nothing at all.

    Tab-separated, matching every other line: the client splits on the first
    tab for the round-number column.
    """
    lines: list[str] = []
    run: dict | None = None

    def emit(round_num: int, text: str, continuing: bool) -> None:
        # A run is exactly one line, rewritten in place as it grows, so
        # continuing one means replacing the line it already wrote.
        if continuing:
            lines.pop()
        lines.append(f"{round_num}\t{text}")

    for event in events:
        action, actor, round_num = event.action, event.actor, event.round_num
        kind = action.type
        who = _who(actor, labels)

        if kind in _BUILD_KIND:
            verb, item = _BUILD_KIND[kind]
            key = ("build", actor, round_num)
            continuing = run is not None and run["key"] == key
            if not continuing:
                run = {"key": key, "verb": verb, "items": [], "extra": []}
            run["items"].append(item)
            if kind is ActionType.SETUP_SETTLEMENT:
                gains = _hand_gains(event.before.hands[actor], event.after.hands[actor])
                if gains:
                    run["extra"].append(f"{who} collects {gains}.")
            line = f"{who} {run['verb']} {_list_with_counts(run['items'])}."
            emit(round_num, " ".join([line, *run["extra"]]), continuing)
            continue

        if kind is ActionType.DISCARD:
            # The engine takes discards one card at a time (see
            # `legal_actions` under Phase.DISCARD, which deliberately keeps
            # the action space linear in resources rather than combinatorial
            # in hand size), so one seven can cost a full hand half a dozen
            # steps in a row — and for a bot every one of them said the same
            # six words. They collapse to a single line.
            #
            # Which resources went is the discarding seat's own line only. A
            # collapsed line is exactly where a whole hand would leak at once.
            key = ("discard", actor, round_num)
            continuing = run is not None and run["key"] == key
            if not continuing:
                run = {"key": key, "counts": [0] * NUM_RESOURCES}
            run["counts"][action.a] += 1
            total = sum(run["counts"])
            if omniscient or actor == viewer:
                # Same wording as the "collects" half of a roll line, since
                # it's the same fact pointed the other way.
                line = f"{who} discarded {_resource_counts(run['counts'])}."
            else:
                line = f"{who} discarded {total} card{'' if total == 1 else 's'}."
            emit(round_num, line, continuing)
            continue

        if kind is ActionType.BANK_TRADE:
            # Ports make this the one action people repeat back to back —
            # four wood for a wheat, then four more for another — and each
            # step was its own line. Same pair in a row sums into one; a
            # different pair is a different trade and starts its own (the
            # pair is part of the run's key).
            key = ("bank", actor, round_num, action.a, action.b)
            continuing = run is not None and run["key"] == key
            if not continuing:
                run = {"key": key, "given": 0, "got": 0}
            run["given"] += event.before.hands[actor][action.a] - event.after.hands[actor][action.a]
            run["got"] += 1
            line = (
                f"{who} traded {run['given']} {RESOURCE_NAMES[action.a]} "
                f"for {run['got']} {RESOURCE_NAMES[action.b]} with the bank."
            )
            emit(round_num, line, continuing)
            continue

        run = None  # anything else ends whatever run was open

        if kind is ActionType.END_TURN:
            # Whatever line comes next (the following seat's roll, build,
            # ...) already implies the previous turn ended — a dedicated
            # "X ended the turn." line for every single turn was pure noise,
            # not information.
            continue

        lines.append(
            f"{round_num}\t{_describe(event, board, labels, viewer, omniscient=omniscient)}"
        )
        for line in _trade_lines(event, labels):
            lines.append(f"{round_num}\t{line}")

    return lines


# --- The session ---------------------------------------------------------------


@dataclass
class GameSession:
    """One in-progress game: the engine state and which seats are claimed.

    A seat is claimed or it isn't — never "a bot's" as a special case the
    session itself knows about. A table dealt for one human against three
    checkpoints and a table of four humans are the same object here,
    differing only in `claimed_seats` and who's actually driving each one
    from outside this session (a browser, an LLM over MCP, or a bot runner —
    see `botclient.py`). Whoever holds a seat submits its actions through
    `submit`, decides for itself when its own turn is over, and is asked for
    per seat: whose turn it is to act, what they may legally do, what they
    are shown, and what they may take back.

    All mutation goes through `submit`, which routes every action through
    `hexset.actions.apply` after checking it against a fresh
    `legal_actions(game)` — the one enforcement point the hard constraint
    asks for. Nothing runs a further seat's turn on another's behalf: there
    is no cascade for this session to drive, only the one action a caller
    just submitted.

    `seed` is the integer that seeded `game.rng`, and `journal` is where every
    action is written down as it happens, hidden cards and all (see
    `hexset.server.journal`). A session built without one plays exactly the same and
    keeps no account of itself, which is what the tests that only care about
    the rules want.
    """

    game: Game
    # Every seat somebody is playing, whichever kind of client it is — a
    # browser, a script on the HTTP API, an LLM over MCP, or a bot runner
    # (embedded or external, see `botclient.py`). Nothing here distinguishes
    # them: a seat submits its own actions through `submit` the same way
    # regardless of who or what is behind it, so there is no `bot: Bot` field
    # to route a turn to any more — a bot plays by calling `submit` from
    # outside this session, exactly as a human's client does.
    claimed_seats: set[int]
    seed: int = 0
    journal: Journal | None = None
    # Seat -> the model-picker display name playing it, for `state_view` to
    # echo back so the client can label seats by bot rather than by number.
    # Empty for human seats and for any caller that never set it.
    bot_names: dict[int, str] = field(default_factory=dict)
    # Seat -> the entrant spec that built the bot on it, which the display
    # name above does not always give back (an .onnx entry is named after its
    # file, not its path). Journalled so a resumed game can put the same
    # opponents back, and unused by play itself.
    bot_specs: dict[int, str] = field(default_factory=dict)
    # Seat -> whatever the person playing it registered as, for the same
    # labelling `bot_names` does for the other seats. A seat with no entry
    # here is one nobody named, which the log and the journal just say.
    player_names: dict[int, str] = field(default_factory=dict)
    # The join code of the table this game was dealt for, journalled in the
    # header so a restart can find this game again by the code people already
    # have. Nothing about play reads it, and a session dealt outside a table
    # (every test, for one) has none.
    code: str | None = None
    # Seat -> the dice total that seat rolled on its own most recent turn.
    # `game.last_roll` is one global value, whoever rolled it last; this is
    # what lets the player list show each seat's own roll instead of just
    # whoever moved most recently.
    last_roll_by_seat: dict[int, int] = field(default_factory=dict)
    # Every action applied so far, in full and unredacted — the sidebar
    # transcript is folded out of these per reader (see `render_log`) rather
    # than accumulated as text, because different seats are owed different
    # accounts of the same game.
    events: list[_Event] = field(default_factory=list)
    # How many actions have been applied, which is the step number the next
    # one is journalled under. Distinct from `len(events)` only in intent:
    # this is the journal's own numbering and follows it through an undo.
    _steps: int = field(default=0, repr=False)
    # Set the first time the game is seen to be over, so the game is filed
    # away exactly once however many more times _apply runs afterwards.
    _ended: bool = field(default=False, repr=False)
    # The one action that could still be taken back, and by whom — set only
    # right after a qualifying human action, cleared by anything else. See
    # _apply and undo_last_build.
    _undo: _UndoPoint | None = field(default=None, repr=False)
    # Seat -> whatever answers that seat's private gate (`hexset.trading`):
    # an embedded bot itself for a bot seat, a `PostedValuation` for a seat
    # a person is playing, nothing at all for an empty one. Kept here rather
    # than on `Game` directly because a seat can change hands mid-game
    # (`api.Tables.seat_bot`), and `set_trader` is the one place that
    # rewrites the engine's tuple.
    traders: dict[int, object] = field(default_factory=dict, repr=False)
    # Seats opted into confirm mode at seat-up (`POST /api/games`/`/api/join`'s
    # `confirm` flag). `publish` reads this to decide which gate a seat's own
    # vector installs -- `PendingGate` here, `PostedValuation` otherwise.
    # Never populated for a bot seat; nothing reads it for one.
    #
    # The *default*, when a request omits `confirm`, is transport-dependent
    # by design: those two routes default it to `True` for a caller that
    # says nothing -- a human's gate is the explicit submit, so nothing
    # auto-clears against one without an opt-out (`Tables.create`/`join`'s
    # docstrings, `api.py`). `hexset.server.mcp`'s `new_game`/`join` tools
    # send `confirm` explicitly on every call instead, keeping an LLM seat's
    # own default the opt-in one from PI ratification decision 3.
    confirm_seats: set[int] = field(default_factory=set)

    def set_trader(self, seat: int, trader: object | None) -> None:
        """Seat (or unseat) what answers `seat`'s side of a trade's gate."""
        if trader is None:
            self.traders.pop(seat, None)
        else:
            self.traders[seat] = trader
        self.game.gates = tuple(
            self.traders.get(s) for s in range(self.game.num_players)
        )

    def valuation_of(self, seat: int) -> tuple[float, ...]:
        """What `seat` has published, all-zero if it has published nothing."""
        return tuple(self.game.valuations[seat])

    def confirm_mode(self, seat: int) -> None:
        """`seat` opted into confirm mode at seat-up: gate it now, on the
        vector it has not published (`hexset.trading.NO_VALUATION`).

        Installing the gate here rather than waiting for a first `publish`
        is what makes "a person at the web page does not trade" a property
        of sitting down, not of what the page happens to call. The page
        offers no way to publish (owner, 2026-09-03 -- human trading
        surfaces are withheld; `docs/negotiation-interface.md`), so without
        this a human seat's gate stays `None` for the whole game and the
        rule rests entirely on the vector.

        Both halves say no, and either alone would be enough:

        *The vector.* All-zero, so every bundle this seat is party to scores
        a public surplus of exactly 0 on its side, and
        `hexset.trading._best_clearing` keeps only candidates whose surplus
        is *strictly* positive for both sides -- in either role, proposer or
        counterparty. A zero-vector seat is dropped at ranking and its gate
        is never even asked, which is the sense in which a human here is
        simply not a counterparty.

        *The gate.* `PendingGate` never clears on its own. Should this seat
        ever publish (`PUT /api/games/<code>/valuation` is untouched, for an
        LLM or an API client), what it advertises still cannot be taken
        without an explicit `POST .../trade/confirm`.

        Bots are unaffected: their own gates and vectors are seated by
        `api.Tables._spawn_local_bots`/`seat_bot`, and they go on trading
        with each other through the same engine event.
        """
        self.confirm_seats.add(seat)
        self.set_trader(seat, PendingGate(self.game, seat, NO_VALUATION))

    def publish(self, seat: int, vector: Sequence[float]) -> None:
        """`PUT /api/games/<code>/valuation`: `seat` sets its own vector.

        Takes effect at the next trade event, which is the next time the
        main phase opens (`hexset.trading`) -- publishing does not move any
        cards by itself. `Game.publish` validates and records it; the
        (checked) vector then becomes this seat's gate -- a `PostedValuation`,
        unconditionally accepting once the engine's own public-surplus test
        already says this seat wants the bundle, for most seats; a
        `PendingGate` instead for a seat that opted into confirm mode at
        seat-up (`confirm_seats`), which never clears on its own and records
        the candidate to `game.pending` for this seat to confirm or decline.
        """
        self.game.publish(seat, vector)
        vector = tuple(self.game.valuations[seat])
        if seat in self.confirm_seats:
            self.set_trader(seat, PendingGate(self.game, seat, vector))
        else:
            self.set_trader(seat, PostedValuation(vector))

    def __post_init__(self) -> None:
        # Written here rather than on the first action because the header's
        # whole point is the deal — the shuffled development deck in
        # particular, which `start` has already made and the first
        # BUY_DEV_CARD will already have taken a card off.
        if self.journal is not None:
            self.journal.start(
                self.game,
                seed=self.seed,
                # `setup_queue[0]` is `game.start`'s own `first` argument,
                # read back off the queue it built rather than threaded
                # through as a second copy — the two can never disagree.
                first=self.game.setup_queue[0],
                human_seats=sorted(self.claimed_seats),
                bot_names=self.bot_names,
                bot_specs=self.bot_specs,
                player_names=self.player_names,
                code=self.code,
            )

    @property
    def seat_labels(self) -> dict[int, str]:
        """Seat -> what to call whoever holds it, people and bots in one map.

        The log and the client both want a name per seat and neither cares
        which kind of player it belongs to, so the two sources are merged
        here rather than at each of the half-dozen call sites. A claimed
        seat with no bot label falls back to its own registered name, or
        plain "player" if it never gave one: every seat gets a label, so
        `_who` never has to invent one.
        """
        labels = dict(self.bot_names)
        for seat in self.claimed_seats:
            if seat not in labels:
                labels[seat] = self.player_names.get(seat) or "player"
        return labels

    def claim(self, seat: int, name: str | None) -> None:
        """A seat somebody just joined, after the deal — the one seat this
        session's own header (see `__post_init__`) could not have named
        because nobody had taken it yet. Journalled the same way a mid-game
        bot swap is (`Journal.seated`, with an empty `spec` — there is no
        checkpoint to name for a person), so a resumed table knows this seat
        was somebody's without needing its lost token back (see `api.py`'s
        module docstring: a token never touches disk)."""
        self.claimed_seats.add(seat)
        if name:
            self.player_names[seat] = name
        if self.journal is not None:
            self.journal.seated(seat=seat, name=name or "", spec="")

    @property
    def round(self) -> int:
        """One full lap of the table, 1-indexed — what a human watching the
        log means by "turn", distinct from `game.turns`, which counts
        per-seat and stays that way (it's a trained policy input feature; see
        hexset.encoding's TURN_SCALE). Every seat's actions within a lap
        share one round number, unlike `game.turns` where each gets its own.

        0 during setup: the placement snake isn't a lap of the table in the
        normal sense (order is 1,2,3,4,4,3,2,1, not 1,2,3,4 repeating), and
        `game.turns` doesn't move at all until end_turn() first runs, which
        `Phase.MAIN` requires — setup can't reach it.
        """
        if self.game.phase in (Phase.SETUP_SETTLEMENT, Phase.SETUP_ROAD):
            return 0
        # true state: `num_players` is a fixed, public board property.
        return self.game.turns // self.game.state(0, hidden=False).num_players + 1

    def restore(
        self,
        steps: list[tuple[int, Action, tuple[Trade, ...]]],
        journal: Journal | None = None,
    ) -> None:
        """Re-apply a journalled game's actions, bringing this session up to
        where it left off (see `hexset.server.journal.replayable`).

        Every step goes through `_apply` like any other, so the sidebar log,
        the per-seat rolls and the round numbering are rebuilt as a
        consequence of replaying rather than being stored and restored — there
        is one way this session reaches a state, and this is still it.

        `journal` is attached only once the replay is done: it is the file
        these steps were just read out of, and a session journalling as it
        restores would write the whole game into it a second time.
        """
        if self.journal is not None:
            raise ValueError("restore would rewrite the journal it is reading")
        for actor, action, trades in steps:
            if not is_legal(self.game, action, legal_actions(self.game)):
                raise ResumeError(
                    f"step {self._steps}: {action} is not legal in {self.game.phase.name}"
                )
            self._apply(actor, action, replay=trades)
        self.journal = journal
        if journal is not None:
            journal.reopened(at_step=self._steps)

    def legal_wire_actions(self, viewer: int | None) -> list[dict]:
        """What `viewer` may play right now — empty unless it is their turn,
        which is also what a seat that isn't theirs, or no seat at all, gets."""
        if is_over(self.game) or viewer is None or to_move(self.game) != viewer:
            return []
        return [action_to_wire(a) for a in legal_actions(self.game)]

    def submit(self, seat: int, wire: dict) -> None:
        """Play `wire` as `seat`. The seat is the caller's to prove (it comes
        off a player token, not off the request body — see `api.py`), and
        every other check happens here — the one enforcement point every
        client funnels through, human, LLM, or bot alike. Whoever holds
        `seat` decides for themselves when their own turn is over (`END_TURN`
        is a submitted action like any other); nothing here runs a further
        seat's turn on this call's behalf."""
        if is_over(self.game):
            raise ValueError("the game is already over")
        if seat not in self.claimed_seats:
            raise ValueError(f"seat {seat} is not yours to play")
        if to_move(self.game) != seat:
            raise ValueError("it is not your turn to act")
        action = wire_to_action(wire)
        options = legal_actions(self.game)
        if not is_legal(self.game, action, options):
            raise ValueError(f"{action} is not a legal action right now")
        self._apply(seat, action)

    def _apply(
        self, actor: int, action: Action, replay: tuple[Trade, ...] | None = None
    ) -> None:
        # Captured before apply(), not after: end_turn() increments
        # game.turns (and so self.round, derived from it), so the line for
        # the END_TURN action itself would otherwise be prefixed with the
        # *next* round's number instead of the one that just ended.
        round_num = self.round
        before = _snapshot(self.game)
        # Any claimed seat's own build/placement/trade is its own to take
        # back — bot or human, no special case: nothing here privileges one
        # kind of client over another (see `submit`).
        undoable = action.type in _UNDOABLE_BUILDS
        # Taken before apply() runs, alongside `before` above, for the same
        # reason: it has to be the instant *before* this action, and nothing
        # between here and apply() touches state/events/_steps.
        undo_point = (
            _UndoPoint(
                state=copy_state(self.game.state(0, hidden=False)),
                ledger=self.game.ledger.copy(),
                free_roads=self.game.free_roads,
                phase=self.game.phase,
                current_player=self.game.current_player,
                setup_step=self.game.setup_step,
                last_settlement=self.game.last_settlement,
                events=len(self.events),
                steps=self._steps,
                actor=actor,
            )
            if undoable
            else None
        )
        seating_before = snapshot(self.game)
        trades_before = len(self.game.trades)
        if replay is None:
            apply(self.game, action)
        else:
            # Replaying a journalled game: the seats that published the
            # vectors this game traded on are not here, so the engine's own
            # event would clear a different set (usually none). The recorded
            # exchanges are re-executed instead -- see `trading.apply_trades`.
            live, self.game.gates = self.game.gates, None
            try:
                apply(self.game, action)
            finally:
                self.game.gates = live
            apply_trades(self.game, replay)
        # See `hexset.server.seating`'s module docstring: turn order is seat
        # order, seat 0 first, and this re-points it past a retired seat.
        settle(self.game, seating_before)
        if action.type is ActionType.ROLL:
            self.last_roll_by_seat[actor] = self.game.last_roll
        self.events.append(
            _Event(
                round_num=round_num,
                actor=actor,
                action=action,
                before=before,
                after=_snapshot(self.game),
                last_roll=self.game.last_roll,
                trades=tuple(self.game.trades[trades_before:]),
            )
        )

        if self.journal is not None:
            self.journal.action(
                self.game,
                step=self._steps,
                round_num=round_num,
                actor=actor,
                action=action,
                before_hands=before.hands,
                before_held=before.held,
                trades=tuple(self.game.trades[trades_before:]),
            )
        self._steps += 1

        if is_over(self.game) and not self._ended:
            self._ended = True
            if self.journal is not None:
                self.journal.finish(self.game)

        # Every action decides this fresh: a qualifying placement that didn't
        # win the game becomes the new (and only) undo point, anything else
        # — a different seat's move, a second placement — clears whatever
        # was there. A win is excluded because the record above may already
        # be on disk by now.
        self._undo = undo_point if (undo_point is not None and not is_over(self.game)) else None

    def undo_last_build(self, seat: int) -> None:
        """Reverts `seat`'s most recent placement, bank/port trade, or Road
        Building play back to exactly how the session stood the instant
        before it: piece removed and resources refunded (including a second
        setup settlement's grant) or traded resources returned, or the card
        handed back and free_roads zeroed, longest road/largest army
        recomputed from the restored board, whose turn it is un-advanced if
        the action handed off to someone else, the event (and so its log
        line) dropped, step count wound back. Only ever available since that
        seat's own last qualifying action — see _apply.

        Somebody else's undoable action is not offered here even though the
        session only ever holds one: at a table with several people, the most
        recent take-back-able move frequently belongs to a different player,
        and reaching it would rewind the board out from under them.

        The journal is the one thing not reverted: it is append-only, so the
        undo goes into it as its own entry (see `journal.Journal.undo`).
        """
        if self._undo is None or self._undo.actor != seat:
            raise ValueError("nothing to undo")
        point = self._undo
        if point.ledger is None:  # pragma: no cover -- defensive, see the docstring
            raise ValueError("this action cannot be undone")
        self.game.set_state(point.state)
        self.game.ledger = point.ledger
        self.game.free_roads = point.free_roads
        self.game.phase = point.phase
        self.game.current_player = point.current_player
        self.game.setup_step = point.setup_step
        self.game.last_settlement = point.last_settlement
        del self.events[point.events :]
        self._steps = point.steps
        if self.journal is not None:
            self.journal.undo(self.game, back_to=point.steps)
        self._undo = None

    def _log_result(self, round_num: int) -> str:
        """The closing line, appended once the game is over.

        The board itself already says who won — the client draws it across the
        phase banner — but the log is the only part of a game that outlives it
        on disk, and a transcript that stops mid-turn without saying how it
        ended is no use for counting anything afterwards. Names the winner the
        same way every other line names a seat.
        """
        winner = self.game.won_by
        if winner is None:
            return f"{round_num}\tGame over. Nobody won."
        who = _who(winner, self.seat_labels)
        # true state: victory points include hidden VP dev cards.
        points = victory_points(self.game.state(winner, hidden=False), winner)
        return f"{round_num}\t{who} wins with {points} points."

    def _public_mover(self, viewer: int | None) -> int:
        """Whose move it is. Plain `to_move` now that no phase hands the
        decision to a seat chosen by what it holds.

        This used to filter `TRADE_RESPOND`, where `to_move` was the head of
        the engine's eligibility list and publishing it told every poller who
        held the wanted card. There is no such phase any more: trading is one
        engine event, not a decision anybody is asked for (`hexset.trading`),
        so the filter has nothing left to hide. `viewer` is kept in the
        signature because the caller is per-viewer and a future filter would
        land here.
        """
        del viewer
        game = self.game
        if is_over(game):
            return game.current_player
        return to_move(game)

    def state_view(self, viewer: int | None = None, *, omniscient: bool = False) -> dict:
        """The whole game as `viewer` is allowed to see it.

        `viewer` is a seat at this table, or `None` for someone watching
        without one. It decides three things and nothing else: whose hand and
        true victory-point count come back in full, which seat's port ratios
        the trade panel gets, and how the transcript redacts (see
        `render_log`). Everything else here is public and identical to every
        reader, which is why it is computed once regardless of who is asking.

        `omniscient` drops the first and third of those: every hand, every
        dev card, every true victory-point count, and a transcript that
        redacts nothing. **It is only ever for a reader outside the game.**
        `legal_actions` is empty for a viewer-less reader anyway, so a view
        built this way cannot be played from — but nothing here checks that,
        and handing one to a seat would put every opponent's hand in the hands
        of somebody still choosing moves. `api.Tables.handle` passes it at
        exactly one route, the token-free `GET /api/table/<code>`, and every
        seated route leaves it alone.
        """
        if omniscient and viewer is not None:
            raise ValueError("an omniscient view belongs to no seat")
        labels = self.seat_labels
        game = self.game
        # A poll of the table is one of the engine's three event-trigger
        # points (`Game.event_pending`'s docstring) -- fired here,
        # deliberately, before `trades`/`valuations` below are read, rather
        # than moments later inside `legal_wire_actions`, which would
        # otherwise hand back a snapshot whose `trades` still predates the
        # event it itself just caused to run. Observed regardless of
        # `viewer`: `trades`/`valuations` are public to the whole table
        # regardless of who asks (this function's own docstring, "computed
        # once regardless of who is asking"), so whether the pending event
        # has run cannot depend on it either -- an observer's own seat, or
        # none at all, still deserves an up-to-date table.
        #
        # Calls `run_pending_event` directly rather than `game.state
        # (game.current_player)` -- that used to be how this fired, and it
        # was the regression: it built the *current player's own hidden
        # information-set view* as a side effect of literally any viewer's
        # poll, including a spectator's or an acting bot's own runner
        # checking whose turn it is before that bot ever got to publish --
        # a cross-seat view read this function must never make (this
        # function's own docstring: what comes back depends on `viewer`,
        # never on the current player, except through public fields).
        # `run_pending_event` does exactly the same triggering work with no
        # view built and no seat's hand touched.
        #
        # Whatever it clears is attributed to the last action applied, the
        # way `hexset.record.record_game` already attributes a lazily
        # triggered first event to the previous action's step. Without this
        # the exchanges happen and are simply never told: `_apply` records
        # an action's trades from inside itself (`self.game.trades[
        # trades_before:]`), and a turn's *first* event no longer runs
        # there -- it runs here, on whichever poll reaches the table first
        # -- so every trade it cleared reached `state["trades"]` and the
        # engine's own ledger but no `_Event`, and therefore no line in the
        # transcript. Watching three bots deal with each other for sixteen
        # turns, that was five exchanges in the state and none in the log.
        traded_before = len(game.trades)
        run_pending_event(game)
        cleared = game.trades[traded_before:]
        if cleared and self.events:
            last = self.events[-1]
            last.trades = last.trades + tuple(cleared)
        # true state: the server's own omniscient observer view -- the
        # per-viewer filtering happens below, not here.
        state = game.state(0, hidden=False)
        over = is_over(game)
        players = []
        # Both are public — a route's length and a played Knight count are
        # visible on the board/in front of everyone, unlike hand contents.
        lengths = road_lengths(state)
        for p in range(state.num_players):
            reveal = over or omniscient or p == viewer
            seat_ledger = game.ledger.seats[p]
            entry = {
                "seat": p,
                "bot": self.bot_names.get(p),
                "name": labels.get(p),
                "last_roll": self.last_roll_by_seat.get(p),
                "victory_points": victory_points(state, p)
                if reveal
                else public_victory_points(state, p),
                "knights_played": state.knights_played[p],
                "road_length": lengths[p],
                "longest_road": state.longest_road_holder == p,
                "largest_army": state.largest_army_holder == p,
                "hand_size": sum(state.hands[p]),
                "dev_card_count": sum(holdings(state, p)),
                # The public-knowledge ledger (`hexset.ledger`) — public
                # for every seat, reveal or not: resource *counting* is not
                # hidden information in this game, only a steal's identity
                # and dev-card types are (see `ledger.py`'s module
                # docstring). `known` is a certified per-resource floor;
                # `unknown` is what the public log can't yet type.
                "known": dict(zip(RESOURCE_NAMES, seat_ledger.known)),
                "unknown": seat_ledger.unknown,
            }
            if reveal:
                entry["hand"] = dict(zip(RESOURCE_NAMES, state.hands[p]))
                entry["dev_cards"] = dict(zip(DEV_CARD_NAMES, holdings(state, p)))
            players.append(entry)

        return {
            "phase": game.phase.name,
            "current_player": game.current_player,
            "to_move": None if over else self._public_mover(viewer),
            "seat": viewer,
            "claimed_seats": sorted(self.claimed_seats),
            # Seats the setup snake reached while still empty and waited out
            # — permanently retired, for good, from this game (see
            # `hexset.server.seating.lock_seat`). `api.Table.join` refuses one of
            # these the same way it refuses an already-occupied seat.
            "locked": sorted(locked_of(game)),
            "winner": game.won_by,
            "game_over": over,
            # Whether POST /api/undo would succeed right now — see
            # undo_last_build. A session convenience, not a rule, so it isn't
            # in legal_actions alongside everything hexset.actions offers.
            "can_undo": self._undo is not None and self._undo.actor == viewer,
            # "round" — one lap of the table — not game.turns' per-seat count
            # (see the `round` property docstring). The only client reader
            # is the sidebar log's current-round filter, which now needs
            # this to match the log lines' own round-number tags.
            "round": self.round,
            "last_roll": game.last_roll,
            "robber": state.robber,
            "vertex_owner": state.vertex_owner,
            "vertex_building": state.vertex_building,
            "edge_owner": state.edge_owner,
            "bank": dict(zip(RESOURCE_NAMES, state.bank)),
            "dev_cards_remaining": len(state.deck),
            "discard_quota": list(game.discard_quota),
            "trade_ratios": dict(zip(RESOURCE_NAMES, trade_ratios(state, viewer)))
            if viewer is not None
            else {},
            "players": players,
            # The trade mechanic, in full and public to everyone
            # (`hexset.trading`): what every seat has advertised each
            # resource is worth to it, and what the engine cleared this
            # turn. Both are things a table hears, so neither is filtered.
            "valuations": [list(v) for v in game.valuations],
            "trades": [
                {
                    "a": t.a,
                    "b": t.b,
                    "gave": [max(0, -n) for n in t.received],
                    "got": [max(0, n) for n in t.received],
                }
                for t in game.trades
            ],
            # This turn's *pending* offers (`Game.pending`): a snapshot of
            # the last trade event against a confirm-mode seat, filtered per
            # viewer -- only the seat named `a` (the one the offer is
            # standing against) ever sees an entry, since it names a private
            # exchange nobody has agreed to yet (`docs/negotiation-interface.md`
            # §2). Empty for a spectator (`viewer is None`).
            "pending": [
                {
                    "counterparty": t.b,
                    "gave": [max(0, -n) for n in t.received],
                    "got": [max(0, n) for n in t.received],
                }
                for t in game.pending
                if t.a == viewer
            ]
            if viewer is not None
            else [],
            "legal_actions": self.legal_wire_actions(viewer),
            "log": self.log_for(viewer, omniscient=omniscient),
        }

    def log_for(self, viewer: int | None, *, omniscient: bool = False) -> list[str]:
        """The sidebar transcript as `viewer` should see it, `None` for a
        reader with no seat, who is owed the least of anyone — or, with
        `omniscient`, the most (see `render_log` and `state_view`)."""
        # true state: the board is public.
        lines = render_log(
            self.events,
            self.game.state(0, hidden=False).board,
            self.seat_labels,
            viewer,
            omniscient=omniscient,
        )
        if is_over(self.game):
            # The round the final action fell in, not self.round: a game that
            # ended on an END_TURN has already ticked over to the next one.
            lines.append(self._log_result(self.events[-1].round_num if self.events else 0))
        return lines
