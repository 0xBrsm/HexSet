"""Game session and wire protocol for the local human-vs-policy web board.

Deliberately torch-free: `hexset.server.web` imports the network bot lazily, so
this module — the board layout math, the wire-format mapping and the session
that drives a game — can be imported and tested without PyTorch, the same way
`hexset.actions` and `hexset.game` can. Anything with a
`choose(game) -> Action` method is all a session needs of its opponent:
`NetworkBot` from `hexset.clients.netbot`, `Heximax` or `RandomBot` from
`hexset.bots`.

## The wire format

An `Action` is a `NamedTuple` — a plain tuple under the hood — so it survives a
JSON round trip if its fields are given JSON-friendly types. `action_to_wire`
does that (the enum becomes its name, the tuples become lists); `wire_to_action`
undoes it. Round-tripping is exact: `wire_to_action(action_to_wire(a)) == a` for
every `a` `legal_actions` can produce, and `test_webplay.py` pins that across a
played-out game rather than a handful of hand-picked shapes.

## Never build an action the engine did not offer

`GameSession.submit` decodes the wire action and checks it against
a *fresh* list from `hexset.actions.legal_actions`, not merely against what was on offer at
some earlier poll. A UI bug, a stale page, or a tampered request all fail the same
way: the action is rejected before it reaches `hexset.actions.apply`. That is
also why every clickable thing in the frontend is one of the literal wire
objects `state_view()` already sent, echoed back unchanged — the client never
constructs an `Action` from parts, it only ever repeats one the server offered.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, NamedTuple, Sequence

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
from hexset.game import Game, Phase, is_over, may_act, players_owing_discards, to_move
from hexset.ledger import PublicLedger
from hexset.roads import road_lengths
from hexset.state import MAX_CITIES, MAX_ROADS, MAX_SETTLEMENTS, GameState, copy_state
from hexset.trading import (
    MAX_TRADE_CARDS,
    RESPONSE_COUNTER,
    RESPONSE_PASS,
    Bundle,
    Offer,
    Response,
    Trade,
    _candidates,
    apply_trades,
    default_offer,
    default_pick,
    default_respond,
    execute_agreed,
)
from hexset.victory import public_victory_points, victory_points

from .journal import Journal
from .seating import locked_of, settle, snapshot

RESOURCE_NAMES: tuple[str, ...] = tuple(r.name.title() for r in Resource)
DEV_CARD_NAMES: tuple[str, ...] = tuple(c.name.title().replace("_", " ") for c in DevCard)


class ResumeError(Exception):
    """A journalled game would not replay — its actions no longer describe a
    legal game under this engine. Recoverable, and by design: the caller deals
    a fresh game rather than failing the request (see `api.reopen_session`), so
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
class PendingGate:
    """The gate of a manual seat -- a person at the page or an LLM over MCP
    -- installed at seat-up (`GameSession.confirm_mode`). It never agrees to
    anything on its own: every answer a manual seat gives goes through the
    server (`GameSession.answer_round`, `execute_round_choice`), where the
    submission itself is the consent (`hexset.trading.execute_agreed`).

    Under the trade round (`hexset.trading`, "The trade round") `offer`
    passes -- a person composes an offer through `POST .../trade/round`,
    never automatically; `respond` records the broadcast to `game.pending`
    for the seat to answer later and passes for the round's own
    bookkeeping; `pick` declines -- a person picks through
    `POST .../trade/round/choose`. `gains_many`, the clearing house's
    surface, refuses everything; a served table never reaches it
    (`game.max_trades = 0`, `api.build_session`).
    """

    game: "Game"
    seat: int

    def gains_many(self, view, received: Sequence[Bundle], counterparties: Sequence[int]) -> list[float]:
        del view, counterparties
        return [-1.0] * len(received)

    def offer(self, view, candidates):
        del view, candidates
        return None

    def respond(self, view, offer: Offer) -> Response:
        del view
        self.game.pending.append(Trade(offer.actor, self.seat, offer.received))
        return Response(self.seat, RESPONSE_PASS)

    def pick(self, view, responses):
        del view, responses
        return None


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


# --- Wire format for the trade round (`hexset.trading`, "The trade round") ---
#
# Positional counts in `RESOURCE_NAMES` order, never named amounts: a round's
# bundles are echoed back verbatim by the client at every later step (an
# offer's `received`, a response's `bundle`), so the wire carries them exactly
# as the engine does and no client reconstructs one from named parts.


def round_bundle_from_wire(give: list, want: list) -> Bundle:
    """A signed `Bundle` for `POST .../trade/round`'s own body, positive
    towards the proposer -- `give`/`want` are each `NUM_RESOURCES`-long
    lists of nonnegative counts, on disjoint resources, `MAX_TRADE_CARDS`
    cards or fewer a side (`hexset.trading`'s own cap, checked again here
    rather than only at broadcast time, so a malformed request is refused
    before anything is asked of a gate). Raises `ValueError` naming the
    first check that fails."""
    give = _int_list(give, "give")
    want = _int_list(want, "want")
    if any(n < 0 for n in give) or any(n < 0 for n in want):
        raise ValueError("give/want must be nonnegative")
    if any(g and w for g, w in zip(give, want)):
        raise ValueError("give and want must not share a resource")
    if not 0 < sum(give) <= MAX_TRADE_CARDS:
        raise ValueError(f"give must be 1-{MAX_TRADE_CARDS} cards")
    if not 0 < sum(want) <= MAX_TRADE_CARDS:
        raise ValueError(f"want must be 1-{MAX_TRADE_CARDS} cards")
    return tuple(w - g for w, g in zip(want, give))


def signed_bundle_from_wire(values: list) -> Bundle:
    """A signed `Bundle` echoed straight off the wire -- `.../trade/round/
    answer`'s `received` and `.../trade/round/choose`'s `bundle`, both of
    which name an *exact* offer or response the client read out of a
    previous `state_view` rather than composing a fresh one (`GameSession.
    answer_round`/`execute_round_choice` match it verbatim, never a
    reconstructed approximation)."""
    return tuple(_int_list(values, "bundle"))


def _int_list(values: list, name: str) -> list[int]:
    if not isinstance(values, list) or len(values) != NUM_RESOURCES:
        raise ValueError(f"{name} must be a {NUM_RESOURCES}-length list")
    try:
        return [int(n) for n in values]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be integers") from exc


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
# server-side (unlike a Knight, which now resolves through the same forced
# robber phase a seven does the instant it is played, and is no more
# undoable than that: a seven's own robber move has never had an undo point
# either).
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

    `action` is `None` for the one kind of event that is not a board action
    at all: a manually executed trade (`GameSession._execute_round_trade`,
    `POST .../trade` or a confirmed pending offer) -- nothing about the
    phase or the turn changed, only two hands, so there is nothing for
    `_describe` to say and `render_log` skips straight to `_trade_lines`.
    """

    round_num: int
    actor: int
    action: Action | None
    before: _Snapshot
    after: _Snapshot
    last_roll: int | None
    # The exchanges the engine cleared inside this action (`hexset.trading`).
    # The trade event now runs exactly once a turn, on the way into the main
    # phase -- so only the roll or robber move that actually enters `MAIN`
    # ever carries any (`hexset.game.run_trade_event`). Empty for setup,
    # discards, every later MAIN action, and every other seat's actions --
    # except a manual trade's own event (`action is None`), which is never
    # empty: that is the entire reason it exists.
    trades: tuple[Trade, ...] = ()
    # One step of the trade round (`GameSession._note`): the offer, a seat's
    # answer, the actor declining, everyone having passed. Discrete here and
    # in the journal -- the record keeps every step -- and collapsed by
    # `render_log` into the one line a reader wants. Announced at a real
    # table, so nothing about it is redacted per reader. Only ever set on an
    # `action is None` event, never alongside `trades`.
    note: RoundNote | None = None


class RoundNote(NamedTuple):
    """One step of a trade round, as the event record and the journal keep
    it. `kind` is one of `ROUND_NOTE_KINDS`; `seat` is who did it (the
    actor for `offer`/`decline`/`nobody`, the answering seat otherwise);
    `actor` is the round's actor throughout; `bundle` is signed towards the
    actor (`hexset.trading`'s convention) for `offer`/`accept`/`counter`
    and `None` otherwise."""

    kind: str
    seat: int
    actor: int
    bundle: Bundle | None


# offer: the actor's broadcast. accept/counter/pass: one seat's answer.
# decline: the actor turned down what was on the table. nobody: every seat
# asked has passed.
ROUND_NOTE_KINDS = ("offer", "accept", "counter", "pass", "decline", "nobody")


def _round_clause(note: RoundNote, labels: dict[int, str], *, standalone: bool) -> str | None:
    """The sentence one round step adds to the round's line -- or, when the
    round's line is no longer the last one written (`standalone`), a
    sentence that names the offer it answers. `None` for a pass: a pass is
    kept in the record and shown on the page, but the transcript says
    nothing about it until everyone has passed (`nobody`) -- see
    `render_log`."""
    who = _who(note.seat, labels)
    actor = _who(note.actor, labels)
    if note.kind == "offer":
        gave = tuple(max(0, -n) for n in note.bundle or ())
        got = tuple(max(0, n) for n in note.bundle or ())
        return f"{who} offers {_bundle_text(gave)} for {_bundle_text(got)}."
    if note.kind == "accept":
        return f"{who} accepts {actor}'s offer." if standalone else f"{who} accepts."
    if note.kind == "counter" and note.bundle is not None:
        # Signed towards the actor: positive is what the actor gets, so it
        # is what the answering seat hands over.
        gives = tuple(max(0, n) for n in note.bundle)
        wants = tuple(max(0, -n) for n in note.bundle)
        head = f"{who} counters {actor}'s offer with" if standalone else f"{who} counters with"
        return f"{head} {_bundle_text(gives)} for {_bundle_text(wants)}."
    if note.kind == "decline":
        return f"{who} declines every answer." if standalone else f"{who} declines."
    if note.kind == "nobody":
        return f"Everyone declines {actor}'s offer." if standalone else "Everyone declines."
    return None


class SeatLabels(dict):
    """Seat -> label, plus the seat's number among the seats still in the game:
    a closed seat keeps no number, so the table reads Player 1, 2, 3."""

    def __init__(self, labels: dict[int, str], locked: Iterable[int] = ()) -> None:
        super().__init__(labels)
        self.locked = frozenset(locked)

    def number(self, seat: int) -> int:
        return seat + 1 - sum(1 for s in self.locked if s < seat)


def _who(seat: int, labels: dict[int, str]) -> str:
    """"Player N (label)": N is the seat's number among the seats still in the
    game when `labels` is a `SeatLabels`, else the 1-indexed seat."""
    number = labels.number(seat) if isinstance(labels, SeatLabels) else seat + 1
    return f"Player {number} ({labels.get(seat, 'bot')})"


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

    if kind is ActionType.PLAY_KNIGHT:
        return f"{who} played a Knight."

    if kind is ActionType.MOVE_ROBBER:
        victim = action.b if action.b < num_players else None
        line = f"{who} moved the robber to {_hex_label(board, action.a)}"
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
    discards_open: bool = False,
) -> list[str]:
    """Every event as the sidebar transcript `viewer` should see.

    A pure fold, run fresh per reader, because two humans at one table are
    owed two different transcripts (see `_Event`). That it is recomputed
    rather than appended to is also what makes undo trivial: dropping the
    events drops their lines, with no separate log to wind back.

    Two kinds of action arrive as a burst of engine steps that a reader
    would only ever want as one sentence, and each collapses into a run
    rewritten in place as it grows:

      builds     one "placed/built ..." per actor per round
      bank       consecutive trades of the same pair, summed

    Only ever one run is open at a time — anything that doesn't continue the
    current one clears it, so a run can never reach back across an intervening
    line to join something older.

    **Discards are held back until the whole round has resolved.** A seven's
    discards are simultaneous, not a sequence of turns (see
    `hexset.game.to_move`): every seat over the limit gives up cards at the
    same instant, and the engine only resolves them one submission at a time
    because a request is one submission. Writing each seat's line as its
    submission landed said something that never happened — that seat 0
    discarded and *then* seat 3 did — and told the table half a round while
    the other half was still choosing. So the round's discards accumulate
    here, one entry per seat however many cards and however interleaved, and
    are written out together, in seat order, the moment it is over: on the
    first event that follows it (nothing else can be played while cards are
    owed, so any later action is proof the round closed), or at the end of
    the fold if `discards_open` says no seat still owes. `discards_open` is
    the caller's live `any(game.discard_quota)` — see `GameSession.log_for`,
    which is where a mid-round read gets its answer from.

    Redaction is unchanged and is still per reader: a seat sees the cards it
    lost named, everyone else sees a count (`omniscient` names them all).

    Trades are not actions and so are not events of their own: the engine
    clears them inside the roll or the robber move that opened the main
    phase, and `_trade_lines` writes one public sentence per exchange
    straight after that action's own line. END_TURN writes nothing at all.

    Tab-separated, matching every other line: the client splits on the first
    tab for the round-number column.
    """
    lines: list[str] = []
    run: dict | None = None
    # The open discard round: seat -> what it has given up so far, and the
    # round number its lines will carry. Written to no reader until the round
    # closes (see the docstring).
    owed: dict[int, list[int]] = {}
    owed_round = 0

    def flush_discards() -> None:
        # Seat order, the order the player list is already in — the order the
        # submissions actually arrived in is exactly what a simultaneous
        # round has no business reporting.
        nonlocal owed
        for actor in sorted(owed):
            counts = owed[actor]
            who = _who(actor, labels)
            if omniscient or actor == viewer:
                # Same wording as the "collects" half of a roll line, since
                # it's the same fact pointed the other way.
                text = f"{who} discarded {_resource_counts(counts)}."
            else:
                # Which resources went is the discarding seat's own line
                # only. A collapsed line is exactly where a whole hand would
                # leak at once.
                total = sum(counts)
                text = f"{who} discarded {total} card{'' if total == 1 else 's'}."
            lines.append(f"{owed_round}\t{text}")
        owed = {}

    def emit(round_num: int, text: str, continuing: bool) -> None:
        # A run is exactly one line, rewritten in place as it grows, so
        # continuing one means replacing the line it already wrote.
        if continuing:
            lines.pop()
        lines.append(f"{round_num}\t{text}")

    for event in events:
        if owed and (event.action is None or event.action.type is not ActionType.DISCARD):
            # Nothing else can be played while a seven's cards are still
            # owed, so any other event is proof the round closed before it.
            flush_discards()
        if event.action is None:
            # No board action happened: a trade-round step
            # (`GameSession._note`) or a manually executed trade
            # (`GameSession._execute_round_trade`). The round is one line,
            # rewritten in place as it goes -- the offer, then each accept or
            # counter as it lands, then how it ended: the trade the actor
            # took, "declines." when it turned the answers down, or
            # "Everyone declines." when every seat passed. Passes are not
            # written one by one: the record and the page keep them, the
            # transcript says only the one thing a reader wants to know.
            # An answer that arrives after some other line has been written
            # (the actor kept playing while a person made up their mind)
            # stands alone and names the offer it answers.
            if event.note is not None:
                note = event.note
                key = ("round", note.actor, event.round_num)
                continuing = run is not None and run["key"] == key and note.kind != "offer"
                clause = _round_clause(note, labels, standalone=not continuing and note.kind != "offer")
                if clause is None:
                    continue
                if note.kind == "offer":
                    run = {"key": key, "text": clause}
                    emit(event.round_num, clause, False)
                elif continuing:
                    run["text"] = f"{run['text']} {clause}"
                    emit(event.round_num, run["text"], True)
                else:
                    run = None
                    lines.append(f"{event.round_num}\t{clause}")
                continue
            trade_lines = _trade_lines(event, labels)
            key = ("round", event.actor, event.round_num)
            if run is not None and run["key"] == key and len(event.trades) == 1:
                # The bundle is already on the line -- in the offer clause
                # for a taken accept, in the counter clause for a taken
                # counter -- so the close names only who it was with.
                run["text"] = f"{run['text']} Traded with {_who(event.trades[0].b, labels)}."
                emit(event.round_num, run["text"], True)
                continue
            run = None
            for line in trade_lines:
                lines.append(f"{event.round_num}\t{line}")
            continue
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
            # steps in a row — and several seats can be spending those steps
            # at once. Every one of them accumulates into the open round and
            # says nothing until `flush_discards` writes the lot.
            run = None  # a discard ends whatever build/bank run was open
            owed_round = round_num
            owed.setdefault(actor, [0] * NUM_RESOURCES)[action.a] += 1
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

    # A round that finished on the last event ever applied — every owing seat
    # cleared its quota and nobody has moved the robber yet — is over all the
    # same, and its lines are owed to the table now rather than at whatever
    # the next action turns out to be.
    if owed and not discards_open:
        flush_discards()
    return lines


def _is_split_knight_play(
    steps: list[tuple[int, Action | None, tuple[Trade, ...]]], index: int, actor: int
) -> bool:
    """Whether `steps[index]`, a `PLAY_KNIGHT` that did not end the game
    outright, is today's bare card play followed by its own `MOVE_ROBBER`
    step (`hexset.game.play_knight_card` enters `Phase.ROBBER` and nothing
    else resolves it, so that follow-up is the only thing legal right
    after) -- as opposed to one journalled before the knight/robber split
    (commit `3034778`, "play a knight, then move the robber"), whose `a`/`b`
    were the target hex and victim of a robber move this engine no longer
    folds into the same step (`GameSession._apply_knight`).

    Only ever asked once the knight itself is already known not to have won
    the game (`_apply_knight`) -- a bare play that does is a real next step
    short exactly the same way an old one always is (no robber move follows
    a win either engine's way; `play_knight_card`'s own docstring), which is
    what makes checking the *next* step, rather than guessing from operand
    values here, the exact test: an old knight that robbed hex 0 from seat 0
    wrote `a=0, b=0`, the same as today's operand-less card
    (`Action(PLAY_KNIGHT)` defaults both to 0), and either shape can name
    any other hex/victim too, so no threshold on `a`/`b` alone can tell them
    apart. Nothing in this engine's turn structure ever lands two of this
    actor's own actions back to back with no other action between them
    except this one case, so a real next step can only be that follow-up
    `MOVE_ROBBER`.
    """
    if index + 1 >= len(steps):
        return False
    next_actor, next_action, _ = steps[index + 1]
    return (
        next_actor == actor
        and next_action is not None
        and next_action.type is ActionType.MOVE_ROBBER
    )


# --- The session ---------------------------------------------------------------


@dataclass
class _OpenRound:
    """The current turn's broadcast still being negotiated (`GameSession.
    open_round`; `None` when nothing is open). `responses` holds at most one
    answer per seat, passes included (a pass's `bundle` is `None`) -- a
    seat answering again replaces its earlier one.
    `awaiting` is the manual seats that have not answered yet: while a bot
    actor's round has any, the table holds that bot's turn (`api.Table.view`
    reports `trade_wait` and a `to_move` of `None`) so a person gets to
    accept, counter or pass before the bot picks."""

    offer: Offer
    responses: list[Response] = field(default_factory=list)
    awaiting: set[int] = field(default_factory=set)


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
    # Seat -> `{"id", "kind"}` of the client that claimed it (see
    # `api.parse_client`), for the journal only -- nothing about play reads
    # this, and a seat nobody has claimed, or a bot's, has no entry.
    clients: dict[int, dict] = field(default_factory=dict)
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
    # The `(turns, current_player)` a bot actor last broadcast in, so a
    # second entry into MAIN in the same turn (a knight's robber move) does
    # not open a second round (`begin_round`).
    _broadcast_turn: tuple[int, int] | None = field(default=None, repr=False)
    # Set the first time the game is seen to be over, so the game is filed
    # away exactly once however many more times _apply runs afterwards.
    _ended: bool = field(default=False, repr=False)
    # The one action that could still be taken back, and by whom — set only
    # right after a qualifying human action, cleared by anything else. See
    # _apply and undo_last_build.
    _undo: _UndoPoint | None = field(default=None, repr=False)
    # The seat that has placed its setup road and not yet said it is done, or
    # None. A setup road is the one handoff in the game the engine makes on
    # its own: the snake moves `current_player` to the next seat as part of
    # applying the placement, where every Main-phase handoff waits for that
    # seat's own explicit END_TURN. That left setup as the one phase where a
    # seat could not take its placement back -- `_apply` drops the undo point
    # the instant another seat moves, and with peer-client bots the next seat
    # moves in milliseconds (`hexset.clients.botclient.BotRunner` acts the
    # moment `to_move` names it), so the button never survived long enough to
    # press. Holding the turn here gives setup the same shape as everywhere
    # else: nobody moves until the seat on move says it is done.
    #
    # This restores `awaiting_confirm`, added in 4f9dbe4 and lost in 82f4bd1
    # when the no-cascade rewrite removed `advance_bots()` -- the old hold was
    # written as "don't run the cascade driver", so it went out with the
    # driver. There is no driver to hold now, so the turn itself is held
    # instead, which gates every client through one rule rather than asking
    # each of them to be polite.
    awaiting_confirm: int | None = field(default=None)
    # Seat -> whatever answers that seat's private gate (`hexset.trading`):
    # an embedded bot itself for a bot seat, a `PendingGate` for every manual
    # (human or LLM) seat, nothing at all for a claimed seat this session has
    # not yet gated (a brief window at seat-up between `claim`/build_session
    # and `confirm_mode`, below). Kept here rather than on `Game` directly
    # because a seat can change hands mid-game (`api.Tables.seat_bot`), and
    # `set_trader` is the one place that rewrites the engine's tuple.
    traders: dict[int, object] = field(default_factory=dict, repr=False)
    # Every seat `confirm_mode` has gated with a `PendingGate` -- which is to
    # say, every manual seat this game has ever seated: `agents/reference/
    # trading-final.md` item 5 ("human and LLM seats are direct gates") made
    # this the *only* mode a manual seat gets, so there is no longer a flag
    # controlling whether a claimed seat lands here. Never populated for a
    # bot seat; nothing reads it for one.
    confirm_seats: set[int] = field(default_factory=set)
    # This turn's own broadcast, still being negotiated -- see `_OpenRound`.
    # `None` whenever nothing is open (before the first broadcast, after one
    # executes or is declined, or once the turn ends).
    open_round: "_OpenRound | None" = field(default=None, repr=False)

    def set_trader(self, seat: int, trader: object | None) -> None:
        """Seat (or unseat) what answers `seat`'s side of a trade's gate."""
        if trader is None:
            self.traders.pop(seat, None)
        else:
            self.traders[seat] = trader
        self.game.gates = tuple(
            self.traders.get(s) for s in range(self.game.num_players)
        )

    def confirm_mode(self, seat: int) -> None:
        """Gate `seat` with a `PendingGate` -- every manual seat, at seat-up
        (`api.Tables.create`/`Table.join`): a person or an LLM only ever
        trades through its own explicit answers on the round."""
        self.confirm_seats.add(seat)
        self.set_trader(seat, PendingGate(self.game, seat))

    def pending_for(self, seat: int) -> list[Trade]:
        """The broadcasts standing against `seat`, unanswered:
        `PendingGate.respond` records one as `Trade(actor, seat, received)`,
        signed towards the actor."""
        return [t for t in self.game.pending if t.b == seat]

    def is_manual(self, seat: int) -> bool:
        return isinstance(self.traders.get(seat), PendingGate)

    def trade_wait(self) -> list[int]:
        """The manual seats a bot actor's open round is still waiting on.
        Empty for a manual actor (a person picks whenever they like) and
        whenever no round is open."""
        round_ = self.open_round
        if round_ is None or self.is_manual(round_.offer.actor):
            return []
        return sorted(round_.awaiting)

    # -- the trade round (`hexset.trading`, "The trade round"): the session
    # drives it itself so a round survives between calls -- a person's answer
    # lands well after the broadcast that invited it.

    def begin_round(self) -> None:
        """A bot actor's broadcast at MAIN entry (`_apply`): its gate's
        `offer` over every coverable candidate, or nothing. A manual actor
        broadcasts through `open_round_for` instead.

        Once a turn. `_apply` calls this on every entry into MAIN, and a
        knight re-enters MAIN after its robber move -- which used to give a
        bot a second broadcast in the same turn. The turn is keyed by
        `(turns, current_player)`, so a new turn always gets its one."""
        game = self.game
        me = game.current_player
        if self._broadcast_turn == (game.turns, me):
            return
        self._broadcast_turn = (game.turns, me)
        self._close_round()
        if me in game.locked:
            return
        gate = self.traders.get(me)
        if gate is None or isinstance(gate, PendingGate):
            return
        # true state: the engine enumerates coverable candidates as the referee.
        candidates = list(_candidates(game.state(0, hidden=False), me, game.locked))
        if not candidates:
            return
        view = game.state(me)
        offer_fn = getattr(gate, "offer", None)
        index = offer_fn(view, candidates) if offer_fn is not None else default_offer(gate, view, candidates)
        if index is None or not (0 <= index < len(candidates)):
            return
        _them, received = candidates[index]
        self._broadcast(Offer(me, received))

    def open_round_for(self, actor: int, received: Bundle) -> None:
        """A manual actor's own broadcast (`POST .../trade/round`), already
        validated by the caller. Replaces any round it had open."""
        self._close_round()
        self._broadcast(Offer(actor, received))

    def _broadcast(self, offer: Offer) -> None:
        """Put `offer` to every other seated gate. A bot answers at once and
        its answer -- a pass included -- is recorded and logged; a manual
        seat is `awaiting` and answers later through `answer_round`. The
        offer itself is the first line the log writes about the round."""
        game = self.game
        self._note(RoundNote("offer", offer.actor, offer.actor, offer.received))
        responses: list[Response] = []
        awaiting: set[int] = set()
        for seat in range(game.num_players):
            if seat == offer.actor or seat in game.locked:
                continue
            gate = self.traders.get(seat)
            if gate is None:
                continue
            if isinstance(gate, PendingGate) and not any(game.state(0, hidden=False).hands[seat]):
                # true state: the engine is the referee for coverage. A seat
                # with no cards can neither accept nor counter, so it is not
                # asked: its pass is recorded at once rather than holding a
                # bot actor's turn on an answer that could only be no.
                response = Response(seat, RESPONSE_PASS)
                responses.append(response)
                self._note(RoundNote(response.kind, seat, offer.actor, None))
                continue
            view = game.state(seat)
            respond_fn = getattr(gate, "respond", None)
            response = respond_fn(view, offer) if respond_fn is not None else default_respond(gate, view, offer)
            if isinstance(gate, PendingGate):
                awaiting.add(seat)  # recorded to `game.pending`; answers through `answer_round`
            else:
                responses.append(response)
                self._note(RoundNote(response.kind, seat, offer.actor, response.bundle))
        self.open_round = _OpenRound(offer, responses, awaiting)
        self._note_if_everyone_passed()
        self._resolve()

    def _note(self, note: RoundNote, *, round_num: int | None = None) -> None:
        """One step of the trade round, into the event record and the
        journal. Not an action and not a trade: nothing on the board moved,
        so the event carries the same snapshot before and after, and
        `render_log` folds the step into the round's one line. Journalled
        (`Journal.note`) so a restored session's log reads the same as the
        live one did (`restore`)."""
        if round_num is None:
            round_num = self.round
        snapshot = _snapshot(self.game)
        self.events.append(
            _Event(
                round_num=round_num,
                actor=note.seat,
                action=None,
                before=snapshot,
                after=snapshot,
                last_roll=self.game.last_roll,
                note=note,
            )
        )
        if self.journal is not None:
            self.journal.note(step=self._steps, round_num=round_num, note=note)

    def _note_if_everyone_passed(self) -> None:
        """The round is answered in full and every answer was a pass: say
        so once, the moment it is known -- a person's offer to a table of
        bots reads "Everyone declines." at once, rather than after they
        close a pane of passes."""
        round_ = self.open_round
        if round_ is None or round_.awaiting or not round_.responses:
            return
        if all(r.kind == RESPONSE_PASS for r in round_.responses):
            self._note(RoundNote("nobody", round_.offer.actor, round_.offer.actor, None))

    def _resolve(self) -> Trade | None:
        """A bot actor's pick once every manual seat has answered: its gate's
        `pick` over the responses, executed through `_execute_round_trade`;
        the round closes either way. Nothing happens for a manual actor
        (who picks through `execute_round_choice`) or while `awaiting` is
        non-empty."""
        round_ = self.open_round
        if round_ is None:
            return None
        actor = round_.offer.actor
        gate = self.traders.get(actor)
        if gate is None or isinstance(gate, PendingGate) or round_.awaiting:
            return None
        trade = None
        view = self.game.state(actor)
        pick_fn = getattr(gate, "pick", None)
        chosen = pick_fn(view, round_.responses) if pick_fn is not None else default_pick(gate, view, round_.responses)
        if chosen is not None and 0 <= chosen < len(round_.responses):
            response = round_.responses[chosen]
            if response.kind != RESPONSE_PASS and response.bundle is not None:
                try:
                    trade = self._execute_round_trade(actor, response.seat, response.bundle)
                except ValueError:
                    trade = None
        self._close_round(traded=trade is not None)
        return trade

    def _close_round(self, *, traded: bool = False) -> None:
        """Drop the open round and every unanswered `pending` entry it left
        with a manual seat -- an offer nobody can act on any more must not
        keep showing on a page. A round closing with accepts or counters
        still on the table and nothing `traded` is the actor declining
        them, whichever way it closed (its own decline, its turn ending, a
        fresh broadcast replacing it), and is recorded as such."""
        round_ = self.open_round
        if round_ is not None:
            offer = round_.offer
            if not traded and any(r.kind != RESPONSE_PASS for r in round_.responses):
                self._note(RoundNote("decline", offer.actor, offer.actor, None))
            self.game.pending = [
                t for t in self.game.pending
                if not (t.a == offer.actor and t.received == offer.received)
            ]
        self.open_round = None

    def answer_round(self, seat: int, actor: int, received: Bundle, kind: str, bundle: Bundle | None) -> None:
        """A manual `seat` answers the open broadcast from `actor` -- the
        exact offer its `pending` showed (`received`, signed towards
        `actor`): `"accept"`, `"counter"` (`bundle` is the counter, signed
        towards `actor`) or `"pass"`. `ValueError` for an offer that is no
        longer open, or a counter `seat` cannot cover. A fresh answer
        replaces the seat's earlier one; then a bot actor resolves the
        round once nobody is left to wait on (`_resolve`)."""
        round_ = self.open_round
        if round_ is None or round_.offer.actor != actor or round_.offer.received != received:
            raise ValueError("that offer is no longer open")
        if kind == RESPONSE_COUNTER:
            if bundle is None:
                raise ValueError("a counter needs a bundle")
            give = [max(0, n) for n in bundle]  # what `seat` hands over: positive towards the actor
            take = [max(0, -n) for n in bundle]
            if not any(give) or not any(take) or any(g and t for g, t in zip(give, take)):
                raise ValueError("a counter gives and gets on disjoint resources")
            if sum(give) > MAX_TRADE_CARDS or sum(take) > MAX_TRADE_CARDS:
                raise ValueError(f"a trade moves at most {MAX_TRADE_CARDS} cards a side")
            # true state: the engine is the referee for coverage.
            hand = self.game.state(0, hidden=False).hands[seat]
            if any(hand[r] < n for r, n in enumerate(give)):
                raise ValueError("you cannot cover your side of that counter")
        self.game.pending = [
            t for t in self.game.pending
            if not (t.a == actor and t.b == seat and t.received == received)
        ]
        round_.responses = [r for r in round_.responses if r.seat != seat]
        response = Response(
            seat, kind, None if kind == RESPONSE_PASS else (received if kind != RESPONSE_COUNTER else bundle)
        )
        round_.responses.append(response)
        self._note(RoundNote(kind, seat, actor, response.bundle))
        round_.awaiting.discard(seat)
        self._note_if_everyone_passed()
        self._resolve()

    def execute_round_choice(self, actor: int, seat: int, bundle: Bundle) -> Trade:
        """A manual `actor`'s own pick (`POST .../trade/round/choose`):
        `seat`'s recorded response whose `bundle` matches exactly, executed
        through `_execute_round_trade`. `ValueError` for no open round or no
        such response; the round closes on success."""
        round_ = self.open_round
        if round_ is None or round_.offer.actor != actor:
            raise ValueError("there is no open round to choose from")
        if not any(r.seat == seat and r.kind != RESPONSE_PASS and r.bundle == bundle for r in round_.responses):
            raise ValueError("no such response is open")
        trade = self._execute_round_trade(actor, seat, bundle)
        self._close_round(traded=True)
        return trade

    def decline_round(self, actor: int) -> None:
        """`actor` declines every response on its open round. Nothing moves."""
        round_ = self.open_round
        if round_ is None or round_.offer.actor != actor:
            raise ValueError("there is no open round to decline")
        self._close_round()

    def _execute_round_trade(self, actor: int, counterparty: int, bundle: Bundle) -> Trade:
        """Execute an agreed exchange (`hexset.trading.execute_agreed`): a
        bot side has its gate re-asked fresh, a manual side's consent is its
        submission. Recorded the two ways an automatic clearing is -- one
        `_Event` of its own for the log (`_trade_lines`) and one journal
        line (`Journal.manual_trade`, replayed by `restore`) -- and clears
        any pending take-back, since cards moved."""
        round_num = self.round
        before = _snapshot(self.game)
        trade = execute_agreed(
            self.game, actor, counterparty, bundle,
            ask_actor=not self.is_manual(actor),
            ask_counterparty=not self.is_manual(counterparty),
        )
        self.events.append(
            _Event(
                round_num=round_num,
                actor=actor,
                action=None,
                before=before,
                after=_snapshot(self.game),
                last_roll=self.game.last_roll,
                trades=(trade,),
            )
        )
        if self.journal is not None:
            self.journal.manual_trade(self.game, step=self._steps, round_num=round_num, trade=trade)
        self._steps += 1
        self._undo = None
        return trade

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
                clients=self.clients,
                code=self.code,
            )

    @property
    def seat_labels(self) -> SeatLabels:
        """Seat -> what to call whoever holds it, people and bots in one map.

        The log and the client both want a name per seat and neither cares
        which kind of player it belongs to, so the two sources are merged
        here rather than at each of the half-dozen call sites. A claimed
        seat with no bot label falls back to its own registered name -- every
        seat gets a label, so `_who` never has to invent one.
        """
        labels = dict(self.bot_names)
        for seat in self.claimed_seats:
            if seat not in labels:
                # `api.py` resolves a name at claim time now, so this is only
                # ever hit by a session built directly (tests) with no name.
                labels[seat] = self.player_names.get(seat) or "api"
        return SeatLabels(labels, locked_of(self.game))

    def claim(self, seat: int, name: str | None, client: dict | None = None) -> None:
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
        if client is not None:
            self.clients[seat] = client
        if self.journal is not None:
            self.journal.seated(seat=seat, name=name or "", spec="", client=client)

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

        A lap is the seats *still in the game*, not the seats the board was
        dealt for: a retired seat is skipped by turn rotation (`Game.locked`,
        via `game._next_unlocked`) and so never takes one of the lap's turns.
        Dividing by the dealt `num_players` instead counted those skipped
        turns as if somebody played them, which on a table with two seats
        retired — the shape every 1v1 here has, since a 1v1 is a four-seat
        board with two seats locked — made a lap four turns long when only
        two seats were taking them, so each player showed up twice per round
        and the count came out at half the laps actually played.

        Only laps shrink, so this stays monotonic across a mid-game
        retirement: locking a seat divides by less and the round can jump
        forward, never back. `seating.unlock_seat` is the one way it could go
        the other way; nothing in the server calls it on a live game.
        """
        if self.game.phase in (Phase.SETUP_SETTLEMENT, Phase.SETUP_ROAD):
            return 0
        # true state: `num_players` is a fixed, public board property, and the
        # retired set is read through `locked_of` exactly as `SeatLabels` and
        # the wire view below read it. `max(..., 1)` is for the table nobody
        # is left at: `lock_seat` places no restriction on retiring the last
        # seat (see `Game.locked`), and a display property is the wrong place
        # to raise about it.
        playing = self.game.state(0, hidden=False).num_players - len(locked_of(self.game))
        return self.game.turns // max(playing, 1) + 1

    def restore(
        self,
        steps: list[tuple[int, Action | None, tuple[Trade, ...]]],
        journal: Journal | None = None,
        notes: dict[int, list[tuple[int, RoundNote]]] | None = None,
    ) -> None:
        """Re-apply a journalled game's actions, bringing this session up to
        where it left off (see `hexset.server.journal.replayable`).

        `notes` (`hexset.server.journal.notes_of`) are the trade-round steps
        the live session recorded, keyed by the step they preceded; each is
        put back into the record at the same place, under the round number
        it was written with, so the restored transcript reads as the live
        one did. They move nothing.

        Every step with a real `action` goes through `_apply` like any
        other, so the sidebar log, the per-seat rolls and the round
        numbering are rebuilt as a consequence of replaying rather than
        being stored and restored. A step with `action is None` is a
        manually executed trade (`Journal.manual_trade`) replaying on its
        own, not attached to any action -- there is nothing to check
        against `legal_actions` (nothing about the phase or the turn moved),
        so the recorded `Trade` is simply re-executed (`apply_trades`, not
        `Game.execute_trade` -- replaying trusts the journal outright, the
        same as an automatic clearing's own replay does) and logged the
        same way `_execute_round_trade` logs a live one.

        A `PLAY_KNIGHT` step goes through `_apply_knight` instead, which
        also handles one journalled before the knight/robber split
        (`_is_split_knight_play`): played as the two actions this engine
        now wants, but still counted as the one step the file actually
        recorded, so every `undo.back_to`/`note.step` after it in this same
        file still lands where it was written to land.

        `journal` is attached only once the replay is done: it is the file
        these steps were just read out of, and a session journalling as it
        restores would write the whole game into it a second time.
        """
        if self.journal is not None:
            raise ValueError("restore would rewrite the journal it is reading")
        notes = notes or {}
        for index, (actor, action, trades) in enumerate(steps):
            for note_round, note in notes.get(self._steps, ()):
                self._note(note, round_num=note_round)
            if action is None:
                round_num = self.round
                before = _snapshot(self.game)
                apply_trades(self.game, trades)
                self.events.append(
                    _Event(
                        round_num=round_num,
                        actor=actor,
                        action=None,
                        before=before,
                        after=_snapshot(self.game),
                        last_roll=self.game.last_roll,
                        trades=trades,
                    )
                )
                self._steps += 1
                continue
            if action.type is ActionType.PLAY_KNIGHT:
                self._apply_knight(actor, action, trades, steps, index)
                continue
            if action not in legal_actions(self.game, actor):
                raise ResumeError(
                    f"step {self._steps}: {action} is not legal in {self.game.phase.name}"
                )
            self._apply(actor, action, replay=trades)
        for note_round, note in notes.get(self._steps, ()):
            self._note(note, round_num=note_round)
        self.journal = journal
        if journal is not None:
            journal.reopened(at_step=self._steps)

    def _apply_knight(
        self,
        actor: int,
        action: Action,
        trades: tuple[Trade, ...],
        steps: list[tuple[int, Action | None, tuple[Trade, ...]]],
        index: int,
    ) -> None:
        """Replay one journalled `PLAY_KNIGHT` step -- today's bare card, or
        one written before the knight/robber split (commit `3034778`, "play
        a knight, then move the robber"), whose `action.a`/`action.b` are a
        robber move's target hex and victim that the file folded into the
        same action this engine now resolves as a separate `MOVE_ROBBER`.

        Always plays the bare card first (`hexset.game.play_knight_card`
        ignores whatever operands the action carries, so this is safe
        either way) and stops there if that already ended the game -- a
        winning knight never moves the robber, its own docstring's rule.
        The pre-split engine moved the robber unconditionally, ahead of its
        own win check, so an old file recording a winning knight really did
        see one move; replayed today, it does not. Left that way rather than
        chased: the position is already decided the instant the game ends,
        so nothing past this step ever reads the difference.

        Otherwise checks `_is_split_knight_play`: today's shape already has
        its own `MOVE_ROBBER` as the very next step, which this leaves for
        the ordinary path in `restore`'s own loop to apply and log; an old
        one does not; synthesized here from `action.a`/`action.b` and
        applied the same way instead, with the recorded step's own `trades`
        -- attached to this `MOVE_ROBBER`, not the bare card, which is when
        this engine would actually have cleared them.

        Plays both through the ordinary `_apply` each still is on its own --
        so `settle`, the sidebar log and `begin_round`'s own MAIN-entry rule
        all run exactly as they would live -- and then folds `self._steps`
        back down by one whenever it does both: two calls to `_apply` each
        count a step, but the file only ever recorded this as one, and every
        `undo.back_to`/`note.step` written after it in the same file was
        counted that way too.
        """
        bare = Action(ActionType.PLAY_KNIGHT)
        if bare not in legal_actions(self.game, actor):
            raise ResumeError(
                f"step {self._steps}: {action} is not legal in {self.game.phase.name}"
            )
        self._apply(actor, bare, replay=())
        if is_over(self.game) or _is_split_knight_play(steps, index, actor):
            return
        move = Action(ActionType.MOVE_ROBBER, action.a, action.b)
        if move not in legal_actions(self.game, actor):
            raise ResumeError(
                f"step {self._steps}: {move} is not legal in {self.game.phase.name}"
            )
        self._apply(actor, move, replay=trades)
        self._steps -= 1

    def legal_wire_actions(self, viewer: int | None) -> list[dict]:
        """What `viewer` may play right now — empty unless it is their turn,
        which is also what a seat that isn't theirs, or no seat at all, gets.

        "Their turn" is `hexset.game.may_act`, not `to_move`: a seven's
        discards are owed by several seats at once and none of them is
        anybody's turn, so each owing seat is offered its own cards for as
        long as it still owes some, whatever the others have done."""
        if is_over(self.game) or viewer is None:
            return []
        # A seat holding its setup turn open has exactly one move: end it.
        # Offered as an ordinary END_TURN rather than a route or a control of
        # its own, because every client already knows what that means -- the
        # board's End Turn button reads this list, and so does an LLM seat
        # through MCP. `may_act` says no here (the engine has already moved
        # the snake on), which is the whole reason it needs saying.
        if self.awaiting_confirm == viewer:
            return [action_to_wire(Action(ActionType.END_TURN))]
        if not may_act(self.game, viewer):
            return []
        return [action_to_wire(a) for a in legal_actions(self.game, viewer)]

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
        # `may_act`, not `to_move`: a seven's discards are simultaneous, so
        # seat 3 submitting one while seat 0 still owes its own is not out of
        # turn — there is no turn — and refusing it was this gate's one real
        # bug (see `hexset.game.to_move`).
        # The held seat's own END_TURN, which is what releases the table --
        # see `legal_wire_actions`. Checked before `may_act`, which refuses it
        # (the engine moved the snake on when the road was placed), and before
        # the legality check, since the engine offers nothing in that phase.
        if self.awaiting_confirm == seat:
            if wire_to_action(wire).type is not ActionType.END_TURN:
                raise ValueError("end your setup turn first")
            self.awaiting_confirm = None
            return
        # Somebody else is still finishing their setup turn. Refused here
        # rather than left to each client to respect, so nobody can take a
        # seat's placement away by being quicker than its button: this is the
        # one enforcement point every client funnels through, and the hold is
        # worth exactly as much as its least polite reader.
        if self.awaiting_confirm is not None:
            raise ValueError(
                f"seat {self.awaiting_confirm} has not finished its setup turn"
            )
        if not may_act(self.game, seat):
            raise ValueError("it is not your turn to act")
        action = wire_to_action(wire)
        options = legal_actions(self.game, seat)
        if action not in options:
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
        # `seat=actor` matters for exactly one action type, DISCARD, whose
        # actor the position does not fix (see `hexset.actions.apply`): the
        # seat that submitted it is the seat that loses the card, even when a
        # lower-numbered seat is still owing. It is also what makes a
        # journalled game replay the discards back to the seats that actually
        # made them, since the journal records the actor per step.
        if replay is None:
            apply(self.game, action, seat=actor)
        else:
            # Replaying a journalled game: the seats that published the
            # vectors this game traded on are not here, so the engine's own
            # event would clear a different set (usually none). The recorded
            # exchanges are re-executed instead -- see `trading.apply_trades`.
            live, self.game.gates = self.game.gates, None
            try:
                apply(self.game, action, seat=actor)
            finally:
                self.game.gates = live
            apply_trades(self.game, replay)
        # See `hexset.server.seating`'s module docstring: turn order is seat
        # order, seat 0 first, and this re-points it past a retired seat.
        settle(self.game, seating_before)
        if action.type is ActionType.ROLL:
            self.last_roll_by_seat[actor] = self.game.last_roll
        if action.type is ActionType.END_TURN:
            self._close_round()  # a round is a per-turn thing
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

        # Recomputed on every action, so the ordinary case clears it without
        # anything having to remember to: only a setup road that actually
        # handed the snake on leaves a turn owing a confirm. `to_move` still
        # naming `actor` is the turn of the snake, where a seat places twice
        # in a row -- it never gave the turn away, so it has nothing to end.
        self.awaiting_confirm = (
            actor
            if (
                action.type is ActionType.SETUP_ROAD
                and not is_over(self.game)
                and to_move(self.game) != actor
                # A browser seat, and only a browser seat. Undo is a
                # misclick affordance -- it exists because a person can put a
                # settlement somewhere they did not mean to -- so the hold
                # that protects it is scoped to the client that has the
                # button. Everything else plays setup exactly as before:
                #
                # - a bot decides from `onnx_record`'s `action_mask`, built
                #   from the engine's own action space. An END_TURN only this
                #   session knows about is not in it, so a held bot seat
                #   would have no legal move at all -- and widening the mask
                #   would be a new action every checkpoint has to be trained
                #   on, for a button no bot will ever press.
                # - an LLM seat (`kind` "mcp"/"api") is manual, and so is in
                #   `confirm_seats`, but drives itself through a turn without
                #   a person watching; holding it just stalls the table until
                #   it thinks to end a turn the engine already moved past.
                and self.clients.get(actor, {}).get("kind") == "web"
            )
            else None
        )

        if (
            replay is None
            and action.type in (ActionType.ROLL, ActionType.MOVE_ROBBER)
            and self.game.phase is Phase.MAIN
        ):
            # MAIN entry -- where the clearing house used to fire for a served
            # table (`game.max_trades = 0` switches it off, `api.build_session`)
            # -- opens this turn's round for a bot actor. After this action's own
            # event and journal line, so a trade the round executes at once is
            # recorded once, as its own step (`_execute_round_trade`). Never on
            # a journal replay, which asks no gate.
            self.begin_round()

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
        # Undoing a setup road always lands back before the handoff it was
        # holding, so there is no longer a turn to end -- and leaving it set
        # would wedge the table behind a confirm for a road that no longer
        # exists. The seat places again and owes a fresh confirm then.
        self.awaiting_confirm = None

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

        During `Phase.DISCARD` this is one owing seat out of possibly
        several, and it is a label rather than a permission: what a client
        may actually do comes off `legal_actions` (see
        `legal_wire_actions`/`may_act`), which offers every owing seat its
        own cards at once. `discard_quota` in the same view already says
        exactly which seats the table is waiting on.
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

        A finished game (`over`) gets the same two drops as `omniscient`,
        seat or no seat: nobody is still choosing moves once `game_over` is
        true, so the concern above no longer applies, and every seat's own
        route reveals exactly what the spectator link already showed for
        this game. The hand/dev-card/victory-point half of that has always
        followed `over` (see `reveal` below); the transcript now does too.
        """
        if omniscient and viewer is not None:
            raise ValueError("an omniscient view belongs to no seat")
        labels = self.seat_labels
        game = self.game
        # A gate is a pure function of the position, asked fresh once a
        # turn, and that one event fires eagerly, inside whichever action
        # transitions the turn into `MAIN` (`enter_main` for a non-seven
        # roll, `move_robber_to` once the robber phase resumes) -- so
        # `_apply`'s own `self.game.trades[trades_before:]` bookkeeping
        # already attributes every trade to the `_Event` it belongs to, and
        # a poll of the table has nothing left to trigger here.
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
            # Seats somebody closed outright rather than sit empty forever
            # (`api.Tables.close_seat`, `hexset.game.lock_seat`) — permanently
            # retired, for good, from this game. `api.Table.join` refuses one
            # of these the same way it refuses an already-occupied seat.
            "locked": sorted(locked_of(game)),
            # Whether the first move has been made. Until then the table is
            # still being set: an empty seat can be closed and a closed one
            # reopened or given a bot (`api.Tables.close_seat`/`open_seat`/
            # `seat_bot`); from the first move on the seats are fixed, so
            # the numbering the log uses never shifts mid-game.
            "started": self._steps > 0,
            "winner": game.won_by,
            "game_over": over,
            # Whether POST /api/undo would succeed right now — see
            # undo_last_build. A session convenience, not a rule, so it isn't
            # in legal_actions alongside everything hexset.actions offers.
            "can_undo": self._undo is not None and self._undo.actor == viewer,
            # The seat whose setup turn is placed but not yet ended, or None.
            # Every seat sees which one it is, not just a bool for itself:
            # a spectator's banner and the other seats' "waiting for..." read
            # the same field the holder's own End Turn button does, and
            # `BotRunner` needs to know whether the seat owing it is its own.
            "awaiting_confirm": self.awaiting_confirm,
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
            # What the engine cleared this turn (`hexset.trading`): public
            # to everyone, so it is not filtered.
            "trades": [
                {
                    "a": t.a,
                    "b": t.b,
                    "gave": [max(0, -n) for n in t.received],
                    "got": [max(0, n) for n in t.received],
                }
                for t in game.trades
            ],
            # The broadcasts standing against `viewer` unanswered (`pending_for`):
            # `bundle` is the offer's own `received`, signed towards `actor`,
            # echoed back verbatim by `POST .../trade/round/answer`.
            "pending": [
                {"actor": t.a, "bundle": list(t.received)}
                for t in self.pending_for(viewer)
            ]
            if viewer is not None
            else [],
            # `viewer`'s own open round, only for the seat that broadcast it:
            # the offer, every answer so far -- accept/counter (`bundle`
            # signed towards the actor, echoed back by
            # `.../trade/round/choose`) and pass (`bundle` null), so a seat
            # that turned the offer down is told apart from one still to
            # answer -- and the manual seats still to answer.
            #
            # `trade_round`, not `round`: this dict already carries a `round`
            # — the lap number the log lines are tagged with — and two keys of
            # one name in one literal silently keep the second. That is what
            # happened: the lap number never reached a client, so the sidebar
            # log's current-round filter matched nothing and the log read
            # empty from the day the trade round landed.
            "trade_round": (
                {
                    "offer": {"actor": self.open_round.offer.actor, "bundle": list(self.open_round.offer.received)},
                    "responses": [
                        {"seat": r.seat, "kind": r.kind, "bundle": None if r.bundle is None else list(r.bundle)}
                        for r in self.open_round.responses
                    ],
                    "awaiting": sorted(self.open_round.awaiting),
                }
                if self.open_round is not None and self.open_round.offer.actor == viewer
                else None
            ),
            "legal_actions": self.legal_wire_actions(viewer),
            "log": self.log_for(viewer, omniscient=omniscient or over),
        }

    def log_for(self, viewer: int | None, *, omniscient: bool = False) -> list[str]:
        """The sidebar transcript as `viewer` should see it, `None` for a
        reader with no seat, who is owed the least of anyone — or, with
        `omniscient`, the most (see `render_log` and `state_view`).

        A discard round still in progress is told to nobody, spectator
        included: `discards_open` carries the engine's own "somebody still
        owes cards" to the fold, which holds that round's lines back until
        it closes (see `render_log`). It is the live quota rather than
        anything recorded per event because a round can also close without
        an action — `hexset.game.lock_seat` zeroes a retired seat's quota —
        and the reveal follows the round, not the last submission."""
        # true state: the board is public.
        lines = render_log(
            self.events,
            self.game.state(0, hidden=False).board,
            self.seat_labels,
            viewer,
            omniscient=omniscient,
            discards_open=bool(players_owing_discards(self.game)),
        )
        if is_over(self.game):
            # The round the final action fell in, not self.round: a game that
            # ended on an END_TURN has already ticked over to the next one.
            lines.append(self._log_result(self.events[-1].round_num if self.events else 0))
        return lines
