"""Game session and wire protocol for the local human-vs-policy web board.

Deliberately torch-free: `catan.webserver` imports the network bot lazily, so
this module — the board layout math, the wire-format mapping and the session
that drives a game — can be imported and tested without PyTorch, the same way
`catan.actions` and `catan.game` can. `catan.bots.Bot` (anything with a
`choose(game) -> Action` method) is all a session needs of its opponent; a
`NetworkBot` from `catan.netbot` satisfies that, and so does `catan.bots.RandomBot`,
which is what the tests use.

## The wire format

An `Action` is a `NamedTuple` — a plain tuple under the hood — so it survives a
JSON round trip if its fields are given JSON-friendly types. `action_to_wire`
does that (the enum becomes its name, the tuples become lists); `wire_to_action`
undoes it. Round-tripping is exact: `wire_to_action(action_to_wire(a)) == a` for
every `a` `legal_actions` can produce, and `test_webplay.py` pins that across a
played-out game rather than a handful of hand-picked shapes.

## Never build an action the engine did not offer

`GameSession.apply_human_action` decodes the wire action and checks it against
a *fresh* call to `legal_actions`, not merely against what was on offer at some
earlier poll. A UI bug, a stale page, or a tampered request all fail the same
way: the action is rejected before it reaches `catan.actions.apply`. That is
also why every clickable thing in the frontend is one of the literal wire
objects `state_view()` already sent, echoed back unchanged — the client never
constructs an `Action` from parts, it only ever repeats one the server offered.
"""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass, field
from typing import Protocol

from .actions import (
    ONE_RESOURCE,
    YEAR_OF_PLENTY_PAIRS,
    Action,
    ActionType,
    apply,
    is_legal,
    legal_actions,
)
from .board.board import Board
from .board.coords import Hex
from .board.terrain import NUM_RESOURCES, TERRAIN_RESOURCE, Resource
from .board.topology import Topology
from .cards import NUM_DEV_CARDS, DevCard
from .devcards import holdings
from .economy import trade_ratios
from .game import MAX_OFFERS_PER_TURN, Game, Phase, is_over, to_move
from .record import Record, append_step, board_fields, write as write_records
from .roads import road_lengths
from .state import MAX_CITIES, MAX_ROADS, MAX_SETTLEMENTS, GameState, copy_state
from .victory import public_victory_points, victory_points

RESOURCE_NAMES: tuple[str, ...] = tuple(r.name.title() for r in Resource)
DEV_CARD_NAMES: tuple[str, ...] = tuple(c.name.title().replace("_", " ") for c in DevCard)

# A cascade of bot moves between two human decisions is bounded so a runaway
# bot (or an engine bug that never hands the turn back) surfaces as an error
# rather than a request that never returns. Arena duels cap a whole game's
# actions at 20000 (`catan.arena.MAX_ACTIONS`); one cascade is at most a few
# players' worth of a turn, so a far smaller number is already generous.
MAX_CASCADE_STEPS = 2000


class Bot(Protocol):
    def choose(self, game: Game) -> Action: ...


# --- Hex-to-pixel layout -----------------------------------------------------
#
# Pointy-top orientation (textbook redblobgames algebra): a hex's six corners
# sit at angles 60*i - 30 degrees around its center, and `Topology.hex_vertices`
# already lists a hex's vertices in that same i = 0..5 order (corner i is shared
# with the neighbours in directions i and i+1, and the direction vectors in
# `catan.board.coords` place direction i at angle 60*(i-1) under this same
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
        # The engine's own supply caps (`catan.state`), so the frontend's
        # remaining-piece HUD can't drift from what `can_place_*` actually
        # enforces.
        "piece_supply": {
            "road": MAX_ROADS,
            "settlement": MAX_SETTLEMENTS,
            "city": MAX_CITIES,
        },
    }


# --- Wire format for actions --------------------------------------------------


def action_to_wire(action: Action) -> dict:
    return {
        "type": action.type.name,
        "a": action.a,
        "b": action.b,
        "give": list(action.give),
        "want": list(action.want),
        "ask": list(action.ask),
    }


def wire_to_action(data: dict) -> Action:
    try:
        kind = ActionType[str(data["type"])]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"unknown action type {data.get('type')!r}") from exc
    try:
        return Action(
            type=kind,
            a=int(data.get("a", 0)),
            b=int(data.get("b", 0)),
            give=tuple(int(x) for x in data.get("give", ())),
            want=tuple(int(x) for x in data.get("want", ())),
            ask=tuple(int(x) for x in data.get("ask", ())),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"malformed action payload: {data!r}") from exc


def _proposable_options(game: Game) -> list[Action]:
    """Every one-for-one (give, want) pair the current player could open a
    trade proposal for — from public information alone: their own hand and
    the turn's offer count.

    Deliberately not `catan.actions.legal_actions`'s own `PROPOSE_TRADE`
    sample, which also skips any pair no opponent could currently cover.
    That's a fair thing for a bot to lean on when picking a search target —
    the engine already sees every hand, it's one shared `GameState` — but
    Catan hands are private at a real table, and this list is what tells a
    *human* what they may propose. Reflecting that omniscient filter here,
    whether by omission or by an "isn't available" message, would hand them
    the one thing the actual board never does: proof of what's in a
    specific opponent's hand. `propose_trade` doesn't require coverage
    either — a proposal nobody can cover is still legal, it just gets no
    takers (see its own `if not willing: return`) — so this is the accurate
    rule for what a human may attempt, not a laxer one.
    """
    state = game.state
    player = game.current_player
    if game.offers_made >= MAX_OFFERS_PER_TURN:
        return []
    hand = state.hands[player]
    return [
        Action(ActionType.PROPOSE_TRADE, give=ONE_RESOURCE[given], want=ONE_RESOURCE[wanted])
        for given in range(NUM_RESOURCES)
        if hand[given]
        for wanted in range(NUM_RESOURCES)
        if wanted != given
    ]


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
# per actor instead of a line each — see GameSession._log_action.
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


# The five placement actions the human can take back — never a bot's (an
# opponent's misclick isn't the human's to undo). Road Building's free roads
# need no special case: they still arrive as ordinary BUILD_ROAD actions (see
# game.build_road), so restoring game.free_roads alongside the board covers
# them too.
_UNDOABLE_BUILDS: frozenset[ActionType] = frozenset(
    {
        ActionType.SETUP_SETTLEMENT,
        ActionType.SETUP_ROAD,
        ActionType.BUILD_ROAD,
        ActionType.BUILD_SETTLEMENT,
        ActionType.BUILD_CITY,
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
    free_roads: int
    phase: Phase
    current_player: int
    setup_step: int
    last_settlement: int
    log: list[str]
    run: dict | None
    actions_count: int


def _snapshot(game: Game) -> _Snapshot:
    state = game.state
    return _Snapshot(
        hands=[hand[:] for hand in state.hands],
        held=[holdings(state, p)[:] for p in range(state.num_players)],
    )


def _who(seat: int, human_seat: int, bot_names: dict[int, str]) -> str:
    """"Player N (label)" — the label is "human" or the seat's model name,
    the same one `state_view` already puts on that seat (see
    `GameSession.bot_names`). One form for every seat, actor or bystander,
    rather than a special case for the human that the rest of the log has to
    match. Lowercase to match the bot names (search2, linear805, ...).
    Player numbers are 1-indexed for the log even though `seat` itself
    stays 0-indexed everywhere else — "Player 0" reads as a bug to anyone
    who isn't a programmer.
    """
    label = "human" if seat == human_seat else bot_names.get(seat, "bot")
    return f"Player {seat + 1} ({label})"


def _describe(
    game: Game,
    actor: int,
    action: Action,
    before: _Snapshot,
    human_seat: int,
    bot_names: dict[int, str],
) -> str:
    state = game.state
    kind = action.type
    who = _who(actor, human_seat, bot_names)

    if kind is ActionType.ROLL:
        line = f"{who} rolled {game.last_roll}."
        gains = [
            f"{_who(p, human_seat, bot_names)} collects {g}."
            for p in range(state.num_players)
            if (g := _hand_gains(before.hands[p], state.hands[p])) is not None
        ]
        return " ".join([line, *gains])

    if kind is ActionType.SETUP_SETTLEMENT:
        line = f"{who} placed a settlement."
        gains = _hand_gains(before.hands[actor], state.hands[actor])
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
        gained = [
            c for c in range(NUM_DEV_CARDS) if before.held[actor][c] < holdings(state, actor)[c]
        ]
        if actor == human_seat and gained:
            return f"{who} bought a {DEV_CARD_NAMES[gained[0]]}."
        return f"{who} bought a development card."

    if kind is ActionType.PLAY_ROAD_BUILDING:
        return f"{who} played Road Building."

    if kind in (ActionType.MOVE_ROBBER, ActionType.PLAY_KNIGHT):
        prefix = f"{who} played a Knight and " if kind is ActionType.PLAY_KNIGHT else f"{who} "
        victim = action.b if action.b < state.num_players else None
        line = f"{prefix}moved the robber to {_hex_label(state.board, action.a)}"
        if victim is None:
            return line + "."
        stolen = _hand_gains(before.hands[victim], state.hands[victim])
        if human_seat in (actor, victim) and stolen:
            resource = next(
                RESOURCE_NAMES[r]
                for r in range(NUM_RESOURCES)
                if state.hands[victim][r] < before.hands[victim][r]
            )
            return line + f" and stole {resource} from {_who(victim, human_seat, bot_names)}."
        return line + f" and stole a card from {_who(victim, human_seat, bot_names)}."

    if kind is ActionType.PLAY_MONOPOLY:
        resource = RESOURCE_NAMES[action.a]
        swept = state.hands[actor][action.a] - before.hands[actor][action.a]
        return f"{who} played Monopoly on {resource} and collected {swept} card(s)."

    if kind is ActionType.PLAY_YEAR_OF_PLENTY:
        r0, r1 = YEAR_OF_PLENTY_PAIRS[action.a]
        return f"{who} played Year of Plenty for {RESOURCE_NAMES[r0]} and {RESOURCE_NAMES[r1]}."

    if kind is ActionType.PROPOSE_TRADE:
        return f"{who} offered {_bundle_text(action.give)} for {_bundle_text(action.want)}."

    return f"{who} played {kind.name}."


# --- The session ---------------------------------------------------------------


@dataclass
class GameSession:
    """One in-progress game: the engine state, the human's seat and the bot
    playing every other seat.

    All mutation goes through `apply_human_action` and `advance_bots`, and both
    route every action through `catan.actions.apply` after checking it against
    a fresh `legal_actions(game)` — the one enforcement point the hard
    constraint asks for.

    `seed` is the integer that seeded `game.rng`, and `record_path` is where to
    append a `catan.record.Record` once the game ends — the format the project
    already has for a played-out game, rather than a second one invented here.
    Recording is opt-in (`record_path` defaults to `None`) so nothing is written
    unless a caller asks for it.
    """

    game: Game
    human_seat: int
    bot: Bot
    seed: int = 0
    record_path: str | None = None
    # Seat -> the model-picker display name playing it, for `state_view` to
    # echo back so the client can label seats by bot rather than by number.
    # Empty for the human seat and for any caller that never set it.
    bot_names: dict[int, str] = field(default_factory=dict)
    # Seat -> the dice total that seat rolled on its own most recent turn.
    # `game.last_roll` is one global value, whoever rolled it last; this is
    # what lets the player list show each seat's own roll instead of just
    # whoever moved most recently.
    last_roll_by_seat: dict[int, int] = field(default_factory=dict)
    log: list[str] = field(default_factory=list)
    _actions: list[tuple[int, int, int]] = field(default_factory=list, repr=False)
    _offers: list[tuple[int, tuple[int, ...], tuple[int, ...], tuple[int, ...]]] = field(
        default_factory=list, repr=False
    )
    _recorded: bool = field(default=False, repr=False)
    # A run of build actions by the same actor in the same round, folded
    # into one log line instead of one each — see _log_action. Cleared by
    # any other action from any seat.
    _run: dict | None = field(default=None, repr=False)
    # A trade's proposal line, held back from `log` until the offer
    # concludes — see _log_action. `None` whenever no trade is in flight.
    # Deliberately just the proposal, not a running list of responses:
    # who was actually asked (and how many) is exactly the hidden-hand
    # information a human must not be able to read off the log.
    _trade_buffer: str | None = field(default=None, repr=False)
    # Set only right after a human BUILD_ROAD/BUILD_SETTLEMENT/BUILD_CITY,
    # cleared by literally anything else — see _apply and undo_last_build.
    _undo: _UndoPoint | None = field(default=None, repr=False)
    # True only in the instant right after the human's own setup road handed
    # the turn to someone else — the one handoff in the game with no
    # explicit "I'm done" the way END_TURN is everywhere else. Public (no
    # underscore) since the webserver reads it directly to decide whether to
    # run advance_bots() itself or wait for POST /api/confirm — see
    # apply_human_action/confirm_setup_turn.
    awaiting_confirm: bool = field(default=False)

    @property
    def round(self) -> int:
        """One full lap of the table, 1-indexed — what a human watching the
        log means by "turn", distinct from `game.turns`, which counts
        per-seat and stays that way (it's a trained policy input feature and
        a replay-verified Record field; see catan.encoding's TURN_SCALE and
        catan.record). Every seat's actions within a lap share one round
        number, unlike `game.turns` where each gets its own.

        0 during setup: the placement snake isn't a lap of the table in the
        normal sense (order is 1,2,3,4,4,3,2,1, not 1,2,3,4 repeating), and
        `game.turns` doesn't move at all until end_turn() first runs, which
        `Phase.MAIN` requires — setup can't reach it.
        """
        if self.game.phase in (Phase.SETUP_SETTLEMENT, Phase.SETUP_ROAD):
            return 0
        return self.game.turns // self.game.state.num_players + 1

    def legal_wire_actions(self) -> list[dict]:
        if is_over(self.game) or to_move(self.game) != self.human_seat:
            return []
        options = [a for a in legal_actions(self.game) if a.type is not ActionType.PROPOSE_TRADE]
        options += _proposable_options(self.game)
        return [action_to_wire(a) for a in options]

    def apply_human_action(self, wire: dict) -> None:
        if is_over(self.game):
            raise ValueError("the game is already over")
        if to_move(self.game) != self.human_seat:
            raise ValueError("it is not your turn to act")
        action = wire_to_action(wire)
        options = legal_actions(self.game)
        if not is_legal(self.game, action, options):
            raise ValueError(f"{action} is not a legal action right now")
        self._apply(self.human_seat, action)
        # place_initial_road hands current_player straight to the next seat
        # in the snake with no separate step of its own — unlike Main phase,
        # where that handoff only ever happens on the human's own explicit
        # END_TURN. The caller (webserver._handle_action) reads this to
        # decide whether to run advance_bots() itself or leave it for
        # POST /api/confirm, so the human gets the same one-button pause
        # before it as everywhere else, not a same-request cascade they
        # never got to see, let alone react to.
        self.awaiting_confirm = (
            action.type is ActionType.SETUP_ROAD
            and not is_over(self.game)
            and to_move(self.game) != self.human_seat
        )

    def confirm_setup_turn(self) -> None:
        """Lets the cascade `apply_human_action` held back actually run.
        A harmless no-op, not an error, if nothing is actually pending —
        there's nothing to protect by rejecting a stale or doubled-up
        confirm the way undo_last_build rejects a stale undo."""
        self.awaiting_confirm = False
        self.advance_bots()

    def _default_ask_order(self, proposer: int) -> tuple[int, ...]:
        """Who to ask first when a proposer didn't say: lowest victory
        points, tied broken by fewest development cards, then random —
        trading with whoever's behind rather than the engine's own neutral
        default (`ask=()`, clockwise seat order — see
        `catan.game.propose_trade`'s docstring). Applied to every seat's
        proposals in `_apply`, not just the human's: no bot sets `ask`
        itself today (a training-side gap, not something to paper over
        here), so leaving this human-only would have meant every bot kept
        the engine's neutral default while only the human got the better
        one.

        Victory points are `public_victory_points`, not the true count: a
        hidden victory-point development card is exactly the information the
        real board wouldn't hand the proposer either, so this shouldn't see
        it. Dev card *count* has no such issue — that stack is visible
        whatever's in it.
        """
        state = self.game.state
        others = [p for p in range(state.num_players) if p != proposer]
        random.Random().shuffle(others)  # only breaks byte-for-byte ties below
        others.sort(key=lambda p: (public_victory_points(state, p), sum(holdings(state, p))))
        return tuple(others)

    def advance_bots(self) -> None:
        steps = 0
        while not is_over(self.game) and to_move(self.game) != self.human_seat:
            if steps >= MAX_CASCADE_STEPS:
                raise RuntimeError(
                    f"bot cascade did not yield the turn within {MAX_CASCADE_STEPS} actions"
                )
            seat = to_move(self.game)
            options = legal_actions(self.game)
            action = self.bot.choose(self.game)
            if not is_legal(self.game, action, options):
                raise RuntimeError(f"bot chose an action legal_actions did not offer: {action}")
            self._apply(seat, action)
            steps += 1

    def _apply(self, actor: int, action: Action) -> None:
        # Every seat's proposal gets the same treatment — human or bot,
        # whichever `actor` is — since this is the one choke point both
        # apply_human_action and advance_bots funnel every action through.
        if action.type is ActionType.PROPOSE_TRADE and not action.ask:
            action = action._replace(ask=self._default_ask_order(actor))
        # Captured before apply(), not after: end_turn() increments
        # game.turns (and so self.round, derived from it), so the line for
        # the END_TURN action itself would otherwise be prefixed with the
        # *next* round's number instead of the one that just ended.
        round_num = self.round
        before = _snapshot(self.game)
        undoable = actor == self.human_seat and action.type in _UNDOABLE_BUILDS
        # Taken before apply() runs, alongside `before` above, for the same
        # reason: it has to be the instant *before* this action, and nothing
        # between here and apply() touches state/log/_run/_actions.
        # self._run is whatever run was open *before* this action — a build
        # run from earlier this same round, a bank-trade or discard run left
        # over from something else entirely, or None — not necessarily a
        # build-shaped dict, since _run_for only replaces it once _log_action
        # runs for *this* action, further down. copy.deepcopy rather than
        # reaching into it by name is what makes this correct regardless of
        # which kind it happens to be.
        undo_point = (
            _UndoPoint(
                state=copy_state(self.game.state),
                free_roads=self.game.free_roads,
                phase=self.game.phase,
                current_player=self.game.current_player,
                setup_step=self.game.setup_step,
                last_settlement=self.game.last_settlement,
                log=list(self.log),
                run=copy.deepcopy(self._run),
                actions_count=len(self._actions),
            )
            if undoable
            else None
        )
        apply(self.game, action)
        if action.type is ActionType.ROLL:
            self.last_roll_by_seat[actor] = self.game.last_roll
        self._log_action(round_num, actor, action, before)

        append_step(self._actions, self._offers, action)

        if is_over(self.game) and self.record_path and not self._recorded:
            self._recorded = True
            record = Record(
                num_players=self.game.state.num_players,
                seed=self.seed,
                actions=tuple(self._actions),
                offers=tuple(self._offers),
                winner=self.game.won_by,
                turns=self.game.turns,
                **board_fields(self.game.state.board),
            )
            write_records(self.record_path, [record])

        # Every action decides this fresh: a qualifying human placement that
        # didn't win the game becomes the new (and only) undo point, anything
        # else — a bot's move, a second placement — clears whatever was
        # there. A win is excluded because the record above may already be
        # on disk by now. Correct even for a setup road handing off to a
        # bot: apply_human_action holds that handoff's advance_bots() back
        # until confirm_setup_turn (see awaiting_confirm), so no bot action
        # ever actually runs between the human's placement and their chance
        # to undo it.
        self._undo = undo_point if (undo_point is not None and not is_over(self.game)) else None

    def undo_last_build(self) -> None:
        """Reverts the human's most recent placement back to exactly how the
        session stood the instant before it: piece removed, resources
        refunded (including a second setup settlement's grant), longest
        road/largest army recomputed from the restored board, whose turn it
        is un-advanced if the placement handed off to someone else, log line
        shortened or dropped, replay bookkeeping truncated. Only ever
        available since the human's own last placement — see _apply.
        """
        if self._undo is None:
            raise ValueError("nothing to undo")
        point = self._undo
        self.game.state = point.state
        self.game.free_roads = point.free_roads
        self.game.phase = point.phase
        self.game.current_player = point.current_player
        self.game.setup_step = point.setup_step
        self.game.last_settlement = point.last_settlement
        self.log = point.log
        self._run = point.run
        del self._actions[point.actions_count :]
        self._undo = None
        # A setup road is the only action that ever sets this, and only
        # after it runs — so whatever's being undone, it was false going
        # in. Left alone, undoing a road would leave it stuck true: to_move
        # is back to the human (the whole point), but the client would
        # still show the confirm button over the real board instead.
        self.awaiting_confirm = False

    def _run_for(self, kind: str, key: tuple, fresh: dict) -> tuple[dict, bool]:
        """The open run for `key`, or a new one seeded from `fresh`.

        Returns `(run, continuing)`. When continuing, the last line in the
        log is this run's own and the caller's rewritten line replaces it
        rather than following it — which is the whole trick: a run is always
        exactly one line, rewritten in place as it grows.

        Only ever one run open at a time. Anything that doesn't continue the
        current one replaces or clears it (see `_log_action`), so a run can
        never reach back across an intervening line to join something older
        — a second seven in the same round starts a fresh discard line
        rather than swelling the first.
        """
        run = self._run
        if run is not None and run["kind"] == kind and run["key"] == key:
            return run, True
        self._run = {"kind": kind, "key": key, **fresh}
        return self._run, False

    def _log_run(self, round_num: int, line: str, continuing: bool) -> None:
        if continuing:
            self.log.pop()
        self.log.append(f"{round_num}\t{line}")

    def _log_action(self, round_num: int, actor: int, action: Action, before: _Snapshot) -> None:
        """Appends this action's line to `self.log`.

        Three kinds of action arrive as a burst of engine steps that a reader
        would only ever want as one sentence, and each collapses into a run
        (see `_run_for`) rewritten in place as it grows:

          builds     one "placed/built ..." per actor per round
          discards   one line per actor per discard, however many cards
          bank       consecutive trades of the same pair, summed

        PROPOSE_TRADE is held back differently — buffered in `_trade_buffer`
        and only written once the whole offer concludes, as "accepted" naming
        who took it or as a uniform "Everyone declined." that never says who
        was asked or how many (see those branches for why). END_TURN writes
        nothing at all.

        Tab-separated, matching every other line: the client splits on the
        first tab for the round-number column.
        """
        kind = action.type
        who = _who(actor, self.human_seat, self.bot_names)

        if kind in _BUILD_KIND:
            verb, item = _BUILD_KIND[kind]
            run, continuing = self._run_for(
                "build", (actor, round_num), {"verb": verb, "items": [], "extra": []}
            )
            run["items"].append(item)
            if kind is ActionType.SETUP_SETTLEMENT:
                gains = _hand_gains(before.hands[actor], self.game.state.hands[actor])
                if gains:
                    run["extra"].append(f"{who} collects {gains}.")
            line = f"{who} {run['verb']} {_list_with_counts(run['items'])}."
            self._log_run(round_num, " ".join([line, *run["extra"]]), continuing)
            return

        if kind is ActionType.DISCARD:
            # The engine takes discards one card at a time (see
            # `legal_actions` under Phase.DISCARD, which deliberately keeps
            # the action space linear in resources rather than combinatorial
            # in hand size), so one seven can cost a full hand half a dozen
            # steps in a row — and for a bot every one of them said the same
            # six words. They collapse to a single line.
            #
            # Which resources went is the human's own line only. `_describe`
            # drew that line for a single discard and it matters more here,
            # not less: a collapsed line is exactly where a whole hand would
            # leak at once.
            run, continuing = self._run_for(
                "discard", (actor, round_num), {"counts": [0] * NUM_RESOURCES}
            )
            run["counts"][action.a] += 1
            total = sum(run["counts"])
            if actor == self.human_seat:
                # Same wording as the "collects" half of a roll line, since
                # it's the same fact pointed the other way.
                line = f"{who} discarded {_resource_counts(run['counts'])}."
            else:
                line = f"{who} discarded {total} card{'' if total == 1 else 's'}."
            self._log_run(round_num, line, continuing)
            return

        if kind is ActionType.BANK_TRADE:
            # Ports make this the one action people repeat back to back —
            # four wood for a wheat, then four more for another — and each
            # step was its own line. Same pair in a row sums into one; a
            # different pair is a different trade and starts its own (the
            # pair is part of the run's key).
            given = before.hands[actor][action.a] - self.game.state.hands[actor][action.a]
            run, continuing = self._run_for(
                "bank", (actor, round_num, action.a, action.b), {"given": 0, "got": 0}
            )
            run["given"] += given
            run["got"] += 1
            line = (
                f"{who} traded {run['given']} {RESOURCE_NAMES[action.a]} "
                f"for {run['got']} {RESOURCE_NAMES[action.b]} with the bank."
            )
            self._log_run(round_num, line, continuing)
            return

        self._run = None  # anything else ends whatever run was open

        if kind is ActionType.END_TURN:
            # Whatever line comes next (the following seat's roll, build,
            # ...) already implies the previous turn ended — a dedicated
            # "X ended the turn." line for every single turn was pure noise,
            # not information.
            return

        if kind is ActionType.PROPOSE_TRADE:
            line = _describe(self.game, actor, action, before, self.human_seat, self.bot_names)
            if self.game.offer is None:
                # propose_trade() found nobody eligible and concluded the
                # offer immediately, with no ACCEPT_TRADE/DECLINE_TRADE ever
                # coming — logged the same as the "everyone who was asked
                # said no" case below, for the same reason: see there.
                self.log.append(f"{round_num}\t{line} Everyone declined.")
            else:
                self._trade_buffer = line
            return

        if kind is ActionType.ACCEPT_TRADE and self._trade_buffer is not None:
            self.log.append(f"{round_num}\t{self._trade_buffer} {who} accepted.")
            self._trade_buffer = None
            return

        if kind is ActionType.DECLINE_TRADE and self._trade_buffer is not None:
            if self.game.offer is None:
                # Only who's *eligible* to cover an offer is ever asked
                # (`catan.game.propose_trade`'s own `responders`/`willing`),
                # in ask order, one at a time, stopping at the first accept
                # — so naming each individual decliner, or even just their
                # count, would tell a human exactly how many opponents held
                # what was wanted before the queue ran out. Catan hands are
                # private, so every "nobody took it" offer reads identically
                # regardless of how many were actually asked, or whether any
                # were: "Everyone declined." every time, full stop.
                self.log.append(f"{round_num}\t{self._trade_buffer} Everyone declined.")
                self._trade_buffer = None
            return

        line = _describe(self.game, actor, action, before, self.human_seat, self.bot_names)
        self.log.append(f"{round_num}\t{line}")

    def state_view(self) -> dict:
        game = self.game
        state = game.state
        over = is_over(game)
        players = []
        # Both are public — a route's length and a played Knight count are
        # visible on the board/in front of everyone, unlike hand contents.
        lengths = road_lengths(state)
        for p in range(state.num_players):
            reveal = over or p == self.human_seat
            entry = {
                "seat": p,
                "bot": self.bot_names.get(p),
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
            }
            if reveal:
                entry["hand"] = dict(zip(RESOURCE_NAMES, state.hands[p]))
                entry["dev_cards"] = dict(zip(DEV_CARD_NAMES, holdings(state, p)))
            players.append(entry)

        offer = None
        if game.offer is not None:
            offer = {
                "proposer": game.offer.proposer,
                "give": list(game.offer.give),
                "want": list(game.offer.want),
                # Deliberately no `responders`: `game.pending_responders` is
                # exactly who's eligible to cover the offer, in ask order —
                # sending it live, before anyone has actually responded,
                # would leak the same hidden hand information the log fix
                # (see _log_action) exists to hide, just earlier and over a
                # different channel. Nothing on the client reads it either.
            }

        return {
            "phase": game.phase.name,
            "current_player": game.current_player,
            "to_move": None if over else to_move(game),
            "human_seat": self.human_seat,
            "winner": game.won_by,
            "game_over": over,
            # Whether POST /api/undo would succeed right now — see
            # undo_last_build. A session convenience, not a rule, so it isn't
            # in legal_actions alongside everything catan.actions offers.
            "can_undo": self._undo is not None,
            # True while a setup road's handoff is waiting on POST
            # /api/confirm — see apply_human_action. `to_move` already moved
            # on to whoever's next by this point, so the client reads this
            # (not to_move) to know the human still has a decision to make.
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
            "trade_ratios": dict(
                zip(RESOURCE_NAMES, trade_ratios(state, self.human_seat))
            ),
            "players": players,
            "offer": offer,
            "legal_actions": self.legal_wire_actions(),
            "log": self.log,
        }
