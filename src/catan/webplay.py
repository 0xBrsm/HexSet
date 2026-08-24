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

import math
from dataclasses import dataclass, field
from typing import Protocol

from .actions import (
    YEAR_OF_PLENTY_PAIRS,
    Action,
    ActionType,
    apply,
    legal_actions,
)
from .board.board import Board
from .board.coords import Hex
from .board.terrain import NUM_RESOURCES, TERRAIN_RESOURCE, Resource
from .board.topology import Topology
from .cards import NUM_DEV_CARDS, DevCard
from .devcards import holdings
from .economy import trade_ratios
from .game import Game, Phase, is_over, to_move
from .record import Record, append_step, board_fields, write as write_records
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


def _hand_gains(before: list[int], after: list[int]) -> str | None:
    parts = [
        f"{after[r] - before[r]} {RESOURCE_NAMES[r]}"
        for r in range(NUM_RESOURCES)
        if after[r] != before[r]
    ]
    return ", ".join(parts) if parts else None


@dataclass
class _Snapshot:
    hands: list[list[int]]
    held: list[list[int]]
    offer: object


def _snapshot(game: Game) -> _Snapshot:
    state = game.state
    return _Snapshot(
        hands=[hand[:] for hand in state.hands],
        held=[holdings(state, p)[:] for p in range(state.num_players)],
        offer=game.offer,
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

    if kind is ActionType.END_TURN:
        return f"{who} ended the turn."

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

    if kind is ActionType.BANK_TRADE:
        given = before.hands[actor][action.a] - state.hands[actor][action.a]
        return (
            f"{who} traded {given} {RESOURCE_NAMES[action.a]} "
            f"for 1 {RESOURCE_NAMES[action.b]} with the bank."
        )

    if kind is ActionType.DISCARD:
        if actor == human_seat:
            return f"{who} discarded 1 {RESOURCE_NAMES[action.a]}."
        return f"{who} discarded a card."

    if kind is ActionType.PROPOSE_TRADE:
        return f"{who} offered {_bundle_text(action.give)} for {_bundle_text(action.want)}."

    if kind is ActionType.ACCEPT_TRADE:
        offer = before.offer
        if offer is None:
            return f"{who} accepted a trade."
        proposer = _who(offer.proposer, human_seat, bot_names)
        return (
            f"{who} accepted {proposer}'s trade: "
            f"{proposer} gives {_bundle_text(offer.give)} and gets {_bundle_text(offer.want)}."
        )

    if kind is ActionType.DECLINE_TRADE:
        return f"{who} declined the trade."

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
        return [action_to_wire(a) for a in legal_actions(self.game)]

    def apply_human_action(self, wire: dict) -> None:
        if is_over(self.game):
            raise ValueError("the game is already over")
        if to_move(self.game) != self.human_seat:
            raise ValueError("it is not your turn to act")
        action = wire_to_action(wire)
        options = legal_actions(self.game)
        if action not in options:
            raise ValueError(f"{action} is not a legal action right now")
        self._apply(self.human_seat, action)

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
            if action not in options:
                raise RuntimeError(f"bot chose an action legal_actions did not offer: {action}")
            self._apply(seat, action)
            steps += 1

    def _apply(self, actor: int, action: Action) -> None:
        # Captured before apply(), not after: end_turn() increments
        # game.turns (and so self.round, derived from it), so the line for
        # the END_TURN action itself would otherwise be prefixed with the
        # *next* round's number instead of the one that just ended.
        round_num = self.round
        before = _snapshot(self.game)
        apply(self.game, action)
        if action.type is ActionType.ROLL:
            self.last_roll_by_seat[actor] = self.game.last_roll
        line = _describe(self.game, actor, action, before, self.human_seat, self.bot_names)
        # Tab-separated rather than "Round N: " — the client splits on the
        # first tab and puts the number in its own fixed-width column so
        # every line's text lines up regardless of digit count, which a
        # baked-in "Round N: " prefix of varying length couldn't do.
        self.log.append(f"{round_num}\t{line}")

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

    def state_view(self) -> dict:
        game = self.game
        state = game.state
        over = is_over(game)
        players = []
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
                "responders": list(game.pending_responders),
            }

        return {
            "phase": game.phase.name,
            "current_player": game.current_player,
            "to_move": None if over else to_move(game),
            "human_seat": self.human_seat,
            "winner": game.won_by,
            "game_over": over,
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
