# SPDX-License-Identifier: GPL-3.0-only
"""Mirrors a hexset `Game` as a live catanatron `Game` (`to_catanatron`).

The mirror is rebuilt fresh on every decision rather than kept in lockstep,
since the two turn machines differ in enough small ways (catanatron asks one
`PLAY_TURN` prompt on both sides of the roll) that lockstep would be more
fragile. `to_catanatron` writes the `State` fields directly rather than
replaying a game into them, because `State.__init__` shuffles seating and the
deck, neither of which a mirror may do. Player-to-player trading is not
translated: `generate_playable_actions` never emits `OFFER_TRADE` during a
normal turn, so `Phase.TRADE_RESPOND` is unreachable and `bot.py`'s entrant
declines every exchange.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
import random

from hexset.cards import DevCard
from hexset.game import Game, Phase, to_move, pending_free_roads
from hexset.state import NO_OWNER, Building, GameState

from catanatron.game import Game as CatanatronGame
from catanatron.models.actions import generate_playable_actions
from catanatron.models.board import (
    STATIC_GRAPH,
    Board as CatanatronBoard,
    longest_acyclic_path,
)
from catanatron.models.enums import CITY, ROAD, SETTLEMENT, ActionPrompt
from catanatron.models.player import Color, Player
from catanatron.state import PLAYER_INITIAL_STATE, State

from ._board import BoardMapping
from ._names import DEV_CARD_NAMES, RESOURCE_NAMES


@dataclass(frozen=True)
class Seating:
    """catanatron plays by `Color`; hexset plays by seat index. One order."""

    color_of: dict[int, Color]
    seat_of: dict[Color, int]


def seating(colors: tuple[Color, ...], seats: "Sequence[int] | None" = None) -> Seating:
    """One colour per seat. `seats` names the hexset seats being mirrored, in
    the order the colours are given, and defaults to `0..len(colors)-1`.

    It is not always every seat: a table with retired seats mirrors only the
    ones that can still act, so the reference engine searches against the
    opponents it actually has rather than against colours that never move.
    `Seating` is therefore the record of *which* seats are in the mirror, and
    the mirror's other halves read the seat list off it.
    """
    order = tuple(range(len(colors))) if seats is None else tuple(seats)
    if len(order) != len(colors):
        raise ValueError("one colour per mirrored seat")
    color_of = dict(zip(order, colors))
    seat_of = {c: s for s, c in color_of.items()}
    return Seating(color_of=color_of, seat_of=seat_of)


def mirrored_seats(seats: Seating) -> tuple[int, ...]:
    """The hexset seats this mirror holds, in colour order."""
    return tuple(sorted(seats.color_of))


# The prompt catanatron asks for each of hexset's phases. Not a bijection:
# catanatron asks `PLAY_TURN` both before and after the roll, telling them
# apart by `HAS_ROLLED`, where hexset has a phase for each.
_PHASE_TO_PROMPT = {
    Phase.SETUP_SETTLEMENT: "BUILD_INITIAL_SETTLEMENT",
    Phase.SETUP_ROAD: "BUILD_INITIAL_ROAD",
    Phase.ROBBER: "MOVE_ROBBER",
    Phase.DISCARD: "DISCARD",
    Phase.ROLL: "PLAY_TURN",
    Phase.MAIN: "PLAY_TURN",
}

_SETUP = (Phase.SETUP_SETTLEMENT, Phase.SETUP_ROAD)


def _robber_coordinate(state, mapping):
    """Where Catanatron's robber stands. Its board always holds the robber on
    a tile, so a robber still beside a desertless board (`state.OFF_BOARD`)
    has no Catanatron position: `ValueError`."""
    if state.robber < 0:
        raise ValueError("Catanatron cannot play a board whose robber has not been placed")
    return mapping.coord_of[state.robber]


def _catanatron_board(state: GameState, mapping: BoardMapping, seats: Seating):
    """catanatron's `Board` holding this position, caches and all.
    `connected_components` and `road_lengths` are recomputed through
    catanatron's own `dfs_walk` and `longest_acyclic_path`, so the answer stays
    theirs."""
    board = CatanatronBoard(mapping.catan_map)
    for vertex, owner in enumerate(state.vertex_owner):
        if owner == NO_OWNER:
            continue
        node = mapping.node_of[vertex]
        kind = SETTLEMENT if state.vertex_building[vertex] == Building.SETTLEMENT else CITY
        board.buildings[node] = (seats.color_of[owner], kind)
        board.board_buildable_ids.discard(node)
        board.board_buildable_ids.difference_update(STATIC_GRAPH.neighbors(node))
    for edge, owner in enumerate(state.edge_owner):
        if owner == NO_OWNER:
            continue
        a, b = mapping.catanatron_edge_of[edge]
        board.roads[(a, b)] = board.roads[(b, a)] = seats.color_of[owner]
    board.robber_coordinate = _robber_coordinate(state, mapping)

    for seat in mirrored_seats(seats):
        color = seats.color_of[seat]
        seeds = {node for node, (c, _) in board.buildings.items() if c == color}
        seeds.update(
            node for edge, c in board.roads.items() if c == color
            for node in edge if not board.is_enemy_node(node, color)
        )
        components: list[set[int]] = []
        for seed in sorted(seeds):
            if not any(seed in component for component in components):
                components.append(board.dfs_walk(seed, color))
        board.connected_components[color] = components
        board.road_lengths[color] = max(
            (len(longest_acyclic_path(board, c, color)) for c in components), default=0
        )

    # Who *holds* longest road is hexset's answer, not a recomputation: both
    # engines award it to the incumbent on a tie, so only history can say who.
    if state.longest_road_holder != NO_OWNER:
        board.road_color = seats.color_of[state.longest_road_holder]
        board.road_length = board.road_lengths[board.road_color]
    return board


def _player_state(game: Game, state: GameState, seats: Seating, board) -> dict:
    """catanatron's flat per-seat feature dictionary for this position."""
    out = {}
    # `P<n>` is catanatron's index into its own colour tuple, which is the
    # position in the mirrored-seat order -- not the hexset seat number. They
    # differ as soon as a seat is left out of the mirror.
    for index, seat in enumerate(mirrored_seats(seats)):
        color = seats.color_of[seat]
        pieces = Counter(
            state.vertex_building[v]
            for v, owner in enumerate(state.vertex_owner)
            if owner == seat
        )
        settlements, cities = pieces[Building.SETTLEMENT], pieces[Building.CITY]
        values = dict(PLAYER_INITIAL_STATE)
        for r, name in enumerate(RESOURCE_NAMES):
            values[f"{name}_IN_HAND"] = state.hands[seat][r]
        for card, name in DEV_CARD_NAMES.items():
            matured = state.dev_cards[seat][card]
            values[f"{name}_IN_HAND"] = matured + state.new_dev_cards[seat][card]
            # catanatron's maturity is a per-type boolean, not hexset's
            # per-copy count: any matured copy sets it, and a seat holding
            # only fresh ones cannot play.
            if f"{name}_OWNED_AT_START" in values:
                values[f"{name}_OWNED_AT_START"] = matured > 0
        values["PLAYED_KNIGHT"] = state.knights_played[seat]
        values["ROADS_AVAILABLE"] -= state.edge_owner.count(seat)
        values["SETTLEMENTS_AVAILABLE"] -= settlements
        values["CITIES_AVAILABLE"] -= cities
        values["HAS_ROAD"] = state.longest_road_holder == seat
        values["HAS_ARMY"] = state.largest_army_holder == seat
        values["LONGEST_ROAD_LENGTH"] = board.road_lengths[color]
        mine = seat == game.current_player
        values["HAS_ROLLED"] = mine and game.phase not in (*_SETUP, Phase.ROLL)
        values["HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN"] = mine and game.dev_card_played
        values["VICTORY_POINTS"] = (
            settlements + 2 * cities + 2 * values["HAS_ROAD"] + 2 * values["HAS_ARMY"]
        )
        values["ACTUAL_VICTORY_POINTS"] = (
            values["VICTORY_POINTS"] + values["VICTORY_POINT_IN_HAND"]
        )
        out.update({f"P{index}_{field}": value for field, value in values.items()})
    return out


class BoardMirrorCache:
    """Reuse road-network reconstruction until occupancy or its holder changes.
    The template stays private, since native players may mutate the board they
    are handed."""

    def __init__(self, mapping: BoardMapping, seats: Seating) -> None:
        self.mapping = mapping
        self.seats = seats
        self._key = None
        self._board = None

    def board(self, state: GameState):
        key = (
            tuple(state.vertex_owner), tuple(state.vertex_building),
            tuple(state.edge_owner), state.longest_road_holder,
        )
        if key != self._key:
            self._board = _catanatron_board(state, self.mapping, self.seats)
            self._key = key
        board = self._board.copy()
        board.robber_coordinate = _robber_coordinate(state, self.mapping)
        return board


def to_catanatron(
    game: Game, mapping: BoardMapping, seats: Seating,
    *, board_cache: BoardMirrorCache | None = None, rng: random.Random | None = None,
) -> CatanatronGame:
    """The catanatron `Game` mirroring `game` right now.

    `rng` is the mirror's random stream, shared by its `Game`, its `State`
    and every copy a search makes; without one the mirror clones `game`'s
    own, without advancing it."""
    # true state: a catanatron `Player` reads the whole table, so this is the
    # sanctioned true-state read rather than a `View`.
    state = game.state(0, hidden=False)
    colors = tuple(seats.color_of[seat] for seat in mirrored_seats(seats))

    cstate = State([], None, initialize=False)
    cstate.players = [Player(color) for color in colors]
    cstate.colors = colors
    cstate.color_to_index = {color: seat for seat, color in enumerate(colors)}
    cstate.discard_limit = state.rules.discard_limit
    cstate.friendly_robber = state.rules.friendly_robber
    cstate.board = (
        _catanatron_board(state, mapping, seats)
        if board_cache is None else board_cache.board(state)
    )
    cstate.player_state = _player_state(game, state, seats, cstate.board)
    cstate.resource_freqdeck = list(state.bank)
    cstate.development_listdeck = [DEV_CARD_NAMES[DevCard(c)] for c in state.deck]
    cstate.action_records = []
    cstate.num_turns = game.turns

    cstate.buildings_by_color = {color: defaultdict(list) for color in colors}
    for node, (color, kind) in cstate.board.buildings.items():
        cstate.buildings_by_color[color][kind].append(node)
    for edge, color in cstate.board.roads.items():
        if edge[0] < edge[1]:
            cstate.buildings_by_color[color][ROAD].append(edge)
    if game.phase is Phase.SETUP_ROAD:
        # `initial_road_possibilities` reads the *last* settlement in this
        # list, which is hexset's `last_settlement`; order is otherwise
        # immaterial.
        settlements = cstate.buildings_by_color[seats.color_of[game.current_player]][SETTLEMENT]
        last = mapping.node_of[game.last_settlement]
        settlements.remove(last)
        settlements.append(last)

    # Catanatron indexes its own colour tuple; hexset indexes seats. They
    # coincide only when every seat is mirrored, which a table with retired
    # seats is not.
    cstate.current_player_index = cstate.color_to_index[seats.color_of[to_move(game)]]
    cstate.current_turn_index = cstate.color_to_index[
        seats.color_of[game.current_player]
    ]
    cstate.current_prompt = ActionPrompt[_PHASE_TO_PROMPT[game.phase]]
    cstate.is_initial_build_phase = game.phase in _SETUP
    cstate.is_discarding = game.phase is Phase.DISCARD
    cstate.discard_counts = [
        game.discard_quota[seat] if seat < len(game.discard_quota) else 0
        for seat in mirrored_seats(seats)
    ]
    cstate.is_moving_knight = game.phase is Phase.ROBBER
    cstate.is_road_building = game.free_roads > 0
    cstate.free_roads_available = game.free_roads
    cstate.is_resolving_trade = False
    cstate.current_trade = (0,) * 11
    cstate.acceptees = tuple(False for _ in colors)

    cgame = CatanatronGame([], initialize=False)
    cgame.seed = 0
    # Arena callers supply the entrant's own stream; standalone translation
    # clones the host's without advancing live chance draws. Game, State and
    # search copies intentionally share it.
    if rng is None:
        rng = random.Random(0)
        rng.setstate(game.rng.getstate())
    cgame.random = cstate.random = rng
    cgame.id = ""
    cgame.vps_to_win = state.rules.winning_points
    cgame.friendly_robber = state.rules.friendly_robber
    cgame.state = cstate
    cgame.playable_actions = generate_playable_actions(cstate)
    if not cgame.playable_actions and cstate.is_road_building and not pending_free_roads(game):
        # HexSet lets unplaceable free-road credit expire, where catanatron's
        # road-building prompt would then offer nothing at all. Repairs only
        # that case; ordinary offers are unchanged.
        cstate.is_road_building = False
        cstate.free_roads_available = 0
        cgame.playable_actions = generate_playable_actions(cstate)
    return cgame
