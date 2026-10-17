# SPDX-License-Identifier: GPL-3.0-only
"""Turn a game state into the heterogeneous graph the model reads.

*Seat-relative*: seats are rotated so the perspective seat -- the player to
move unless `encode` is told another -- is always seat 0.
*Information-set correct*: own hand and development cards are exact; every
other seat's hand is its expected count of each resource on the perspective's
own view (`hexset.view.View`: `hexset.ledger`'s rows from public events and
the steals the perspective saw), and its development cards a count, their
ages, and the odds of each kind among the cards the perspective has not seen.
Nothing of trading or robber deals is read: those stay with the trade gate.

Record contract 8.1: contract 8 with the table's rules and the next roll's
odds appended to the globals, and each seat's points read against the
points the game is played to. Closeness reads (`1 / (1 + n)`: a seat's distance to a
vertex, to Longest Road, to Largest Army) read 1 at the goal and 0 where it is
out of reach, so 0 is the far end of the scale, never a stand-in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Sequence

import numpy as np

from .board.board import MAX_ROLL, MIN_ROLL, Board, pips
from .board.terrain import NUM_RESOURCES, Terrain
from .board.topology import Topology
from .cards import NUM_DEV_CARDS, DECK_SIZE, DevCard
from .devcards import dev_count
from .game import Game, Phase, roll_odds, to_move
from .roads import reach_all, roads_to_take_all
from .state import NO_OWNER, Building, GameState
from .victory import MIN_LARGEST_ARMY, award_points
from .view import View

__all__ = [
    "NUM_TERRAIN",
    "NUM_BUILDINGS",
    "NUM_PHASES",
    "MAX_TOKEN_PIPS",
    "BANK_SCALE",
    "HAND_SCALE",
    "TURN_SCALE",
    "DEV_SCALE",
    "AGE_SCALE",
    "AGE_SLOTS",
    "AGE_FILL",
    "PLAYABLE_KINDS",
    "ENCODED_PHASES",
    "RULE_BLOCKS",
    "dev_age_rows",
    "knights_needed_all",
    "expected_hands",
    "StaticGraph",
    "static_graph",
    "Observation",
    "HEX_FEATURES",
    "vertex_features",
    "edge_features",
    "global_features",
    "global_columns",
    "to_frame",
    "from_frame",
    "encode_batch",
    "encode",
]


NUM_TERRAIN = len(Terrain)
NUM_BUILDINGS = len(Building)
NUM_PHASES = len(Phase)

MAX_TOKEN_PIPS = 5
BANK_SCALE = 19.0
HAND_SCALE = 10.0
TURN_SCALE = 200.0
DEV_SCALE = 5.0
POINTS_SCALE = 10.0
FREE_ROADS_SCALE = 2.0
# A held card's age, in its holder's own turns, reads (age + 1) / AGE_SCALE,
# so a card bought this turn reads above 0, which means no card.
AGE_SCALE = 20.0
# How many of each opponent's longest-held cards are read, oldest first.
AGE_SLOTS = 3
# Where a seat's card ages are unknown (a state built without its history),
# each card it holds reads the typical age of a card in that slot: unknown,
# not absent.
AGE_FILL = (5.2, 3.1, 2.2)
# The kinds a seat can play, in `DevCard` order: every kind but the VP card.
PLAYABLE_KINDS = (DevCard.KNIGHT, DevCard.ROAD_BUILDING, DevCard.YEAR_OF_PLENTY, DevCard.MONOPOLY)
# `Phase` one-hot columns: every phase a decision is made in, so all but
# `GAME_OVER`, the last.
ENCODED_PHASES = NUM_PHASES - 1


@dataclass(frozen=True)
class StaticGraph:
    """Adjacency that depends only on the board, so it is built once per board
    and kept apart from features, which change at every node of a search."""

    num_hexes: int
    num_vertices: int
    num_edges: int
    hex_vertex: np.ndarray
    vertex_edge: np.ndarray
    hex_hex: np.ndarray
    vertex_vertex: np.ndarray


def _pairs(groups) -> np.ndarray:
    out = [(i, j) for i, members in enumerate(groups) for j in members]
    return np.array(out, dtype=np.int64).reshape(-1, 2).T


@lru_cache(maxsize=8)
def static_graph(topology: Topology) -> StaticGraph:
    """The `StaticGraph` of `topology`, cached per topology."""
    return StaticGraph(
        num_hexes=topology.num_hexes,
        num_vertices=topology.num_vertices,
        num_edges=topology.num_edges,
        hex_vertex=_pairs(topology.hex_vertices),
        vertex_edge=_pairs(topology.vertex_edges),
        hex_hex=_pairs(topology.hex_neighbors),
        vertex_vertex=_pairs(topology.vertex_neighbors),
    )


@dataclass(frozen=True)
class Observation:
    """One position as the network reads it (`encode`): per-hex, per-vertex,
    per-edge and global feature arrays over the board's `graph`."""

    hexes: np.ndarray
    vertices: np.ndarray
    edges: np.ndarray
    globals: np.ndarray
    graph: StaticGraph
    # Set by `encode_batch`: the one `(positions, width)` buffer the whole
    # batch was written into and this position's row in it, for a consumer
    # that moves a batch as one block. Not pickled (`__reduce__`).
    _packed: np.ndarray | None = field(default=None, repr=False, compare=False)
    _row: int = field(default=-1, repr=False, compare=False)

    def __reduce__(self):
        """Serialize only this position, never the other rows from its tick."""
        return (
            Observation,
            (self.hexes, self.vertices, self.edges, self.globals, self.graph),
        )

    @property
    def shapes(self) -> dict[str, tuple[int, ...]]:
        return {
            "hexes": self.hexes.shape,
            "vertices": self.vertices.shape,
            "edges": self.edges.shape,
            "globals": self.globals.shape,
        }

# Terrain columns: the five that produce, in `Resource` order (`Terrain`'s
# first five); the desert, sea and gold set none.
HEX_TERRAIN = NUM_RESOURCES
HEX_FEATURES = HEX_TERRAIN + 2
_PIPS = HEX_TERRAIN
_ROBBER = HEX_TERRAIN + 1


def vertex_features(players: int) -> int:
    """Width of a vertex row for `players` seats: per seat its building here
    (1 a settlement, 2 a city), a generic port and one port per resource, and
    per seat its reach (`roads.reach_all`, read `1 / (1 + d)`, 0 unreachable)."""
    return players + 1 + NUM_RESOURCES + players


def edge_features(players: int) -> int:
    """Width of an edge row for `players` seats: per seat, 1 for its road here."""
    return players


def _port_base(players: int) -> int:
    return players


def _reach_base(players: int) -> int:
    return players + 1 + NUM_RESOURCES


def _global_blocks(players: int) -> list[tuple[str, int]]:
    """Every block `_encode_globals` writes, in write order, named and sized.
    `own_` is the perspective, `opponent_` each other seat in seat-relative
    order, `seat_` every seat with the perspective first, no prefix the table."""
    opponents = players - 1
    return [
        ("own_hand", NUM_RESOURCES),
        ("own_dev", NUM_DEV_CARDS),
        ("own_dev_playable", len(PLAYABLE_KINDS)),
        ("opponent_hand", opponents * NUM_RESOURCES),
        ("opponent_dev_count", opponents),
        ("opponent_dev_ages", opponents * AGE_SLOTS),
        ("seat_points", players),
        ("seat_longest_road", players),
        ("seat_largest_army", players),
        ("opponent_to_move", opponents),
        ("bank", NUM_RESOURCES),
        ("deck", 1),
        ("dev_odds", NUM_DEV_CARDS),
        ("phase", ENCODED_PHASES),
        ("free_roads", 1),
        ("turn", 1),
        # Contract 8.1, appended so every contract-8 column keeps its place.
        ("rules", 4),
        ("roll_odds", MAX_ROLL - MIN_ROLL + 1),
    ]


#: The blocks contract 8.1 appends to contract 8's globals, in order: a
#: contract-8 net reads exactly the columns before them.
RULE_BLOCKS: tuple[str, ...] = ("rules", "roll_odds")


def global_features(players: int) -> int:
    """Width of `Observation.globals` for `players` seats; `global_columns`
    names its blocks."""
    return sum(width for _, width in _global_blocks(players))


def global_columns(players: int) -> dict[str, slice]:
    """Name every block of `global_features(players)` as a `slice` into
    `Observation.globals`, tiling it exactly: contiguous and non-overlapping."""
    columns: dict[str, slice] = {}
    offset = 0
    for name, width in _global_blocks(players):
        columns[name] = slice(offset, offset + width)
        offset += width
    return columns


def _seat(seat: int, perspective: int, players: int) -> int:
    """Rotate so the perspective player is seat 0."""
    return (seat - perspective) % players


def to_frame(values: Sequence[float], seat: int) -> tuple[float, ...]:
    """Board-order values rotated into `seat`'s perspective frame: slot `i` is
    board seat `(seat + i) % players`, so `seat` lands on slot 0. Reversing the
    direction still type-checks and silently swaps whose number is whose."""
    players = len(values)
    return tuple(values[(seat + i) % players] for i in range(players))


def from_frame(values: Sequence[float], seat: int) -> tuple[float, ...]:
    """Inverse of `to_frame`: a perspective-frame vector (slot 0 is `seat`) back
    in board-seat order. `from_frame(to_frame(v, seat), seat) == tuple(v)`."""
    players = len(values)
    return tuple(
        values[(board_seat - seat) % players] for board_seat in range(players)
    )


# --- rules-level reads, shared with `hexset.onnx_record` ----------------------


def dev_age_rows(state: GameState) -> np.ndarray:
    """`(players, AGE_SLOTS)` float: per seat, the ages of its longest-held
    development cards (`GameState.dev_ages`, in its own turns), oldest first,
    `-1` past the cards it holds. A seat whose ages the state was built
    without reads `AGE_FILL` for each card it holds."""
    players = state.num_players
    out = np.full((players, AGE_SLOTS), -1.0, dtype=np.float32)
    for seat in range(players):
        ages = state.dev_ages[seat]
        if ages is None:
            held = dev_count(state, seat)
            for k in range(min(held, AGE_SLOTS)):
                out[seat, k] = AGE_FILL[k]
            continue
        for k, age in enumerate(sorted(ages, reverse=True)[:AGE_SLOTS]):
            out[seat, k] = age
    return out


def knights_needed_all(state: GameState, perspective: int, unseen_knights: int) -> tuple[int, ...]:
    """Per seat, in seat order, the knights it must still play to take
    Largest Army: 0 for the holder, otherwise enough to beat the holder and
    at least `MIN_LARGEST_ARMY`. `-1` where the perspective can prove it out
    of reach: more than the knights it has not seen (`unseen_knights`), plus,
    for the perspective itself, its own."""
    holder = state.largest_army_holder
    played = state.knights_played
    target = MIN_LARGEST_ARMY if holder == NO_OWNER else max(MIN_LARGEST_ARMY, played[holder] + 1)
    out = []
    for seat in range(state.num_players):
        if seat == holder:
            out.append(0)
            continue
        need = max(1, target - played[seat])
        can = unseen_knights
        if seat == perspective:
            can += state.dev_cards[seat][DevCard.KNIGHT] + state.new_dev_cards[seat][DevCard.KNIGHT]
        out.append(need if need <= can else -1)
    return tuple(out)


def expected_hands(view: View) -> np.ndarray:
    """`(players, NUM_RESOURCES)` float: every seat's expected count of each
    resource on `view` (`View.expected_counts`), the perspective's own hand
    exact."""
    return np.asarray(
        [view.expected_counts(seat) for seat in range(view.num_players)], dtype=np.float64
    )


def _closeness(n: int) -> float:
    """`1 / (1 + n)`, and 0 for `-1`, out of reach."""
    return 0.0 if n < 0 else 1.0 / (1.0 + n)


# --- board blocks -------------------------------------------------------------


@dataclass(frozen=True)
class _Template:
    """The part of an observation no move can change: terrain, pips, ports."""

    hexes: np.ndarray
    vertices: np.ndarray


# One entry per board, and a vectorised collector holds one board per lane, so
# this must outsize the widest lane count in use or every call misses and
# rebuilds the template the cache exists to avoid. A template is about 3 KB.
@lru_cache(maxsize=4096)
def _template_by_value(board: Board, players: int) -> _Template:
    hexes = np.zeros((board.num_hexes, HEX_FEATURES), dtype=np.float32)
    for h in range(board.num_hexes):
        terrain = int(board.terrain[h])
        if terrain < HEX_TERRAIN:
            hexes[h, terrain] = 1.0
        hexes[h, _PIPS] = pips(board.tokens[h]) / MAX_TOKEN_PIPS

    port_base = _port_base(players)
    vertices = np.zeros(
        (board.topology.num_vertices, vertex_features(players)), dtype=np.float32
    )
    for port in board.ports:
        column = port_base if port.resource is None else port_base + 1 + int(port.resource)
        for v in port.vertices:
            vertices[v, column] = 1.0

    # Handed out by reference, so a stray write would corrupt every later
    # encode on this board rather than one observation.
    hexes.flags.writeable = False
    vertices.flags.writeable = False
    return _Template(hexes=hexes, vertices=vertices)


_TEMPLATE_IDENTITIES: dict[tuple[int, int], tuple[Board, _Template]] = {}


def _template(board: Board, players: int) -> _Template:
    """Board template, keyed by identity before structural hashing: hashing a
    `Board` walks the whole topology, and a collector re-asks the same board."""
    key = (id(board), players)
    cached = _TEMPLATE_IDENTITIES.get(key)
    if cached is not None and cached[0] is board:
        return cached[1]

    template = _template_by_value(board, players)
    if len(_TEMPLATE_IDENTITIES) >= 4096:
        _TEMPLATE_IDENTITIES.pop(next(iter(_TEMPLATE_IDENTITIES)))
    _TEMPLATE_IDENTITIES[key] = (board, template)
    return template


def _slots(players: int, perspective: int) -> list[int]:
    """Seat column per raw owner value, with NO_OWNER last -- being -1 it lands
    on the last row under numpy's negative indexing, without a branch."""
    return [_seat(owner, perspective, players) for owner in range(players)] + [players]


def _vertex_key(building: int, owner: int, players: int) -> int:
    """One index standing for a vertex's building and owner together."""
    return building * (players + 1) + owner + 1


@lru_cache(maxsize=32)
def _vertex_rows(players: int, perspective: int) -> np.ndarray:
    """`rows[key]` is the `seat_building` block of a vertex: the building's
    points (`Building`'s value: 1 a settlement, 2 a city) in its owner's
    seat-relative column."""
    slots = _slots(players, perspective)
    rows = np.zeros((NUM_BUILDINGS * (players + 1), players), dtype=np.float32)
    for building in range(1, NUM_BUILDINGS):
        for owner in range(players):
            rows[_vertex_key(building, owner, players), slots[owner]] = float(building)
    rows.flags.writeable = False
    return rows


@lru_cache(maxsize=32)
def _edge_rows(players: int, perspective: int) -> np.ndarray:
    """`rows[owner]` is the whole feature row of an edge; `NO_OWNER`'s, the
    last, is all 0."""
    rows = np.zeros((players + 1, edge_features(players)), dtype=np.float32)
    for owner, slot in enumerate(_slots(players, perspective)[:players]):
        rows[owner, slot] = 1.0
    rows.flags.writeable = False
    return rows


@lru_cache(maxsize=8)
def _vertex_rows_all(players: int) -> np.ndarray:
    """The vertex lookup tables stacked for batched perspective indexing."""
    rows = np.stack([_vertex_rows(players, seat) for seat in range(players)])
    rows.flags.writeable = False
    return rows


@lru_cache(maxsize=8)
def _edge_rows_all(players: int) -> np.ndarray:
    """The edge lookup tables stacked for batched perspective indexing."""
    rows = np.stack([_edge_rows(players, seat) for seat in range(players)])
    rows.flags.writeable = False
    return rows


@lru_cache(maxsize=1 << 12)
def _reach_block(reach: tuple[tuple[int, ...], ...], perspective: int) -> np.ndarray:
    """`(vertices, players)` `seat_reach` block, seat-relative, from
    `roads.reach_all`."""
    players = len(reach)
    out = np.zeros((len(reach[0]), players), dtype=np.float32)
    for i, seat in enumerate(to_frame(range(players), perspective)):
        d = np.asarray(reach[seat], dtype=np.float32)
        out[:, i] = np.where(d < 0, 0.0, 1.0 / (1.0 + np.maximum(d, 0.0)))
    out.flags.writeable = False
    return out


def _encode_hexes(state: GameState, template: _Template) -> np.ndarray:
    out = template.hexes.copy()
    if state.robber >= 0:  # `OFF_BOARD` marks no hex
        out[state.robber, _ROBBER] = 1.0
    return out


def _encode_vertices(
    state: GameState, perspective: int, template: _Template
) -> np.ndarray:
    players = state.num_players
    keys = np.asarray(
        [
            _vertex_key(building, owner, players)
            for building, owner in zip(state.vertex_building, state.vertex_owner)
        ],
        dtype=np.intp,
    )
    out = template.vertices.copy()
    out[:, :players] = _vertex_rows(players, perspective)[keys]
    base = _reach_base(players)
    out[:, base : base + players] = _reach_block(reach_all(state), perspective)
    return out


def _encode_edges(state: GameState, perspective: int) -> np.ndarray:
    owners = np.asarray(state.edge_owner, dtype=np.intp)
    return _edge_rows(state.num_players, perspective)[owners]


# --- globals ------------------------------------------------------------------


def _global_parts(game: Game, perspective: int, building_points: Sequence[float]) -> list[float]:
    """Every global block but the ones `encode_batch` writes as arrays, in
    `_global_blocks` order: the single source of their semantics, so the
    batched path stays byte-identical to `encode` by construction."""
    state = game._state
    rules = state.rules
    players = state.num_players
    seats = to_frame(range(players), perspective)
    view = View.from_game(game, perspective)
    unseen = view.unseen_dev_cards()

    parts: list[float] = []
    parts.extend(n / HAND_SCALE for n in state.hands[perspective])
    held = state.dev_cards[perspective]
    parts.extend(
        (h + f) / DEV_SCALE for h, f in zip(held, state.new_dev_cards[perspective])
    )
    parts.extend(
        0.0 if game.dev_card_played else held[kind] / DEV_SCALE for kind in PLAYABLE_KINDS
    )

    # The float blocks are rounded in float32 from the record's float32
    # values, as `hexn.export_onnx.RecordEncoder` computes them.
    hands = expected_hands(view).astype(np.float32)
    for s in seats[1:]:
        parts.extend(hands[s] / np.float32(HAND_SCALE))
    parts.extend(dev_count(state, s) / DEV_SCALE for s in seats[1:])
    ages = dev_age_rows(state)
    for s in seats[1:]:
        parts.extend(
            np.where(ages[s] < 0, np.float32(0.0), (ages[s] + np.float32(1.0)) / np.float32(AGE_SCALE))
        )

    parts.extend(
        (building_points[i] + award_points(state, s)) / rules.winning_points
        for i, s in enumerate(seats)
    )
    roads = roads_to_take_all(state)
    parts.extend(_closeness(roads[s]) for s in seats)
    knights = knights_needed_all(state, perspective, unseen[DevCard.KNIGHT])
    parts.extend(_closeness(knights[s]) for s in seats)
    mover = _seat(to_move(game), perspective, players)
    parts.extend(1.0 if mover == i else 0.0 for i in range(1, players))

    parts.extend(n / BANK_SCALE for n in state.bank)
    parts.append(len(state.deck) / DECK_SIZE)
    total = sum(unseen)
    parts.extend(n / total if total else 0.0 for n in unseen)
    phase = int(game.phase)
    parts.extend(1.0 if phase == i else 0.0 for i in range(ENCODED_PHASES))
    parts.append(game.free_roads / FREE_ROADS_SCALE)
    parts.append(min(game.turns / TURN_SCALE, 1.0))

    # The table's rules, which every seat is told: the points the game is
    # played to (10 reads 1), the hand size above which a seven discards, the
    # friendly robber and the balanced dice.
    parts.append(rules.winning_points / POINTS_SCALE)
    parts.append(rules.discard_limit / HAND_SCALE)
    parts.append(float(rules.friendly_robber))
    parts.append(float(rules.balanced_dice))
    # The chance of each sum 2..12 on the next roll of the seat on turn: the
    # plain two-dice odds, or under balanced dice the deck's, read off the
    # public rolls (`game.roll_odds`).
    odds = dict(roll_odds(game))
    parts.extend(odds.get(total, 0.0) for total in range(MIN_ROLL, MAX_ROLL + 1))
    return parts


def _building_points(vertices: np.ndarray, players: int) -> np.ndarray:
    """Every seat's building victory points, in seat-relative order: the
    `seat_building` block summed over the board, since 1 and 2 are both the
    `Building` enum and the points it scores."""
    return vertices[..., :players].sum(axis=-2)


def encode_batch(
    games: Sequence[Game], perspectives: Sequence[int]
) -> list[Observation]:
    """Encode a collector tick with one set of vector operations, pinned to the
    plain single-position `encode` as its oracle."""
    if len(games) != len(perspectives):
        raise ValueError("one perspective is required per game")
    if not games:
        return []

    states = [game._state for game in games]
    players = states[0].num_players
    if any(state.num_players != players for state in states):
        raise ValueError("one batch cannot mix player counts")

    perspective = np.asarray(perspectives, dtype=np.intp)
    if np.any((perspective < 0) | (perspective >= players)):
        raise ValueError("a perspective does not name a player")

    batch = len(games)
    rows = np.arange(batch)
    templates = [_template(state.board, players) for state in states]

    shapes = (
        (states[0].board.num_hexes, HEX_FEATURES),
        (states[0].board.topology.num_vertices, vertex_features(players)),
        (states[0].board.topology.num_edges, edge_features(players)),
        (global_features(players),),
    )
    packed = np.empty(
        (batch, sum(int(np.prod(shape)) for shape in shapes)), dtype=np.float32
    )
    blocks = []
    start = 0
    for shape in shapes:
        stop = start + int(np.prod(shape))
        block = packed[:, start:stop].reshape(batch, *shape)
        if block.base is None:
            raise AssertionError("packed observation slice did not stay a view")
        blocks.append(block)
        start = stop
    hexes, vertices, edges, globals_ = blocks

    np.stack([template.hexes for template in templates], out=hexes)
    robbers = np.asarray([state.robber for state in states], dtype=np.intp)
    placed = robbers >= 0  # `OFF_BOARD` marks no hex
    hexes[rows[placed], robbers[placed], _ROBBER] = 1.0

    buildings = np.asarray(
        [state.vertex_building for state in states], dtype=np.intp
    )
    owners = np.asarray([state.vertex_owner for state in states], dtype=np.intp)
    keys = buildings * (players + 1) + owners + 1
    np.stack([template.vertices for template in templates], out=vertices)
    vertices[:, :, :players] = _vertex_rows_all(players)[perspective[:, None], keys]
    base = _reach_base(players)
    for i, (state, p) in enumerate(zip(states, perspective)):
        vertices[i, :, base : base + players] = _reach_block(reach_all(state), int(p))

    edge_owners = np.asarray([state.edge_owner for state in states], dtype=np.intp)
    edges[:] = _edge_rows_all(players)[perspective[:, None], edge_owners]

    building_points = _building_points(vertices, players)
    for i, (game, p) in enumerate(zip(games, perspective)):
        globals_[i] = np.asarray(_global_parts(game, int(p), building_points[i]), dtype=np.float32)

    graph = static_graph(states[0].board.topology)
    return [
        Observation(hexes[i], vertices[i], edges[i], globals_[i], graph, packed, i)
        for i in range(batch)
    ]


def encode(game: Game, perspective: int | None = None) -> Observation:
    """Encode the position as seen by `perspective`, defaulting to the mover."""
    state = game._state
    if perspective is None:
        perspective = to_move(game)
    if not 0 <= perspective < state.num_players:
        raise ValueError(f"no such player: {perspective}")

    template = _template(state.board, state.num_players)
    vertices = _encode_vertices(state, perspective, template)
    building_points = _building_points(vertices, state.num_players)

    return Observation(
        hexes=_encode_hexes(state, template),
        vertices=vertices,
        edges=_encode_edges(state, perspective),
        globals=np.asarray(_global_parts(game, perspective, building_points), dtype=np.float32),
        graph=static_graph(state.board.topology),
    )
