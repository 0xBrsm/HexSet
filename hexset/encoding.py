# SPDX-License-Identifier: GPL-3.0-only
"""Turn a game state into the heterogeneous graph the model reads.

*Seat-relative*: seats are rotated so the perspective seat -- the player to
move unless `encode` is told another -- is always seat 0.
*Information-set correct*: own hand and development cards are exact, opponents
contribute counts plus what `hexset.ledger` reconstructs from public events.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Sequence

import numpy as np

from .board.board import Board, pips
from .board.terrain import NUM_RESOURCES, Terrain
from .board.topology import Topology
from .cards import NUM_DEV_CARDS, DECK_SIZE
from .devcards import dev_count
from .economy import hand_size
from .game import Game, Phase, to_move
from .state import NO_OWNER, Building, GameState

__all__ = [
    "NUM_TERRAIN",
    "NUM_BUILDINGS",
    "NUM_PHASES",
    "MAX_TOKEN_PIPS",
    "BANK_SCALE",
    "HAND_SCALE",
    "TURN_SCALE",
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


HEX_FEATURES = NUM_TERRAIN + 3


def vertex_features(players: int) -> int:
    """Width of a vertex row for `players` seats: the building, the
    seat-relative owner or nobody, a generic port, and one resource port per
    resource."""
    return NUM_BUILDINGS + (players + 1) + 1 + NUM_RESOURCES


def edge_features(players: int) -> int:
    """Width of an edge row for `players` seats: the seat-relative road owner
    or nobody."""
    return players + 1


def _global_blocks(players: int) -> list[tuple[str, int]]:
    """Every block `_encode_globals` writes, in write order, named and sized."""
    return [
        ("own_hand", NUM_RESOURCES),
        ("opponent_hand_sizes", players - 1),
        ("bank", NUM_RESOURCES),
        ("own_dev_cards", NUM_DEV_CARDS),
        ("opponent_dev_card_counts", players - 1),
        ("knights_played", players),
        ("victory_points", players),
        ("longest_road_holder", players + 1),
        ("largest_army_holder", players + 1),
        ("phase", NUM_PHASES),
        ("free_roads", 1),
        ("deck_size", 1),
        ("turn", 1),
        # known[5] + unknown per opponent.
        ("ledger", (players - 1) * (NUM_RESOURCES + 1)),
    ]


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


@dataclass(frozen=True)
class _Template:
    """The part of an observation no move can change: terrain, tokens, pips, ports."""

    hexes: np.ndarray
    vertices: np.ndarray


# One entry per board, and a vectorised collector holds one board per lane, so
# this must outsize the widest lane count in use or every call misses and
# rebuilds the template the cache exists to avoid. A template is about 3 KB.
@lru_cache(maxsize=4096)
def _template_by_value(board: Board, players: int) -> _Template:
    hexes = np.zeros((board.num_hexes, HEX_FEATURES), dtype=np.float32)
    for h in range(board.num_hexes):
        hexes[h, int(board.terrain[h])] = 1.0
        token = board.tokens[h]
        hexes[h, NUM_TERRAIN] = 1.0 if token else 0.0
        hexes[h, NUM_TERRAIN + 1] = pips(token) / MAX_TOKEN_PIPS

    port_base = NUM_BUILDINGS + players + 1
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
    """`rows[key]` is the building and owner block of a vertex."""
    width = NUM_BUILDINGS + players + 1
    slots = _slots(players, perspective)
    rows = np.zeros((NUM_BUILDINGS * (players + 1), width), dtype=np.float32)
    for building in range(NUM_BUILDINGS):
        for owner in range(NO_OWNER, players):
            key = _vertex_key(building, owner, players)
            rows[key, building] = 1.0
            rows[key, NUM_BUILDINGS + slots[owner]] = 1.0
    rows.flags.writeable = False
    return rows


@lru_cache(maxsize=32)
def _edge_rows(players: int, perspective: int) -> np.ndarray:
    """`rows[owner]` is the whole feature row of an edge."""
    rows = np.zeros((players + 1, edge_features(players)), dtype=np.float32)
    for owner, slot in enumerate(_slots(players, perspective)):
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


_BUILDING_VALUE = np.arange(NUM_BUILDINGS, dtype=np.float32)


def _building_points(vertices: np.ndarray, players: int) -> np.ndarray:
    """Every seat's building victory points, in seat-relative order: a
    contraction of the encoded vertex block, since 1 and 2 are both the
    `Building` enum and the points it scores."""
    unowned = NUM_BUILDINGS + players
    return (vertices[:, :NUM_BUILDINGS] @ _BUILDING_VALUE) @ vertices[
        :, NUM_BUILDINGS:unowned
    ]


def _encode_hexes(state: GameState, template: _Template) -> np.ndarray:
    out = template.hexes.copy()
    if state.robber >= 0:  # `OFF_BOARD` marks no hex
        out[state.robber, NUM_TERRAIN + 2] = 1.0
    return out


def _encode_vertices(
    state: GameState, perspective: int, template: _Template
) -> np.ndarray:
    players = state.num_players
    span = players + 1
    keys = np.asarray(
        [
            building * span + owner + 1
            for building, owner in zip(state.vertex_building, state.vertex_owner)
        ],
        dtype=np.intp,
    )
    out = template.vertices.copy()
    out[:, : NUM_BUILDINGS + span] = _vertex_rows(players, perspective)[keys]
    return out


def _encode_edges(state: GameState, perspective: int) -> np.ndarray:
    owners = np.asarray(state.edge_owner, dtype=np.intp)
    return _edge_rows(state.num_players, perspective)[owners]


def _ledger_parts(game: Game, perspective: int) -> list[float]:
    """Each opponent's reconstructed hand composition (`hexset.ledger`), in
    seat-relative order, own seat excluded: per opponent `known[5]` scaled like a
    hand then `unknown`, so `(players - 1) * (NUM_RESOURCES + 1)` floats."""
    players = game._state.num_players
    parts: list[float] = []
    for seat in to_frame(range(players), perspective)[1:]:
        seat_ledger = game.ledger.seats[seat]
        parts.extend(k / HAND_SCALE for k in seat_ledger.known)
        parts.append(seat_ledger.unknown / HAND_SCALE)
    return parts


def _encode_globals(
    game: Game, perspective: int, building_points: np.ndarray
) -> np.ndarray:
    from .victory import award_points

    state = game._state
    players = state.num_players
    seats = to_frame(range(players), perspective)

    parts: list[float] = []
    parts.extend(n / HAND_SCALE for n in state.hands[perspective])
    parts.extend(hand_size(state, s) / HAND_SCALE for s in seats[1:])
    parts.extend(n / BANK_SCALE for n in state.bank)

    own_cards = [
        held + fresh
        for held, fresh in zip(
            state.dev_cards[perspective], state.new_dev_cards[perspective]
        )
    ]
    parts.extend(n / 5.0 for n in own_cards)
    parts.extend(dev_count(state, s) / 5.0 for s in seats[1:])

    parts.extend(state.knights_played[s] / 5.0 for s in seats)
    parts.extend(
        (building_points[i] + award_points(state, s)) / 10.0
        for i, s in enumerate(seats)
    )

    for holder in (state.longest_road_holder, state.largest_army_holder):
        slot = players if holder == NO_OWNER else _seat(holder, perspective, players)
        one_hot = [0.0] * (players + 1)
        one_hot[slot] = 1.0
        parts.extend(one_hot)

    phase = [0.0] * NUM_PHASES
    phase[int(game.phase)] = 1.0
    parts.extend(phase)

    parts.append(game.free_roads / 2.0)
    parts.append(len(state.deck) / DECK_SIZE)
    parts.append(min(game.turns / TURN_SCALE, 1.0))

    parts.extend(_ledger_parts(game, perspective))

    return np.array(parts, dtype=np.float32)


def _encode_globals_batch(
    games: Sequence[Game], perspectives: np.ndarray, building_points: np.ndarray
) -> np.ndarray:
    """Write the small global blocks once per batch instead of once per game."""
    batch = len(games)
    players = games[0]._state.num_players
    rows = np.arange(batch)[:, None]
    seats = (perspectives[:, None] + np.arange(players)) % players

    hands = np.asarray([game._state.hands for game in games], dtype=np.int16)
    banks = np.asarray([game._state.bank for game in games], dtype=np.int16)
    cards = np.asarray([game._state.dev_cards for game in games], dtype=np.int16)
    fresh = np.asarray(
        [game._state.new_dev_cards for game in games], dtype=np.int16
    )
    knights = np.asarray(
        [game._state.knights_played for game in games], dtype=np.int16
    )

    out = np.zeros((batch, global_features(players)), dtype=np.float32)
    cursor = 0

    def append(values: np.ndarray, scale: float) -> None:
        nonlocal cursor
        block = np.asarray(values, dtype=np.float64)
        if block.ndim == 1:
            block = block[:, None]
        width = block.shape[1]
        out[:, cursor : cursor + width] = block / scale
        cursor += width

    append(hands[np.arange(batch), perspectives], HAND_SCALE)
    hand_sizes = hands.sum(axis=2)
    append(hand_sizes[rows, seats[:, 1:]], HAND_SCALE)
    append(banks, BANK_SCALE)

    all_cards = cards + fresh
    append(all_cards[np.arange(batch), perspectives], 5.0)
    card_counts = all_cards.sum(axis=2)
    append(card_counts[rows, seats[:, 1:]], 5.0)
    append(knights[rows, seats], 5.0)

    longest = np.asarray(
        [game._state.longest_road_holder for game in games], dtype=np.intp
    )
    army = np.asarray(
        [game._state.largest_army_holder for game in games], dtype=np.intp
    )
    awards = 2 * (longest[:, None] == seats) + 2 * (army[:, None] == seats)
    append(building_points.astype(np.float64) + awards, 10.0)

    batch_rows = np.arange(batch)
    for holders in (longest, army):
        slots = np.where(
            holders == NO_OWNER,
            players,
            (holders - perspectives) % players,
        )
        out[batch_rows, cursor + slots] = 1.0
        cursor += players + 1

    phases = np.asarray([int(game.phase) for game in games], dtype=np.intp)
    out[batch_rows, cursor + phases] = 1.0
    cursor += NUM_PHASES

    append(np.asarray([game.free_roads for game in games]), 2.0)
    append(np.asarray([len(game._state.deck) for game in games]), DECK_SIZE)
    turns = np.minimum(
        np.asarray([game.turns for game in games], dtype=np.float64) / TURN_SCALE,
        1.0,
    )
    append(turns, 1.0)

    # `_ledger_parts` is the single source of the ledger block's semantics, so
    # this fast path stays byte-identical to the oracle by construction.
    ledger_block = np.asarray(
        [_ledger_parts(game, int(p)) for game, p in zip(games, perspectives)],
        dtype=np.float64,
    )
    append(ledger_block, 1.0)

    if cursor != out.shape[1]:
        raise AssertionError(f"wrote {cursor} global features into {out.shape[1]}")
    return out


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
    hexes[rows[placed], robbers[placed], NUM_TERRAIN + 2] = 1.0

    buildings = np.asarray(
        [state.vertex_building for state in states], dtype=np.intp
    )
    owners = np.asarray([state.vertex_owner for state in states], dtype=np.intp)
    keys = buildings * (players + 1) + owners + 1
    np.stack([template.vertices for template in templates], out=vertices)
    dynamic = NUM_BUILDINGS + players + 1
    vertices[:, :, :dynamic] = _vertex_rows_all(players)[
        perspective[:, None], keys
    ]

    edge_owners = np.asarray([state.edge_owner for state in states], dtype=np.intp)
    edges[:] = _edge_rows_all(players)[perspective[:, None], edge_owners]

    building_value = vertices[:, :, :NUM_BUILDINGS] @ _BUILDING_VALUE
    building_points = np.sum(
        building_value[:, :, None]
        * vertices[:, :, NUM_BUILDINGS : NUM_BUILDINGS + players],
        axis=1,
    )
    globals_[:] = _encode_globals_batch(games, perspective, building_points)

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

    return Observation(
        hexes=_encode_hexes(state, template),
        vertices=vertices,
        edges=_encode_edges(state, perspective),
        globals=_encode_globals(
            game, perspective, _building_points(vertices, state.num_players)
        ),
        graph=static_graph(state.board.topology),
    )
